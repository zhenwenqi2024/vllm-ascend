# SPDX-License-Identifier: Apache-2.0
"""Existing MemFabric data path with a small JSON/ZMQ lease control plane."""

from __future__ import annotations

import threading
import time

import zmq

from .configuration import DEFAULT_URMA_SWAP_MIB, configure_urma_staging, parse_urma_swap_mib
from .protocol import STAGE_BYTES, LeaseRegistry, validate_wire

# Failure-only process owners. Each reader serializes mutation with its lock;
# entries disappear only after both transport teardown stages have succeeded.
# Keeping the actual owners here is necessary if an engine drops its connector.
_retained_readers: dict[int, MemfabricReader] = {}


class TransferCancelled(Exception):
    """Cancellation before any transport read; safe to fail the async load."""


class LeaseServer:
    def __init__(self, endpoint: str, registry: LeaseRegistry, stage=None):
        self.registry = registry
        self.stage = stage
        self.endpoint = endpoint
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="hisparse-pd-control", daemon=True)
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError("HiSparse P/D control server failed to bind") from self.error

    def _run(self):
        try:
            with zmq.Context() as context, context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, 8 * 1024 * 1024)
                socket.bind(self.endpoint)
                self.ready.set()
                while not self.stop.is_set():
                    if not socket.poll(100):
                        continue
                    try:
                        message = socket.recv_json()
                        validate_wire(message)
                        if message["op"] == "lookup":
                            lookup = getattr(self.registry, "lookup_wire", None)
                            if lookup is None:
                                raise ValueError("This producer does not publish shared snapshots")
                            response = lookup(message)
                        elif message["op"] == "acquire":
                            response = self.registry.acquire(
                                message["lease_id"],
                                message["consumer"],
                                message["identity"],
                                message["rank"],
                                message["world_size"],
                            )
                        elif message["op"] == "release":
                            self.registry.release(
                                message["lease_id"], message["consumer"], drained=message["drained"] is True
                            )
                            response = {"status": "released"}
                        elif message["op"] in ("stage_prepare", "stage_release"):
                            if self.stage is None:
                                raise ValueError("Producer HBM staging is not available")
                            method = self.stage.prepare if message["op"] == "stage_prepare" else self.stage.release
                            response = method(message)
                        else:
                            raise ValueError("Unknown HiSparse P/D control operation")
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        response = {"status": "error", "error": str(exc)}
                    socket.send_json(response)
        except BaseException as exc:
            self.error = exc
            self.ready.set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("HiSparse P/D control server did not stop")


def rpc(endpoint: str, message: dict, timeout_seconds: float) -> dict:
    with zmq.Context() as context, context.socket(zmq.REQ) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
        socket.connect(endpoint)
        socket.send_json(message)
        if not socket.poll(max(1, int(timeout_seconds * 1000))):
            raise TimeoutError("HiSparse P/D control request timed out; source lease remains retained")
        result = socket.recv_json()
        if result.get("status") == "error":
            raise ValueError(result["error"])
        return result


def acquire(endpoint: str, message: dict, timeout_seconds: float, cancelled: threading.Event) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while not cancelled.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Producer did not publish its completed KV before the deadline")
        result = rpc(endpoint, {"op": "acquire", **message}, min(remaining, 5))
        if result["status"] == "ready":
            return result["manifest"]
        if result["status"] != "pending":
            raise ValueError("Unexpected producer lease state")
        cancelled.wait(min(remaining, 0.1))
    raise TransferCancelled("HiSparse P/D request was cancelled before acquisition")


def prepare_stage(endpoint, message, timeout_seconds):
    """Poll a busy slot only. An uncertain prepare/reply keeps its source lease."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Producer stage remains owned; no timeout reuse is permitted")
        response = rpc(endpoint, {"op": "stage_prepare", **message}, min(remaining, 5))
        if response.get("status") == "ready":
            validate_wire(response)
            if (
                response.get("chunk_id") != message["chunk_id"]
                or response.get("length") != message["length"]
                or type(response.get("generation")) is not int
                or response["generation"] <= 0
                or type(response.get("stage_address")) is not int
                or response["stage_address"] <= 0
                or response.get("capacity") != STAGE_BYTES
            ):
                raise ValueError("Producer stage response differs from the bounded request")
            return response
        if response.get("status") != "pending":
            raise ValueError("Unexpected producer stage state")
        time.sleep(min(remaining, 0.01))


class MemfabricReader:
    """Register only real HBM; Host SOURCE ranges never enter MemFabric.

    Producer HBM staging is separate from MemFabric's own 16 MiB URMA scratch.
    Consumer SOURCE destinations remain raw Host addresses. Whole real owners
    remain retained until device drain plus both engine teardown stages pass.
    """

    def __init__(
        self,
        *,
        producer: bool,
        device: int,
        host: str,
        protocol: str,
        regions: dict,
        owners: list,
        urma_swap_space_mib=DEFAULT_URMA_SWAP_MIB,
    ):
        # Construction only captures owners. The worker must store this reader
        # before initialize() can register pointers or raise during rollback.
        self.producer = producer
        self.device = device
        self.host = host
        self.protocol = protocol
        if protocol != "device_urma":
            raise ValueError("HiSparse P/D mapped-to-host transport requires device_urma")
        self.urma_swap_space_mib = parse_urma_swap_mib(urma_swap_space_mib)
        self.staging_config = None
        self.regions = dict(regions)
        self.owners = list(owners)
        self.engine = None
        self.session = None
        self.initialized = False
        self.closed = False
        self._started = False
        self._closing = False
        self._destroyed = False
        self._uninitialized = False
        self._reads_started = False
        self._failed_reads = []
        self._lock = threading.RLock()

    def _teardown_methods(self):
        raw_engine = self.engine._engine
        methods = tuple(getattr(raw_engine, name, None) for name in ("destroy", "unInitialize"))
        if not all(callable(method) for method in methods):
            raise RuntimeError("MemFabric requires both destroy and unInitialize before borrowing HiSparse mappings")
        return methods

    def initialize(self):
        with self._lock:
            if self._started or self.closed:
                raise RuntimeError("MemFabric reader cannot be initialized twice")
            self._started = True
            try:
                self.staging_config = configure_urma_staging(self.urma_swap_space_mib)
                from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import global_memfabric_te

                global_memfabric_te.configure(
                    role="Prefill" if self.producer else "Decode",
                    device_id=self.device,
                    transfer_protocol=self.protocol,
                )
                self.engine = global_memfabric_te.get_transfer_engine(self.host)
                self.session = global_memfabric_te.unique_id
                self._teardown_methods()  # Reject incomplete APIs before registration.
                for (address, size), role in self.regions.items():
                    if role != "hisparse_source":
                        if self.engine.register_memory(address, size) != 0:
                            raise RuntimeError(
                                f"MemFabric cannot register {role} allocation; no alternate source address is attempted"
                            )
                self.initialized = True
            except BaseException as initialize_error:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise initialize_error from cleanup_error
                raise

    def read(self, session: str, descriptors: list[tuple[int, int, int]]) -> None:
        with self._lock:
            if not self.initialized or self.closed or self._closing:
                raise RuntimeError("MemFabric reader is not available for reads")
            if not descriptors:
                raise ValueError("Empty transfer")
            destinations, sources, lengths = map(list, zip(*descriptors))
            self._reads_started = True
            try:
                if self.engine.batch_transfer_sync_read(session, destinations, sources, lengths) != 0:
                    raise RuntimeError("MemFabric read failed; source and destination ownership must be retained")
            except BaseException:
                # A failed synchronous API is not proof that partially queued
                # work has drained. Do not reuse this reader or its descriptors.
                self._failed_reads.append((session, destinations, sources, lengths))
                self._closing = True
                _retained_readers[id(self)] = self
                raise

    def close(self, *, device_drained: bool = False):
        with self._lock:
            if self.closed:
                return
            self._closing = True
            try:
                if self._reads_started and not device_drained:
                    raise RuntimeError("Device drain must succeed before MemFabric read owners can be released")
                if self.engine is not None:
                    destroy, uninitialize = self._teardown_methods()
                    if not self._destroyed:
                        result = destroy()
                        if result not in (None, 0):
                            raise RuntimeError(f"MemFabric destroy returned {result}")
                        self._destroyed = True
                    if not self._uninitialized:
                        result = uninitialize()
                        if result not in (None, 0):
                            raise RuntimeError(f"MemFabric unInitialize returned {result}")
                        self._uninitialized = True
            except BaseException:
                _retained_readers[id(self)] = self
                raise
            self.closed = True
            self.initialized = False
            self._failed_reads.clear()
            self.regions.clear()
            self.owners.clear()
            _retained_readers.pop(id(self), None)
