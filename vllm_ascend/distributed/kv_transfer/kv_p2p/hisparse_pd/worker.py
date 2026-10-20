# SPDX-License-Identifier: Apache-2.0
"""Worker transfer orchestration; all KV allocation stays upstream."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from .configuration import validate_transport_options
from .layout import build_layouts
from .protocol import VERSION, LayerLayout, LeaseRegistry, encode_layouts, plan_reads
from .transport import LeaseServer, MemfabricReader, TransferCancelled, acquire, rpc

logger = logging.getLogger(__name__)

# Keep failed workers, their real cache owners and the live ACK server until a
# successful close. Worker lifecycle operations run on the engine thread; the
# single receive thread may also retain its own worker when a read fails.
_retained_workers: dict[int, HiSparsePDWorker] = {}


class HiSparsePDWorker:
    def __init__(self, config, kv_cache_config, *, identity: str, producer: bool):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        extra = config.kv_transfer_config.kv_connector_extra_config
        self.config = config
        self.kv_cache_config = kv_cache_config
        self.identity = identity
        self.producer = producer
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = config.parallel_config.tensor_parallel_size
        self.device = torch.npu.current_device()
        self.host = extra["host"]
        self.port = int(extra["control_port"]) + self.rank
        self.peer_host = extra.get("producer_host", self.host)
        self.peer_port = int(extra.get("producer_control_port", extra["control_port"])) + self.rank
        self.protocol, self.urma_swap_space_mib = validate_transport_options(extra)
        self.timeout = float(extra.get("transfer_timeout_seconds", 120))
        self.max_descriptors = int(extra.get("max_descriptors_per_read", 1024))
        if not 0 < self.max_descriptors <= 65536 or not 0 < self.timeout <= 3600:
            raise ValueError("Invalid HiSparse P/D transfer limits")
        self.registry = LeaseRegistry()
        self.server = None
        self.reader = None
        self.layouts = []
        self.owners = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hisparse-pd-read")
        self.pending = []
        self.futures = {}
        self.closed = False
        self.closing = False

    def register(self, kv_caches):
        if self.reader is not None or self.closing or self.closed:
            raise RuntimeError("HiSparse P/D caches can only be registered once")
        context = self.config.compilation_config.static_forward_context
        self.layouts, regions, self.owners = build_layouts(
            kv_caches, self.kv_cache_config, context, producer=self.producer
        )
        self.reader = MemfabricReader(
            producer=self.producer,
            device=self.device,
            host=self.host,
            protocol=self.protocol,
            regions=regions,
            owners=self.owners,
            urma_swap_space_mib=self.urma_swap_space_mib,
        )
        try:
            self.reader.initialize()
            if self.producer:
                self.server = LeaseServer(f"tcp://{self.host}:{self.port}", self.registry)
        except BaseException as initialize_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise initialize_error from cleanup_error
            raise

    def queue(self, metadata):
        if self.closed or self.closing:
            raise RuntimeError("HiSparse P/D worker is closing")
        self.pending.append(metadata)

    def poll(self, cancelled_ids: set[str]):
        if self.closed or self.closing:
            raise RuntimeError("HiSparse P/D worker is closing or has shut down")
        # Called after all MultiConnector start_load/finish_forward hooks, so
        # the existing HiSparse worker has already submitted host mirrors and
        # host block copies. Eager-only: drain these before exporting pointers.
        for metadata in self.pending:
            if metadata.exports:
                torch.npu.synchronize(self.device)
            for export in metadata.exports:
                manifest = {
                    "version": VERSION,
                    "identity": self.identity,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "session": self.reader.session,
                    "ticket": export["ticket"],
                    "blocks": export["blocks"],
                    "layers": encode_layouts(self.layouts),
                }
                self.registry.publish(export["ticket"]["lease_id"], export["request_id"], manifest)
            for job in metadata.imports:
                request_id = job["request_id"]
                if request_id in self.futures:
                    raise RuntimeError("Duplicate HiSparse P/D load")
                cancelled = threading.Event()
                future = self.executor.submit(self._load, job, cancelled)
                self.futures[request_id] = future, cancelled
        self.pending.clear()
        received, failed = set(), set()
        for request_id, (future, cancelled) in list(self.futures.items()):
            if request_id in cancelled_ids:
                cancelled.set()
            if not future.done():
                continue
            success = future.result()  # DMA errors remain fatal; never fake a drained completion.
            received.add(request_id)
            if not success:
                failed.add(request_id)
            del self.futures[request_id]
        return self.registry.take_finished(), received, failed

    def _load(self, job, cancelled):
        try:
            torch.npu.set_device(self.device)
            return self._load_request(job, cancelled)
        except BaseException:
            _retained_workers[id(self)] = self
            raise

    def _load_request(self, job, cancelled):
        endpoint = f"tcp://{self.peer_host}:{self.peer_port}"
        ticket = job["ticket"]
        consumer = f"{job['request_id']}:{self.rank}:{uuid.uuid4().hex}"
        message = {
            "lease_id": ticket["lease_id"],
            "consumer": consumer,
            "identity": self.identity,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        try:
            manifest = acquire(endpoint, message, self.timeout, cancelled)
            if manifest["version"] != VERSION or manifest["ticket"] != ticket:
                raise ValueError("Producer ticket differs from the requested prefix")
            remote = [LayerLayout(**item) for item in manifest["layers"]]
            source_blocks = {int(key): value for key, value in manifest["blocks"].items()}
            local_blocks = {int(key): value for key, value in job["blocks"].items()}
            descriptors = plan_reads(
                remote,
                self.layouts,
                source_blocks,
                local_blocks,
                job["start_token"],
                ticket["num_tokens"],
                include_host=not self.kv_cache_config.hisparse_shared_host_pool or self.rank == 0,
            )
        except TransferCancelled:
            return False  # No acquire was sent and no DMA could have started.
        except (TimeoutError, ValueError, KeyError, TypeError) as error:
            # Even a lost acquire reply may have claimed the producer lease.
            # No read has been submitted yet, so releasing that claim is safe.
            try:
                rpc(endpoint, {"op": "release", **message, "drained": True}, min(self.timeout, 5))
            except (TimeoutError, ValueError):
                logger.warning("HiSparse P/D pre-read failure left an unconfirmed producer lease: %s", error)
            return False
        # Each call is synchronous. Only a successful return proves that its
        # remote reads no longer own the source. Do not ACK an uncertain error.
        for first in range(0, len(descriptors), self.max_descriptors):
            if cancelled.is_set():
                break
            self.reader.read(manifest["session"], descriptors[first : first + self.max_descriptors])
        torch.npu.synchronize(self.device)
        try:
            response = rpc(endpoint, {"op": "release", **message, "drained": True}, self.timeout)
            if response["status"] != "released":
                raise ValueError("Producer did not acknowledge drained transfer")
        except (TimeoutError, ValueError):
            # All local reads are known drained. Report a failed request through
            # the common async-load path instead of crashing the engine. The
            # producer conservatively retains an unconfirmed acquired lease.
            logger.warning("HiSparse P/D completed its reads but could not confirm producer lease release")
            return False
        return not cancelled.is_set()

    def close(self):
        if self.closed:
            return
        self.closing = True
        transfer_error = None
        try:
            for _, cancelled in self.futures.values():
                cancelled.set()
            self.executor.shutdown(wait=True, cancel_futures=False)
            for future, _ in self.futures.values():
                try:
                    future.result()
                except BaseException as error:
                    if transfer_error is None:
                        transfer_error = error
            # A failed read still needs both device drain and complete transport
            # teardown. Do not let future.result() prevent those safe retries.
            torch.npu.set_device(self.device)
            torch.npu.synchronize(self.device)
            # Failed admission leaves the ACK server operational for active
            # remote readers. Only a drained ACK can release their source lease.
            self.registry.begin_shutdown()
            if self.server is not None:
                self.server.close()
            if self.reader is not None:
                self.reader.close(device_drained=True)
        except BaseException as cleanup_error:
            _retained_workers[id(self)] = self
            if transfer_error is not None:
                raise transfer_error from cleanup_error
            raise
        self.closed = True
        self.owners.clear()
        self.pending.clear()
        self.futures.clear()
        _retained_workers.pop(id(self), None)
        if transfer_error is not None:
            # Memory is now safe to release, but the request still failed.
            raise transfer_error
