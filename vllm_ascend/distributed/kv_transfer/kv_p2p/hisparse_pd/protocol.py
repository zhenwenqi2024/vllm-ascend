# SPDX-License-Identifier: Apache-2.0
"""Versioned, JSON-only HiSparse P/D descriptors and transfer leases.

These leases retain a finished producer request through the upstream connector
contract. They do not allocate KV blocks or implement a second cache manager.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class LayerLayout:
    name: str
    group_id: int
    role: str
    block_size: int
    row_bytes: int
    base: int
    block_stride: int
    num_blocks: int

    def __post_init__(self):
        if self.role not in ("hisparse_source", "hisparse_indexer"):
            raise ValueError(f"Unsupported transfer role {self.role}")
        if min(self.block_size, self.row_bytes, self.base, self.block_stride, self.num_blocks) <= 0:
            raise ValueError("Invalid or empty KV address geometry")
        if self.block_stride < self.block_size * self.row_bytes:
            raise ValueError("KV pages overlap")
        if self.base + (self.num_blocks - 1) * self.block_stride + self.block_size * self.row_bytes >= 2**64:
            raise ValueError("KV address exceeds uint64")

    def signature(self) -> tuple:
        # Capacities, addresses and HMA block/layer strides may differ on P/D.
        return self.name, self.group_id, self.role, self.block_size, self.row_bytes

    def address(self, block: int, offset: int = 0) -> int:
        if not 0 <= block < self.num_blocks or not 0 <= offset < self.block_size:
            raise ValueError(f"Out-of-range KV page for {self.name}")
        return self.base + block * self.block_stride + offset * self.row_bytes


def validate_layouts(layouts: list[LayerLayout]) -> None:
    names = [layer.name for layer in layouts]
    if not names or len(names) != len(set(names)):
        raise ValueError("A transfer must contain each layer exactly once")
    roles = {layer.role for layer in layouts}
    if roles != {"hisparse_source", "hisparse_indexer"}:
        raise ValueError("Both SOURCE and INDEXER layers are required")


def plan_reads(
    source: list[LayerLayout],
    destination: list[LayerLayout],
    source_blocks: dict[int, list[int]],
    destination_blocks: dict[int, list[int]],
    start_token: int,
    end_token: int,
    *,
    include_host: bool = True,
) -> list[tuple[int, int, int]]:
    """Return (destination, source, bytes) without int32 address arithmetic.

    Copy only the requested token interval, including a partial tail. The
    scheduler's group IDs index manager blocks, never expanded indexer pages.
    """
    validate_layouts(source)
    validate_layouts(destination)
    if start_token < 0 or end_token <= start_token:
        raise ValueError("The imported token interval must be nonempty")
    remote = {layer.name: layer for layer in source}
    if remote.keys() != {layer.name for layer in destination}:
        raise ValueError("Producer and consumer layer sets differ")
    result = []
    for dst in destination:
        src = remote[dst.name]
        if src.signature() != dst.signature():
            raise ValueError(f"Incompatible KV layer {dst.name}")
        source_ids = source_blocks[dst.group_id]
        destination_ids = destination_blocks[dst.group_id]
        required = (end_token + dst.block_size - 1) // dst.block_size
        if min(len(source_ids), len(destination_ids)) < required:
            raise ValueError(f"Missing block IDs for {dst.name}")
        token = start_token
        while token < end_token:
            page, offset = divmod(token, dst.block_size)
            rows = min(dst.block_size - offset, end_token - token)
            src_ptr = src.address(source_ids[page], offset)
            dst_ptr = dst.address(destination_ids[page], offset)
            if include_host or dst.role != "hisparse_source":
                result.append((dst_ptr, src_ptr, rows * dst.row_bytes))
            token += rows
    if not result:
        raise ValueError("No KV payload was selected for transfer")
    return result


class LeaseRegistry:
    """Retain source ownership until an explicit, drained consumer release.

    A timeout is not proof that a DMA stopped. There is deliberately no TTL
    eviction of acquired leases. A dead peer requires engine/process teardown.
    """

    def __init__(self, unclaimed_timeout: float = 300, ack_retry_window: float = 600, max_tombstones: int = 4096):
        self._lock = threading.Lock()
        self._leases: dict[str, dict] = {}
        self._released: dict[str, tuple[str, float]] = {}
        self._finished: set[str] = set()
        self.unclaimed_timeout = unclaimed_timeout
        self.ack_retry_window = ack_retry_window
        self.max_tombstones = max_tombstones
        self._closing = False

    def _prune_tombstones(self):
        now = time.monotonic()
        for lease_id, (_, expiry) in list(self._released.items()):
            if expiry <= now:
                del self._released[lease_id]
        while len(self._released) > self.max_tombstones:
            del self._released[next(iter(self._released))]

    def publish(self, lease_id: str, request_id: str, manifest: dict) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Producer transport is closing")
            self._prune_tombstones()
            if lease_id in self._leases or lease_id in self._released:
                raise ValueError("Duplicate export lease")
            self._leases[lease_id] = {
                "request_id": request_id,
                "manifest": manifest,
                "consumer": None,
                "expires": time.monotonic() + self.unclaimed_timeout,
            }

    def _expire_unclaimed(self):
        for lease_id, lease in list(self._leases.items()):
            if lease["consumer"] is None and time.monotonic() >= lease["expires"]:
                self._finished.add(lease["request_id"])
                del self._leases[lease_id]

    def acquire(self, lease_id: str, consumer: str, identity: str, rank: int, world_size: int) -> dict:
        with self._lock:
            if self._closing:
                raise ValueError("Producer transport is closing")
            self._expire_unclaimed()
            lease = self._leases.get(lease_id)
            if lease is None:
                return {"status": "pending"}
            manifest = lease["manifest"]
            if (manifest["identity"], manifest["rank"], manifest["world_size"]) != (identity, rank, world_size):
                raise ValueError("Model, rank, or TP size mismatch")
            if not consumer or lease["consumer"] not in (None, consumer):
                raise ValueError("Export lease is already claimed by a different consumer")
            lease["consumer"] = consumer
            return {"status": "ready", "manifest": manifest}

    def release(self, lease_id: str, consumer: str, *, drained: bool) -> None:
        with self._lock:
            self._prune_tombstones()
            if self._released.get(lease_id, (None, 0))[0] == consumer:
                return  # A lost reply can be retried without a duplicate completion.
            lease = self._leases.get(lease_id)
            if lease is None or lease["consumer"] != consumer or not drained:
                raise ValueError("Only the owning, drained reader may release a lease")
            self._finished.add(lease["request_id"])
            self._released[lease_id] = consumer, time.monotonic() + self.ack_retry_window
            self._prune_tombstones()
            del self._leases[lease_id]

    def take_finished(self) -> set[str]:
        with self._lock:
            self._expire_unclaimed()
            result, self._finished = self._finished, set()
            return result

    def active(self) -> int:
        with self._lock:
            return sum(lease["consumer"] is not None for lease in self._leases.values())

    def begin_shutdown(self):
        """Atomically exclude new claims, or leave the ACK server operational."""
        with self._lock:
            if any(lease["consumer"] is not None for lease in self._leases.values()):
                raise RuntimeError("HiSparse P/D still has source leases; drain the peer before cache teardown")
            self._closing = True
            self._leases.clear()  # Unclaimed pointers were never exposed to a reader.


def encode_layouts(layouts: list[LayerLayout]) -> list[dict]:
    validate_layouts(layouts)
    return [asdict(layer) for layer in layouts]
