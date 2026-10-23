# SPDX-License-Identifier: Apache-2.0
"""One explicitly leased producer HBM slot; no timeout can recycle it."""

from __future__ import annotations

from .protocol import SOURCE_TRANSPORT, STAGE_BYTES, VERSION, LeaseRegistry


class StageSlot:
    def __init__(self, registry: LeaseRegistry, address: int, capacity: int, copy_source):
        if type(address) is not int or address <= 0 or type(capacity) is not int or capacity != STAGE_BYTES:
            raise ValueError("Producer staging requires one real 16 MiB HBM allocation")
        if address + capacity >= 2**64 or not callable(copy_source):
            raise ValueError("Invalid producer stage allocation or copy callback")
        self.registry = registry
        self.address, self.capacity = address, capacity
        self.copy_source = copy_source
        self._lock = registry._lock
        self._current = None
        self._generation = 0
        self._released = {}
        self._closing = False

    @staticmethod
    def _key(message):
        chunk = message.get("chunk_id")
        if not isinstance(chunk, str) or not 0 < len(chunk) <= 128:
            raise ValueError("Invalid producer stage chunk identity")
        return tuple(
            message[key]
            for key in (
                "lease_id",
                "consumer",
                "identity",
                "rank",
                "world_size",
                "layer",
                "source",
                "length",
                "chunk_id",
            )
        )

    @staticmethod
    def _position(chunk):
        pieces = chunk.split(":")
        if len(pieces) != 2 or any(not item.isascii() or not item.isdecimal() for item in pieces):
            raise ValueError("Stage chunk identity must be descriptor_index:byte_offset")
        position = tuple(int(item) for item in pieces)
        if any(str(value) != text for value, text in zip(position, pieces)) or position[1] % STAGE_BYTES:
            raise ValueError("Stage chunk identity is not canonical")
        return position

    def prepare(self, message):
        with self._lock:
            if self._closing:
                raise ValueError("Producer stage is closing")
            self.registry.validate_source_claim(message)
            key = self._key(message)
            position = self._position(key[-1])
            if self._current is not None:
                if self._current["key"] != key:
                    if self._current["key"][-1] == key[-1] and self._current["key"][:2] == key[:2]:
                        raise ValueError("Stage chunk identity was reused for different bytes")
                    return {"status": "pending"}
                if self._current["state"] != "ready":
                    raise ValueError("Producer stage copy is uncertain; slot remains retained")
                return dict(self._current["response"])
            previous = self.registry.stage_position(key[0], key[1])
            if previous is not None and position <= previous:
                raise ValueError("Stage chunk has already been released")
            self._generation += 1
            response = {
                "status": "ready",
                "version": VERSION,
                "source_transport": SOURCE_TRANSPORT,
                "stage_address": self.address,
                "capacity": self.capacity,
                "generation": self._generation,
                "chunk_id": key[-1],
                "length": message["length"],
            }
            self.registry._stage_claim = key[:2]
            self._current = {"key": key, "state": "copying", "response": response}
            try:
                # Reserve before copying. The callback sets the control thread's
                # device/stream and synchronizes H2D before this reply is sent.
                self.copy_source(message["layer"], message["source"], message["length"])
            except BaseException as error:
                self._current["state"] = "poisoned"
                raise ValueError("Producer stage copy failed; slot and source lease remain retained") from error
            self._current["state"] = "ready"
            return dict(response)

    def release(self, message):
        with self._lock:
            self.registry.validate_source_claim(message)
            key = self._key(message)
            generation = message.get("generation")
            if type(generation) is not int or message.get("drained") is not True:
                raise ValueError("Stage release requires an exact generation and drained read")
            chunk_identity = key[0], key[1], key[-1]
            if self._released.get(chunk_identity) == (key, generation):
                return {"status": "stage_released", "generation": generation, "chunk_id": key[-1]}
            if (
                self._current is None
                or self._current["key"] != key
                or self._current["response"]["generation"] != generation
                or self._current["state"] != "ready"
            ):
                raise ValueError("Only this stage generation's owning drained consumer may release it")
            self._released[chunk_identity] = key, generation
            # A constant-size per-reader high-water mark forbids reuse even after
            # the bounded release-reply history below has been pruned.
            self.registry.record_stage_release(key[0], key[1], self._position(key[-1]))
            while len(self._released) > 4096:
                del self._released[next(iter(self._released))]
            self._current = None
            self.registry._stage_claim = None
            return {"status": "stage_released", "generation": generation, "chunk_id": key[-1]}

    def begin_shutdown(self):
        with self._lock:
            self._closing = True
            if self._current is not None:
                raise RuntimeError("Producer stage is still owned by a peer; drain it before shutdown")
