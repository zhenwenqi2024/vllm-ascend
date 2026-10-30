# SPDX-License-Identifier: Apache-2.0
"""PTA queued Host mirrors retaining payloads until the common DMA fence."""

from __future__ import annotations

from pathlib import Path

import torch


class QueuedHostDMA:
    """Use the caller's stream and completion event; allocate no new stream."""

    MAX_DESCRIPTORS = 256

    def __init__(self):
        import vllm_ascend

        library = Path(vllm_ascend.__file__).parent / "ops/hisparse_pta_host2511.so"
        if not hasattr(torch.ops.hisparse_pta_host2511, "batch_copy"):
            torch.ops.load_library(str(library))
        self.extension = torch.ops.hisparse_pta_host2511
        self.cookie = id(self)
        self.pending = {}
        self.uncertain = False
        self.binding = None
        self.owners = []
        self.device = None
        self.submitted_batches = self.completed_batches = 0
        self.last_statuses = []

    def _bind(self, owners):
        binding = tuple(
            (t.device, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
            for t in owners
        )
        if self.binding is not None:
            if binding != self.binding:
                raise RuntimeError("Host DMA payload binding changed before shutdown")
            return
        retained = []
        devices = set()
        storages = set()
        for tensor in owners:
            if tensor.device.type == "cpu":
                if not tensor.is_contiguous():
                    raise ValueError("Host mirror payload must be contiguous")
                # Keep the actual registered Host extent, not allocator padding.
                retained.append(tensor.view(torch.uint8).reshape(-1))
            elif tensor.device.type == "npu":
                storage = tensor.untyped_storage()
                key = (tensor.device, storage.data_ptr(), storage.nbytes())
                devices.add(tensor.device.index)
                if key not in storages:
                    storages.add(key)
                    # Packed resident layer views are strided. Retain an alias
                    # of their real storage without copying or changing layout.
                    alias = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    alias.set_(storage, 0, (storage.nbytes(),), (1,))
                    retained.append(alias)
            else:
                raise ValueError("Unsupported Host mirror payload device")
        if len(devices) != 1 or not any(t.device.type == "cpu" for t in retained):
            raise ValueError("Host DMA needs Host owners and exactly one NPU")
        self.device = next(iter(devices))
        self.owners = retained
        self.binding = binding

    def submit(self, source, destination, sizes, owners, completion_event):
        if self.uncertain or completion_event in self.pending:
            raise RuntimeError("Uncertain DMA or reused completion event")
        if source.ndim != 1 or destination.shape != source.shape or sizes.shape != source.shape:
            raise ValueError("Host DMA descriptor vectors differ")
        if source.numel() == 0:
            self.pending[completion_event] = []
            return
        self._bind(owners)
        tickets = []
        self.pending[completion_event] = tickets
        try:
            for offset in range(0, source.numel(), self.MAX_DESCRIPTORS):
                end = offset + self.MAX_DESCRIPTORS
                # The existing C++ bridge snapshots descriptors before enqueue,
                # detaches payload owners and retains both until release_copy.
                ticket = self.extension.batch_copy(
                    source[offset:end],
                    destination[offset:end],
                    sizes[offset:end],
                    1,
                    self.owners,
                    self.device,
                    self.cookie,
                )
                tickets.append(ticket)
                self.submitted_batches += 1
        except BaseException:
            # A throwing enqueue may have created a ticket before returning it.
            # Discoverable C++ state and all Python owners must stay retained.
            self.uncertain = True
            raise

    def release(self, completion_event):
        if self.uncertain or not completion_event.query():
            raise RuntimeError("DMA owners cannot retire before their completion event")
        tickets = self.pending[completion_event]
        statuses = [self.extension.copy_status(ticket) for ticket in tickets]
        if not all(s[4] == 0 and s[6] == 1 for s in statuses):
            self.uncertain = True
            raise RuntimeError("PTA Host DMA submission failed or is incomplete")
        for ticket in tickets:
            self.extension.release_copy(ticket, True, False)
        self.last_statuses = statuses
        self.completed_batches += len(tickets)
        del self.pending[completion_event]

    def drain(self):
        # Called after the common worker has successfully drained its DMA stream.
        # Failed submissions retain mappings and payloads for explicit teardown.
        if self.uncertain:
            raise RuntimeError("Retain uncertain queued DMA until process teardown")
        for event in tuple(self.pending):
            self.release(event)
        if self.extension.pending_copies(self.cookie):
            raise RuntimeError("Unaccounted native DMA tickets remain")
        self.owners.clear()
        self.binding = None
