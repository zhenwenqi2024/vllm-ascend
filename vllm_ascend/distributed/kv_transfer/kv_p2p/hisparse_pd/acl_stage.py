# SPDX-License-Identifier: Apache-2.0
"""An explicit ACL owner for the producer's bounded remote-read stage."""

from __future__ import annotations

import ctypes

from .protocol import STAGE_BYTES


class ACLStageBuffer:
    """Keep the original allocation pointer until transport teardown succeeds.

    Construction does not allocate. The worker must retain this owner before
    calling allocate(), including when allocation/query or subsequent cleanup
    fails. There is deliberately no destructor or device/context destruction.
    """

    def __init__(self, device, set_device, capacity=STAGE_BYTES):
        if type(device) is not int or device < 0 or type(capacity) is not int or capacity != STAGE_BYTES:
            raise ValueError("ACL producer stage requires a device and 16 MiB logical capacity")
        self.device = device
        self.set_device = set_device
        self.capacity = capacity
        self.requested_bytes = capacity
        self.address = 0
        self.context = 0
        self.allocation_range = None
        self.records = []
        self.closed = False
        self._started = False
        self._ready = False
        self._allocated_success = False
        self._api = None

    def _bind(self):
        api = ctypes.CDLL("libascendcl.so")
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        signatures = {
            "aclrtMallocAlign32": (
                [ctypes.POINTER(pointer), size, ctypes.c_int],
                ctypes.c_int,
            ),
            "aclrtMemGetAddressRange": (
                [pointer, ctypes.POINTER(pointer), ctypes.POINTER(size)],
                ctypes.c_int,
            ),
            "aclrtGetCurrentContext": ([ctypes.POINTER(pointer)], ctypes.c_int),
            "aclrtSetCurrentContext": ([pointer], ctypes.c_int),
            "aclrtMemcpy": ([pointer, size, pointer, size, ctypes.c_int], ctypes.c_int),
            "aclrtFree": ([pointer], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(api, name)  # Reject missing ABI before allocation.
            function.argtypes, function.restype = arguments, result
        self._api = api

    def _checked(self, operation, result, *, audit=True, **fields):
        if audit or result != 0:
            self.records.append({"operation": operation, "return_code": result, **fields})
        if result != 0:
            raise RuntimeError(f"{operation} failed with ACL status {result}")

    def allocate(self):
        if self._started or self.closed:
            raise RuntimeError("ACL producer stage can only be allocated once")
        self._started = True
        self._bind()
        self.set_device(self.device)
        context = ctypes.c_void_p()
        result = self._api.aclrtGetCurrentContext(ctypes.byref(context))
        self.context = context.value or 0
        self._checked("aclrtGetCurrentContext", result, context=self.context)
        if not self.context:
            raise RuntimeError("ACL producer stage requires the existing worker context")
        pointer = ctypes.c_void_p()
        # HUGE_ONLY=1, with no fallback. Align32 has no extra padding; the actual range is still
        # queried independently and is never substituted for the free pointer.
        result = self._api.aclrtMallocAlign32(ctypes.byref(pointer), self.requested_bytes, 1)
        self.address = pointer.value or 0  # Retain even a nonzero pointer on failure.
        self._allocated_success = result == 0
        self._checked(
            "aclrtMallocAlign32",
            result,
            address=self.address,
            requested_bytes=self.requested_bytes,
            policy=1,
        )
        if not 0 < self.address < 2**64 - self.capacity:
            raise RuntimeError("ACL producer stage returned an invalid allocation pointer")
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        result = self._api.aclrtMemGetAddressRange(self.address, ctypes.byref(base), ctypes.byref(size))
        self.allocation_range = {"base": base.value or 0, "bytes": size.value}
        self._checked("aclrtMemGetAddressRange", result, **self.allocation_range)
        if not 0 < (base.value or 0) <= self.address or self.address + self.capacity > (base.value or 0) + size.value:
            raise RuntimeError("ACL allocation range does not contain the logical producer stage")
        self._ready = True

    def _activate(self):
        self.set_device(self.device)
        self._checked(
            "aclrtSetCurrentContext",
            self._api.aclrtSetCurrentContext(self.context),
            audit=False,
            context=self.context,
        )

    def copy_from_host(self, source, length):
        if self.closed or not self._ready:
            raise RuntimeError("ACL producer stage is unavailable")
        if (
            type(source) is not int
            or not 0 < source < 2**64
            or type(length) is not int
            or not 0 < length <= self.capacity
            or source + length >= 2**64
        ):
            raise ValueError("Invalid ACL producer stage Host extent")
        self._activate()
        # aclrtMemcpy is synchronous. No success reply can precede this return.
        # H2D=1; source remains the actual retained CPU cache, never an alias.
        result = self._api.aclrtMemcpy(self.address, self.capacity, source, length, 1)
        self._checked("aclrtMemcpy", result, audit=False, source=source, bytes=length)

    def copy_to_host(self, destination, length):
        """Synchronous D2H before acknowledging reuse of this receive slot."""
        if self.closed or not self._ready:
            raise RuntimeError("ACL receive stage is unavailable")
        if (
            type(destination) is not int or not 0 < destination < 2**64
            or type(length) is not int or not 0 < length <= self.capacity
            or destination + length >= 2**64
        ):
            raise ValueError("Invalid ACL receive stage Host extent")
        self._activate()
        result = self._api.aclrtMemcpy(destination, length, self.address, length, 2)
        self._checked("aclrtMemcpyD2H", result, audit=False, destination=destination, bytes=length)

    def close(self):
        """Called only after device drain and both MemFabric teardown stages."""
        if self.closed:
            return
        if self.address:
            if not self._allocated_success:
                raise RuntimeError("ACL failed allocation returned an uncertain pointer; owner must be retained")
            self._activate()
            self._checked("aclrtFree", self._api.aclrtFree(self.address), address=self.address)
            self.address = 0  # A failed free leaves the exact pointer retryable.
        self.closed = True
        self._ready = False
