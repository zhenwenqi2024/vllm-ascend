# SPDX-License-Identifier: Apache-2.0
"""Mooncake TENT Pull transport for the existing Offload Host allocation."""

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


class _MemfabricPullEngine:
    def __init__(self, engine):
        self._engine = engine

    def get_rpc_port(self):
        return self._engine.get_rpc_port()

    def batch_transfer_sync_read(self, session_id, local_buffers, peer_buffers, lengths):
        return self._engine.batch_transfer_sync_read(
            session_id, local_buffers, peer_buffers, lengths, transport_hint="memfabric"
        )


class HiSparseMooncakeTE(GlobalTE):
    def __init__(self):
        super().__init__(protocol="memfabric")
        self._pull_engine = None

    def get_transfer_engine(self, hostname: str, device_name: str | None = None):
        if envs.MC_USE_TENT != "1" or envs.MF_PROTOCOL != "device_urma":
            raise ValueError(
                "HiSparse Mooncake PD requires MC_USE_TENT=1 and MF_PROTOCOL=device_urma; SDMA is disabled"
            )
        engine = super().get_transfer_engine(hostname, device_name)
        with self.transfer_engine_lock:
            if self._pull_engine is None:
                self._pull_engine = _MemfabricPullEngine(engine)
            return self._pull_engine


global_hisparse_te = HiSparseMooncakeTE()
