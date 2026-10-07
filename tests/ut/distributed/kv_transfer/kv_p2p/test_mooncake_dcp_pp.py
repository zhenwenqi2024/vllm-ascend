from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    GroupPull,
    MooncakeConnectorWorker,
)


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("remote_dcp", [4, 16])
@pytest.mark.parametrize("local_dcp", [1, 4])
def test_dcp_routing_covers_all_pipeline_stages(pp_size, remote_dcp, local_dcp):
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.use_mla = worker.use_sparse = True
    worker.pcp_size = 1
    worker.dcp_size = local_dcp
    worker.tp_size = 4
    worker.tp_rank = worker.pcp_rank = worker.dcp_rank = 0
    worker._prefill_tp_size = 16
    worker._prefill_pp_size = pp_size
    worker._is_hma_required = True
    worker.local_remote_block_port_mapping = {}
    worker.remote_port_send_num = {}
    worker.block_size = 128
    worker.block_size_scale = [[1]]
    worker.num_key_value_heads = 1
    worker.side_channel_port = worker.handshake_port = 30200
    worker.kv_group2layeridx = {0: ({"kv_cache_spec_type": "AscendMLAAttentionSpec"}, [0])}
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_port=30000))
    worker._get_tp_num_need_pulls = lambda size: 1
    worker._get_remote_host_info_by_port = lambda base, port, host, engine, mapping: (host, engine)
    meta = SimpleNamespace(
        remote_pcp_size=1,
        remote_dcp_size=remote_dcp,
        remote_ptp_size=16,
        remote_port=30000,
        remote_block_ids=(list(range(10, 10 + 32 // remote_dcp)),),
        local_block_ids=(list(range(100, 100 + 32 // local_dcp)),),
        num_external_tokens=32 * 128,
        num_prompt_blocks=32,
        num_computed_tokens=0,
        remote_engine_id="prefill",
        remote_host="localhost",
        remote_block_size=128,
        remote_multi_nodes_meta_mapping={},
    )
    ports, local_ids, remote_ids = worker._get_kv_split_metadata("pp-dcp", meta)
    pulls = worker._get_group_pulls_metadata("pp-dcp", ports, 16, 30000, 1, remote_dcp)
    assert sum(len(ids[0]) for ids in local_ids) == 32 // local_dcp
    assert all(len(local[0]) == len(remote[0]) for local, remote in zip(local_ids, remote_ids))
    for sources, stage_pulls in zip(ports, pulls):
        assert {(port - 30000) // 16 for port in sources} == set(range(pp_size))
        assert {pull.prefill_pp_rank for group in stage_pulls for pull in group} == set(range(pp_size))
        assert all(pull.remote_tp_offset == 0 and pull.is_group_transfer_end for group in stage_pulls for pull in group)
        assert all(worker.remote_port_send_num["prefill"][port]["num"] > 0 for port in sources)


def test_replicated_indexer_has_one_transfer_per_pipeline_stage():
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.kv_send_thread = None
    worker.kv_recv_thread = MagicMock()
    worker._prefill_tp_size = 16
    worker.kv_group2layeridx = {
        0: ({"kv_cache_spec_type": "AscendMLAAttentionSpec"}, [0]),
        1: ({"kv_cache_spec_type": "AscendSFAIndexerCacheSpec"}, [156]),
    }
    ports = [[30000, 30016], [30001, 30017]]
    worker.remote_port_send_num = {"prefill": {}}
    worker._get_sfa_replicate_k_block_ids = MagicMock(return_value=(([40],), ([20],)))
    worker._get_kv_split_metadata = MagicMock(return_value=(ports, [([10],), ([11],)], [([30],), ([31],)]))
    worker._get_group_pulls_metadata = MagicMock(
        return_value=[
            [
                [GroupPull(group_id=g, remote_tp_offset=0, num_group_pulls=1, prefill_pp_rank=pp) for g in (0, 1)]
                for pp in (0, 1)
            ]
            for _ in ports
        ]
    )
    worker._get_remote_host_info_by_port = MagicMock(return_value=("localhost", "prefill"))
    meta = SimpleNamespace(
        remote_request_id="p-request",
        remote_engine_id="prefill",
        remote_host="localhost",
        remote_port=30000,
        remote_pcp_size=1,
        remote_dcp_size=16,
        remote_ptp_size=16,
        remote_multi_nodes_meta_mapping={},
        remote_block_size=128,
        local_block_ids=([10],),
        remote_block_ids=([30],),
        num_computed_tokens=0,
        num_external_tokens=128,
        do_virtual=False,
    )
    worker.start_load_kv(SimpleNamespace(reqs_in_batch=["request"], requests={"request": meta}))
    calls = [call.kwargs for call in worker.kv_recv_thread.add_request.call_args_list]
    indexer_calls = [call for call in calls if any(p.group_id == 1 for p in call["group_pulls"])]
    assert [call["remote_handshake_port"] for call in indexer_calls] == [30000, 30016]
    assert all(call["local_block_ids_replicate_k"] == ([40],) for call in indexer_calls)
    assert all(call["remote_block_ids_replicate_k"] == ([20],) for call in indexer_calls)
