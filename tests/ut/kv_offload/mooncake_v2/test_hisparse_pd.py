# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd import _HiSparsePDWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import MooncakePushConnectorWorker


def test_hisparse_pd_worker_selects_source_and_indexer() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.num_blocks = 8
    worker.kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=32,
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, host_resident=True),
            SimpleNamespace(enable_kv_transfer=True, host_resident=False),
            SimpleNamespace(enable_kv_transfer=False, host_resident=False),
        ],
    )

    assert [worker._is_transfer_group(i) for i in range(3)] == [True, True, False]
    assert [worker._logical_num_blocks(i) for i in range(2)] == [32, 8]


def test_hisparse_pd_worker_slices_replicated_indexer_blocks() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker.kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=4,
        hisparse_host_block_stride=4,
        kv_cache_groups=[
            SimpleNamespace(
                role="hisparse_source", layer_names=["source"], host_resident=True, enable_kv_transfer=True
            ),
            SimpleNamespace(
                role="hisparse_indexer", layer_names=["indexer"], host_resident=False, enable_kv_transfer=True
            ),
        ],
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"indexer": SimpleNamespace(hisparse_block_stride=2)})
    )
    indexer = torch.arange(16).reshape(4, 4)
    source = torch.zeros((4, 4), dtype=torch.uint8)
    with patch.object(MooncakePushConnectorWorker, "register_kv_caches") as register:
        worker.register_kv_caches({"source": source, "indexer": indexer})

    transferred = register.call_args.args[0]["indexer"][0]
    assert torch.equal(transferred, indexer[::2])
    staged_source = register.call_args.args[0]["source"][0]
    assert staged_source.data_ptr() != source.data_ptr()
    assert staged_source.shape == source.shape


def test_hisparse_pd_push_packs_mla_components_by_token() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_cache_specs = [SimpleNamespace(block_size=2)]
    worker.layer_name_to_spec_index = {"source": 0}
    nope = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    rope = torch.tensor([5, 6], dtype=torch.uint8)
    staging = torch.zeros(12, dtype=torch.uint8)
    local = SimpleNamespace(
        kv_caches_base_addr=[[nope.data_ptr(), rope.data_ptr()]], block_lens=[[4, 2]], block_strides=[[4, 2]]
    )
    pp = SimpleNamespace(block_lens=[[6]], block_strides=[[6]])
    remote = SimpleNamespace(kv_caches_base_addr=[[staging.data_ptr()]])
    batch = SimpleNamespace(
        layer_name="source", requests=[SimpleNamespace(local_block_ids=(1,), remote_block_ids=(1,))]
    )
    src, dst, lengths = [], [], []
    assert worker._append_packed_mla_addresses(batch, 0, 0, local, pp, remote, src, dst, lengths)
    assert lengths == [2, 1, 2, 1]
    assert [address - staging.data_ptr() for address in dst] == [6, 8, 9, 11]


def test_hisparse_pd_push_copies_completed_page() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    stage = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 5, 3, 4, 6], dtype=torch.uint8)
    source = torch.zeros(12, dtype=torch.uint8)
    worker._host_stage_range = (stage.data_ptr(), stage.data_ptr() + 12, source.data_ptr(), source.data_ptr() + 12)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(host_resident=True, kv_cache_spec=SimpleNamespace(block_size=2))]
    )
    worker.transfer_metadata = SimpleNamespace(
        layer_names=["source"],
        group_indices=[0],
        block_size_scales=[[1]],
        kv_caches_base_addr=[[stage.data_ptr()]],
        block_strides=[[6]],
        block_lens=[[6]],
    )
    request = SimpleNamespace(local_block_ids=((1,),), num_computed_tokens=0, num_external_tokens=2)
    worker._copy_imported_source(request)
    assert torch.equal(source, stage)


def test_hisparse_pd_reports_copy_failure_to_core() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker._source_imports = {"request": object()}
    results = SimpleNamespace(finished_recving={"request"}, failed_recving=set())
    with (
        patch.object(MooncakePushConnectorWorker, "get_transfer_results", return_value=results),
        patch.object(worker, "_copy_imported_source", side_effect=ValueError("bad page")),
    ):
        assert worker.get_transfer_results() is results
    assert results.failed_recving == {"request"}
