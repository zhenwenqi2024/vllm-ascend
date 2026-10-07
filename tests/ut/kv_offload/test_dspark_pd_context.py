# SPDX-License-Identifier: Apache-2.0
"""DSpark context ownership/readiness, independent of network and model forward."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextChunk,
    DSparkContextDescriptor,
    DSparkContextReceiver,
    DSparkContextSubmission,
    find_dspark_context_connector,
    resident_mla_context_group_ids,
)


def _descriptor(request_id="request", generation="allocation-1", prompt_tokens=5):
    return DSparkContextDescriptor(request_id, generation, prompt_tokens, (2, 22, 38, 58, 74), 4)


def _chunk(descriptor, offset, tokens):
    return DSparkContextChunk(descriptor, offset, tokens), torch.zeros(
        (tokens, descriptor.feature_width), dtype=torch.bfloat16
    )


def _receiver(*descriptors, budget=400, capacity=4):
    receiver = DSparkContextReceiver(max_pending_bytes=budget, max_requests=capacity)
    for descriptor in descriptors:
        receiver.register_request(descriptor)
    return receiver


def _context_connector():
    return SimpleNamespace(get_dspark_context_descriptor=Mock(), send_dspark_context_chunk=Mock())


def _context_scheduler(draft_spec, *, method="dspark", wrapped=False):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler

    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(method=method)
    )
    scheduler.main_group_idx, scheduler.indexer_group_idx = 0, 1
    host_spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True
    )
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=host_spec),
            SimpleNamespace(kv_cache_spec=SimpleNamespace()),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(kv_cache_specs={"draft": draft_spec}) if wrapped else draft_spec
            ),
        ]
    )
    return scheduler


@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_scheduler_discovers_resident_mla_independent_of_attention_causality(non_causal, wrapped):
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        store_on_host=False,
        non_causal_multi_token_decode=non_causal,
    )
    assert _context_scheduler(spec, wrapped=wrapped)._find_dspark_context_groups() == (2,)
    assert spec.non_causal_multi_token_decode is non_causal


@pytest.mark.parametrize("host_mla", [False, True])
def test_scheduler_rejects_missing_resident_mla_groups(host_mla):
    spec = (
        AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True)
        if host_mla
        else SimpleNamespace(store_on_host=False)
    )
    with pytest.raises(RuntimeError, match="loader-created resident MLA"):
        _context_scheduler(spec)._find_dspark_context_groups()


@pytest.mark.parametrize("method", [None, "mtp"])
def test_scheduler_does_not_require_dspark_groups_on_legacy_paths(method):
    assert _context_scheduler(SimpleNamespace(), method=method)._find_dspark_context_groups() == ()


def test_real_scheduler_conversion_retains_resident_draft_in_shared_block_group():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    from vllm_ascend.patch.platform import patch_kv_cache_utils

    host_spec = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True
    )
    draft_spec = replace(host_spec, store_on_host=False)
    uniform = UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"target": host_spec, "draft": draft_spec})
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["target", "draft"], kv_cache_spec=uniform)],
    )
    vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method="dspark"), kv_transfer_config=None)
    with (
        patch.object(patch_kv_cache_utils, "_orig_get_kv_cache_config_from_groups", return_value=config),
        patch.object(patch_kv_cache_utils, "is_deepseek_v41_cache", return_value=False),
        patch.object(patch_kv_cache_utils, "_get_glm5_next_cache_layout", return_value=None),
        patch.object(patch_kv_cache_utils, "_is_deepseek_v4_groups", return_value=False),
    ):
        physical = patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(vllm_config, config.kv_cache_groups, 1)
    assert physical.dspark_context_group_ids == (0,)
    scheduler_config = generate_scheduler_kv_cache_config([physical])
    assert scheduler_config.kv_cache_groups[0].kv_cache_spec.store_on_host is True
    assert scheduler_config.dspark_context_group_ids == (0,)
    scheduler = _context_scheduler(draft_spec)
    scheduler.kv_cache_config = scheduler_config
    scheduler.main_group_idx = scheduler.indexer_group_idx = 0
    assert scheduler._find_dspark_context_groups() == (0,)
    assert physical.kv_cache_groups[0].kv_cache_spec is uniform
    assert resident_mla_context_group_ids(physical.kv_cache_groups) == (0,)


@pytest.mark.parametrize("group_ids", [(3,), (-1,), (True,), (0, 0)])
def test_scheduler_rejects_corrupt_resident_group_ownership(group_ids):
    scheduler = _context_scheduler(SimpleNamespace())
    scheduler.kv_cache_config.dspark_context_group_ids = group_ids
    with pytest.raises(RuntimeError, match="ownership is invalid"):
        scheduler._find_dspark_context_groups()


@pytest.mark.parametrize("pending", [0, 3, 4, 5])
@pytest.mark.parametrize("already_owned", [False, True])
def test_scheduler_bounds_remote_waiting_contexts_without_local_recompute(pending, already_owned):
    scheduler = _context_scheduler(SimpleNamespace())
    scheduler.vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=4)
    scheduler.block_size = [128, 128, 128]
    scheduler._dspark_context_groups = (2,)
    scheduler._dspark_context_requests = {f"other-{index}": object() for index in range(pending)}
    request_id = "other-0" if already_owned and pending else "new-request"
    scheduler._copy_sfa_slot_allocator = SimpleNamespace(can_bind=Mock(return_value=True))
    request = SimpleNamespace(
        request_id=request_id, kv_transfer_params={"do_remote_prefill": True}, prompt_token_ids=[1] * 5
    )
    should_defer = pending >= 4 and not already_owned
    assert scheduler.get_num_new_matched_tokens(request, 0) == ((None, False) if should_defer else (5, True))
    assert len(scheduler._dspark_context_requests) == pending


@pytest.mark.parametrize("ret", [0, -1])
def test_network_reader_does_not_ack_or_initialize_failed_memfabric_read(monkeypatch, ret):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import DSPARK_CONTEXT_CHUNK
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import MembPullReadThread

    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    reader = MembPullReadThread.__new__(MembPullReadThread)
    reader._state = SimpleNamespace(dspark_context_receiver=receiver)
    reader._lock = threading.Lock()
    reader._accepted_context_chunks = set()
    reader._failed_requests = set()
    reader._p_sessions = {b"p": "p-session"}
    reader._p_pp_topology = {b"p": (1, 2)}
    reader.engine = SimpleNamespace(batch_transfer_sync_read=Mock(return_value=ret))
    reader._send_dspark_context_ack = Mock()
    tensor = torch.zeros((5, descriptor.feature_width), dtype=torch.bfloat16)
    monkeypatch.setattr(torch, "empty", Mock(return_value=tensor))
    message = (
        DSPARK_CONTEXT_CHUNK,
        descriptor.request_id,
        descriptor.generation,
        descriptor.prompt_tokens,
        descriptor.aux_layer_ids,
        descriptor.hidden_size,
        0,
        5,
        1024,
        tensor.numel() * 2,
    )
    reader._handle_dspark_context_chunk(b"p", message, Mock(), Mock())
    assert reader._send_dspark_context_ack.call_args.args[-1] == (b"accepted" if ret == 0 else b"failed")
    initialize = Mock(return_value=None)
    assert receiver.drain(initialize) == (5 if ret == 0 else 0)
    assert reader._failed_requests == (set() if ret == 0 else {descriptor.request_id})
    assert len(reader._accepted_context_chunks) == (1 if ret == 0 else 0)


@pytest.mark.parametrize("nested", [False, True])
def test_multiconnector_routes_context_to_its_exact_child_metadata(nested):
    pd = _context_connector()
    pd_meta = object()
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), pd])
    metadata = SimpleNamespace(metadata=(object(), pd_meta))
    if nested:
        wrapper = SimpleNamespace(_connectors=[wrapper])
        metadata = SimpleNamespace(metadata=(metadata,))
    assert find_dspark_context_connector(wrapper, metadata) == (pd, pd_meta)


@pytest.mark.parametrize("children", [[], [_context_connector(), _context_connector()]])
def test_context_routing_rejects_missing_or_ambiguous_pd_child(children):
    with pytest.raises(RuntimeError, match="exactly one"):
        find_dspark_context_connector(
            SimpleNamespace(_connectors=children), SimpleNamespace(metadata=tuple(object() for _ in children))
        )


def test_context_routing_rejects_misaligned_multiconnector_metadata():
    with pytest.raises(RuntimeError, match="not aligned"):
        find_dspark_context_connector(SimpleNamespace(_connectors=[_context_connector()]), SimpleNamespace(metadata=()))


def test_producer_preserves_partial_context_across_unscheduled_step():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import SfaPDProducerMetadata
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HProducerWorker

    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker._backend = "memfabric"
    worker._dspark_next_offsets = {("request", "allocation-1"): 64}
    worker.bind_connector_metadata(SfaPDProducerMetadata())
    assert worker._dspark_next_offsets == {("request", "allocation-1"): 64}
    worker.get_finished({"request"})
    assert worker._dspark_next_offsets == {}


def test_runner_sends_only_real_prompt_rows_through_multiconnector(monkeypatch):
    from vllm_ascend.worker.v2 import model_runner as module

    descriptor = _descriptor(prompt_tokens=71)
    pd = _context_connector()
    pd.get_dspark_context_descriptor.return_value = descriptor
    pd_meta = SimpleNamespace(requests={"request": SimpleNamespace(dspark_context_generation="allocation-1")})
    wrapper = SimpleNamespace(_connectors=[SimpleNamespace(), pd])
    metadata = SimpleNamespace(metadata=(SimpleNamespace(), pd_meta))
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: wrapper)
    runner = module.NPUModelRunner.__new__(module.NPUModelRunner)
    runner.pd_dspark_aux_layer_ids = descriptor.aux_layer_ids
    runner.is_last_pp_rank = True
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: descriptor.hidden_size)
    aux = [torch.full((75, 4), index, dtype=torch.bfloat16) for index in range(5)]
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request"],
        prefill_len_np=[71],
        num_computed_tokens_np=[0],
        num_scheduled_tokens=[73],
        query_start_loc_np=[2, 75],
    )
    runner.execute_model_state = SimpleNamespace(aux_hidden_states=aux, input_batch=batch)
    runner._send_pd_dspark_context_chunks(SimpleNamespace(kv_connector_metadata=metadata))
    calls = pd.send_dspark_context_chunk.call_args_list
    assert [(call.args[1].token_offset, call.args[1].num_tokens) for call in calls] == [(0, 64), (64, 7)]
    actual = torch.cat([call.args[2] for call in calls])
    torch.testing.assert_close(actual, torch.cat(aux, dim=-1)[2:73])


def test_runner_skips_aux_transport_on_non_last_prefill_stage():
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pd_dspark_aux_layer_ids = (2, 22, 38)
    runner.is_last_pp_rank = False
    runner._send_pd_dspark_context_chunks(object())


@pytest.mark.parametrize("target_first", [False, True])
def test_target_kv_done_does_not_release_before_all_context_writes(target_first):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    releases = [Mock(), Mock()]
    if target_first:
        receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    chunk, tensor = _chunk(descriptor, 0, 3)
    assert receiver.submit(chunk, tensor, releases[0]) is DSparkContextSubmission.ACCEPTED
    initializer = Mock(return_value=None)
    assert receiver.ready_requests() == set()
    assert receiver.drain(initializer) == 3
    releases[0].assert_called_once_with()
    assert receiver.ready_requests() == set()
    chunk, tensor = _chunk(descriptor, 3, 2)
    assert receiver.submit(chunk, tensor, releases[1]) is DSparkContextSubmission.ACCEPTED
    assert receiver.ready_requests() == set()
    assert receiver.drain(initializer) == 2
    if not target_first:
        assert receiver.ready_requests() == set()
        receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    assert receiver.ready_requests() == {descriptor.request_id}
    assert receiver.pending_bytes == 0
    assert [(call.args[0].token_offset, call.args[0].num_tokens) for call in initializer.call_args_list] == [
        (0, 3),
        (3, 2),
    ]
    releases[1].assert_called_once_with()


def test_backpressure_retains_reader_ownership_and_retry_offset():
    descriptor = _descriptor()
    receiver = _receiver(descriptor, budget=120)
    first, tensor = _chunk(descriptor, 0, 3)
    first_release, retry_release = Mock(), Mock()
    assert receiver.submit(first, tensor, first_release) is DSparkContextSubmission.ACCEPTED
    tail, tail_tensor = _chunk(descriptor, 3, 2)
    assert receiver.submit(tail, tail_tensor, retry_release) is DSparkContextSubmission.BACKPRESSURE
    retry_release.assert_not_called()
    assert receiver.pending_bytes == 120
    assert receiver.drain(Mock(return_value=None)) == 3
    assert receiver.submit(tail, tail_tensor, retry_release) is DSparkContextSubmission.ACCEPTED
    assert receiver.drain(Mock(return_value=None)) == 2
    first_release.assert_called_once_with()
    retry_release.assert_called_once_with()


def test_interleaved_requests_cancel_and_generation_reuse():
    first = _descriptor("a")
    other = _descriptor("b")
    receiver = _receiver(first, other)
    releases = [Mock() for _ in range(3)]
    for descriptor, offset, tokens, release in [
        (first, 0, 2, releases[0]),
        (other, 0, 5, releases[1]),
        (first, 2, 3, releases[2]),
    ]:
        chunk, tensor = _chunk(descriptor, offset, tokens)
        assert receiver.submit(chunk, tensor, release) is DSparkContextSubmission.ACCEPTED
    receiver.discard_request("a", first.generation)
    releases[0].assert_called_once_with()
    releases[2].assert_called_once_with()
    replacement = replace(first, generation="allocation-2")
    receiver.register_request(replacement)
    stale, tensor = _chunk(first, 0, 5)
    stale_release = Mock()
    assert receiver.submit(stale, tensor, stale_release) is DSparkContextSubmission.STALE
    stale_release.assert_not_called()
    receiver.mark_target_kv_done("a", first.generation)
    receiver.discard_request("a", first.generation)
    receiver.mark_target_kv_done("b", other.generation)
    initializer = Mock(return_value=None)
    assert receiver.drain(initializer) == 5
    assert receiver.ready_requests() == {"b"}
    assert initializer.call_args.args[0].descriptor == other
    assert receiver.pending_bytes == 0


@pytest.mark.parametrize("offset,tokens", [(1, 2), (-1, 2), (True, 2), (0, 0), (0, -1), (0, 6), (0, True)])
def test_reject_gaps_bad_ranges_and_noninteger_offsets(offset, tokens):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk = DSparkContextChunk(descriptor, offset, tokens)
    with pytest.raises(ValueError):
        receiver.submit(chunk, torch.zeros((2, 20), dtype=torch.bfloat16), Mock())
    assert receiver.pending_bytes == 0


def test_duplicate_chunk_cannot_overwrite_initialized_context():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 3)
    receiver.submit(chunk, tensor, Mock())
    receiver.drain(Mock(return_value=None))
    with pytest.raises(ValueError, match="duplicates"):
        receiver.submit(chunk, tensor, Mock())


@pytest.mark.parametrize("changed", [{"prompt_tokens": 6}, {"hidden_size": 8}, {"aux_layer_ids": (2, 22, 38, 74)}])
def test_reject_changed_checkpoint_or_prompt_schema(changed):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(replace(descriptor, **changed), 0, 2)
    with pytest.raises(ValueError, match="schema"):
        receiver.submit(chunk, tensor, Mock())


@pytest.mark.parametrize(
    "tensor",
    [torch.zeros((2, 20)), torch.zeros((2, 4), dtype=torch.bfloat16), torch.zeros((20, 2), dtype=torch.bfloat16).T],
)
def test_reject_lossy_or_noncontiguous_aux_tensor(tensor):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    with pytest.raises(ValueError):
        receiver.submit(DSparkContextChunk(descriptor, 0, 2), tensor, Mock())


def test_staging_budget_and_request_admission_are_bounded():
    descriptor = _descriptor()
    receiver = _receiver(descriptor, budget=40, capacity=1)
    chunk, tensor = _chunk(descriptor, 0, 2)
    with pytest.raises(ValueError, match="split"):
        receiver.submit(chunk, tensor, Mock())
    with pytest.raises(ValueError, match="previous"):
        receiver.register_request(descriptor)
    with pytest.raises(RuntimeError, match="capacity"):
        receiver.register_request(_descriptor("another"))


def _consumer(receiver):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HConsumerWorker

    worker = SFAPDRD2HConsumerWorker.__new__(SFAPDRD2HConsumerWorker)
    worker.tp_size = 1
    worker.tp_rank = 0
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._cpu_blocks_by_req = {}
    worker.copy_sfa_slots_by_req = {}
    worker._copy_sfa_tail_by_req = {}
    worker._pending_done = set()
    worker._terminal_ext_ids = set()
    worker._dspark_draft_blocks_by_req = {}
    worker._dspark_context_receiver = receiver
    worker._dspark_context_initializer = Mock(return_value=None)
    worker._mf_read_thread = Mock()
    worker._mf_read_thread.get_and_clear_done.return_value = set()
    worker._mf_read_thread.get_and_clear_failed.return_value = set()
    return worker


def test_completed_context_retires_before_next_load_and_late_finished_cleanup():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import SfaPDConsumerMetadata

    # Real worker load/completion hooks: a new batch is loaded before the old
    # batch's finished IDs reach get_finished, as in upstream no_forward.
    receiver = _receiver(capacity=4, budget=800)
    worker = _consumer(receiver)
    previous = set()
    for wave in range(4):
        metadata = SfaPDConsumerMetadata()
        descriptors = [_descriptor(f"w{wave}r{index}") for index in range(4)]
        for descriptor in descriptors:
            metadata.add_request(
                descriptor.request_id + "-internal",
                [1],
                [2],
                dspark_context_descriptor=descriptor,
                dspark_draft_group_ids=(0,),
                dspark_draft_block_ids_by_group={0: (3,)},
            )
        worker.start_load_kv(metadata)
        for descriptor in descriptors:
            chunk, tensor = _chunk(descriptor, 0, 5)
            assert receiver.submit(chunk, tensor, Mock()) is DSparkContextSubmission.ACCEPTED
        current = {descriptor.request_id for descriptor in descriptors}
        worker._mf_read_thread.get_and_clear_done.return_value = current
        internal_ids = {request_id + "-internal" for request_id in current}
        assert worker.get_finished(previous) == (set(), internal_ids)
        assert receiver.pending_bytes == 0
        assert set(worker.request_map) == current
        assert set(worker._dspark_draft_blocks_by_req) == current
        for descriptor in descriptors:
            assert worker._dspark_draft_blocks_by_req[descriptor.request_id] == {0: (3,)}
        previous = internal_ids
    assert receiver.ready_requests() == set()
    for descriptor in descriptors:
        assert receiver.get_descriptor(descriptor.request_id) is None
    assert worker._dspark_context_initializer.call_count == 16


def test_ingress_retirement_waits_for_all_tp_ranks():
    descriptor = _descriptor()
    receiver = _receiver(descriptor, capacity=1)
    worker = _consumer(receiver)
    worker.request_map[descriptor.request_id] = descriptor.request_id
    worker._dspark_draft_blocks_by_req[descriptor.request_id] = {0: (3,)}
    chunk, tensor = _chunk(descriptor, 0, 5)
    assert receiver.submit(chunk, tensor, Mock()) is DSparkContextSubmission.ACCEPTED
    worker._mf_read_thread.get_and_clear_done.side_effect = [{descriptor.request_id}, set(), set()]
    ready = ({descriptor.request_id}, set())
    worker._gather_tp_read_status = Mock(side_effect=[[ready, (set(), set())], [ready, ready], [(set(), set())] * 2])
    assert worker.get_finished() == (set(), set())
    assert receiver.get_descriptor(descriptor.request_id) == descriptor
    with pytest.raises(RuntimeError, match="capacity"):
        receiver.register_request(_descriptor("next"))
    assert worker.get_finished() == (set(), {descriptor.request_id})
    assert receiver.get_descriptor(descriptor.request_id) is None
    receiver.register_request(_descriptor("next"))
    assert worker.get_finished() == (set(), set())


@pytest.mark.parametrize("target_done,context_done", [(False, False), (False, True), (True, False)])
def test_incomplete_ingress_cannot_retire(target_done, context_done):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    if context_done:
        chunk, tensor = _chunk(descriptor, 0, 5)
        receiver.submit(chunk, tensor, Mock())
        receiver.drain(Mock(return_value=None))
    if target_done:
        receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    with pytest.raises(RuntimeError, match="Cannot retire"):
        receiver.retire_ready_request(descriptor.request_id, descriptor.generation)
    assert receiver.get_descriptor(descriptor.request_id) == descriptor


def test_retirement_does_not_remove_newer_generation():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    receiver.retire_ready_request(descriptor.request_id, "older-generation")
    assert receiver.get_descriptor(descriptor.request_id) == descriptor


def test_failed_initializer_quarantines_storage_and_never_reports_ready():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 5)
    release = Mock()
    receiver.submit(chunk, tensor, release)
    receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    with pytest.raises(RuntimeError, match="device write"):
        receiver.drain(Mock(side_effect=RuntimeError("device write failed")))
    release.assert_not_called()
    assert receiver.pending_bytes == 200
    assert receiver.ready_requests() == set()
    with pytest.raises(RuntimeError, match="quarantined"):
        receiver.drain(Mock(return_value=None))
    receiver.discard_request(descriptor.request_id, descriptor.generation)
    release.assert_called_once_with()
    assert receiver.pending_bytes == 0


def test_unwaited_initializer_result_cannot_signal_readiness():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 5)
    release = Mock()
    receiver.submit(chunk, tensor, release)
    receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    with pytest.raises(RuntimeError, match="synchronously"):
        receiver.drain(Mock(return_value=object()))
    release.assert_not_called()
    assert receiver.ready_requests() == set()


def test_failed_staging_release_is_not_reported_as_success():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 5)
    receiver.submit(chunk, tensor, Mock(side_effect=RuntimeError("staging release failed")))
    receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation)
    with pytest.raises(RuntimeError, match="release failed"):
        receiver.drain(Mock(return_value=None))
    assert receiver.ready_requests() == set()


def test_background_thread_can_submit_but_cannot_run_model_or_recycle_blocks():
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 5)
    errors = []

    def reader():
        assert receiver.submit(chunk, tensor, Mock()) is DSparkContextSubmission.ACCEPTED
        for action in [
            lambda: receiver.drain(Mock(return_value=None)),
            lambda: receiver.discard_request(descriptor.request_id, descriptor.generation),
            lambda: receiver.mark_target_kv_done(descriptor.request_id, descriptor.generation),
            lambda: receiver.register_request(_descriptor("other")),
            receiver.ready_requests,
        ]:
            try:
                action()
            except RuntimeError as error:
                errors.append(str(error))

    thread = threading.Thread(target=reader)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 5
    assert all("model worker thread" in error for error in errors)
    assert receiver.drain(Mock(return_value=None)) == 5


@pytest.mark.parametrize("action", ["discard", "drain"])
def test_reentrant_callback_cannot_recycle_a_live_write(action):
    descriptor = _descriptor()
    receiver = _receiver(descriptor)
    chunk, tensor = _chunk(descriptor, 0, 5)
    release = Mock()
    receiver.submit(chunk, tensor, release)

    def initialize(*_):
        if action == "discard":
            receiver.discard_request(descriptor.request_id, descriptor.generation)
        else:
            receiver.drain(Mock(return_value=None))

    with pytest.raises(RuntimeError):
        receiver.drain(initialize)
    release.assert_not_called()
    assert receiver.ready_requests() == set()


@pytest.mark.parametrize(
    "changed",
    [
        {"request_id": ""},
        {"request_id": 1},
        {"generation": ""},
        {"generation": True},
        {"prompt_tokens": 0},
        {"prompt_tokens": True},
        {"hidden_size": 0},
        {"aux_layer_ids": ()},
        {"aux_layer_ids": (22, 2)},
        {"aux_layer_ids": (2, 2)},
        {"aux_layer_ids": (True,)},
    ],
)
def test_invalid_descriptor_rejected(changed):
    with pytest.raises(ValueError):
        replace(_descriptor(), **changed)


@pytest.mark.parametrize("budget,capacity", [(0, 4), (100, 0), (True, 4), (100, True)])
def test_invalid_receiver_bounds_rejected(budget, capacity):
    with pytest.raises(ValueError):
        _receiver(budget=budget, capacity=capacity)
