# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


def test_dflash_graph_reuses_same_proposal_metadata_without_rebuilding(monkeypatch):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.num_query_per_req = 4
    group_cache = object()
    metadata = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8, 8, 8], block_tables=group_cache)}
    speculator._prepared_draft_attn_metadata = (4, metadata)
    build = MagicMock(side_effect=AssertionError("Same proposal must reuse metadata"))
    monkeypatch.setattr(speculator, "_build_uniform_attn_metadata", build)
    result = speculator.build_draft_attn_metadatas(4, object())
    assert result[0] is metadata
    assert metadata["draft"].actual_seq_lengths_q == [4, 8, 12, 16]
    assert metadata["draft"].block_tables is group_cache
    build.assert_not_called()


@pytest.mark.parametrize("cached_count", [None, 2])
def test_dflash_graph_builds_metadata_when_cache_missing_or_shape_changes(monkeypatch, cached_count):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.num_query_per_req = 4
    speculator.input_batch = SimpleNamespace(num_reqs=2)
    speculator._group_causal = False
    stale = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8])}
    speculator._prepared_draft_attn_metadata = None if cached_count is None else (cached_count, stale)
    metadata = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8, 8, 8])}
    build = MagicMock(return_value=metadata)
    monkeypatch.setattr(speculator, "_build_uniform_attn_metadata", build)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", lambda: nullcontext())
    bounds = object()
    result = speculator.build_draft_attn_metadatas(4, bounds)
    assert result[0] is metadata
    assert metadata["draft"].actual_seq_lengths_q == [4, 8, 12, 16]
    assert stale["draft"].actual_seq_lengths_q == [4, 8]
    build.assert_called_once()
    assert build.call_args.kwargs["seq_lens_cpu_upper_bound"] is bounds
    assert build.call_args.kwargs["batch_desc"].num_tokens == 16


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("padded_count", [None, 4])
def test_dflash_caches_full_metadata_after_upstream_builder_refresh(monkeypatch, mode, padded_count):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator._prepared_draft_attn_metadata = None
    descriptor = SimpleNamespace(cg_mode=mode, num_reqs=padded_count)
    query, bounds, local_lengths = object(), object(), object()
    metadata = {"draft": object()}
    calls = []

    def build(self, *args):
        assert self is speculator
        calls.append(args)
        return metadata

    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", build)
    result = speculator._build_attn_metadata(2, descriptor, query, bounds, 4, False, local_lengths)
    assert calls == [(2, descriptor, query, bounds, 4, False, local_lengths)]
    assert result is metadata
    if mode == CUDAGraphMode.FULL:
        assert speculator._prepared_draft_attn_metadata == (padded_count or 2, metadata)
    else:
        assert speculator._prepared_draft_attn_metadata is None


def test_proposal_invalidates_previous_metadata_before_upstream_execution(monkeypatch):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator._prepared_draft_attn_metadata = (4, object())
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", lambda: nullcontext())
    batch, result = object(), object()

    def propose(self, *args, **kwargs):
        assert self.input_batch is batch
        assert self._prepared_draft_attn_metadata is None
        return result

    monkeypatch.setattr(DFlashSpeculator, "propose", propose)
    assert speculator.propose(batch, *([None] * 10)) is result
