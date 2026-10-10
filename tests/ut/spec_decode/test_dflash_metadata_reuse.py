# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
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


@pytest.mark.parametrize("num_reqs,padded", [(0, 4), (1, 1), (1, 4), (2, 4)])
def test_dflash_cpu_mirror_reads_valid_rows_and_zeroes_padding(num_reqs, padded):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor([73, 22666, 99999, 99999], dtype=torch.int32))
    actual = spec._prepare_draft_seq_lens_cpu(num_reqs, padded)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int32
    assert actual.tolist() == [73, 22666][:num_reqs] + [0] * (padded - num_reqs)
    assert spec.input_buffers.seq_lens.tolist() == [73, 22666, 99999, 99999]


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_dflash_metadata_shares_exact_lengths_and_refreshes_after_rejection(monkeypatch, mode):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._use_cpu_seq_lens = True
    spec._prepared_draft_attn_metadata = None
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([73, 22666, 99999, 99999], dtype=torch.int32),
        positions=torch.arange(32),
    )
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    descriptor = SimpleNamespace(cg_mode=mode, num_reqs=4, num_tokens=32)
    query = np.array([0, 8, 16], dtype=np.int32)
    upper_bound = torch.tensor([99, 99999], dtype=torch.int32)
    supplied = []

    @contextmanager
    def factory(positions, pad, **kwargs):
        assert positions is spec.input_buffers.positions
        assert pad == (32 if mode == CUDAGraphMode.FULL else 16)
        assert kwargs["seq_lens_cpu_is_exact"]
        supplied.append(kwargs["seq_lens_cpu"])
        yield

    monkeypatch.setattr(dflash_module, "build_attn_metadata_factory", factory)
    parent = MagicMock(return_value={"draft": object()})
    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", parent)
    spec._build_attn_metadata(2, descriptor, query, upper_bound, 8, False)
    spec.input_buffers.seq_lens[:2].copy_(torch.tensor([74, 22674]))
    spec._build_attn_metadata(2, descriptor, query, upper_bound, 8, False)
    assert [value.tolist() for value in supplied] == [[73, 22666, 0, 0], [74, 22674, 0, 0]]
    assert upper_bound.tolist() == [99, 99999]
    assert parent.call_count == 2
