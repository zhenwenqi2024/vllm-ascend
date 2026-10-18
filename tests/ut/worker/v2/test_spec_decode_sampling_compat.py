# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for the current vLLM draft-sampling operator contracts."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import lmhead_tp_utils, rejection_sampler_utils


@pytest.mark.parametrize("query_lengths", [[4], [2, 4, 1]])
def test_rejection_statistics_cover_all_non_bonus_logits(monkeypatch, query_lengths):
    stats = MagicMock()
    monkeypatch.setattr(rejection_sampler_utils, "_compute_block_stats_kernel", stats)
    monkeypatch.setattr(rejection_sampler_utils, "_probabilistic_rejection_kernel", MagicMock())
    monkeypatch.setattr(rejection_sampler_utils, "resample", MagicMock())
    num_reqs = len(query_lengths)
    num_logits = sum(query_lengths)
    local_pos = torch.cat([torch.arange(length) for length in query_lengths])
    mapping = torch.repeat_interleave(torch.arange(num_reqs), torch.tensor(query_lengths))
    boundaries = torch.tensor([0, *query_lengths]).cumsum(0)

    sampled, counts = rejection_sampler_utils.rejection_sample(
        target_logits=torch.zeros(num_logits, 16),
        draft_logits=None,
        draft_sampled=torch.zeros(num_logits, dtype=torch.int64),
        cu_num_logits=boundaries,
        pos=torch.arange(num_logits),
        idx_mapping=torch.arange(num_reqs),
        expanded_idx_mapping=mapping,
        expanded_local_pos=local_pos,
        temperature=torch.zeros(num_reqs),
        seed=torch.zeros(num_reqs, dtype=torch.int64),
        num_speculative_steps=3,
    )

    _, kwargs = stats.__getitem__.return_value.call_args
    kernel_num_logits = kwargs["num_logits"]
    assert kernel_num_logits == num_logits
    # Match upstream _is_bonus_logit's next-local-position mask: only the
    # final row of each request can skip verification statistics.
    next_positions = [local_pos[i + 1].item() if i + 1 < kernel_num_logits else 0 for i in range(num_logits)]
    bonus_rows = [i for i, position in enumerate(next_positions) if position == 0]
    assert bonus_rows == (boundaries[1:] - 1).tolist()
    assert sampled.shape == (num_reqs, 4)
    assert counts.shape == (num_reqs,)


@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_draft_sampling_forwards_step_and_watermark_width(monkeypatch, lmhead_tp):
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: lmhead_tp)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_max_num_logits", lambda *_: 4)
    monkeypatch.setattr(
        lmhead_tp_utils,
        "lmhead_tp_pad_rows",
        lambda hidden, capacity, _: torch.cat((hidden, hidden.new_zeros(capacity - hidden.shape[0], hidden.shape[1]))),
    )

    class Parent:
        def sample_draft(self, hidden, positions, mapping, temperature, seeds, step, logits, **kwargs):
            self.forwarded = (hidden, positions, mapping, temperature, seeds, step, logits, kwargs)
            return torch.arange(hidden.shape[0])

    class Drafter(lmhead_tp_utils.LmheadTPDraftSamplingMixin, Parent):
        max_num_reqs = 2
        num_speculative_steps = 3

    drafter = Drafter()
    hidden = torch.zeros(2, 8)
    positions = torch.tensor([5, 9])
    sampled = drafter.sample_draft(
        hidden_states=hidden,
        sample_src_positions=positions,
        idx_mapping=torch.tensor([0, 1]),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        draft_step=torch.tensor(1),
        draft_logits=None,
        spec_step_idx=2,
        num_steps=3,
    )

    assert drafter.forwarded[0].shape[0] == (4 if lmhead_tp else 2)
    assert drafter.forwarded[1] is positions
    assert drafter.forwarded[-1] == {"spec_step_idx": 2, "num_steps": 3}
    torch.testing.assert_close(sampled, torch.arange(2))
