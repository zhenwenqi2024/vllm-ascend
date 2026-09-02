import pytest
import torch

from vllm_ascend.distributed.hybrid_attention_parallel import (
    gather_token_counts,
    pad_tokens,
)


class FakeGroup:
    world_size = 2

    def all_gather(self, tensor, dim):
        assert dim == 0
        return torch.tensor([tensor.item(), 5], dtype=tensor.dtype)


def test_gather_token_counts_supports_uneven_dp_batches():
    counts, capacity = gather_token_counts(torch.empty(3, 4), FakeGroup())
    assert counts == [3, 5]
    assert capacity == 5


def test_pad_tokens_preserves_values_and_clears_padding():
    local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded = pad_tokens(local, 3)
    assert torch.equal(padded[:2], local)
    assert torch.equal(padded[2], torch.zeros(2))


def test_token_exchange_rejects_insufficient_capacity():
    with pytest.raises(ValueError, match="capacity is 2"):
        pad_tokens(torch.empty(3, 4), 2)
