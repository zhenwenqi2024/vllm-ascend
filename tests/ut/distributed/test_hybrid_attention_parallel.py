from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.hybrid_attention_parallel import (
    gather_dp_tokens,
    make_global_state_slot,
    reduce_scatter_dp_tokens,
)


class FakeGroup:
    world_size = 2

    def all_gather(self, tensor, dim):
        assert dim == 0
        return torch.cat((tensor, tensor + 10), dim=0)

    def reduce_scatter(self, tensor, dim):
        assert dim == 0
        return tensor.chunk(self.world_size, dim=0)[0]


def test_global_state_slot_is_unique_across_owner_ranks():
    assert make_global_state_slot(0, 3, 8) == 3
    assert make_global_state_slot(1, 3, 8) == 11


@pytest.mark.parametrize("owner_rank, local_slot, slots_per_rank", [(-1, 0, 8), (0, -1, 8), (0, 8, 8)])
def test_global_state_slot_rejects_invalid_values(owner_rank, local_slot, slots_per_rank):
    with pytest.raises(ValueError):
        make_global_state_slot(owner_rank, local_slot, slots_per_rank)


def test_token_exchange_preserves_local_token_count():
    local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    group = FakeGroup()

    gathered, layout = gather_dp_tokens(local, group, exchange_num_tokens=3)

    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[:2], local)
    assert torch.equal(gathered[2], torch.zeros(2))

    restored = reduce_scatter_dp_tokens(gathered, group, layout)
    assert torch.equal(restored, local)


def test_token_exchange_rejects_insufficient_capacity():
    group = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="exceeds exchange_num_tokens"):
        gather_dp_tokens(torch.empty(3, 4), group, exchange_num_tokens=2)
