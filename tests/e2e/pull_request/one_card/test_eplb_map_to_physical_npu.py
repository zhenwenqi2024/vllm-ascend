# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.fused_moe.eplb import map_to_physical_and_record


def test_map_to_physical_and_record_runs_on_npu_without_host_gating():
    routing_table = torch.tensor(
        [[0, 3], [2, 1], [0, 3], [2, 1]],
        dtype=torch.int32,
        device="npu",
    )
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=torch.int32,
        device="npu",
    )
    expert_load = torch.zeros(4, dtype=torch.int32, device="npu")
    record_enabled = torch.tensor(True, device="npu")
    num_unpadded_tokens = torch.tensor(3, dtype=torch.int32, device="npu")

    physical_ids = map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled,
        num_unpadded_tokens,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        physical_ids.cpu(),
        torch.tensor([[0, 3], [2, 1], [0, 3], [2, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )

    record_enabled.fill_(False)
    map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled,
        num_unpadded_tokens.fill_(4),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )


@pytest.mark.parametrize("tokens", [1, 32, 33, 1025])
@pytest.mark.parametrize("capture_enabled", [False, True])
def test_recording_gate_changes_on_graph_replay(tokens, capture_enabled):
    ids_cpu = torch.arange(tokens * 8, dtype=torch.int32).reshape(tokens, 8) % 4
    ids_cpu[0, 0] = -1
    ids_cpu[0, 1] = 4
    ids = ids_cpu.to("npu")
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    table = mapping.unsqueeze(0).to("npu")
    load = torch.full((4,), 7, dtype=torch.int32, device="npu")
    enabled = torch.tensor(capture_enabled, device="npu")
    real_tokens = torch.tensor(tokens, dtype=torch.int32, device="npu")
    for _ in range(3):
        map_to_physical_and_record(ids, table, load, enabled, real_tokens)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = map_to_physical_and_record(ids, table, load, enabled, real_tokens)
    load.fill_(7)
    expected_load = torch.full((4,), 7, dtype=torch.int32)
    expected_ids = torch.where((ids_cpu >= 0) & (ids_cpu < 4), mapping[ids_cpu.clamp(0, 3)], -1)
    # Preserve existing counts while toggling both directions without recapture.
    for record, count in ((False, tokens), (True, tokens), (False, tokens), (True, tokens // 2), (True, 0)):
        enabled.fill_(record)
        real_tokens.fill_(count)
        graph.replay()
        torch.npu.synchronize()
        if record:
            selected = expected_ids[:count].flatten()
            expected_load += torch.bincount(selected[selected >= 0].long(), minlength=4).int()
        torch.testing.assert_close(output.cpu(), expected_ids)
        torch.testing.assert_close(load.cpu(), expected_load)
