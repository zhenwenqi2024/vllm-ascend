# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny real MRv2 state, expert migration, graph counters and ON MLP calibration."""

import io
import logging
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("torch_npu")

from vllm.distributed.eplb import eplb_state as upstream_state
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

from vllm_ascend.distributed.eplb import state as ascend_state
from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator
from vllm_ascend.eplb.diagnostics.assessment import calibrate
from vllm_ascend.eplb.diagnostics.config import EplbDiagnosticsConfig
from vllm_ascend.eplb.diagnostics.model_calibration import capture_template
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe
from vllm_ascend.eplb.diagnostics.runtime import initialize_diagnostics, start_diagnostics
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.eplb import map_to_physical_and_record
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


def _tiny_model(local, device):
    # Retain the actual wrapper/router/weight-owner classes while bypassing
    # serving setup and model loading. All tensors and compute below are real.
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "tiny.layer"
    layer.moe_config = SimpleNamespace(num_experts=4, num_logical_experts=4, experts_per_token=1, ep_size=2)
    layer._use_v2_model_runner = True
    layer._diagnostic_route_ownership_supported = True
    layer.mix_placement = False
    layer.eplb_diagnostic_probe = ExpertLoadProbe(4, device)
    layer.w13_weight = torch.stack(
        [torch.full((128, 256), local * 2 + slot + 1, dtype=torch.bfloat16, device=device) for slot in range(2)]
    )
    layer.w2_weight = torch.stack(
        [torch.full((128, 128), local * 2 + slot + 1, dtype=torch.bfloat16, device=device) for slot in range(2)]
    )
    layer_state = ascend_state.AscendEplbLayerState()
    router = FusedTopKRouter(top_k=1, global_num_experts=4, eplb_state=layer_state)
    layer.router = router
    wrapper = AscendMoERunner.__new__(AscendMoERunner)
    torch.nn.Module.__init__(wrapper)
    wrapper.layer_name = "tiny.layer"
    wrapper.router, wrapper.routed_experts = router, layer
    model = torch.nn.Module()
    model.add_module("layer", wrapper)
    model.moe_layers = [wrapper]
    model.num_moe_layers = 1
    return model, layer, layer_state


def _template(layer):
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(method)
    method.moe = SimpleNamespace(has_bias=False)
    method.dynamic_eplb = False  # Actual MRv2 NONE path keeps dense MLP weights.
    method._lora_routing = method.lora_context = None
    payload = MoEMlpComputeInput(
        hidden_states=torch.ones((16, 128), dtype=torch.bfloat16, device=layer.w13_weight.device),
        group_list=torch.tensor([8, 8], dtype=torch.int64, device=layer.w13_weight.device),
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(w1=layer.w13_weight, w2=layer.w2_weight),
        quant=MoEQuantParams(quant_type=QuantType.NONE),
        fusion=False,
        activation=MoEActivation.SILU,
        layer=layer,
        dynamic_eplb=False,
    )
    return method, payload


@torch.inference_mode()
def _worker(rank, rendezvous):
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        members = ([0, 2], [1, 3])
        groups = [dist.new_group(ranks, backend="gloo") for ranks in members]
        stage, local = rank % 2, members[rank % 2].index(rank)
        group = SimpleNamespace(
            ranks=members[stage], rank_in_group=local, cpu_group=groups[stage], device_group=groups[stage]
        )
        with pytest.MonkeyPatch.context() as patch, torch_cuda_wrapper():
            # Substitute only group lookup/configuration, not communication,
            # routing, state updates, device timing, or MLP implementations.
            patch.setattr(ascend_state, "get_ep_group", lambda: group)
            patch.setattr(upstream_state, "get_ep_group", lambda: group)
            patch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(ranks=[rank]))
            patch.setattr("vllm.distributed.parallel_state.get_pp_group", lambda: SimpleNamespace(rank_in_group=stage))
            patch.setattr("vllm_ascend.distributed.parallel_state.get_mc2_group", lambda: group)
            _run_case(rank, local, device, group)
    finally:
        dist.destroy_process_group()


def _run_case(rank, local, device, group):
    model, layer, layer_state = _tiny_model(local, device)
    parallel = SimpleNamespace(enable_expert_parallel=True, enable_eplb=True, enable_elastic_ep=False)
    state = ascend_state.AscendEplbState(parallel, device)
    state.expert_load_window_size = state.expert_rearrangement_step_interval = 8
    state.should_record_tensor = torch.tensor(True, device=device)
    mapping = torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4, 1)
    replica_count = torch.ones((1, 4), dtype=torch.int32, device=device)
    load_pass = torch.zeros((1, 4), dtype=torch.int32, device=device)
    layer_state.set_layer_state(0, load_pass, mapping, replica_count)
    communicator = AscendGlooEplbCommunicator(group.cpu_group, torch.npu.current_stream())
    model_state = upstream_state.EplbModelState(
        physical_to_logical_map=torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4),
        logical_to_physical_map=mapping,
        logical_replica_count=replica_count,
        expert_load_pass=load_pass,
        expert_load_window=torch.zeros((8, 1, 4), dtype=torch.int32, device=device),
        model_name="tiny",
        model=model,
        expert_buffer=[],
        rebalanced=False,
        eplb_stats=None,
        cuda_device_index=rank,
        communicator=communicator,
    )
    state.model_states["tiny"] = model_state
    runner = SimpleNamespace(
        model=model,
        device=device,
        eplb=SimpleNamespace(state=state),
        ascend_config=SimpleNamespace(
            eplb_diagnostics=EplbDiagnosticsConfig(mode="benefit", window_size=2, warmup_steps=0)
        ),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=parallel,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16]),
        ),
    )
    initialize_diagnostics(runner)
    start_diagnostics(runner)
    recorder = runner._eplb_diagnostics_recorder
    probe = layer.eplb_diagnostic_probe
    assert recorder.layers == [("tiny.layer", layer)]
    assert state._eplb_diagnostic_monitor is model_state._eplb_diagnostic_monitor is recorder.monitor
    assert communicator._eplb_diagnostic_monitor is recorder.monitor
    assert probe.logical_to_physical.data_ptr() == layer_state.logical_to_physical_map.data_ptr()
    method, payload = _template(layer)
    assert capture_template(probe, payload, method, "MC2CommImpl") is None
    logical_ids = torch.tensor([[0], [0], [0], [1], [1], [2], [3]], dtype=torch.int32, device=device)

    def record():
        physical_ids = map_to_physical_and_record(
            logical_ids,
            layer_state.expert_replica_routing_table,
            layer_state.expert_load_view,
            state.should_record_tensor,
            probe.source_token_count,
        )
        probe.record_routes(physical_ids)
        probe.record_comm(1)

    record()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        record()
    load_pass.zero_()
    table_pointer = layer_state.expert_replica_routing_table.data_ptr()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("vllm.eplb.diagnostics")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        for step in range(4):
            if step == 2:
                # Move actual tiny experts using the production staged Gloo
                # communicator and refresh the real MRv2 routing state.
                slot = 1 if local == 0 else 0
                tensors = [layer.w13_weight[slot], layer.w2_weight[slot]]
                communicator.set_transfer_context(None, 0)
                communicator.add_send(tensors, dst_rank=1 - local, expert_id=slot)
                communicator.add_recv(tensors, src_rank=1 - local, expert_id=slot)
                assert recorder.monitor.observing and not recorder.monitor.active
                communicator.execute()
                recorder.monitor.active = True
                mapping.copy_(torch.tensor([[[0], [2], [1], [3]]], dtype=torch.int32, device=device))
                model_state.physical_to_logical_map.copy_(mapping[..., 0])
                ascend_state.refresh_model_routing_tables(model_state)
                recorder.monitor.active = False
                assert probe.generation == 1
                assert layer_state.expert_replica_routing_table.data_ptr() == table_pointer
                assert tensors[0][0, 0].item() == (3 if local == 0 else 2)
            recorder.begin(6)
            recorder.metadata[-1] = {"phase": "decode", "graph_mode": "FULL", "padded_tokens": 7}
            graph.replay()
            recorder.end()
            # MRv2 advances EPLB after execute_model, when end() already ran.
            state.step()
        original_weights = [value.cpu().clone() for value in (layer.w13_weight, layer.w2_weight)]
        original_mapping = mapping.cpu().clone()
        calibrate(recorder, samples=2, repeats=2)
        assert recorder.calibrated
        costs = recorder.costs[local]
        assert costs["pending"] == costs["dropped"] == 0
        assert costs["transfers"] == [
            dict(
                layer="tiny.layer",
                direction=direction,
                locality="same_node",
                payload_bytes=98304,
                tensor_ops=2,
                submissions=1,
            )
            for direction in ("send", "recv")
        ]
        components = {(item["kind"], item["component"]): item for item in costs["totals"]}
        assert components[("device", "eplb_step")]["count"] == 4
        assert components[("device", "migration_pipeline")]["sum_ms"] > 0
        assert components[("device", "routing_table_refresh")]["sum_ms"] > 0
        torch.testing.assert_close(mapping.cpu(), original_mapping, rtol=0, atol=0)
        for actual, expected in zip((layer.w13_weight, layer.w2_weight), original_weights):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if local == 0:
            job = recorder.jobs["tiny.layer"]
            assert job["loads"] == [[6, 4, 2, 0], [6, 4, 2, 0]]
            assert job["initial_placement"] == ((0, 1), (2, 3))
            assert job["current_placement"] == ((0, 2), (1, 3))
            assert "calibrated_layers=1" in output.getvalue()
            assert "[EPLB migration]" in output.getvalue() and "98304" in output.getvalue()
            assert "estimated_net_saving_ms=unknown" in output.getvalue()
        else:
            assert not output.getvalue()
    finally:
        logger.removeHandler(handler)


def test_real_mrv2_commit_graph_history_and_enabled_mlp_assessment(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "mrv2-init").as_uri(),), nprocs=4, join=True)
