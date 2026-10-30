"""Qualify only the changed generic prepare kernel, under the existing carrier."""

import importlib.util
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def qualify(candidate_root, frozen_runtime):
    import torch
    import torch_npu  # noqa: F401

    candidate = load("prepare60103", Path(candidate_root) / "source/vllm-ascend/vllm_ascend/ops/triton/hisparse.py")
    baseline = load("prepare_api12", Path(frozen_runtime) / "vllm-ascend/vllm_ascend/ops/triton/hisparse.py")
    device = torch.device("npu", torch.npu.current_device())
    results = []
    for owner, rows, query in [("target", 32, 4), ("first_draft", 8, 1), ("later_draft", 8, 1)]:
        requests, topk, hot, pages, block = 8, 2048, 10240, 32, 128
        raw_requests = torch.arange(rows, dtype=torch.int32) // query
        ids = (torch.arange(rows * topk, dtype=torch.int32).reshape(rows, topk) * 7) % (pages * block)
        ids[:, 0] = -1
        ids[:, 1] = pages * block + 1
        indices = ids.to(device)
        request_ids = raw_requests.to(device)
        reference_ids = request_ids.clone()
        states = torch.arange(requests - 1, -1, -1, dtype=torch.int32).to(device)
        source = torch.arange(1, requests * pages + 1, dtype=torch.int32).reshape(requests, pages).to(device)
        resident = source.clone()
        resident[:, ::3] = 0
        dgi = (torch.arange(requests * hot, dtype=torch.int32).reshape(requests, hot) % 3000).to(device)
        valid = torch.tensor([rows], dtype=torch.int32, device=device)

        def outputs(rows=rows, topk=topk, hot=hot):
            return [
                torch.full((rows, topk), -77, dtype=torch.int32, device=device),
                torch.full((rows, topk), -77, dtype=torch.int32, device=device),
                torch.full((rows,), -77, dtype=torch.int32, device=device),
                torch.full((rows, topk), -77, dtype=torch.int64, device=device),
                torch.full((rows, hot), -77, dtype=torch.int64, device=device),
                torch.full((rows, 8), -77, dtype=torch.int32, device=device),
            ]

        actual, expected = outputs(), outputs()
        geometry = dict(
            ROWS=rows,
            TOP_K=topk,
            HOT_SIZE=hot,
            INPUT_STRIDE=topk,
            COUNT_STRIDE=1,
            STATE_ROWS=requests,
            STATE_COUNT=requests,
            HOST_ROWS=(requests * pages + 1) * block,
            SOURCE_STRIDE=pages,
            SOURCE_ROWS=requests,
            SOURCE_COLS=pages,
            SOURCE_BLOCK=block,
            RESIDENT_STRIDE=pages,
            RESIDENT_ROWS=requests,
            RESIDENT_COLS=pages,
            RESIDENT_BLOCK=block,
            RESIDENT_NULL=0,
            HAS_REQUESTS=True,
            HAS_STATE=True,
            HAS_SOURCE=True,
            HAS_RESIDENT=True,
            HAS_RESOLVED=True,
            HAS_COUNTS=True,
            BK=topk,
        )

        def run_candidate(
            rows=rows,
            indices=indices,
            request_ids=request_ids,
            states=states,
            source=source,
            resident=resident,
            dgi=dgi,
            actual=actual,
            valid=valid,
            geometry=geometry,
        ):
            candidate.prepare_residency_kernel[(rows,)](
                indices, request_ids, states, source, resident, dgi, *actual, valid, HAS_VALID_ROWS=True, **geometry
            )

        def run_baseline(
            rows=rows,
            indices=indices,
            reference_ids=reference_ids,
            states=states,
            source=source,
            resident=resident,
            dgi=dgi,
            expected=expected,
            geometry=geometry,
        ):
            baseline.prepare_residency_kernel[(rows,)](
                indices, reference_ids, states, source, resident, dgi, *expected, **geometry
            )

        run_candidate()
        run_baseline()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run_candidate()
        # Clamp boundaries, shrinking active regions, then growing them again.
        for count in [rows, 1, rows // 2, 0, -3, rows + 4]:
            active = max(0, min(rows, count))
            valid.copy_(torch.tensor([count], dtype=torch.int32))
            masked = raw_requests.clone()
            masked[active:] = -1
            reference_ids.copy_(masked)
            graph.replay()
            run_baseline()
            torch.npu.synchronize()
            # Inactive metadata grouping bounds may differ; its invalid state
            # and all material outputs must agree with API12's explicit mask.
            for index in range(5):
                assert torch.equal(actual[index].cpu(), expected[index].cpu()), (owner, count, index)
            assert torch.equal(actual[5][:active].cpu(), expected[5][:active].cpu()), (owner, count, "active metadata")
            if active < rows:
                assert (actual[5][active:, 0:2].cpu() == -1).all()
                assert (actual[0][active:].cpu() == -1).all() and (actual[2][active:].cpu() == 0).all()
            results.append(
                dict(
                    owner=owner,
                    valid_rows=count,
                    effective_rows=active,
                    rows=rows,
                    topk=topk,
                    Hot=hot,
                    graph_replay=True,
                    material_outputs_equal=True,
                )
            )
    return dict(
        passed=True,
        cases=results,
        actual_generic_graph_owners=3,
        changing_replays=18,
        scope="Generic prepare valid-row masking; no model correctness or performance claim",
    )
