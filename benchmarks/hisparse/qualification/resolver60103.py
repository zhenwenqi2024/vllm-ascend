"""Payload for an existing exclusive-lease NPU qualification launcher.

No accelerator is initialized on import. This is primitive correctness, not a
model accuracy/performance test. The caller owns admission, identity and reap.
"""

import importlib.util
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualify(candidate_root, frozen_runtime):
    import torch
    import torch_npu  # noqa: F401

    candidate_root = Path(candidate_root)
    ops = candidate_root / "source/vllm-ascend/vllm_ascend/ops"
    frozen_ops = Path(frozen_runtime) / "vllm-ascend/vllm_ascend/ops"
    torch.ops.load_library(str(frozen_ops / "torch_binding.so"))
    torch.ops.load_library(str(candidate_root / "native/torch_binding.so"))
    before = load("frozen_resolver", frozen_ops / "hisparse_ascendc.py")
    after = load("candidate_resolver", ops / "hisparse_ascendc.py")
    invalidator = load("candidate_invalidator", ops / "triton/hisparse_invalidate_groups.py")
    results = []
    device = torch.device("npu", torch.npu.current_device())

    def npu(value):
        return value.to(device)

    # Each graph owns its descriptor addresses, counters and scratch. Inputs
    # change through the same storage for every replay, including valid_rows.
    def qualify_owner(owner, rows, query):
        requests = rows // query
        topk, hot, pages, block = 2048, 10240, 32, 128
        raw_ids = torch.arange(rows, dtype=torch.int32) // query
        source = torch.arange(1, requests * pages + 1, dtype=torch.int32).reshape(requests, pages)
        hot_table = torch.arange(
            1 + requests * pages, 1 + requests * (pages + hot // block), dtype=torch.int32
        ).reshape(requests, hot // block)
        cache = torch.empty(
            (1 + requests * (pages + hot // block), block, 656), dtype=torch.float8_e4m3fn, device=device
        )
        host = torch.empty(((requests * pages + 1) * block, 656), dtype=torch.float8_e4m3fn)

        def make():
            result = dict(
                host_cache=host,
                hot_cache=cache,
                hot_block_table=npu(hot_table),
                global_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
                hot_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
                device_global_indices=torch.empty((requests, hot), dtype=torch.int32, device=device),
                lru_slots=torch.empty((requests, hot), dtype=torch.int16, device=device),
                request_state_indices=torch.empty(requests, dtype=torch.int32, device=device),
                region_stride=hot,
                max_union_rows=8192,
                miss_mask=torch.empty((rows, topk), dtype=torch.int32, device=device),
                stats=torch.empty(2, dtype=torch.int64, device=device),
                attention_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
                attention_block_stride=5 * block,
                request_ids=torch.empty(rows, dtype=torch.int32, device=device),
                source_block_table=npu(source),
                source_block_size=block,
                resolved_global_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
                valid_counts=torch.empty(rows, dtype=torch.int32, device=device),
                swap_host_physical_rows=torch.empty((rows, topk), dtype=torch.int32, device=device),
                swap_device_physical_rows=torch.empty((rows, topk), dtype=torch.int32, device=device),
                swap_counts=torch.empty(rows, dtype=torch.int32, device=device),
                resident_block_table=torch.empty((requests, pages), dtype=torch.int32, device=device),
                resident_block_size=block,
                resident_null_block=0,
                num_valid_rows=torch.empty(1, dtype=torch.int32, device=device),
            )
            return result

        old, new = make(), make()

        def reset(generation):
            kind = generation % 4  # Resident, Hot hits, all misses, mixed hits.
            valid = [rows, rows - 1, 0, 1, rows // 2, rows][generation % 6]
            states = torch.arange(requests, dtype=torch.int32).roll(generation % requests)
            resident = torch.zeros((requests, pages), dtype=torch.int32)
            dgi = torch.full((requests, hot), -1, dtype=torch.int32)
            for request in range(requests):
                state = int(states[request])
                if kind == 0:
                    resident[state] = source[request]
                if kind in (1, 3):
                    values = source[request, 0] * block + torch.arange(pages * block, dtype=torch.int32)
                    if kind == 3:
                        values[::2] = -1
                    dgi[state, : pages * block] = values
            indices = (
                torch.arange(topk, dtype=torch.int32)[None, :]
                + torch.arange(rows, dtype=torch.int32)[:, None] * 31
                + generation * 7
            ) % (pages * block)
            common = dict(
                global_indices=indices,
                device_global_indices=dgi,
                resident_block_table=resident,
                lru_slots=torch.arange(hot, dtype=torch.int16).repeat(requests, 1),
                request_state_indices=states,
            )
            for data in (old, new):
                for name, value in common.items():
                    data[name].copy_(value)
                for name in [
                    "hot_indices",
                    "resolved_global_indices",
                    "attention_indices",
                    "miss_mask",
                    "swap_host_physical_rows",
                    "swap_device_physical_rows",
                ]:
                    data[name].fill_(-33)
                data["swap_counts"].fill_(-33)
                data["valid_counts"].fill_(-33)
                data["stats"].zero_()
                data["num_valid_rows"].fill_(valid)
            old["request_ids"].copy_(torch.where(torch.arange(rows) < valid, raw_ids, -1))
            new["request_ids"].copy_(raw_ids)
            return kind, valid

        reset(0)
        old_bound = before.BoundResolver(old, None)
        new_bound = after.BoundResolver(new, None)
        graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
        torch.npu.synchronize()
        with torch.npu.graph(graphs[0]):
            old_bound()
        with torch.npu.graph(graphs[1]):
            new_bound()
        for generation in range(18):
            kind, valid = reset(generation)
            graphs[0].replay()
            graphs[1].replay()
            torch.npu.synchronize()
            for name in [
                "hot_indices",
                "device_global_indices",
                "lru_slots",
                "resolved_global_indices",
                "attention_indices",
                "miss_mask",
                "valid_counts",
                "swap_host_physical_rows",
                "swap_device_physical_rows",
                "swap_counts",
                "stats",
            ]:
                torch.testing.assert_close(old[name], new[name], rtol=0, atol=0, msg=f"{owner}/{name}")
            torch.testing.assert_close(old_bound.errors, new_bound.errors, rtol=0, atol=0)
            if kind == 0:
                assert torch.count_nonzero(new["swap_counts"]).cpu().item() == 0
            results.append(dict(owner=owner, generation=generation, kind=kind, valid_rows=valid, exact_equal=True))

    for owner, rows, query in [("target", 32, 4), ("first_draft", 8, 1), ("later_draft", 8, 1)]:
        qualify_owner(owner, rows, query)

    # Independent group allocations and excluded groups catch accidental use
    # of the first group's address for every program. IDs remain integers.
    for hot in [17, 257, 10240]:
        for dtype in [torch.int32, torch.int64]:
            cpu = [torch.arange(4 * hot, dtype=torch.int32).reshape(4, hot).remainder(23) for _ in range(4)]
            tables = [npu(t) for t in cpu]
            states = torch.tensor([1, 0, 3, -1], dtype=torch.int32)
            reqs = torch.tensor([0, 0, -1, 7, 1, 2, 3], dtype=torch.int32)
            slots = torch.tensor([7, 7, 4, 5, 9, 11, -1], dtype=dtype)
            for request, slot in zip(reqs.tolist(), slots.tolist()):
                if 0 <= request < len(states) and 0 <= int(states[request]) < 4 and slot >= 0:
                    for table in cpu[:3]:
                        row = table[states[request]]
                        row[row == slot] = -1
            addresses = torch.tensor([t.data_ptr() for t in tables[:3]], dtype=torch.int64, device=device)
            invalidator.invalidate_written_groups(addresses, npu(states), npu(reqs), npu(slots), 4, hot)
            torch.npu.synchronize()
            for table, expected in zip(tables, cpu):
                torch.testing.assert_close(table.cpu(), expected, rtol=0, atol=0)
            results.append(dict(kind="multi_group_invalidation", hot=hot, dtype=str(dtype), exact_equal=True))
    return dict(passed=True, primitive_only=True, model_qualified=False, performance_qualified=False, cases=results)


if __name__ == "__main__":
    raise SystemExit(
        "Run only through an existing exclusive-resource qualification controller; import performs no NPU work."
    )
