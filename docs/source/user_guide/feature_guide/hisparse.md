# HiSparse on Ascend

## Overview

HiSparse keeps sparse MLA KV pages in host memory and uses NPU memory for resident pages and a per-request hot cache. During decode, the sparse indexer selects logical tokens. The NPU resolves those tokens to resident or hot-cache rows and reads missing rows from mapped host memory. This allows the main KV history to exceed the resident device cache.

This experimental implementation requires the matching vLLM platform-backend changes. It has no released version yet. The development environment uses Ascend 950, CANN 9.2, torch-npu 2.10, and Triton-Ascend 3.6 development builds.

```text
Sparse indexer -> logical top-k tokens -> NPU residency resolver
                                             |
                               resident row / hot-cache hit / host miss
                                             |
                              mapped-host gather for missing KV rows
                                             |
                              shared packed KV + physical top-k -> direct sparse attention
```

The resolver protects the union of tokens selected by all MTP query rows before choosing eviction slots. It uses device sorting and Triton-Ascend kernels with persistent scratch buffers. Decode does not copy top-k indices to the CPU for transfer planning.

The adapter reuses upstream scheduling, prefix-cache ownership, page spills, and host-mirror completion. The direct operator reads the shared packed latent/RoPE cache through explicit 64-bit page and row addresses. It moves invalid selections to the end while preserving valid order and duplicates, and excludes padding from softmax using a per-query valid count. The indexer supplies causal selections before physical address resolution: physical row numbers are not sequence positions.

One direct call processes the complete query batch without materializing a separate selected KV tensor for every query. The old bridge, retained as a test oracle, split the batch into groups of 32 queries and copied selected rows into latent and RoPE buffers. Stable compaction still has a cost; throughput and latency must be measured for the intended workload. Prefill still uses the upstream historical-page staging path before attention, so direct attention does not eliminate that transfer or staging allocation.

KV preprocessing uses the same fused RMSNorm/RoPE operator as native SFA before copying the result into HiSparse's packed cache. A reusable split cache preserves the native operator's BF16 rounding and is allocated during profiling, before graph capture. Local temporary slots include padding; the final packed write masks invalid slots and inactive rows.

## Requirements and limits

| Area | Requirement |
| --- | --- |
| Hardware | Ascend 950 with AIV access to registered host memory; runtime capability is checked |
| Runner | Model Runner V2 |
| Model path | Sparse MLA/DSA with a sparse-index group |
| KV formats | BF16 indexer KV; BF16 main KV or experimental native A5 attention C8 (see validation status below) |
| Context parallelism | Prefill and decode context-parallel sizes must both be 1 |
| Sparse selection | SFA requires top-k divisible by 128 |
| Direct operator | BF16 latent 512 + RoPE 64, 8–64 local heads divisible by 8; selected capacity at most 32768 |
| Hot cache | Representation supports at most 32768 aligned rows; NPU tests cover capacities through 10240. Capacity must cover the complete query-row union |
| Host allocation | Private pool per worker rank; account for the sum across TP ranks |
| Native extension | Build the Ascend 950 extension and `custom_transformer` vendor, including `hisparse_flash_attention` |

Mapped-host allocations remain registered until work using their addresses has completed. Host pools must remain alive across graph replay. Initialization fails if the hardware lacks the required mapping capability.

The normal native extension registers `hisparse_direct::sparse_flash_attention` once. Its matching CANN operator is included in the A5 custom-op build list. Rebuild both after updating these sources; a Python-only install cannot provide the new operator. Do not load a standalone private binding in a process that already imports the normal extension. Private bindings are limited to separate operator probes.

## Configuration

Set `VLLM_USE_V2_MODEL_RUNNER=1`, then add these options to a compatible model's launch command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

The example requests 4 GiB of host pool per rank, or 32 GiB across TP8. Size the pool for the model's layers and intended context capacity. Select the model's usual weight-quantization option separately.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `kv_connector_extra_config.host_pool_gib` | Required, positive | Host pool capacity in GiB per rank |
| `attention_config.indexer_kv_dtype` | Model dependent | Set `bf16` for this implementation |
| `attention_config.hisparse_config.device_buffer_size` | Derived from top-k and query-row count | Per-request hot-cache capacity, rounded to the cache block size |
| `attention_config.hisparse_config.eager_host_mirror` | `true` | Mirror decode KV writes during forward; prefill writes are always mirrored |

Omit `device_buffer_size` initially so the upstream allocator accounts for the number of MTP query rows. Changing MTP length changes the required union capacity.

## Operator validation

From the source checkout, with the Ascend environment and project virtual environment active:

```bash
cd /path/to/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_direct_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py \
  tests/e2e/pull_request/one_card/test_hisparse_selection_order.py \
  tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py -v
```

These tests compare state transitions and KV payloads with a CPU reference. They cover resident hits, hot-cache hits, host misses, reordered request state, padded rows, physical cache strides, compact gathers, invalidation after KV writes, and ACL graph replay. Large cases include top-k 2048 and four query rows with an 8192-row union. The attention tests independently compute QK, masked softmax, and weighted values in FP32 on CPU, including duplicate selections and empty rows. `--confcutdir` keeps these tests independent of the model-download fixtures. Passing operator tests alone does not establish full-model accuracy or performance.

The prefill staging tests check resident and host rows across shared-layer cache views and different resident page sizes. The selection-order tests check interleaved padding, duplicates, empty rows, and changed selections during graph replay.

The KV preprocessing tests include a fixed BF16 activation row that exposes a rounding difference between standalone and fused RMSNorm. They require native fused results to match exactly, including strided packed destinations, invalid slots, padding, and graph replay with changed inputs.


## Platform backend integration

The adapter targets vLLM PR [#62592](https://github.com/vllm-project/vllm/pull/62592),
commit `d9f25459725c8fd9db07b7d9571a3f5be39db592`. Keep both repository
revisions pinned when validating this draft interface.

`NPUPlatform.get_hisparse_backend_cls()` selects
`vllm_ascend.distributed.kv_transfer.hisparse.backend.AscendHiSparseBackend`.
The common runtime retains request scheduling, TP coordination, cache allocation
ownership and synchronization before release. The plugin implements mapped Host
registration, DMA, resolver/gather operators, KV view binding and NPU streams/events.
No monkey patch of the common runtime is needed for resident FP8 row indexing:
`gather_resident_rows()` reinterprets FP8 tensors as `uint8` before indexing and
copies the complete packed payload without numerical conversion. Physical page
strides and layer storage offsets are preserved.

Native Ascend SFA prefill calls `stage_prefill_rows()`, which combines resident
rows through this backend hook with Host rows through `HiSparseNPUOps`.
For native attention C8, the resulting opaque rows retain the Ascend order of
512 NoPE bytes, 128 RoPE bytes and 16 scale bytes; SFA consumes that layout.
The CUDA `fp8_ds_mla` prefill conversion hook uses a different layout and is
explicitly rejected by this backend. It must not silently reinterpret Ascend C8.
This adapter does not add another quantizer or change indexer precision.

Host pools remain private per worker rank (`supports_shared_host_pool=False`).
Local Event creation supports the common `interprocess=False` signature. NPU
IPC Event creation/export/import explicitly reject calls until their ownership
and completion protocol is implemented and validated. This does not affect
ordinary local copy-stream completion events.

### Validation status

The existing integration evidence predates this interface update. The
`docs/hisparse_integration_validation_20261020.json` manifest describes its
original frozen snapshot; it does not qualify the adapter changes above.
CPU payload tests cover BF16/E4M3/E5M2 bytes, page gaps, layer offsets, duplicate
and padded row IDs, geometry errors and private-event behavior. The NPU prefill
staging suite includes graph capture/replay with changed row IDs and payloads.
These NPU cases and the full model have not yet been rerun against the paired
vLLM revision. C8 model accuracy remains pending; no speedup is claimed here.
