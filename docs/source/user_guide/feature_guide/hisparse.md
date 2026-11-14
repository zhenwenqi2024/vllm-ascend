# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend requires a SIMT-capable CANN 9.2
toolchain on Ascend 950.

### Working Principle

HiSparse keeps recent KV rows in the NPU Resident cache and uses the NPU Hot
cache for rows refilled from Host memory. The resolver checks the selected rows
against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

### Usage Scenarios

Use HiSparse with Model Runner V2 for sparse RoPE MLA/SFA models serving long
contexts. BF16 KV and native Ascend 950 C8 (E4M3FN values with FP32 scales) are
supported, with independent Main and Indexer KV dtype selection.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | Ascend 950 with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | C4, TurboQuant and INT8-format C8 are outside the current support scope. |
| Parallelism and transport | Context parallelism is unsupported. For P/D transport, use `HiSparsePDConnector` with a Mooncake TransferEngine build supporting Ascend and matching TP sizes. The selected Ascend transport backend depends on the Mooncake build and runtime configuration. |
| Host storage | Use private Host pools per tensor-parallel rank and Core eager Host mirroring. |
| P/D Host import | The D worker registers a separate pinned Host staging region for Mooncake, with capacity equal to the Host Source pool. Allow for this additional Host memory per TP rank. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native A5 C8, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

### P/D with a Memcache Prefill pool

Run a Prefill instance without HiSparse and a Decode instance with HiSparse.
Use the layerwise P/D proxy with Decode allocation before Prefill. The
Prefill connector saves ordinary KV to Memcache and pushes the Prefill KV
directly to Decode. The Decode instance does not read Memcache.

Prefill `--kv-transfer-config`:

```json
{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"HiSparsePDConnector","kv_role":"kv_producer"},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","use_layerwise":false}}]}}
```

Decode `--kv-transfer-config`:

```json
{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"HiSparseConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"host_pool_gib":4}},{"kv_connector":"HiSparsePDConnector","kv_role":"kv_consumer"}]}}
```

Set distinct Mooncake `kv_port` values for P and D. Use the
[layerwise Push proxy](../../developer_guide/Design_Documents/disaggregated_prefill.md)
so D publishes its allocation before P starts Prefill. This path requires
matching TP sizes, DCP=1, and no KVPP. Both instances must use the same model
and KV format. P writes MLA main KV into D's pinned Host staging and Indexer
KV directly into D NPU. After all P workers report completion, D copies the
imported Host pages into Source and notifies Core; Core restores the last page
to Resident. Account for the additional pinned Host staging capacity on D.

### Verifying the Feature

Send a request and check the server's metrics endpoint for the HiSparse miss
counter. The counter can stay at zero when selected rows are already on the NPU.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:hisparse_cache_misses_total'
```

## Configuration Parameters

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Host pool size in GiB per tensor-parallel rank. Size it for the planned context lengths and concurrency; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |
