# Kimi K3 KDA State Parallelism

## Goal

KDA state parallelism (KSP) keeps MLA and the model's external execution data
parallel, but shards the KDA head dimension over a fine-grained group carved
from the DP axis. The KDA projections, convolution, recurrent calculation, and
state are all head-local inside that group.

```text
                   KSP group (formed from DP ranks)
DP owner input ── all-gather ──> local KDA head shard
                                  │ in_proj / conv / SSM / O partial
DP owner output <─ reduce-scatter ┘

MLA path: DP-local input ──> MLA ──> DP-local output (unchanged)
```

This replaces the weight-only prototype's communication around every KDA
linear with one activation all-gather and one output reduce-scatter per KDA
layer.

## Configuration

```json
{
  "finegrained_tp_config": {
    "kda_tensor_parallel_size": 8
  }
}
```

Standard tensor parallel size must be one. `kda_tensor_parallel_size` must
divide both the data-parallel size and KDA head count. All ranks in one KSP
group must use the same model, cache, and speculative-decoding configuration.

For MTP or DSpark, use the native multi-slot state path:

```text
--mamba-cache-mode all
```

Speculative `align` mode and RecoverSSM are rejected until their postprocess
kernels can address an owner dimension. PD disaggregation is also rejected in
this first version; transferring a whole physical cache block would mix
independent DP owners.

## State ownership

DP schedulers allocate block IDs independently, so `(owner=0, block=5)` and
`(owner=1, block=5)` are different logical states. KSP encodes that distinction
inside every physical cache page:

```text
conv:      [num_blocks, ksp_size, local_conv_channels, conv_width]
recurrent: [num_blocks, ksp_size, local_heads, head_dim, head_dim]

kernel_slot = logical_block_id * ksp_size + owner_rank
```

The KDA kernel consumes a zero-copy flattened `[num_blocks * ksp_size, ...]`
view. Prefix-cache block moves are exported by a small optional vLLM callback,
gathered once per model step, and then applied only to the matching owner slice
on every head-shard rank.

Because each KSP rank processes requests from every DP owner, the owner
dimension exactly offsets the head sharding. KDA state bytes per device are
therefore approximately unchanged. The memory saving comes from the large KDA
weights (including convolution weights), which are divided by `ksp_size`.

## Step execution

```text
model runner
  ├─ DP-wide token padding (existing path)
  ├─ build local attention metadata
  ├─ gather KDA metadata + prefix-copy plans once
  └─ model forward
       ├─ MLA layer: unchanged, local owner only
       └─ KDA layer
            1. all-gather hidden states in rank-major owner order
            2. run local in_proj, f_b, conv, KDA state update, norm
               for each owner's metadata and owner state slice
            3. run local O-projection partial
            4. reduce-scatter partials back to each DP owner
```

Uniform DP padding gives the two activation collectives fixed split sizes in
eager and ACL graph execution. Metadata federation is deliberately outside the
captured graph. The first implementation uses the CPU control group for
correctness; replacing it with persistent packed device buffers is the main
decode-latency optimization left for hardware tuning.

## Compatibility

| Feature | Status | Notes |
| --- | --- | --- |
| Prefix cache | Supported | Owner-aware logical block-copy plans |
| ACL graph | Structurally supported | Fixed activation shapes; NPU capture validation required |
| MTP / DSpark | Supported with `mamba_cache_mode=all` | Speculative align and RecoverSSM fail closed |
| PD disaggregation | Not yet supported | Needs owner-aware KDA-state transfer |
| Mixed KDA quantization | Supported by construction | Existing mixed projection loaders use the KSP rank |
| MLA | Unchanged | Remains DP-local; MLA O weight-only sharding remains optional in the existing path |

## Required vLLM extension

The paired vLLM change adds an optional `copy_plan_callback` to
`preprocess_mamba`. With no callback, behavior is byte-for-byte the existing
pointer-copy path. KSP supplies the callback to export logical source and
destination block indices before vLLM turns them into device-specific pointers.
