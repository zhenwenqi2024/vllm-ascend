# Kimi K3 Hybrid Attention Parallel Prototype

## Goal

This prototype keeps MLA request execution and latent KV cache data parallel,
while preparing a fine-grained tensor-parallel group for KDA. The same existing
fine-grained O-projection group can shard MLA `o_proj` weights.

```text
DP-local MLA -> KDA group gather -> KDA head shards -> reduce-scatter -> DP-local output
```

## Configuration

```json
{
  "finegrained_tp_config": {
    "oproj_tensor_parallel_size": 8,
    "kda_tensor_parallel_size": 8
  }
}
```

The initial topology requires standard tensor parallel size one. Both
fine-grained sizes divide the data-parallel size and are formed along the DP
axis. They may therefore reuse the same physical ranks without changing MLA KV
ownership.

## Implemented foundation

- Typed `kda_tensor_parallel_size` configuration and validation.
- A dedicated KDA process group initialized with the existing fine-grained TP
  group builder.
- Fixed-capacity token gather/reduce-scatter helpers for graph-safe integration.
- Collision-free `(owner_rank, local_slot)` KDA state slot mapping.
- Existing `oproj_tensor_parallel_size` remains the MLA O-projection sharding
  mechanism.

## Remaining runtime integration

The Kimi K3 KDA module must use the KDA group for projection weight loading and
head/state sharding. Its metadata builder must gather request boundaries,
positions, and state slots in the same rank order as hidden states. Until that
integration lands, setting `kda_tensor_parallel_size` only initializes the
experimental group and must not be used for production serving.
