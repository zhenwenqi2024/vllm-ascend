# Kimi K3 Weight-Only Attention Parallelism

## Goal

This feature keeps MLA and KDA execution, metadata, and state data parallel,
while sharding the large KDA projections and MLA output projection over a
fine-grained group formed along the DP axis.

```text
DP-local input -> distributed weight shards -> DP-owner output -> DP-local KDA state
```

## Configuration

```json
{
  "finegrained_tp_config": {
    "kda_tensor_parallel_size": 8
  }
}
```

The topology requires standard tensor parallel size one, eager execution, and
`kda_tensor_parallel_size` dividing the data-parallel size. KDA and MLA state
ownership is unchanged.

## Execution

- Column-parallel projections gather padded DP token batches, calculate their
  local weight shard, and all-to-all the projection shards back to each request
  owner. Packed projections restore checkpoint partition order and retain only
  one copy of replicated `f_a` and alignment padding.
- Row-parallel output projections split each owner's full attention result,
  all-to-all input shards to the weight owners, calculate partial outputs, and
  all-to-all the partial results back for summation by the request owner.
- Uneven token counts across DP ranks are supported through a dynamically
  exchanged token capacity.

## Sharded and replicated data

The KDA packed input projections, mixed-quantization QKV/gate projections,
`f_b_proj`, KDA `o_proj`, and MLA `o_proj` are sharded. Convolution weights,
`A_log`, `dt_bias`, normalization weights, convolution state, recurrent state,
MLA KV cache, and attention metadata remain fully DP-local. Consequently no
state index remapping or scheduler changes are required.

The initial implementation is eager-only because reading uneven token counts
introduces a device-to-host synchronization. ACL graph support requires a
fixed-capacity exchange layout and is left for a follow-up.
