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

The topology requires standard tensor parallel size one and
`kda_tensor_parallel_size` dividing the data-parallel size. KDA and MLA state
ownership is unchanged. Eager and full-decode ACL graph execution share the
same communication layout.

## Execution

- Before model execution, the model runner synchronizes the token count over
  DP and pads every rank to the same token capacity. This also aligns graph
  selection across ranks.
- Column-parallel projections gather the uniformly padded DP token batches, calculate their
  local weight shard, and all-to-all the projection shards back to each request
  owner. Packed projections restore checkpoint partition order and retain only
  one copy of replicated `f_a` and alignment padding.
- Row-parallel output projections split each owner's full attention result,
  all-to-all input shards to the weight owners, calculate partial outputs, and
  all-to-all the partial results back for summation by the request owner.
- Uneven DP workloads are represented by padding at the model-runner boundary;
  no projection performs a token-count collective or device-to-host read.

## Sharded and replicated data

The KDA packed input projections, mixed-quantization QKV/gate projections,
`f_b_proj`, KDA `o_proj`, and MLA `o_proj` are sharded. Convolution weights,
`A_log`, `dt_bias`, normalization weights, convolution state, recurrent state,
MLA KV cache, and attention metadata remain fully DP-local. Consequently no
state index remapping or scheduler changes are required.

Full-decode ACL graphs capture fixed-shape all-gather and all-to-all operations.
Prefill outside the graph uses the same DP-wide padded token shape. Future work
can replace the two-way O-projection exchange with a fused GEMM/collective.
