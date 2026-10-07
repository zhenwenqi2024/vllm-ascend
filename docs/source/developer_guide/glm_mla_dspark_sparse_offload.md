# GLM MLA DSpark with V2 sparse KV offload

This path combines a GLM-5.2 target using Sparse Flash Attention with a
separate `Glm5DSparkForCausalLM` MLA draft checkpoint. It does not enable
generic GQA DSpark checkpoints or offload the draft's context KV to the host.

## Configuration boundaries

- Use the V2 model runner. The validated draft configuration is
  `num_speculative_tokens=8`, checkpoint `block_size=8`,
  `sample_from_anchor=true`, `kv_lora_rank=512` and `qk_rope_head_dim=64`.
  These values describe the tested checkpoint, not a fused-SFA configuration
  whitelist. Draft block/sampling semantics belong to speculative/model
  validation; loaded cache/backend and auxiliary-schema checks determine remote
  context compatibility. V1 lacks this remote draft-context initialization path.
- The Prefill producer is target-only, eager, without speculative decoding or
  prefix caching. Set `dspark_aux_hidden_state_layer_ids` in its
  `kv_connector_extra_config` to the ordered target-layer boundaries declared
  by the draft checkpoint. The tested checkpoint uses `[2, 22, 38, 58, 74]`.
- `dspark_context_chunk_tokens` defaults to 64 and must be between 1 and 64.
  P and D must agree on the auxiliary schema. Transfer uses the MemFabric
  SFA remote-D2H path; the final P pipeline stage owns the staging buffer.
- Sparse offload is enabled on the Decode consumer, using `fused_copy_sfa`.
  LIM requires TopK=2048 and supports 1 through 14 target query rows per request.
  For `N` speculative tokens, reserve at least `(N + 1) * 2048` hot tokens;
  the budget must be 256-aligned and no greater than the kernel limit of 32640
  (the largest aligned budget is 32512). This contract is independent of the
  draft architecture. The eight-token draft requires nine verification rows;
  the tested hot budget is 20480. Accepting a configuration within the kernel
  limits does not establish end-to-end NPU coverage for every draft or width.
  These bounds are shared through `vllm_ascend/attention/sfa_contract.py`;
  source-checkout tests compare the LIM values with both kernel headers.
  The stronger 256-token alignment comes from the serving layout's two
  128-token circular-tail blocks, not from LIM's 128-token alignment rule.
  Configuration, tail allocation and addressing share this layout definition.
- Remote draft-context initialization does not support PCP or DCP. Target
  host KV and resident draft KV must use a common block size.

## Ownership and readiness

The loaded draft supplies its attention-layer ownership. Its persistent MLA
KV buffers are separate from target host KV, layerwise reuse buffers and the
target connector's registered caches. Draft KV therefore still consumes HBM
and scales with context length; target offload alone does not make 1M-context
DSpark memory usage bounded.

P captures and relays all configured auxiliary states. D receives bounded,
ordered BF16 chunks, then projects them with the draft's own weights and RoPE
into scheduler-allocated draft slots on the model worker thread. Requests
become ready only after target KV transfer and synchronized draft-context
writes complete on every D TP rank. Generation IDs protect reused request
IDs; failed transfers do not silently recompute the target prompt.

`spec_decode/dspark_utils.py` resolves auxiliary capture boundaries for both
P's explicit transfer schema and D's checkpoint, preserving the upstream
checkpoint layer-to-boundary conversion. The DSpark context backend separately
validates P-side capture constraints before model loading. The model runner
consumes the validated boundaries and connects native capture/PP relay to the
backend; it does not own the transport configuration constraints.

The full MLA draft block is bidirectional. FIA receives actual initialized
draft lengths, with zero lengths for graph padding, rather than optimistic
target upper bounds. The current non-DCP MLA path uses a batched synchronous
device-to-host length copy. Context staging also waits for device completion
before reuse; these are correctness constraints, not throughput claims.

## Validation limits

The pre-merge implementation was exercised on NPUs with a W4A8 GLM-5.2
target and BF16 MLA draft. Matched GSM8K short inputs had 487/503 correct
answers without offload and 490/503 with offload; draft acceptance was
59.0110% and 58.4930%, respectively. Length-limited answers were retained.

Eight synthetic archive-padded GSM8K inputs were also compared. All four
16K inputs were correct in both modes. At 24K, the offload run had one
8192-token truncated incorrect answer, while the colocated run had four
correct answers. These are diagnostic inputs, not a standard long-context
benchmark. Client scheduling and P/D topology also differed, so this is not
an offload-only causal comparison. Quality/alignment has not been accepted,
and this evidence does not validate 1M contexts or the subsequently merged
tree's end-to-end accuracy.
