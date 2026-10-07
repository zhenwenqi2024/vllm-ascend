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
- Both Prefill and Decode load the same DSpark draft checkpoint. Prefill is
  eager, with prefix caching disabled until draft-cache reuse is supported.
  Capture boundaries are resolved from that checkpoint; the tested model uses
  `[2, 22, 38, 58, 74]`. No separate target-only capture option is required.
- P projects prompt auxiliary features in bounded 64-token chunks and writes
  its own persistent draft KV. MemFabric transfers those KV pages directly
  into D's draft buffers, alongside the target KV; features do not leave P.
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
target pool's registered caches. The SFA PD connector registers and transfers
the draft pages separately. Draft KV therefore still consumes HBM
and scales with context length; target offload alone does not make 1M-context
DSpark memory usage bounded.

P captures the configured auxiliary states and projects them with the loaded
draft's weights and RoPE into its scheduler-allocated draft slots. D reads the
valid prompt KV rows directly into its own draft slots; it does not reconstruct
prompt draft KV from transferred hidden states. Requests become ready only
after target and draft KV transfers complete on every D TP rank. Generation
IDs protect reused request IDs; failed transfers do not silently recompute
the target prompt.

`spec_decode/dspark_utils.py` resolves auxiliary capture boundaries for both
P and D from their checkpoint, preserving the upstream
checkpoint layer-to-boundary conversion. The DSpark context backend separately
validates P-side capture constraints before model loading. The model runner
consumes the validated boundaries and connects native capture/PP relay to the
backend; it does not own the transport configuration constraints.

The full MLA draft block is bidirectional. FIA receives actual initialized
draft lengths, with zero lengths for graph padding, rather than optimistic
target upper bounds. The current non-DCP MLA path uses a batched synchronous
device-to-host length copy. P waits for draft KV writes before publishing its
pages to the remote reader; these are correctness constraints, not throughput claims.

## Validation limits

The corrected P-computed draft-KV path was exercised on NPUs with a W4A8
GLM-5.2 target and BF16 MLA draft: V2, DP1/TP16/PP1 on each side,
two concurrent requests, a 32768-token context limit and draft8.
The first 400 GSM8K test questions produced 390 correct answers (97.5%),
zero NULL extractions, nine length-limited responses and zero API errors.
Truncated and incorrect responses were retained; no model request was retried.

Draft acceptance was 302997/508544 (59.5813%) over 63568 rounds.
The eight prefix-through-position counts were
`[56963, 50287, 44263, 38922, 34125, 29828, 25969, 22640]`;
their sum equals accepted, and drafted equals eight times rounds.
Position rates use all rounds as their denominator, not marginal agreement
or only rounds that reached a position. Mean length including the bonus token
was 5.7665. Every request retained all 16 P draft-transfer and 16 D-ready
rank receipts, with no D prompt recomputation or local cache hit.

This used frozen HTTP requests, official AISBench 3.1.20260630 zero-shot CoT
prompts and offline content-only extraction/scoring, not a full AISBench CLI
run. An immediate log-readiness check interrupted the first attempt after
60 completed answers; those answers were reused, and only the remaining
340 requests were sent. The original failure and exit were preserved.
Exact 400-question colocated acceptance is unavailable, so acceptance
equivalence is not claimed.

The 400-question run preceded a final load-guard change requiring a drafter
only on the last PP stage; the guard has identical behavior in PP1.
The final production source was then started on the same NPU configuration
and passed one previously untested GSM8K request (correct answer, normal stop,
all P/D transfer ranks and no D prompt recomputation). That single regression
is separate from the 400-question score and does not validate PP2.

Separate synthetic archive-padded GSM8K diagnostics retained one correct
16K answer and one 24K answer truncated at 8192 tokens with a NULL extraction.
Acceptance was 19.0126% and 2.6412%, respectively. These are not a standard
long-context benchmark or proof of a pure context-length effect. The 24K
failure has not been waived; full quality/alignment remains unaccepted.
The earlier 503-question hidden-feature-transfer results do not validate
this corrected design. No 1M, W8A8, PP2 or performance validation is claimed.
