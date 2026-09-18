# EPLB benefit and cost diagnostics

Run a representative workload such as GSM8K with EP enabled and EPLB disabled:

```json
{"eplb_diagnostics": {"mode": "observe", "window_size": 32, "warmup_steps": 0, "max_windows": 0}}
```

The default mode is `off`, window size is 32 and warmup is 32 calls. Set warmup
to zero to include the first real request. `max_windows=0` observes the whole
run; a positive value limits collection. All workers must use the same settings.
Diagnostics prints logs and does not create JSON files.

There are two stages:

1. Observe real routing, report workload, and evaluate a hypothetical expert
   placement on subsequent work. No weights or routing choices are changed.
2. After generation, optionally run isolated compute and scratch-transfer
   calibration. This measures parts of the benefit/cost equation while EPLB
   remains off. It does not establish the complete performance change from
   enabling EPLB.

## Each EP rank and each MoE layer

The first rank of each EP/MC2 group prints all its members' information. `rank`
identifies the expert owner, not necessarily the logging process. Each layer
and rank includes its cold experts. An illustrative line is:

```text
[EPLB experts] stage=0 layer=model.layers.10.mlp.experts rank=1 window=3 window_work=800 cumulative_work=2100 expert_work=['46:1800', '47:300', '48:0']
```

- `window_work`: valid assignments handled by this rank's experts in this layer
  during the current window.
- `cumulative_work`: its assignments across all valid observed windows.
- `expert_work`: every owned expert as `expert_id:cumulative_assignments`.

A top-k token contributes k assignments in each MoE layer. These are assignment
counts, not unique tokens, FLOPs, or measured execution times. Explicit final
collection also includes a trailing partial window in cumulative work.

## Per-layer workload evidence

After rank details, each layer reports its distribution and persistence:

```text
[EPLB diagnostic] stage=0 layer=model.layers.10.mlp.experts window=3 calls=32 ranks=[0,1] valid_assignments=1000 rank_work=[200,800] window_rank_max_mean=1.600 cumulative_rank_max_mean=1.400 busiest_rank=1 imbalanced_windows=3/3 invalid_windows=0 observed_calls=96 hot_experts=['46(700)'] persistent_hot_experts=['46(3/3)'] hint=persistent_load_skew
```

`window_rank_max_mean` is the busiest rank's work divided by mean rank work in
that window. `cumulative_rank_max_mean` uses each rank's accumulated work,
including the final partial window; it is not the mean of window ratios.
`busiest_rank` is the global rank with the largest cumulative work; ties select
the first rank in `ranks`. Zero-load ranks contribute to the mean. Invalid
windows do not update cumulative work.

Each layer is evaluated independently, so opposite hotspots in different layers
cannot cancel. Max/mean at least 1.2 marks an imbalanced window. Hot experts are
the top four including ties, with at least 1.2 times the layer's all-expert mean.
`hot_experts` displays up to four with current-window counts.

An expert is persistent if it is hot in at least 80% of valid complete windows
and has at least two consecutive comparable hot windows. Invalid windows and
phase/graph changes break streaks. `persistent_hot_experts` prints qualifying
IDs with hot-window hits/valid-window count. Partial windows add work but do not
add persistence evidence.

`hint` is `collecting`, `persistent_load_skew`, `no_persistent_rank_skew`, or
`insufficient_hotspot_evidence`. `persistent_load_skew` requires persistent hot
experts and at least 80% imbalanced complete windows. It replaces the earlier
`consider_eplb` label: load imbalance alone does not justify enabling EPLB.
Invalid layers print `insufficient_evidence=<reason>` independently.

## Candidate placement and subsequent-window evaluation

The diagnostic invokes Ascend's `DefaultEplb` policy on one complete window,
using actual expert owners and the current number of slots, with no redundant
experts. It then evaluates that placement on the next comparable complete
window, not the window used to produce the plan. A single hot expert remains
indivisible when there are no replicas. Hotspot drift can make the candidate
worse than the original placement.

The candidate explicitly uses **default EPLB policy 1**. It does not simulate a
configured policy 2, other Ascend policies, or the upstream MRv2 EPLB algorithm.
The diagnostic is usable from both model runners, but its candidate policy
must not be confused with the policy a future production run would enable.

`[EPLB projection]` reports the source window, candidate rank work, moved-expert
count, and:

- `step_peak_work`: sum of each step's largest rank workload.
- `candidate_step_peak_work`: the same quantity under the preceding candidate.
- `work_reduction`: their signed relative difference; negative means worse.
- `planner_ms`: measured CPU time to compute this candidate.
- `timing=not_estimated`: work reduction is not converted to milliseconds.

Fixed per-step counters avoid hiding hotspots that alternate between ranks
within a window. Changes in phase, graph mode, or ownership, invalid windows,
and missing per-step evidence prevent a valid comparison. The first complete
window can produce a plan but cannot evaluate a preceding plan.

`[EPLB summary]` lists per-layer assessments and `skewed_layers`. Observation
alone always reports `conclusion=insufficient_benefit_evidence` and
`speedup=not_estimated`, including when sustained imbalance is present.

## Finish observation or run isolated calibration

After all offline requests finish, flush the partial window and stop collection:

```python
outputs = llm.generate(prompts, sampling_params)
llm.collective_rpc("finish_eplb_diagnostics")
```

To also measure candidate compute and scratch overheads, call the calibration
RPC after generation instead. It finishes observation before calibrating:

```python
outputs = llm.generate(prompts, sampling_params)
llm.collective_rpc(
    "calibrate_eplb_diagnostics",
    kwargs={"update_interval": 700, "samples": 4, "repeats": 3},
)
```

`700` is illustrative. Supply the actual intended production update interval
in inference steps; it is required and is not `window_size`. Ascend's update
cycle includes heat collection, algorithm execution, and layer-by-layer weight
updates. Use the interval for the intended policy and configuration.

Calibration samples the retained last complete comparable evaluation window.
`samples` limits sampled real-work steps; `repeats` controls repeated isolated
measurements. A trailing partial window still contributes cumulative work but
does not replace the complete evaluation window. No valid preceding plan or
matching kernel template means insufficient calibration evidence.

Every participating worker must enter these RPCs collectively. With separately
controlled DP engines, all instances must invoke the same RPC concurrently
after their requests finish. Keep inference stopped during calibration. Do not
call an individual worker. Final collection is idempotent; no collective is
issued from asynchronous teardown. Without final collection, the trailing
partial window is omitted.

## What calibration measures

Calibration retains model weights read-only and creates fresh synthetic MLP
inputs with the observed per-expert counts. The candidate changes the synthetic
count distribution, not live expert placement or request outputs. Currently
supported compute templates are the matching unquantized (`NONE`) and dynamic
`W8A8` Ascend methods. Other quantization, LoRA, routing-dependent MLP inputs,
ambiguous templates, or unsupported layouts report missing evidence.

MLP calibration measures the existing GMM/activation/quantization path in eager
mode with NPU events. Inputs are allocated before timing, and device execution
completes before scratch inputs are released. Real request graph replay is not
replaced with eager execution; graph workload counters and isolated eager
calibration are separate measurements.

Calibration also measures serial scratch operations: a one-layer load gather
and CPU copy, exact-byte packed expert transfers between candidate peers, and
incoming-buffer/map-copy proxies. Every rank agrees on metadata, memory limits,
and allocation success over the CPU group before device communication. Scratch
limits or a peer's allocation failure make calibration unavailable collectively.
No model weight, map, KV cache, or inference output is overwritten.

The logs report baseline and candidate MLP timings, sampled compute saving,
planner time, serial scratch costs, the supplied update interval, and the
remaining budget after those costs. Per-update costs are amortized over that
production interval. Reported minimum/median/maximum values describe observed
samples; their range is not a statistical confidence interval.

## Calibration log fields

Each `[EPLB calibration]` line identifies its training and evaluation windows,
selected steps, phase, communication method, and observed graph mode.

| Field | Meaning |
| --- | --- |
| `baseline_mlp_ms`, `candidate_mlp_ms` | Observed timing ranges, averaging aligned per-step EP maxima. |
| `compute_saving_ms` | Baseline minus candidate MLP time; includes activation and quantization. |
| `collection_per_update_ms` | Isolated one-layer load-gather and CPU-copy proxy. |
| `transfer_per_update_ms` | Isolated packed scratch transfer proxy. |
| `apply_per_update_ms` | Incoming-buffer and map-copy proxy. |
| `serial_scratch_cost_per_step_ms` | Planner and scratch costs amortized over `production_update_interval`. |
| `compute_budget_after_proxy_cost_ms` | Sampled compute saving minus those proxies; not full EPLB net gain. |
| `compute_budget_status` | `positive`, `nonpositive`, or `uncertain` within that limited budget. |
| `net_saving_ms` | `unknown` while production effects remain unmeasured. |

The final `[EPLB calibration summary]` reports calibrated and uncalibrated
available layers plus `compute_budget_layer_counts`. It does not sum timings
across layers or PP stages. The overall conclusion remains
`insufficient_benefit_evidence`. Failed or incomplete calibration can be retried;
a fully successful call with identical arguments is idempotent. Changing the
scratch budget, sample count, repeat count, or update interval permits a new run.

## Interpreting the remaining budget

A positive compute budget means the sampled MLP saving exceeds the measured
serial cost proxies under the stated calibration assumptions. It is **not a
measured full EPLB net gain**. Missing terms include:

- Changes to dispatch/combine communication and its critical path.
- Compute/communication overlap and migration interference during serving.
- Differences between EPLB-off and EPLB-on kernel paths.
- Differences between isolated eager MLP timing and actual graph execution.
- Production planning delay and the layer-by-layer update schedule: the
  candidate workload projection uses the next diagnostic window.
- Production parameter-message counts, packed all-layer load collection, and
  map-update behavior: scratch operations are proxies, not guaranteed bounds.

Unknown terms remain unknown, rather than being treated as zero. A positive
budget therefore supports further evaluation, not an automatic enable decision
or a performance guarantee. A negative budget shows that this measured compute
component does not pay for these serial proxies; it also does not prove that
all production EPLB configurations are unprofitable.

## Collection scope and overhead

Workload counters execute in eager mode and graph replay for prefill and decode.
`phases` identifies observed phases; mixed batches remain mixed. MC2, AllToAll,
and AllGather follow valid source-token ownership, including TP replicas and
uneven sequence splits. Scheduler validity and MC2 masks exclude padding.
Dummy requests contribute no workload; idle DP ranks still participate in
window coordination. Capture and dummy-run data are not calibration workload.

MRv1/MRv2 routing collection requires EP enabled, EPLB disabled, PCP=1, DCP=1,
no mixed shared-expert placement, no redundant experts, and no forced routing.
FusedMC2/MegaMoE and other unsupported paths report insufficient evidence.
Ownership must remain fixed; replacing a model or its layout requires restart.

Window boundaries copy counters to CPU and gather on the existing EP/MC2 CPU
group. Source conservation, calls, layer alignment, and unique ownership are
checked. Counter history is bounded to a window, with bounded retained evidence
for calibration. These counters, synchronous summaries, CPU planning, and log
output add diagnostic overhead; throughput measured in this mode is not the
uninstrumented baseline. Isolated calibration deliberately synchronizes.

PP stages retain separate summaries and calibration scopes. There is no new
world/PP barrier inside model execution and no whole-pipeline speedup claim.
