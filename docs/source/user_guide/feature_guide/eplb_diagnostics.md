# EPLB adjustment benefit diagnostics

This diagnostic runs with **EP and EPLB enabled**. It records actual expert
placements and live EPLB activity, then reports layout-adjustment saving and
EPLB overhead separately. It does not create hypothetical placements or change
production routing, weights, or update policy.

Enable it alongside the model's production EPLB configuration:

```json
{"eplb_diagnostics": {"mode": "benefit", "window_size": 32, "warmup_steps": 0, "max_windows": 0}}
```

`mode="off"` disables only the diagnostic. The only diagnostic modes are `off`
and `benefit`; the earlier `observe` mode is rejected. Benefit diagnosis requires
EPLB to be enabled. The default window is 32 calls and warmup is 32 calls. Set
warmup to zero to include the first real request. `max_windows=0` observes the
whole run; a positive value bounds collection. All workers need identical
settings. Output consists of log messages; no JSON files are written.

## Three distinct quantities

| Field | Meaning | Measurement scope |
| --- | --- | --- |
| `adjustment_saving_ms` | Initial-layout MLP time minus current-layout MLP time on the same logical expert workload | Per layer, mean of per-step EP maximum times, both using EPLB-on kernel paths |
| `eplb_overhead_ms` | Cost shared by the observed run | Per-layer logs say `shared_run_cost`; the summary says `unknown` until a valid exposed total is available |
| `estimated_net_saving_ms` | Adjustment saving minus EPLB overhead | Available only when comparison and cost scopes are complete and compatible |

Positive adjustment saving means the current placement reduces the measured
MLP component compared with `reference=initial_live_placement`. Negative saving
means it increases that component. The initial placement is an actual observed
live layout. This comparison never substitutes a uniform layout, shadow plan,
or a different EPLB algorithm for an observed layout.

The current log output keeps total overhead and net saving unknown. It prints
measured raw cost components and covered update spans separately. Those spans
are not a complete exposed-overhead measurement, so the diagnostic does not
subtract them from each layer or print them as a net performance improvement.
Missing costs remain listed; unknown values are not replaced with zero.

## Workload and actual placement

Collection retains per-layer expert assignments, real source-token validity,
and per-step workload inside a bounded window. A top-k token contributes k
expert assignments in each MoE layer; assignment counts are not compute times.
Each EP owner's experts and cumulative work remain available in workload logs,
including cold experts. `[EPLB experts]` prints `window_work`, `expert_work`
(current-window logical expert counts), `cumulative_valid_work`, and
`cumulative_expert_work`. Device counters retain logical expert work separately
for every actual EP owner, so work before and after a placement commit is
attributed to the rank that handled it. `[EPLB adjustment]` prints
`rank_max_mean`, `initial_step_peak_work`, `current_step_peak_work`, and
`moved_experts`. Peak/mean describes workload, not a performance benefit; there
is no threshold-based enable recommendation.

The reference layout is the initial live expert placement. Current ownership is
read from the model's actual placement. A placement commit advances the mapping
generation: a window spanning incompatible mappings cannot be treated as one
stable layout for timing comparison. Its valid expert-work counts are still
reported and accumulated; only its timing-comparison sample is excluded.
Calibration uses the latest comparable window, including a valid partial
window when observation ends.

The same recorded **logical expert workload** is distributed according to the
initial and current actual layouts for the timing comparison. This removes the
confounding effect of comparing a light pre-update batch with a heavy
post-update batch. It does not rewrite routes for any real request.

For a sampled step and layer:

```text
initial_step_ms = maximum MLP time across EP ranks under the initial layout
current_step_ms = maximum MLP time across EP ranks under the current layout
adjustment_saving_ms = mean(initial_step_ms - current_step_ms)
```

Take the rank maximum separately for every step, then average the steps. Taking
the maximum of rank averages can hide a busiest rank that changes across steps.
Different layers remain separate; independently sampled layer results are not
summed into a whole-model performance prediction.

If the complete current placement equals the initial placement, its placement
contribution is exactly zero. The diagnostic skips repeated MLP timing and
prints `adjustment_status=no_adjustment`, `calibration_execution=not_required`,
and `ranges=exact_zero`. Timing noise cannot turn an unchanged placement into
a positive adjustment result. Any EPLB overhead remains a separate quantity.

## Measure live EPLB activity

`[EPLB migration]` logs cumulative submitted payload by rank, layer, direction
(`send` or `recv`), and locality (`same_node`, `cross_node`, or `unknown`). Each
record includes `payload_bytes`, `tensor_ops`, and `submissions`. Sizes come from
actual submitted tensor metadata, including weights and scales, without reading
device data. These are payload bytes, excluding local copies and transport
framing. A successful submission is not a committed placement. A send and its
peer's receive describe the same payload: do not add both to count traffic.

Locality uses a host boot identity exchanged once at diagnostic startup. The
identity is never logged; unavailable identity produces `unknown`. Background
transfers remain observable between forwards, with timing events on the actual
staging stream. Warmup and completed observation are excluded. A snapshot taken
before an update finishes can have incomplete cost coverage; use the calibration
RPC after the update cycle completes for a final snapshot.

Graph replay writes the selected history slot directly. History mode avoids a
second, unused cumulative device update; window and run summaries are derived
from that history.

Runtime instrumentation reports durations of actual EPLB work where supported,
such as load aggregation, waiting for the planner, transfer submission or wait,
and map/weight application. Component labels distinguish host and device spans.
Model-runner paths can expose different components, so logs also identify
coverage and missing measurements.

Raw spans are diagnostic evidence, not automatically additive overhead:

- A transfer's full duration can overlap useful work; its exposed wait must not
  be added to that duration as a second cost.
- Planner execution and the corresponding blocking wait may overlap or enclose
  one another.
- A whole rearrangement span can already include collection, transfers, and
  apply work.
- Host and device measurements of the same operation must not be counted twice.

Only non-overlapping, scope-compatible exposed spans could form an overhead
subtotal, normalized by actual observed steps rather than a configured update
interval. The current summary instead prints `covered_update_span_total_ms`:
it selects device `eplb_step` or `eplb_step_before`/`eplb_step_after` spans,
excluding nested components and host duplicates, and reports the minimum and
maximum rank totals. This is descriptive coverage, not a per-step EP critical
path or complete exposed cost. Pending/dropped measurements, missing outer
spans, invalid timings, or mixed outer-span families make this field unknown.
The summary also prints `observed_real_steps`.

`[EPLB overhead]` retains `rank_records` with raw components and
`accounting=overlapping_components_do_not_sum`. Shared run costs are not
subtracted independently from every layer.

Fused per-route collection, asynchronous interference, and unobserved critical
path effects can remain missing even when several component timings exist.
The diagnostic does not turn raw background durations into claimed serving
latency overhead.

## Finish and compare the recorded layouts

After all representative requests finish, stop collection and perform the
isolated MLP comparison collectively:

```python
outputs = llm.generate(prompts, sampling_params)
llm.collective_rpc(
    "calibrate_eplb_diagnostics",
    kwargs={"samples": 4, "repeats": 3, "max_scratch_bytes": 134217728},
)
```

The RPC finishes observation first. `samples` limits sampled real-work steps
and `repeats` controls repeated timing. `max_scratch_bytes` limits the estimate
of explicit calibration activations and intermediates; its default is 128 MiB.
Opaque CANN workspace is excluded from this estimate, so it is not a hard
allocation limit. There is no `update_interval` argument: costs come from the
observed execution. To flush workload logs without running calibration:

```python
llm.collective_rpc("finish_eplb_diagnostics")
```

All participating workers must enter the same RPC collectively after their
requests finish. Separately controlled DP engines must invoke it concurrently.
Keep inference stopped during calibration. Do not call an individual worker.
No collective is issued from asynchronous teardown. Repeating the same
calibration arguments after every available layer succeeds returns
`reason=already_calibrated`. Failed or partial calibration remains retryable;
changing arguments requests another calibration.

Calibration also requires the current EPLB update cycle to have completed.
If execution or an update is still pending, all ranks return
`reason=execution_or_update_pending retry_after_completed_cycle=True` without
finishing observation. Allow normal requests to complete the update cycle,
then retry after those requests finish; the RPC does not drain or discard an
update. `finish_eplb_diagnostics` also refuses pending inference, but can flush
while a background update remains, with incomplete cost coverage identified.

The comparison uses fresh synthetic MLP inputs, observed per-expert counts,
matching EPLB-on compute templates, and already-loaded weights read-only.
Synthetic outputs do not enter request generation or accuracy results. Both
layouts use the same measurement method and logical workload. Insufficient
layout history, invalid workload, unsupported templates, or incomplete timing
produce a reason instead of an invented result.

The currently supported isolated MLP methods are the matching unquantized
(`NONE`) and dynamic `W8A8` Ascend paths. Device events time the actual
GMM/activation/quantization operations, with allocations outside the timed
region. The helpers synchronize during isolated calibration; live collection
does not insert a per-step timing synchronization.

Per-layer `[EPLB benefit]` lines identify `generation`, `window_end_step`,
`sampled_steps`, `phase`, `observed_graph`, and `comm`, followed by:

```text
reference=initial_live_placement calibration_execution=eager adjustment_saving_ms=(4.0, 7.5) adjustment_status=positive eplb_overhead_ms=shared_run_cost estimated_net_saving_ms=unknown
```

These values are illustrative. The summary reports `calibrated_layers`,
`available_layers`, `comparable_layers`, and `adjustment_layer_counts`, as well
as run-level cost coverage. Available layers include layers without a usable
window; those print `reason=no_comparable_window` and remain retryable. A
`no_adjustment` result counts as an assessed layer without running MLP timing.
The summary does not sum independent layer samples into model latency.

Minimum/median/maximum values describe the observed repetitions. The minimum
to maximum range is not a statistical confidence interval. A sampled isolated
MLP difference is not a direct measurement of whole-model serving latency.

## Graph mode, support and remaining scope

Workload counters support eager execution and graph replay for prefill and
decode on supported MC2, AllToAll and AllGather routes. They use valid source
ownership across TP/EP ranks. Padding and dummy-run routes are excluded;
idle DP ranks still join required coordination. Captured Python calls are not
used as a substitute for replayed device counters.

Initial support requires zero redundant experts, fixed EP membership, PCP=1,
DCP=1, supported expert ownership, and a compatible MLP template. Unsupported
communication or fused paths, mixed shared placement, forced synthetic routing,
LoRA, or ambiguous templates report missing evidence. Model runner support
for workload collection does not imply identical overhead coverage in both
EPLB implementations.

Isolated MLP calibration currently uses eager execution. Real-request graph
execution remains unchanged, but its communication/graph effects are not
inferred from isolated eager measurements. Important omitted terms can include:

- Dispatch/combine changes caused by the actual placement adjustment.
- Graph execution differences and compute/communication overlap.
- Embedded or fused expert-load collection work.
- Migration interference and any unmeasured exposed waits or apply operations.

These omissions keep `estimated_net_saving_ms` unknown even when
`adjustment_saving_ms` and some actual overhead components are measurable.
Logs and summaries report calibrated layer/step coverage and missing terms.
They do not promise a throughput gain or derive a full-model result by summing
unmatched layer samples. PP stages keep separate scopes.

Counters, bounded history, host aggregation, timing markers, and logging add
diagnostic overhead of their own. The instrumented run's throughput is not an
uninstrumented baseline, and diagnostic overhead is not mislabeled as EPLB
production overhead. Diagnostic timing markers can also inflate the live spans
they record; their overhead is not assumed to be zero or subtracted without
measurement.
