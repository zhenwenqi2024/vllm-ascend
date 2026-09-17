# EPLB enablement diagnostics

EPLB diagnostics observes a workload with EP enabled and EPLB disabled. It checks
whether expert hotness and rank-work imbalance persist, then evaluates a candidate
placement on later observations. It does not move weights or change routing.

The result is evidence for deciding whether to try EPLB, not a throughput
prediction or a guarantee that enabling EPLB will improve performance.

## Enable collection

Keep your model's supported parallelism and graph configuration. Enable EP,
leave EPLB disabled, and add the following to `--additional-config`:

```json
{
  "eplb_diagnostics": {
    "mode": "observe",
    "run_id": "workload-off-01",
    "output_dir": "/tmp/eplb-workload-off-01",
    "sample_interval": 128,
    "burst_size": 32,
    "warmup_steps": 32,
    "max_samples": 128,
    "max_pending": 2,
    "max_snapshot_mb": 16
  }
}
```

Use the same unique `run_id` on all workers. Each process writes a separate JSONL
file containing its rank and a random suffix. Collect files from every host into
one directory before reporting. Do not combine separate engine runs under one ID.
Allow normal worker shutdown to drain pending writes.

| Option | Default | Meaning |
| --- | --- | --- |
| `mode` | `off` | `observe` enables collection; `off` disables it. |
| `run_id` | empty | Required in observe mode; up to 128 letters, digits, `_`, `-`, or `.`. |
| `output_dir` | `eplb_diagnostics` | Directory for per-process JSONL files. |
| `sample_interval` | 32 | Period between sampling bursts, in model calls. |
| `burst_size` | 1 | Consecutive calls sampled per period; 1 to 64, no greater than the interval. |
| `warmup_steps` | 32 | Initial model calls skipped after worker warmup. |
| `max_samples` | 128 | Maximum accepted samples per worker. |
| `max_pending` | 2 | Maximum outstanding snapshots; 1 to 16. |
| `max_snapshot_mb` | 16 | Per-snapshot size limit in MiB; 1 to 256. |

Default single-call sampling is useful for counters but too sparse for persistence
analysis. Use bursts of at least 24 consecutive valid calls for the default
eight-call windows and two held-out evaluation windows. The example uses 32.
Missing participants, dropped samples, or layout changes can reduce usable calls.
Sampling gaps are never treated as continuous observations.

## What is collected

| Data | Purpose |
| --- | --- |
| Backend per-expert assignments | Work reported by the receiving expert backend. |
| Valid source routes to physical experts | Count real assignments while excluding masked and padded source tokens. |
| Logical, physical, and local expert mappings | Reconstruct actual owners without assuming contiguous placement. |
| EP, MC2, TP, ETP, DP, PP, PCP, and DCP group metadata | Compare matching communication groups and detect missing members. |
| Model-call ordinals and per-layer counter deltas | Align executed calls and detect unexpected call counts. |
| Request phase, scheduled tokens, graph mode, and padded shape | Separate incompatible observations. |
| Local execution-stream event interval | Execution-boundary context, not isolated MoE or end-to-end latency. |
| Invalid counters, missing mappings, and dropped samples | Explain why a sample cannot support a decision. |

Counts are top-k assignments, not unique tokens. Backend counts can include
implementation padding or shared experts; they remain separate from valid routed
work. Source scheduled tokens are not the work received by that rank's experts.

### Dummy execution and padding

`dummy_run` produces no snapshots, timing records, or JSONL rows and consumes no
sample budget. The offline report also excludes historical dummy-marked rows.
Graph-resident counters may still execute during dummy communication; real-call
before/after deltas exclude that work. An internal ordinal advances for alignment.

If excluding a dummy participant makes a group incomplete, the report returns
insufficient evidence. It never invents a zero-load rank or joins observations
across that missing call.

Supported MC2 paths combine the backend token mask, scheduler real-token prefix,
and TP source offset. Padded graph capacity is metadata, not valid workload.
Backend padding is not subtracted by guessing a constant per-rank token count.

### Graph execution and parallelism

Fixed device buffers are allocated before capture. Counter updates execute on
graph replay; host snapshots and JSON serialization remain outside the graph.
Device-to-host copies are bounded and asynchronous. A background writer waits
for the corresponding event. No diagnostic collective is added.

Collection hooks are present in MRv1 and MRv2. Valid-routing analysis currently
requires MC2, PCP=1, DCP=1, no mixed shared-expert placement, and no forced synthetic
load balancing. MRv1 additionally requires dynamic EPLB to be disabled. Enablement
assessment requires EPLB to be disabled.

TP/DP/PP combinations use recorded groups, layouts, and conservation checks.
Recording backend counts does not imply every model, graph bucket, communication
method, or parallel combination supports valid-routing analysis. Unsupported
cases retain raw evidence and return insufficient evidence for affected analyses.

## Generate a report

From a vllm-ascend source checkout:

```bash
python vllm_ascend/eplb_diagnostics/report.py \
  /tmp/eplb-workload-off-01 --output summary.json
```

Reporting uses the Python standard library and needs no NPU. New traces carry
an alignment contract. The legacy `--aligned-collective-ordinals` option is an
explicit caller assertion; use it only after independently verifying counter
origins and stable topology.

Start with `enablement_summary`, then inspect corresponding
`workload_persistence` entries:

| Decision | Interpretation |
| --- | --- |
| `enable_candidate` | Persistent hot experts and rank imbalance exist; a candidate reduces busiest-rank work in later windows. EPLB is worth evaluating for this workload. |
| `not_recommended_from_observed_load` | Sustained rank-work imbalance was not observed. This sample provides no load-balancing reason to enable EPLB. |
| `insufficient_evidence` | Sampling, routing, persistence, or candidate-improvement evidence is inadequate. This does not establish that EPLB is ineffective. |

These are per-layer, per-group assessments. An empty assessment list means
insufficient evidence, not a balanced model. Layer counts and work reductions
must not be added together as a service-level speedup.

### Expert hotness and persistence

`expert_diagnosis` reports each logical expert's assignments, load share, observed
owner ranks, hot-window fraction, and longest consecutive hot-window run. Cold
experts are included in the mean. By default:

- A complete window contains eight consecutive valid calls.
- Expert max/mean of at least 1.2 indicates imbalance in that window.
- Hot experts are the top four, including ties, with nonzero work at least
  1.2 times the all-expert mean.
- A persistent expert is hot in at least 80% of complete windows and in at least
  two consecutive windows.

An expert can remain hot even when other hot experts change. Set overlap does
not replace per-expert persistence. A hot-window run describes observed calls,
not wall-clock duration or hotness on every call. Thresholds are heuristics,
not significance tests.

### Rank imbalance and candidate placement

The report verifies complete groups, stable mappings, aligned single model calls,
and conservation of valid top-k assignments per independent source group.
Unavailable backend counts are not zero work. Rank max/mean is calculated per
call, before aggregation. By default at least 80% of calls in a window and at
least 80% of complete windows must be imbalanced to establish persistent skew.

`placement_diagnosis` uses one window to construct a greedy mapping that preserves
each rank's actual expert-slot count. It verifies that replaying the current
mapping reproduces observed rank work. It then evaluates the candidate on the
next contiguous window, which was not used for planning. Improvements, regressions,
and numbers of moved experts are reported.

The default criterion requires at least two evaluation windows, at least 80%
improving windows, and at least 5% reduction in summed per-call maximum rank work.
Persistent hot experts and rank imbalance are also required for `enable_candidate`.
Multi-replica layouts are outside this candidate model.

The greedy candidate is not the configured EPLB policy. It does not model topology,
redundant replicas, migration latency, or update cost. Its failure does not rule
out other policies. Its work reduction is not a latency or throughput estimate.

`workload_persistence` allows graph-bucket changes within compatible workload
segments for work-count analysis. `persistent_imbalance` separates matching batch
shapes. Neither joins gaps, changed layouts, or incompatible phases.

Adjust thresholds with `--window-size`, `--min-windows`, `--skew-threshold`,
`--persistent-fraction`, and `--min-work-reduction`. The legacy `recommendation`
field concerns measured speedup and remains
`insufficient_evidence_to_claim_EPLB_speedup`; use `enablement_summary` for this
mode. `next_measurements` describes optional later work, not prerequisites.

## Cost and limitations

Observe mode adds device counter operations on every execution of instrumented
layers, including graph replay. Sparse snapshots reduce copying and logging;
they do not remove per-layer device counter operations. A full snapshot queue
drops samples and records the reason.

Diagnostic overhead has not yet been calibrated across models and parallel
configurations. Diagnostic-enabled timing is not an unperturbed baseline. Local
execution-stream time is not isolated MoE time and cannot establish that rank-work
skew is the service's critical bottleneck.

This initial mode supplies workload evidence before enabling EPLB. It does not
implement an actual EPLB-policy dry run, measure migration costs, or prove net
performance gains. Model accuracy, diagnostic overhead, and additional hardware,
model, and parallel combinations require separate validation.
