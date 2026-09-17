# EPLB workload diagnostics

Enable EP, leave EPLB disabled, and add:

```json
{"eplb_diagnostics": {"mode": "observe", "window_size": 32, "warmup_steps": 32, "max_windows": 8}}
```

Diagnostics prints one summary per window from the first rank of each EP/MC2
communication group. It creates no diagnostic files and needs no offline tool.
The default mode is `off`. `window_size` and `max_windows` must be positive;
`warmup_steps` can be zero. Use the same settings on all workers.

## Output

An illustrative log (numbers are examples):

```text
[EPLB diagnostic] stage=0 window=3 calls=32 ranks=[0,1,4,5] valid_assignments=10000 rank_work=[4000,2000,2000,2000] window_rank_max_mean=1.600 imbalanced_windows=3/3 hot_experts=['model.layers.0.mlp.experts:46(700)'] persistent_hot_count=1 persistent_hot_experts=['model.layers.0.mlp.experts:46(3/3)'] hint=consider_eplb
```

- `valid_assignments`: total valid routed expert assignments across **all MoE
  layers in this group/stage** during the window. One top-k token contributes k
  assignments per MoE layer; this is not the number of unique input tokens.
- `rank_work`: those assignments attributed to actual expert owners, in `ranks`
  order. This is aggregate window work, not a per-layer or per-call maximum.
- `window_rank_max_mean`: largest rank total divided by the mean rank total.
- `imbalanced_windows`: windows with max/mean at least 1.2 among the last four
  consecutive comparable windows (or fewer while history accumulates).
- `hot_experts`: at most four displayed layer/expert pairs, with window assignment
  counts. Selection within each layer includes the top four and ties, requiring
  at least 1.2 times that layer's mean expert work; zero-load experts count in
  the mean. Display entries are sorted by assignment count across layers.
- `persistent_hot_count`: total number of persistent hot layer/expert pairs.
  `persistent_hot_experts` shows at most four, with hot-window hits/history size.
  Persistence requires at least 80% hot windows and hotness in both the current
  and preceding window; at least two windows are needed.
- `hint`: `collecting`, `consider_eplb`, `no_persistent_rank_skew`, or
  `insufficient_hotspot_evidence`. `consider_eplb` requires persistent hot experts
  and at least 80% imbalanced windows. It is a workload signal, not a speedup claim.

Invalid data prints `insufficient_evidence=<reason>` and clears persistence.
Changes in phase/graph mode, active source participation, or recorded placement
also restart history. No per-layer log flood is produced.

## Collection and scope

Fixed device counters update during eager execution and graph replay. Real
scheduler-token counts and MC2 masks exclude padding. Dummy requests contribute
no workload, hotspot samples, or timing. Idle DP workers still participate in
window coordination and advance call bookkeeping, so real routes from other
workers can be assigned to their expert owners.

At each window boundary, workers copy counters to CPU and gather them over the
existing MC2 CPU group. The summary checks matching windows/layers, TP source
conservation, supported call counts, and complete unique expert ownership.
This synchronous operation adds diagnostic overhead; do not interpret performance
with diagnostics enabled as an unperturbed serving baseline. Collection and
printing are bounded by `max_windows`; graph counter operations remain captured.

MRv1/MRv2 have collection hooks. Valid routing currently requires MC2, PCP=1,
DCP=1, no mixed shared-expert placement, no redundant experts, and no synthetic
forced load balancing. Unsupported call paths report insufficient evidence.
EP must be enabled and EPLB disabled. Expert ownership is read after warmup and
assumed fixed during observation. Model or placement replacement requires restart.

TP/DP are aggregated within the actual group. PP stages print separate summaries:
there is no world/PP barrier in model execution. One stage's line is not a whole
pipeline total. Counts sum token assignments without weighting different layer
shapes. Aggregate balance can hide alternating per-layer or per-call bottlenecks;
`no_persistent_rank_skew` therefore does not rule out a benefit from EPLB.

This mode does not simulate layouts, run EPLB, time kernels, or predict net gains.
It provides overall work distribution and hot-expert persistence for inspection.
