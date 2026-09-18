# EPLB workload diagnostics

Run a representative workload such as GSM8K with EP enabled and EPLB disabled:

```json
{"eplb_diagnostics": {"mode": "observe", "window_size": 32, "warmup_steps": 0, "max_windows": 0}}
```

The default mode is `off`, window size is 32 and warmup is 32 calls. Set warmup
to zero to include the first real request. `max_windows=0` (default) observes the
whole run; a positive value limits collection. All workers must use the same
settings. Diagnostics only prints logs, with no JSON files or offline report.

## Each EP rank and each MoE layer

The first rank of each EP/MC2 group prints all its members' information. The
`rank` field identifies the expert owner, not necessarily the logging process.
For each layer and rank, including cold experts, an illustrative line is:

```text
[EPLB experts] stage=0 layer=model.layers.10.mlp.experts rank=1 window=3 window_work=800 cumulative_work=2100 expert_work=['46:1800', '47:300', '48:0']
```

- `window_work`: valid assignments handled by this rank's experts in this layer
  in the current window.
- `cumulative_work`: its total assignments over all valid observed windows.
- `expert_work`: **every owned expert**, as `expert_id:cumulative_assignments`.
  Counts include a final partial window when explicitly collected below.

A top-k token contributes k assignments in each MoE layer. These are not unique
token counts, FLOPs, or measured compute times.

## Per-layer diagnosis

After the rank details, print that layer's current rank distribution and
cumulative persistence evidence. Example:

```text
[EPLB diagnostic] stage=0 layer=model.layers.10.mlp.experts window=3 calls=32 ranks=[0,1] valid_assignments=1000 rank_work=[200,800] window_rank_max_mean=1.600 cumulative_rank_max_mean=1.400 busiest_rank=1 imbalanced_windows=3/3 invalid_windows=0 observed_calls=96 hot_experts=['46(700)'] persistent_hot_experts=['46(3/3)'] hint=consider_eplb
```

`window_rank_max_mean` is the busiest rank's work divided by the mean rank work
in this window. `cumulative_rank_max_mean` uses each rank's accumulated work
across all valid observed windows, including the final partial window. It is
not the average of window ratios. `busiest_rank` is the global rank with the
largest cumulative workload; ties select the first rank in `ranks`. Zero-load
ranks are included in the mean. Invalid windows do not update these totals.

Rank work is calculated independently for each layer. Opposite hotspots in two
layers cannot cancel each other. Max/mean at least 1.2 marks an imbalanced window.
Hot experts are the top four including ties, with at least 1.2 times the layer's
all-expert mean. `hot_experts` displays up to four with current-window counts.

Persistence uses all valid complete windows in the run, not just the most recent
windows. An expert must be hot in at least 80% of them and have a run of at least
two consecutive hot windows. Invalid samples and phase/graph changes break
consecutive streaks. `persistent_hot_experts` prints all qualifying expert IDs
with hot-window hits/valid-window count. Partial windows contribute work but do
not count as evidence of persistence.

`hint` is `collecting`, `consider_eplb`, `no_persistent_rank_skew`, or
`insufficient_hotspot_evidence`. `consider_eplb` requires persistent experts and
at least 80% imbalanced complete windows. Invalid layers instead print
`insufficient_evidence=<reason>` and do not invalidate unrelated layers.

## Overall conclusion and end of dataset

Each window ends with `[EPLB summary]`, containing layer-assessment counts,
candidate layer names, invalid-layer-window count, and a conclusion based on
**per-layer evidence**. It does not add different layers' rank loads together.
With candidates it prints `consider_eplb_for_observed_layers`; with every layer
assessed as balanced it prints `no_persistent_rank_skew_observed`; otherwise it
prints `insufficient_evidence`. These are load signals, not speedup guarantees.

After offline generation finishes, explicitly collect the final incomplete window
and print `final=True` summaries on the workers:

```python
# Generate all GSM8K prompts first; wait for generation to finish.
outputs = llm.generate(prompts, sampling_params)
llm.collective_rpc("finish_eplb_diagnostics")
```

This ends collection and is idempotent. Invoke it on **all participating workers**;
with separately controlled DP engines, all DP instances must issue this RPC
concurrently after their requests finish. Never call one worker in isolation.
There is no collective in asynchronous shutdown. Without this final RPC, the
last incomplete window is not included. Warmup, invalid windows and any configured
window limit are also outside the counted work; `observed_calls` and invalid
counts expose the measurement scope.

PP stages print separate group summaries and identify their stage. There is no
world/PP barrier inside model execution and no claim of whole-pipeline speedup.

## Collection scope and cost

Fixed device counters execute in eager mode and graph replay for both prefill
and decode. `phases` lists the phases seen in the window; mixed batches remain
mixed rather than being attributed entirely to prefill or decode. This is a
combined workload assessment, not separate per-phase histograms.

MC2, AllToAll and AllGather use the same valid-source counting rule. AllToAll
restores the original TP split offset, including uneven splits. AllGather counts
only each rank's source segment: non-SP TP replicas count on TP rank zero, while
SP ranks count their own shard using the gather's actual DP/EP layout. No extra
device collective is added for diagnostics. Scheduler validity and MC2 masks
exclude padding. Dummy requests contribute no workload or timing;
idle DP workers still join window coordination and advance call bookkeeping.
At window boundaries counters are copied to CPU and gathered on the existing MC2
CPU group. Matching windows/layers, TP source conservation and unique expert
ownership are checked. This synchronous diagnostic adds overhead.

MRv1/MRv2 hooks support valid routing for these three paths, PCP=1, DCP=1, no mixed
shared-expert placement, no redundant experts, and no synthetic forced routing.
EP must be enabled and EPLB disabled. FusedMC2/MegaMoE and other unsupported paths report insufficient
evidence. Ownership is read after warmup and must stay fixed; replacement of a
model or placement requires a restart. No layout simulation or kernel timing is
performed. Per-window totals can still hide imbalance that alternates between
individual calls, so a negative hint is not proof that EPLB cannot help.
