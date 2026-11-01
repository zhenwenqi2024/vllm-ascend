# HiSparse per-layer wait dependency review

This is an offline source and saved-trace review. No wait was removed and no
new model result is claimed. The active rebased backend and the frozen
experimental snapshot have different qualification scopes.

## What the extra 78 waits represent

The saved Ascend single-D C8, 128-input/8192-output, MTP3 profile contains
78 target attention layers. Across eight ranks and five middle cycles per
rank, the following main-stream waits are additional to the OFF path:

| Next consumer | Count per target | Mean wait per target |
| --- | ---: | ---: |
| Group mapping/query packing | 19 | 0.643175 ms |
| Group mapping/compaction | 2 | 0.065638 ms |
| Follower query packing | 57 | 0.276130 ms |
| Total | 78 | 0.984943 ms |

These are device stream dependencies, not 78 CPU synchronizations or proof
of 78 Host-to-device transfers. Reported settled miss/H2D deltas were zero;
that counter evidence does not establish the absence of all physical DMA.

For rank 0, all 390 waits in five middle targets have the same captured
task sequence: side-stream compact gather, event record, compute-stream wait,
then mapping/packing. The model/launch connection and consecutive task IDs
match; the record finishes within 0.828 microseconds of the wait ending.
The trace does not expose the event handle, so the graph edge is corroborated
with the frozen call sites rather than inferred from timing alone.

## Which dependency is necessary

`runtime_adapter._resolve_and_stage_group` waits for this group's current
TopK, resolves current physical indices, submits gather, and records a
layer-ready event. The common `HiSparseRuntime.swap_in` waits for that event
before returning indices to attention.

Even if gather copies zero rows, the resolver still produces the current
physical index array and zero-swap counts on another stream. Reading that
array before its producer completes is unsafe. A zero reported miss count
from a different snapshot cannot substitute for the current graph dependency.

The dependency is necessary in this implementation. Its full exposed latency
is not necessarily unavoidable. In particular:

- follower Q packing does not read resolved indices or gathered KV and can
  potentially precede the existing wait;
- compatible follower gathers can potentially be submitted at the group
  leader, preserving each layer's readiness event;
- a separately proved all-Resident route could avoid empty gather work and
  redundant follower fences, but must preserve current index generation,
  graph ownership and Resident reuse/write ordering.

No claim is made that all 78 waits can simply be deleted, or that their summed
duration is a TPOT improvement. The first Target AllReduce wait is separate.

## GPU comparison and version distinction

The official vLLM source checked at commit
`e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` retains the same semantic pattern:
`_resolve_and_stage_group`, per-layer ready events,
`_stage_this_layer`, and `wait_for_staged_swap`.
[Pinned source](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/hisparse/runtime.py).

The saved H200 experimental runtime is different: its
`is_hisparse_group_prefetch_batch` allows verification query lengths above one
to prefetch follower gathers. That experiment keeps `decode_batch` and Host
mirror policy separate. The measured Ascend adapter still tests
`resident.decode_batch`, so MTP target verification does not take this group
prefetch path. This local H200 extension must not be attributed to official
main.

Across 96 middle H200 target cycles from ranks 0 and 1, the saved trace still
contains 21 resolver kernels, 78 compact gathers and 78 attention kernels in
each target. CUDA graph replay does not expose the same per-node `EVENT WAIT`
duration as the Ascend export. An absent orange wait bar therefore does not
mean an absent dependency. H200 C4/64K/FP8 and Ascend C8/128/W4A8 are different
workloads; these traces establish structure, not a hardware speed ratio.

## Candidate status

Follower query prepack is stored in the experimental snapshot. Its 48 CPU
proxy cases do not qualify NPU graph behavior or model speed. Verification
group prefetch needs a separate Ascend proof for tentative MTP writes,
Host durability, copy-stream order and all graph owners. Neither change is
promoted here on the strength of this source review.

The active backend already opts into the paired PR62592 revision's
`supports_speculative_group_prefetch` interface. The older measured adapter
used for the C8 profile has a different integration path; its current source
port is preserved under `candidates/mtp-group-prefetch-a6-v1` in the snapshot.
Its 54 CPU cases verify the actual candidate method's policy and call order,
including mixed/empty/oversized batches, piecewise/event fallback and both
Host mirror policies. They do not qualify an NPU graph.

An earlier Ascend port completed model N2, but its original whole-cycle
comparison did not establish a net gain. Reanalysis of both saved profiles,
eight ranks and five middle targets per rank, now confirms actual overlap:

| Observed per target | Parent | Prefetch |
| --- | ---: | ---: |
| Follower gathers completed before their leader attention ended | 0 | 57 |
| Mapping/query-consumer wait | 0.968620 ms | 0.706404 ms |
| First main-stream wait, paired to AllReduce | 0.667246 ms | 1.159374 ms |

Thus prefetch reduced the local exposed wait by about 0.262 ms; a roughly
0.492 ms increase in the first collective wait offset it in that capture.
The latter is not established as a causal effect of prefetch. These are
profile intervals, not new formal TPOT results. The historical port predates
later lazy-MTP and kernel changes and cannot qualify the new combination.

## First AllReduce: where the additional wait comes from

Both actual CPU graph launches were linked to device `MODEL_EXECUTE` tasks
using `connection_id`, then to the first AllReduce. Five middle targets were
checked on every rank in both saved parent/prefetch arms.

| Mean per rank/target | Parent | Prefetch | Difference |
| --- | ---: | ---: | ---: |
| Exposed first wait | 0.667246 ms | 1.159374 ms | +0.492128 ms |
| Arrival lead relative to the last rank entering the collective | 0.518536 ms | 1.025451 ms | +0.506915 ms |
| Wait remaining after subtracting arrival lead | 0.148710 ms | 0.133923 ms | -0.014787 ms |
| CPU submission lead relative to the last submitting rank | 0.515129 ms | 1.010502 ms | +0.495373 ms |

In all ten checked steps, the rank submitting last on CPU also entered the
collective last. CPU-submit/collective-entry correlation exceeds 0.9997 in
every step. Previous drafts ended within 6-17 microseconds across ranks;
collective ends aligned within 12-25 microseconds. In the prefetch arm,
rank 5 submitted last in all five steps; the last rank varied in the parent.

This supports CPU submission skew causing exposed collective waiting in
these captures. It does not establish that prefetch caused the skew.
The remaining-after-arrival term includes communication progress; it is not
an isolated measurement of data transfer time. Arrival lead and first wait
overlap and must not be added as separate savings.

## Cross-check on the latest measured ON/OFF tree

The later `b7bf0486...` source was checked independently across eight ranks
and five middle targets per arm. ON submitted Target with a 1.31-1.53 ms
cross-rank spread; rank 7 was last in every checked step. Its CPU-submit/
collective-entry correlation exceeded 0.9995. OFF could queue its next Target
before the previous device draft finished, whereas ON submitted almost
immediately before its Target started on device.

To include OFF's work that overlaps device execution, host preparation was
measured after the last long `aclrtSynchronizeEvent` completed and before
the next Target submission:

| Observed host interval | OFF | ON | Difference |
| --- | ---: | ---: | ---: |
| Total wall interval | 1.866637 ms | 3.892302 ms | +2.025664 ms |
| Recorded CPU APIs, nested spans counted once | 0.692325 ms | 1.156498 ms | +0.464173 ms |
| Unmarked wall interval | 1.174312 ms | 2.735803 ms | +1.561491 ms |

The recorded increment includes about 0.158 ms of `as_strided`/`slice`,
0.043 ms of `copy_`, and 0.028 ms of copy-workspace preparation. Python
call stacks were not captured for the unmarked time. It can include
Python/NumPy metadata and descriptor work, IPC, shorter waits or OS
descheduling. Attributing all 1.561 ms to the 79-layer descriptor builder
or Resident updates is unsupported. These intervals are not additive TPOT
predictions. The first checked CPU interval also has a larger readiness
spread; it is retained in the means rather than silently discarded.

Existing count-copy/MTP ordering is preserved. The current-source prefetch
plus prefix-view candidate passed ten real cold-module CPU integration
checks and 54 actual-method CPU policy cases with initialization denied.
Model/NPU and paired N2 qualification remain pending; CPU checks do not close
the <=3% target.
