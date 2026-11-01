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
