# gmxpacked candidate leaf collect probe - 2026-07-04

## Scope

This note records the candidate leaf collect decomposition added after
`aa7961a Add dedicated fixed-light count kernel`.

The probe is default-off and only runs on the current peak fixed-shift onepass
path after production candidate leaf collection succeeds. It writes independent
scratch buffers and does not modify:

- `d_sci_candidate_leaf_ids`
- `d_sci_candidate_leaf_offsets`
- `d_candidate_leaf_onepass_cursor`

## Gates

Peak env remains the stable set from
`docs/gmxpacked-peak-performance-env-20260701.md`, including:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1
```

Additional probe gates:

```sh
SPONGE_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_TRAVERSAL_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_SCREEN_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_EMIT_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_STATS=1
```

Default-off node-overlap experiment gate:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_NODEBOX_OPT=1
```

Default-off cooperative traversal probe gates:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COOP_TRAVERSAL_PROBE=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COOP_SCREEN_PROBE=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COOP_EMIT_PROBE=1
```

No cooperative production collector gate is retained in the final code from
this round. The cooperative traversal probe failed the stop rule below, so the
production `COOP_COLLECT` branch was not kept.

## Smoke

Input:

```text
/tmp/sponge-fixed-light-probes-20260704/mdin_probe_trace1.spg.toml
```

One-step smoke with all three probe gates plus stats:

| mode | production leaves | probe total | cursor | overflow |
|---|---:|---:|---:|---:|
| traversal | 5,257,583 | 7,991,009 | -1 | -1 |
| screen | 5,257,583 | 5,257,583 | -1 | -1 |
| emit | 5,257,583 | 5,257,583 | 5,257,583 | 0 |

This confirms that `screen` and `emit` match the production leaf count while
`traversal` intentionally measures the pre-leaf-screen upper bound.

## NCU

Input:

```text
/tmp/sponge-fixed-light-probes-20260704/mdin_probe_trace1.spg.toml
```

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_traversal.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_screen.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_emit.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_production.log
```

Single-launch NCU results:

| kernel | duration | regs/thread | eligible warps/sched | issued warp/sched | DRAM | L2 | spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| traversal probe | 28.038 ms | 31 | 2.67 | 0.58 | 0.38% | 0.29% | 0 |
| screen probe | 29.755 ms | 47 | 2.26 | 0.58 | 0.41% | 0.66% | 0 |
| emit probe | 33.495 ms | 46 | 2.25 | 0.58 | 0.56% | 1.28% | 0 |
| production onepass collect | 30.905 ms | 48 | 2.09 | 0.56 | 0.54% | 2.35% | 0 |

Marginal costs:

- screen minus traversal: `+1.717 ms`
- emit minus screen: `+3.740 ms`
- production minus screen: `+1.150 ms`

## Nsys

Probes-off production baseline after the probe implementation:

```text
/tmp/sponge-fixed-light-probes-20260704/nsys_candidate_leaf_default_10000.nsys-rep
/tmp/sponge-fixed-light-probes-20260704/nsys_candidate_leaf_default_10000.sqlite
/tmp/sponge-fixed-light-probes-20260704/nsys_candidate_leaf_default_10000_stats_cuda_gpu_kern_sum.csv
```

Result:

| metric | value |
|---|---:|
| Core Run Speed | 127.842232 ns/day |
| Calculate_Force | 6.236134 s |
| candidate collect | 630.884 ms x24, avg 26.287 ms |
| dedicated fixed-light count | 441.436 ms x24, avg 18.393 ms |
| stderr | empty |

The probes are default-off and did not add probe output or shift the production
baseline outside the existing aa7961a range.

## Decision

Traversal dominates candidate leaf collect. The traversal-only probe is already
about 91% of the NCU production collector duration, and it is higher than the
Nsys production average. Leaf overlap screening adds only about 1.7 ms, and
scratch emit adds about 3.7 ms over screen.

Next optimization should target traversal/node overlap, not leaf screening or
onepass emit. The first candidates are:

1. Precompute the fixed-shift target box per candidate SCI and avoid rebuilding
   shifted target center/size state inside every node overlap predicate.
2. Replace the generic `cstone::overlap` path in the fixed-shift collector with
   a lighter predicate specialized to the unit cornerstone box and open
   boundary.
3. Only after a traversal probe improves meaningfully, remeasure screen and
   production collect before considering emit buffering.

Stop this line if the optimized traversal path cannot recover at least about
150 ms over a 10000-step run; otherwise promote a dedicated traversal/node
overlap collector behind a default-off gate and validate with the existing
2000-step verify and alternating e2e protocol.

## Node-Overlap Attempt

After the decomposition above, NCU/PTX/SASS analysis was run before editing the
collector:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_production_deep_details.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_production_source_sass.log
```

The production collector is not bandwidth-limited:

| metric | value |
|---|---:|
| instructions executed | 11,353,526,233 |
| memory throughput | 5.11 GB/s |
| mem busy | 2.37% |
| L2 hit rate | 89.43% |
| eligible warps/scheduler | 2.09 |
| issued warps/scheduler | 0.56 |
| stall not selected | 2.73 cycles |
| stall math pipe throttle | 2.70 cycles |
| stall wait | 1.58 cycles |
| long scoreboard | 0.43 cycles |

The source page mostly reports SASS/address rows for this build, so line-level
source attribution would require a separate lineinfo build if needed later.

Two fixed-shift node-overlap variants were tested behind
`SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_NODEBOX_OPT=1`:

1. An interval/open-box predicate using precomputed min/max bounds.
2. A narrower preshift variant that computes `target_center + shift` once per
   SCI and keeps the original cornerstone overlap predicate.

The interval/open-box variant was rejected because it made traversal worse. The
current gate keeps only the preshift variant.

Single-launch NCU comparison:

| kernel | gate | duration | regs/thread | eligible warps/sched | issued warp/sched | spill |
|---|---|---:|---:|---:|---:|---:|
| production onepass collect | off | 30.905 ms | 48 | 2.09 | 0.56 | 0 |
| traversal probe | off | 28.038 ms | 31 | 2.67 | 0.58 | 0 |
| production onepass collect | interval/open-box attempt | 30.221 ms | 47 | 2.23 | 0.57 | 0 |
| traversal probe | interval/open-box attempt | 30.205 ms | 31 | 2.67 | 0.57 | 0 |
| production onepass collect | preshift current gate | 30.148 ms | 47 | 2.22 | 0.58 | 0 |
| traversal probe | preshift current gate | 31.378 ms | 31 | 2.65 | 0.58 | 0 |

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_nodebox_production.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_nodebox_traversal.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_preshift_production.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_preshift_traversal.log
```

The best production result is only about 2.45% faster than the baseline NCU
launch, while the isolated traversal probe regresses. This does not meet the
15% candidate-collect target or the 150 ms / 10000-step stop-rule. Keep the
gate default-off and do not add it to the peak env group.

Smoke with the preshift gate and all candidate-leaf probes still preserves the
screen/emit totals:

```text
/tmp/sponge-fixed-light-probes-20260704/nodebox_opt_probe_smoke.out
/tmp/sponge-fixed-light-probes-20260704/nodebox_opt_probe_smoke.err
```

| mode | production leaves | probe total | cursor | overflow |
|---|---:|---:|---:|---:|
| traversal | 5,257,583 | 7,991,009 | -1 | -1 |
| screen | 5,257,583 | 5,257,583 | -1 | -1 |
| emit | 5,257,583 | 5,257,583 | 5,257,583 | 0 |

Decision: do not continue with simple shifted-center or generic node-box
micro-optimizations. The next candidate-leaf line should inspect
`singleTraversal` itself and the SFC/prefix representation, or move to a more
specialized traversal collector only if NCU/SASS shows a concrete reduction in
the traversal probe first.

## Cooperative Traversal Probe

Implementation:

- `SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COOP_TRAVERSAL_PROBE=1`
  launches a probe where only sublane 0 in each 8-lane subgroup walks the
  `singleTraversal` node/backtrack state.
- The leader broadcasts endpoint leaf ids and traversal state with
  `deviceShfl(subgroup_mask, value, subgroup_lane_base, 32)`.
- `coop_screen` and `coop_emit` reuse the same leaf-screening and scratch-emit
  semantics as the existing candidate-leaf probes.
- The probes write independent scratch buffers only; production candidate leaf
  ids, offsets, and cursor are not modified.

Final-code 1-step smoke:

```text
/tmp/sponge-fixed-light-probes-20260704/coop_probe_final_smoke.out
/tmp/sponge-fixed-light-probes-20260704/coop_probe_final_smoke.err
```

| mode | production leaves | probe total | cursor | overflow |
|---|---:|---:|---:|---:|
| coop_traversal | 5,257,583 | 7,991,009 | -1 | -1 |
| coop_screen | 5,257,583 | 5,257,583 | -1 | -1 |
| coop_emit | 5,257,583 | 5,257,583 | 5,257,583 | 0 |

NCU artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_coop_traversal.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_coop_production.log
```

The production NCU artifact came from the transient gated collector used only
to quantify the candidate path before the gate was removed.

Single-launch NCU comparison:

| kernel | duration | regs/thread | eligible warps/sched | issued warp/sched | DRAM | L2 | spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| traversal probe baseline | 28.038 ms | 31 | 2.67 | 0.58 | 0.38% | 0.29% | 0 |
| production onepass collect baseline | 30.905 ms | 48 | 2.09 | 0.56 | 0.54% | 2.35% | 0 |
| coop traversal probe | 29.673 ms | 38 | 2.80 | 0.60 | 0.42% | 0.29% | 0 |
| transient coop production collect | 32.946 ms | 47 | 2.20 | 0.58 | 0.51% | 2.18% | 0 |

The cooperative probe did not reduce traversal work. It regressed the
traversal probe by about 5.8% versus the 28.038 ms baseline and stayed above
the 24 ms stop rule. The transient production collector also regressed versus
the 30.905 ms baseline.

Transient `COOP_COLLECT` correctness runs, taken before removing the production
gate:

```text
/tmp/sponge-fixed-light-probes-20260704/coop_collect_verify2000.out
/tmp/sponge-fixed-light-probes-20260704/coop_collect_verify2000.err
/tmp/sponge-fixed-light-probes-20260704/coop_collect_10000.out
/tmp/sponge-fixed-light-probes-20260704/coop_collect_10000.err
```

The 2000-step subgroup verify reported zero mismatches at all sampled steps:
`flag_mismatch=0`, `cj_mismatch=0`, `excl_mismatch=0`,
`row_count_mismatch=0`, and `field_mismatch=0`.

The 10000-step run was finite with empty stderr:

| metric | value |
|---|---:|
| final temperature | 294.23 K |
| Calculate_Force | 6.023303 s |
| Core Run Speed | 132.708557 ns/day |

Decision: stop the cooperative traversal line here. Keep only the default-off
cooperative probes as diagnostic surfaces. Do not add `NODEBOX_OPT`, cooperative
traversal, or any cooperative production collector to the peak env. The next
optimization target should move away from candidate-leaf traversal unless a new
SASS/PTX idea can first demonstrate a traversal probe below 24 ms.

## E2E Low-Frequency Distribution Sampling - 2026-07-08

Question: is the high zero-SCI ratio in candidate leaf collection a one-snapshot
artifact, or does it persist during longer e2e runs?

Instrumentation:

- `SPONGE_CLUSTERED_GMXPACKED_CANDIDATE_LEAF_SAMPLE_INTERVAL=N` runs a
  default-off screen probe during candidate leaf collection and prints
  `candidate_sci`, `production_leaves`, `zero_sci`, `zero_pct`,
  `nonzero_sci`, `max_per_sci`, and top-5% leaf share.
- The sampler writes independent scratch counts and does not modify production
  candidate leaf ids, offsets, or force payloads.

Real peak e2e result:

```text
/tmp/sponge-collect-e2e-sampling-20260708/wat160k_peak/e2e_lowfreq.stderr
/tmp/sponge-collect-e2e-sampling-20260708/wat600k_peak/e2e_lowfreq.stderr
/tmp/sponge-collect-e2e-sampling-20260708/dna_peak_padding/e2e_lowfreq.stderr
```

With the normal peak inputs, each 10000-step e2e run produced only one sample
at `step=0`. This is expected: `refresh_interval=0` uses automatic rebuilds
with `reuse_skin=10.00` and `skin_permit=0.50`, and these runs do not re-enter
candidate leaf collection within 10000 steps. Therefore the normal peak e2e
run confirms the initial payload shape but does not provide multiple collect
snapshots.

For distribution stability only, a second run forced periodic rebuilds with:

```toml
[neighbor_list]
refresh_interval = 1000
```

Artifacts:

```text
/tmp/sponge-collect-e2e-sampling-20260708/wat160k_peak/e2e_lowfreq_refresh1000.stderr
/tmp/sponge-collect-e2e-sampling-20260708/wat600k_peak/e2e_lowfreq_refresh1000.stderr
/tmp/sponge-collect-e2e-sampling-20260708/dna_peak_padding/e2e_lowfreq_refresh1000.stderr
```

These forced-refresh runs sampled `step=0,1000,...,10000` and should not be
used as performance baselines. They are only distribution checks.

| system | samples | zero_sci avg | zero_sci range | top5 leaf share avg | nonzero_sci avg | production leaves avg |
|---|---:|---:|---:|---:|---:|---:|
| wat160k | 11 | 90.01% | 89.98-90.05% | 87.33% | 6932.3 | 4668176.8 |
| wat600k | 11 | 92.51% | 92.50-92.53% | 94.09% | 20784.1 | 5771291.8 |
| dna_cou | 11 | 83.17% | 83.05-83.22% | 71.27% | 2249.5 | 639941.1 |

Conclusion: the high `zero_sci` ratio is a stable fixed-shift candidate
collection property, not a rare snapshot. It is strongest for water systems
and especially `wat600k`, where only about 7.5% of candidate SCIs produce
screened leaves and the top 5% of SCIs carry about 94% of accepted leaves. This
supports prioritizing active-candidate compaction or finer-grained dynamic
collect scheduling over optimizing emit/write traffic.

## Active Root-Child Collect Replay - 2026-07-08

Starting point: NCU showed candidate-leaf collect is latency and load-balance
limited, not DRAM bandwidth limited. The production-style `collect-screen`
probe for the wat160k peak force payload had very low memory utilization and
low active/eligible warp counts:

| mode | duration | grid | waves/SM | active warps | eligible warps | DRAM throughput | L1 hit | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline screen | 6.826 ms | 244 | 0.19 | 3.56% | 0.385 | 0.25% | 89.15% | 54.93% |
| active root-child | 4.308 ms | 176 | 0.14 | 4.77% | 0.420 | 0.43% | 90.20% | 48.30% |

The simple active-SCI replay was not enough: it reduced work items too far and
was slower than baseline even with the same accepted leaves. The useful variant
is to compact active SCIs and then split each active SCI into eight root-child
subtree tasks. Each task traverses one root child and atomically accumulates the
per-SCI accepted leaf count. This keeps the replay sidepath default-off and does
not modify the production builder/force path.

Wat160k peak force-payload validation:

| replay mode | candidate SCI | work items | leaves | nonzero SCI | avg time |
|---|---:|---:|---:|---:|---:|
| baseline screen | 3900 | 3900 | 51044 | 352 | 6.062 ms |
| active SCI only | 352 | 352 | 51044 | 352 | 7.183 ms |
| active root-child | 352 | 2816 | 51044 | 352 | 3.793 ms |

The NCU instruction count stayed similar (`5.08e8` baseline vs `5.19e8`
root-child), while kernel duration dropped by about 1.6x. This points to better
scheduling/latency hiding from finer-grained subtree tasks rather than less
memory traffic.

Three-system builder-footer replay check, 3 interleaved rounds per system:

| system | baseline screen | active root-child | speedup | leaves check |
|---|---:|---:|---:|---:|
| wat160k | 5.458 ms | 3.657 ms | 1.49x | 64533 = 64533 |
| wat600k | 4.610 ms | 3.290 ms | 1.40x | 69332 = 69332 |
| dna_cou | 3.614 ms | 0.871 ms | 4.15x | 15812 = 15812 |

Important caveat: the current sidepath builds the active SCI list with an
untimed full collect prepass, so these numbers are replay-kernel upper bounds.
Production integration should derive active SCI work from existing count/shift
flags or builder metadata, not by adding another full collect traversal.

### Root-Child Task Queue Payload

Follow-up sidepath: replace the implicit `active_sci * 8` replay mapping with a
builder/collect payload:

```text
root_child_task_sci_ids[task] = physical candidate SCI
root_child_task_nodes[task]   = root-child subtree node
```

The builder-side task-build kernel tests root and root-child node-box overlap
and emits only overlapping `{sci,node}` tasks. The collect replay consumes this
queue directly and accumulates counts into the physical `sci`, removing the
`logical_sci -> active_sci_indices -> physical_sci` indirection.

One-pass three-system check:

| system | baseline screen | active SCI * 8 replay | root-child queue replay | task build | tasks |
|---|---:|---:|---:|---:|---:|
| wat160k | 5.563 ms | 3.955 ms | 4.434 ms | 0.792 ms | 1099 |
| wat600k | 4.712 ms | 3.448 ms | 3.150 ms | 0.868 ms | 2160 |
| dna_cou | 3.799 ms | 0.936 ms | 0.963 ms | 0.880 ms | 799 |

All modes produced identical accepted leaf totals for each system.

NCU on wat160k shows why the compressed queue is mixed: it removes instructions
but can underfill the GPU.

| kernel | duration | grid | waves/SM | active warps | eligible warps | inst |
|---|---:|---:|---:|---:|---:|---:|
| baseline collect | 6.827 ms | 244 | 0.19 | 3.56% | 0.385 | 5.08e8 |
| active SCI * 8 collect | 4.308 ms | 176 | 0.14 | 4.77% | 0.420 | 5.19e8 |
| root-child queue collect | 5.217 ms | 69 | 0.05 | 5.29% | 0.341 | 4.42e8 |
| task build | 7.424 us | 314 | 0.20 | 14.00% | 0.718 | 1.81e6 |

Decision: `{sci,node}` is the right payload shape for production integration,
but it is too aggressively compact for some systems if each task is just one
root-child subtree. The next queue design should keep direct physical `sci`
payload while preserving enough parallel work, for example by splitting heavy
subtrees into leaf chunks or by padding/replicating only heavy SCI work rather
than globally using `active_sci * 8`.

### Root-Child Queue2: Split One More Octree Level

Follow-up for the queue underfill: keep the same `{sci,node}` payload but build
tasks one octree level deeper. For each overlapping non-leaf root-child node,
the builder task kernel emits overlapping child nodes instead of the root-child
node itself. Leaf root-child nodes are still emitted directly. This preserves
physical `sci` in the payload and raises collect parallelism without going back
to `active_sci * 8`.

Three-system replay check:

| system | baseline screen | active SCI * 8 | queue depth 1 | queue depth 2 | task build depth 2 | tasks depth 2 |
|---|---:|---:|---:|---:|---:|---:|
| wat160k | 5.568 ms | 3.869 ms | 4.502 ms | 1.111 ms | 0.962 ms | 3378 |
| wat600k | 4.574 ms | 3.315 ms | 3.105 ms | 1.521 ms | 0.851 ms | 5354 |
| dna_cou | 3.615 ms | 0.903 ms | 0.951 ms | 0.152 ms | 0.885 ms | 3156 |

All modes produced identical accepted leaf totals for each system.

Wat160k interleaved replay, 3 rounds:

| mode | avg collect replay |
|---|---:|
| baseline screen | 5.474 ms |
| active SCI * 8 | 3.990 ms |
| queue depth 2 | 1.150 ms |

NCU on wat160k:

| kernel | duration | grid | waves/SM | active warps | eligible warps | inst |
|---|---:|---:|---:|---:|---:|---:|
| queue depth 1 collect | 5.217 ms | 69 | 0.05 | 5.29% | 0.341 | 4.42e8 |
| queue depth 2 collect | 1.328 ms | 212 | 0.17 | 10.21% | 0.553 | 3.83e8 |
| queue depth 1 build | 7.424 us | 314 | 0.20 | 14.00% | 0.718 | 1.81e6 |
| queue depth 2 build | 28.256 us | 314 | 0.20 | 8.14% | 0.615 | 5.18e6 |

Decision: depth-2 `{sci,node}` queue fixes the depth-1 underfill and is now the
best sidepath payload candidate. The build kernel cost is small in NCU
microseconds, while the event-timed isolated build in the replay harness is
around `0.85-0.96 ms`; production integration should measure it in the real
builder stream before deciding whether to persist/reuse the queue across force
steps.

Host overhead audit for the same queue-depth-2 build path:

| system | tasks | first build kernel event | first build incl. memset | first build host wall | counter copy host | repeat build+memset GPU | repeat host wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| wat160k | 3378 | 0.894 ms | 0.900 ms | 0.907 ms | 0.028 ms | 0.027 ms | 0.027 ms |
| wat600k | 5354 | 1.470 ms | 1.477 ms | 1.486 ms | 0.025 ms | 0.053 ms | 0.053 ms |
| dna_cou | 3156 | 0.921 ms | 0.926 ms | 0.936 ms | 0.026 ms | 0.014 ms | 0.014 ms |

The repeat column uses 200 back-to-back launches after warmup and includes the
two queue-counter memsets plus the task-build kernel. This matches the NCU
kernel-duration scale and shows that the millisecond-level first-build event is
not the steady-state task-build cost. For production integration, use the
steady-state build+memset cost as the expected device-side overhead, and avoid
host counter copies/synchronization in the main path. The host counter readback
is only a sidepath sizing/diagnostic convenience; the production collect launch
should either use the queue capacity or consume a device-side task counter.

### Queue2 Device-Counter Collect

Sidepath mode:
`production-gmxpacked-collect-screen-rootchild-queue2-devicecounter`.

This keeps the queue2 task-build payload unchanged, but collect no longer uses a
host-read task count to size the launch. Instead, collect launches a fixed
number of blocks and each subgroup atomically takes the next task from a
device-side work cursor until it reaches `*task_counter`. The sidepath still
copies the task count back after collect timing only for diagnostics.

Default fixed collect launch: 256 blocks. It can be overridden with
`SPONGE_ROOT_CHILD_DEVICE_COUNTER_BLOCKS`.

Three-system replay check, 60 timed collect iterations:

| system | host-count queue2 collect | device-counter queue2 collect | leaves check | tasks | device-counter blocks |
|---|---:|---:|---:|---:|---:|
| wat160k | 1.231 ms | 1.247 ms | 64533 = 64533 | 3378 | 256 |
| wat600k | ~1.52 ms | 1.611 ms | 69332 = 69332 | 5354 | 256 |
| dna_cou | ~0.15 ms | 0.202 ms | 15812 = 15812 | 3156 | 256 |

Wat160k block sweep showed 256 blocks as the current best of the tested
settings: 128 blocks took 1.687 ms and 512 blocks took 1.760 ms. The
device-counter path is therefore correct and removes the pre-collect host
counter dependency, but it pays a small extra dynamic-scheduling cost compared
with host-count queue2. This is likely acceptable for production integration if
the priority is preserving an asynchronous builder/collect pipeline.

### Refresh-Owned Queue2 Payload Probe

Sidepath mode:
`production-gmxpacked-refresh-rootchild-queue2`.

This mode keeps the existing refresh replay and appends queue2 payload
generation in the same timed loop. It is not a physically fused refresh kernel,
but it makes the refresh path own queue2 payload production for measuring the
cost model before touching the production builder/refresh code.

Three-system replay, 60 timed iterations:

| system | refresh only | refresh + queue2 payload | delta | queue2 tasks | overflow |
|---|---:|---:|---:|---:|---:|
| wat160k | 0.037 ms | 0.068 ms | 0.031 ms | 3378 | 0 |
| wat600k | 0.138 ms | 0.194 ms | 0.057 ms | 5354 | 0 |
| dna_cou | 0.013 ms | 0.027 ms | 0.014 ms | 3156 | 0 |

The task counts match the queue2 collect sidepath. This gives a cleaner
production-integration estimate than the earlier cold isolated task-build
events: queue2 payload production adds only tens of microseconds when chained
from refresh. A true builder/refresh fusion can still try to remove this
separate launch, but the remaining launch+kernel cost is already small compared
with the collect reduction.

### Production Phase-1 Queue2 Count Gate

Implemented the first production-side integration stage behind a default-off
gate:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_COUNT=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_DEVICE_BLOCKS=256
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_TASK_SPLIT_DEPTH=2
```

Scope:

- replaces only the fixed-shift candidate-leaf count stage;
- builds queue2 `{candidate_sci,node}` tasks on device;
- consumes the task queue with the device-counter queue2 count kernel;
- keeps the existing scan + candidate-leaf fill path for
  `d_sci_candidate_leaf_ids`;
- keeps fixed-light dedicated payload count eligible when candidate leaves came
  from queue2 count instead of onepass.

Wat160k peak 1-step smoke with trace:

```text
/tmp/sponge_queue2_prod_wat160k_peak_trace1_v2.out
/tmp/sponge_queue2_prod_wat160k_peak_trace1_v2.err
```

Key trace:

```text
[clustered candidate leaf queue2 count] step=0 candidate_sci=69417 leaves=4665647 tasks=47155 capacity=4442688 overflow=0 depth=2 blocks=256
[clustered gmxpacked count variant] step=0 variant=subgroup-fixed-light-dedicated fixed_specialized=1 ... onepass=0 ...
[clustered payload count] step=0 ... candidate_leaves=4665647 ... sci=5017 cjpacked=367963 excl=61478 ...
```

The same input and peak env with the gate off reports the same
`candidate_leaves`, `sci`, `cjpacked`, `excl`, `source_rows`, and energy line.
The compact exclusion row count can differ slightly because queue2 count plus
legacy fill changes candidate-leaf ordering before mask dedup, but the 1-step
energy smoke matched:

```text
potential = 3208780.75
LJ_short  = 82512.51
```

Build verification:

```text
cmake --build build-dev-cuda13 --target SPONGE -j 8
```

passed after the integration.

Production 10000-step e2e comparison, single run per case. This is the first
production queue2 count integration check and predates the later
cooperative-count plus cached-fill peak envelope:

```text
/tmp/sponge-queue2-prod-e2e-20260708-164104
```

Environment: the stable peak env at the time; DNA additionally used
`SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1`. Queue2 runs additionally
used `SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_COUNT=1`.

| system | baseline speed | queue2 speed | speed delta | baseline Calculate_Force | queue2 Calculate_Force | force delta | stderr |
|---|---:|---:|---:|---:|---:|---:|---|
| wat160k | 124.551697 ns/day | 119.002571 ns/day | -4.46% | 6.214222 s | 6.528608 s | +5.06% | empty |
| wat600k | 40.516216 ns/day | 39.142693 ns/day | -3.39% | 19.300470 s | 20.083345 s | +4.06% | empty |
| dna_cou | 279.038544 ns/day | 278.584106 ns/day | -0.16% | 4.360749 s | 4.343490 s | -0.40% | same AB-table fallback warning |

Historical conclusion from this first e2e check: the phase-1 production queue2
count gate was runnable, but the isolated gate was not ready to promote by
itself. Replacing onepass candidate-leaf collect with queue2 count plus legacy
fill lost e2e performance on both water systems. The likely reason was that this
stage integrated only the count side: it still paid legacy candidate-leaf fill
and changed the downstream candidate leaf ordering, while the standalone
sidepath win measured a count/collect replay rather than this mixed production
path.

Important baseline note from the 2026-07-08 follow-up: this e2e table did not
enable `SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_COOPERATIVE=1`, so the
wat160k baseline here is a dedicated-count-only baseline rather than the current
post-cooperative-count peak. A same-input retest under
`/tmp/sponge-dedicated-recheck-20260708` measured `122.342957 ns/day` for
dedicated only and `129.247757 ns/day` for dedicated plus cooperative count.
Future queue2 production comparisons should use the latter peak count env.

2026-07-09 superseding peak-env note: queue2 count is part of the current peak
env only in the later combined stack with cooperative fixed-light count, cached
inner-active fill, and refresh block 128. Keep the table above as the traceable
negative result for the intermediate queue2-only production integration stage;
do not use it as the current peak-env recommendation.
