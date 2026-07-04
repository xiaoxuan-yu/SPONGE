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
