# gmxpacked peak performance env - 2026-07-01

This note records the current force-only 160k peak benchmark env so future
profiling does not rediscover the same flag combination or confuse clean runs
with nsys-wrapped runs.

## Case

Use the 160k force-only NVE case:

```text
/tmp/sponge-onepass-capacity-check/mdin_onepass_10000.spg.toml
```

Equivalent inputs are valid if they keep:

- `[PM] MPI_size = 0`
- `[LJ] direct_kernel = "clustered"`
- `step_limit = 10000`
- `cutoff = 8.0`
- default clustered `skin = 10.0`

Do not compare these numbers with the 160k PME case or with nsys-wrapped
`Core Run Speed` values.

## Stable peak env

This is the current stable peak-performance env for the force-only case.

Important: keep rolling source cache disabled until that path is fixed. The
same env with `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=1`
has shown intermittent 10000-step NaN after repeated no-verify runs.

```sh
SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1
SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1
SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0
SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1
SPONGE_CLUSTERED_SHIFT_PARTITIONED_BUILDER=1
SPONGE_CLUSTERED_FIXED_SHIFT_LEAF_SCREENING=1
SPONGE_CLUSTERED_GMXPACKED_FILL_PRUNE_REUSE=1
SPONGE_CLUSTERED_GMXPACKED_FILL_PRUNE_REUSE_LIGHT=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_PARALLEL_ACCUM=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_FRAGMENT_PARALLEL_EMIT=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_COOPERATIVE=1
SPONGE_CLUSTERED_GMXPACKED_DIRTY_J_PARALLEL_SCAN=1
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ZERO_DIRTY_SOURCE_REUSE=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_COUNT=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_DEVICE_BLOCKS=256
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_TASK_SPLIT_DEPTH=2
```

Do not add these flags to the peak env:

```sh
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_SOURCE_CACHE_PATCH=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SASS_OPT=1
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SLIM=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COLLECT_SASS_OPT=1
SPONGE_CLUSTERED_GMXPACKED_PAIR_SHIFT_SIMPLE_REFRESH=1
SPONGE_CLUSTERED_GMXPACKED_SORTED_CLUSTER_MAP=1
SPONGE_CLUSTERED_GMXPACKED_SCI_SHIFT_SPLIT_SKIP_EMPTY=1
SPONGE_CLUSTERED_GMXPACKED_INNER_ACTIVE_CACHED_FILL=1
```

The source-cache patch did not improve the 2026-07-01 peak path. The 2026-07-03
dirty-gate bundle above was stable with rolling cache off, but it was slower
than the default-off path. The 2026-07-08 queue2 current-peak retest promoted
the queue2 count gate only together with the cooperative fixed-light count gate;
the cached inner-active fill gate remains an opt-in comparison target until the
multi-system alternating e2e runs justify promoting it.

## 2026-07-03 rolling source cache decision

Case:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

The worktree was first stashed and rebuilt clean, then the dirty experiment
stash was popped and rebuilt again. The purpose was to separate peak-env
stability from uncommitted experimental code.

| configuration | runs | stability | average speed | speed range |
|---|---:|---:|---:|---:|
| clean HEAD, rolling source cache on | 3 | 2 finite, 1 NaN | finite runs: `104.064915 ns/day` | `103.112076-105.017754 ns/day` |
| clean HEAD, rolling source cache on, zero-dirty source reuse off | 2 | 1 finite, 1 NaN | finite run: `93.731171 ns/day` | n/a |
| clean HEAD, rolling source cache off | 5 | 5 finite, 0 NaN | `107.217345 ns/day` | `105.671165-109.398247 ns/day` |
| dirty experiment code popped, rolling source cache off, new gates default off | 3 | 3 finite, 0 NaN | `106.095558 ns/day` | `104.869461-107.298065 ns/day` |
| dirty experiment code popped, rolling source cache off, new dirty gates on | 2 | 2 finite, 0 NaN | `102.969200 ns/day` | `102.713722-103.224678 ns/day` |

Detailed output files:

```text
/tmp/sponge-clean-stability-10000-r*.out
/tmp/sponge-clean-ablate-zero_off-10000-r*.out
/tmp/sponge-clean-ablate-rolling_off-10000-r*.out
/tmp/sponge-dirty-rollingoff-10000-r*.out
/tmp/sponge-dirty-rollingoff-gateson-10000-r*.out
```

Decision:

- the stable peak env disables rolling source cache;
- the current highest observed stable 10000-step speed before the fixed-count
  dispatch repair was `109.398247 ns/day`;
- the dirty experiment gates are not part of the peak env;
- rolling source cache should be treated as an isolated correctness/stability
  repair target before it is used again in peak runs.

## 2026-07-03 fixed-count specialization dispatch repair

The fixed-shift/light count specialization gate originally required rolling
source cache to be enabled, so it did not match the stable rolling-off peak env.
That dispatch condition was repaired and validated in:

```text
docs/gmxpacked-fixed-count-specialized-stable-peak-20260703.md
```

Updated stable peak result on:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

| configuration | runs | stability | average speed | speed range |
|---|---:|---:|---:|---:|
| fixed-count specialized on, rolling source cache off | 3 | 3 finite, 0 NaN | `111.150754 ns/day` | `110.137749-111.908737 ns/day` |
| fixed-count specialized off, rolling source cache off | 1 | 1 finite, 0 NaN | `106.373154 ns/day` | n/a |

The highest observed stable 10000-step speed before the dedicated fixed-light
count promotion was:

```text
111.908737 ns/day
```

## 2026-07-04 dedicated fixed-light count promotion

The dedicated fixed-light count kernel is now part of the stable peak env:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1
```

This gate remains default-off in code. It is promoted only for the fixed-shift,
onepass candidate-leaf, fill-prune-reuse-light peak path above.

Retest input:

```text
/tmp/sponge-fixed-light-probes-20260704/mdin_dedicated_10000.spg.toml
```

Build and correctness checks:

- `ninja -C build-dev-cuda13 SPONGE` passed.
- `pixi run -e dev-cuda13 compile` passed.
- 2000-step verify had zero count and fill mismatches:
  `/tmp/sponge-fixed-light-probes-20260704/dedicated_verify2000.err`.
- 10000-step dedicated run was finite with empty stderr:
  `/tmp/sponge-fixed-light-probes-20260704/dedicated_10000.out`.

Clean 10000-step retest on the same input:

| configuration | speed | `Calculate_Force` | stderr |
|---|---:|---:|---:|
| peak env without dedicated count | `117.301613 ns/day` | `6.875158 s` | empty |
| peak env with dedicated count | `131.760040 ns/day` | `6.068877 s` | empty |

Commit-time alternating e2e retest:

```text
/tmp/sponge-fixed-light-probes-20260704/alternating-e2e-20260704-1037
```

| order | configuration | speed | `Calculate_Force` | final temperature | stderr |
|---:|---|---:|---:|---:|---:|
| 1 | peak env without dedicated count | `117.054840 ns/day` | `6.882856 s` | `295.14 K` | empty |
| 2 | peak env with dedicated count | `130.189102 ns/day` | `6.150772 s` | `294.44 K` | empty |
| 3 | peak env without dedicated count | `117.026611 ns/day` | `6.891029 s` | `295.05 K` | empty |
| 4 | peak env with dedicated count | `130.491608 ns/day` | `6.126318 s` | `294.53 K` | empty |

Alternating-run averages: baseline `117.040726 ns/day`, dedicated
`130.340355 ns/day` (`+11.363%`); baseline `Calculate_Force`
`6.886942 s`, dedicated `6.138545 s` (`-10.867%`).

2026-07-08 current-binary peak sanity check:

```text
/tmp/sponge-dedicated-recheck-20260708
```

Same wat160k 10000-step force-only input with
`clustered_rebuild_skin = 11.0`:

| configuration | speed | `Calculate_Force` | note |
|---|---:|---:|---|
| dedicated count only | `122.342957 ns/day` | `6.287283 s` | not the current peak count path |
| dedicated + cooperative count | `129.247757 ns/day` | `5.908831 s` | current peak count path |

The peak env for post-cooperative-count comparisons must include both
`SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1` and
`SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_COOPERATIVE=1`. Omitting the
cooperative gate drops wat160k back to the `122-124 ns/day` range and should not
be treated as a regression of the count optimization.

NCU first count launch:

| configuration | count kernel | registers/thread | eligible warps/scheduler | issued warps/scheduler | local spill |
|---|---:|---:|---:|---:|---:|
| peak env without dedicated count | `59.844224 ms` | `71` | `0.12` | `0.11` | `0 B` |
| peak env with dedicated count | `22.289888 ms` | `70` | `0.51` | `0.38` | `0 B` |

Nsys 10000-step retest artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/nsys_baseline_10000.nsys-rep
/tmp/sponge-fixed-light-probes-20260704/nsys_dedicated_10000.nsys-rep
/tmp/sponge-fixed-light-probes-20260704/nsys_baseline_10000_stats_cuda_gpu_kern_sum.csv
/tmp/sponge-fixed-light-probes-20260704/nsys_dedicated_10000_stats_cuda_gpu_kern_sum.csv
```

Nsys-wrapped run speed improved from `114.208397 ns/day` to
`126.263268 ns/day`. The targeted kernel group improved without shifting cost
into materialize/fill:

| group | without dedicated | with dedicated | delta |
|---|---:|---:|---:|
| main count kernel | `1250.114 ms x25` | `460.243 ms x25` | `-63.184%` |
| materialize light fragments | `34.377 ms x25` | `34.521 ms x25` | `+0.418%` |
| record-stream fill total | `174.572 ms x125` | `175.539 ms x125` | `+0.554%` |
| Count + Materialize + Fill | `1459.063 ms` | `670.303 ms` | `-54.059%` |
| total GPU kernels | `6912.057 ms` | `6142.529 ms` | `-11.133%` |

Decision: promote
`SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1` into the stable peak
env. Keep the previous fixed-light SASS/slim gates, source-cache patch, and
rolling source cache out of the peak env.

## 2026-07-03 fixed-shift count metadata experiment

The follow-up gate
`SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_COUNT_METADATA=1` was implemented and
validated, but it is not part of the peak env. Details:

```text
docs/gmxpacked-fixed-shift-count-metadata-20260703.md
```

Result: 2000-step verify was zero-mismatch, but 10000-step nsys showed
`Collect + Count` regressing from `1970.932 ms` to `2004.089 ms`. The extra
metadata memory was about `53.95 MiB`, which is acceptable by itself, but the
kernel timing misses the acceptance gate.

## 2026-07-03 fixed-light slim count experiment

The follow-up gate
`SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SLIM=1` was implemented and
validated, but it is not part of the peak env. Details:

```text
docs/gmxpacked-count-fixed-light-slim-20260703.md
```

Result: 2000-step verify was zero-mismatch. NCU on the first count launch showed
a small improvement over the fixed-light SASS opt variant, `60.948 ms` to
`59.978 ms`, with the same `72` registers/thread and no local spilling. This is
not enough to promote it: the 10000-step sanity run was finite but only reached
`107.187294 ns/day`, below the recorded stable peak of `111.908737 ns/day`.

## Observed results

All rows below are from:

```text
/home/youmans/sidereus/SPONGE-mainline-nbnxm-fixed-builder-specialize
```

commit:

```text
b99dc70 Add gated fixed-shift builder specialization
```

Clean 10000-step matrix:

| configuration | speed | `Calculate_Force` | stderr |
|---|---:|---:|---:|
| historical clean peak before this matrix | `87.843224 ns/day` | `9.175779 s` | empty |
| specialized + onepass/count-fragment only | `83.422737 ns/day` | `9.605787 s` | empty |
| specialized + onepass/count-fragment only, repeat | `83.022064 ns/day` | `9.519245 s` | empty |
| specialized + dirty-J parallel scan | `91.473190 ns/day` | `8.686279 s` | empty |
| specialized + zero-dirty source reuse | `91.191925 ns/day` | `8.697045 s` | empty |
| no specialization + dirty-J parallel scan | `90.561996 ns/day` | `8.779191 s` | empty |
| no specialization + zero-dirty source reuse | `91.395897 ns/day` | `8.728144 s` | empty |
| specialized + dirty-J parallel scan + zero-dirty source reuse | `101.513206 ns/day` | `7.792568 s` | empty |
| specialized + dirty-J parallel scan + zero-dirty source reuse, repeat | `103.863182 ns/day` | `7.667630 s` | empty |

The old `~90 ns/day` memory was not a real 90+ clean result. The artifact found
under `/tmp` was `87.843224 ns/day`; the 90+ results require the combined
dirty-J and zero-dirty source-reuse env above.

## Correctness check

The peak env passed a 2000-step subgroup verify run:

```text
/tmp/sponge-combined-opt-check-20260701/spec_dirtyj_zero_verify_2000.err
```

Observed mismatch lines:

```text
flag_mismatch=0 cj_mismatch=0 excl_mismatch=0
row_count_mismatch=0 field_mismatch=0
```

The 10000-step clean peak repeats had finite final temperatures:

| run | final temperature |
|---|---:|
| `spec_dirtyj_zero_10000` | `294.40 K` |
| `spec_dirtyj_zero_r2_10000` | `293.57 K` |

## nsys anchor

Use the pixi Nsight Systems binary:

```sh
.pixi/envs/dev-cuda13/nsight-compute-2025.3.1/host/target-linux-x64/nsys
```

Profiled peak env result:

```text
/tmp/sponge-combined-opt-check-20260701/nsys_spec_dirtyj_zero_10000.nsys-rep
/tmp/sponge-combined-opt-check-20260701/nsys_spec_dirtyj_zero_10000_stats_cuda_gpu_kern_sum.csv
```

Nsys-wrapped `Core Run Speed`:

```text
95.765167 ns/day
```

Kernel summary:

| kernel group | total time |
|---|---:|
| total GPU kernels | `7840.860 ms` |
| force true variant | `2412.013 ms x10001` |
| force false variant | `629.669 ms x10001` |
| `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true,true,true>` | `1459.261 ms x20` |
| `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass` | `891.425 ms x20` |
| `Mark_Gmxpacked_Incremental_Dirty_J_Candidates_Parallel` | `56.932 ms x813` |
| `Materialize_Gmxpacked_Record_Stream_Sources_From_Light_Count_Fragments` | `38.277 ms x20` |
| `Refresh_Gmxpacked_Pair_Shift_Bits` | `389.941 ms x10022` |

The major delta versus the earlier specialized-only nsys run is not the count
kernel. It is:

- dirty-J marking: about `816 ms -> 57 ms`;
- active source fill: the heavy
  `Fill_Gmxpacked_Record_Stream_Active_View_Sources` path is avoided by
  zero-dirty source reuse.

## Queue2 Peak Cached-Fill Alternating Check

2026-07-08 queue2 current-peak 10000-step alternating runs:

```text
/tmp/sponge-cached-fill-alt5-20260708
```

Environment: the stable peak env above. DNA_COU additionally used
`SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1`. Cached-fill runs additionally
used:

```sh
SPONGE_CLUSTERED_GMXPACKED_INNER_ACTIVE_CACHED_FILL=1
```

All 30 runs completed finite. Water stderr was empty; DNA_COU stderr contained
only the expected AB-table fallback warning.

| system | cached fill | speed mean | speed range | `Calculate_Force` mean | force delta vs off | speed delta vs off |
|---|---|---:|---:|---:|---:|---:|
| wat160k | off | `134.496637 ns/day` | `132.636078-136.152618` | `5.665332 s` | n/a | n/a |
| wat160k | on | `137.364664 ns/day` | `133.901199-141.295807` | `5.540484 s` | `-2.204%` | `+2.132%` |
| wat600k | off | `44.111698 ns/day` | `43.336559-44.776684` | `17.745679 s` | n/a | n/a |
| wat600k | on | `45.340714 ns/day` | `44.251278-45.779572` | `17.255308 s` | `-2.763%` | `+2.786%` |
| dna_cou | off | `308.222406 ns/day` | `302.058014-310.637268` | `3.886263 s` | n/a | n/a |
| dna_cou | on | `312.148608 ns/day` | `310.150604-314.439850` | `3.823580 s` | `-1.613%` | `+1.274%` |

This validates cached fill as a positive opt-in comparison gate under the queue2
current-peak env. It is still listed outside the default peak env until a
separate decision promotes it.

## How to avoid repeating this search

When investigating peak performance, first run the full peak env above. Then
ablate only one layer at a time:

1. remove `SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED`;
2. remove `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED`;
3. remove `SPONGE_CLUSTERED_GMXPACKED_DIRTY_J_PARALLEL_SCAN`;
4. remove `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ZERO_DIRTY_SOURCE_REUSE`;
5. remove onepass/count-fragment flags only if the target is candidate-leaf or
   count-kernel attribution.

If a run reports around `80-84 ns/day`, check first whether dirty-J parallel
scan and zero-dirty source reuse are both missing. If a run reports around
`87-88 ns/day`, it is likely the old clean peak without the combined rolling
source-cache optimizations. If a run reports `95-104 ns/day`, confirm whether it
is clean or nsys-wrapped before comparing absolute speed.
