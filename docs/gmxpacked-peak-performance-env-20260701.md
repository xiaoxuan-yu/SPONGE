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
SPONGE_CLUSTERED_GMXPACKED_DIRTY_J_PARALLEL_SCAN=1
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ZERO_DIRTY_SOURCE_REUSE=1
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
```

The source-cache patch did not improve the 2026-07-01 peak path. The 2026-07-03
dirty-gate bundle above was stable with rolling cache off, but it was slower
than the default-off path.

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

The current highest observed stable 10000-step speed is now:

```text
111.908737 ns/day
```

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

## How to avoid repeating this search

When investigating peak performance, first run the full peak env above. Then
ablate only one layer at a time:

1. remove `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED`;
2. remove `SPONGE_CLUSTERED_GMXPACKED_DIRTY_J_PARALLEL_SCAN`;
3. remove `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ZERO_DIRTY_SOURCE_REUSE`;
4. remove onepass/count-fragment flags only if the target is candidate-leaf or
   count-kernel attribution.

If a run reports around `80-84 ns/day`, check first whether dirty-J parallel
scan and zero-dirty source reuse are both missing. If a run reports around
`87-88 ns/day`, it is likely the old clean peak without the combined rolling
source-cache optimizations. If a run reports `95-104 ns/day`, confirm whether it
is clean or nsys-wrapped before comparing absolute speed.
