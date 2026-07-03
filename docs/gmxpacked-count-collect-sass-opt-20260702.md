# Gmxpacked Count/Collect SASS-Guided Experiment 2026-07-02

## Scope

This round implemented two default-off experimental gates for the current peak
fixed-shift path:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SASS_OPT=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COLLECT_SASS_OPT=1
```

The gates only affect the gmxpacked direct + outer lifecycle + active view +
rolling source cache + fixed-shift leaf screening + onepass candidate leaves +
specialized light count path.

## Code Changes

- Count path:
  - added a `kFixedLightSassOpt` specialization of
    `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup`;
  - precomputes `record_stream_cutoff_sq`;
  - precomputes the source pair-shift vector once per candidate group;
  - routes prune calls through
    `Prune_Gmxpacked_Record_Stream_Source_Imask_With_Shift`;
  - replaces the fixed-light path's shared-memory `atomicAdd` count updates with
    per-thread local counters plus a warp reduction.

- Collect path:
  - added
    `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass_Sass_Opt`;
  - preloads per-sublane i-cluster validity, center, extent, shift vector, and
    cutoff squared before tree traversal;
  - keeps onepass output order and candidate leaf list semantics unchanged.

Both gates are default-off. The existing kernels remain the production path.

## Correctness

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

Result: build passed.

20-step smoke with both gates:

```text
/tmp/sponge-count-collect-sass-opt-20260702/smoke20_both.out
/tmp/sponge-count-collect-sass-opt-20260702/smoke20_both.err
```

Result: exit 0, empty stderr, no NaN.

2000-step verify with both gates and
`SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1`:

```text
/tmp/sponge-count-collect-sass-opt-20260702/verify2000_both.out
/tmp/sponge-count-collect-sass-opt-20260702/verify2000_both.err
```

Verify mismatches were all zero:

```text
step=0    flag_mismatch=0 cj_mismatch=0 excl_mismatch=0 row_count_mismatch=0 field_mismatch=0
step=615  flag_mismatch=0 cj_mismatch=0 excl_mismatch=0 row_count_mismatch=0 field_mismatch=0
step=1131 flag_mismatch=0 cj_mismatch=0 excl_mismatch=0 row_count_mismatch=0 field_mismatch=0
step=1697 flag_mismatch=0 cj_mismatch=0 excl_mismatch=0 row_count_mismatch=0 field_mismatch=0
```

## 10000-Step End-to-End

Input:

```text
/tmp/sponge-onepass-capacity-check/mdin_onepass_10000.spg.toml
```

Baseline env is the recorded peak env from
`docs/gmxpacked-peak-performance-env-20260701.md`, without the two new gates.

| run | `Calculate_Force` | wall | speed | stderr |
|---|---:|---:|---:|---:|
| peak env, no new gate | `7.809169 s` | `8.562722 s` | `100.912582 ns/day` | empty |
| + count SASS opt | `7.817433 s` | `8.514966 s` | `101.478554 ns/day` | empty |
| + collect SASS opt | `8.040880 s` | `8.741190 s` | `98.852264 ns/day` | empty |
| + both gates | `7.807093 s` | `8.532254 s` | `101.272942 ns/day` | empty |

The apparent e2e gain is below the noise level for this case. Collect-only is a
clear regression.

## NCU Comparison

Baseline reports:

```text
/tmp/sponge-count-collect-ncu-20260702/count_peak_core.csv
/tmp/sponge-count-collect-ncu-20260702/collect_peak_current.csv
```

This round:

```text
/tmp/sponge-count-collect-sass-opt-20260702/count_sass_core_app.csv
/tmp/sponge-count-collect-sass-opt-20260702/collect_sass_core.csv
```

Count NCU used application replay because kernel replay produced `nan` metrics
and `LaunchFailed` after replay.

| metric | count baseline | count SASS opt |
|---|---:|---:|
| duration | `76.30 ms` | `78.34 ms` |
| SM throughput | `9.41%` | `9.39%` |
| memory throughput | `13.45%` | `13.19%` |
| DRAM throughput | `1.02%` | `1.03%` |
| issued warp/scheduler | `0.11` | `0.11` |
| eligible warps/scheduler | `0.12` | `0.12` |
| no eligible | `88.68%` | `88.71%` |
| registers/thread | `71` | `72` |
| theoretical occupancy | `58.33%` | `58.33%` |
| achieved occupancy | `19.46%` | `19.40%` |

| metric | collect baseline | collect SASS opt |
|---|---:|---:|
| duration | `48.88 ms` | `49.59 ms` |
| SM throughput | `43.38%` | `48.47%` |
| memory throughput | `3.64%` | `2.85%` |
| DRAM throughput | `0.52%` | `0.57%` |
| issued warp/scheduler | `0.56` | `0.59` |
| eligible warps/scheduler | `2.16` | `2.32` |
| no eligible | `43.51%` | `41.42%` |
| registers/thread | `48` | `45` |
| theoretical occupancy | `83.33%` | `83.33%` |
| achieved occupancy | `32.36%` | `31.67%` |

## Verdict

Do not add either new gate to the peak env.

- Count SASS opt is correct but does not improve the profiled count kernel:
  `76.30 ms -> 78.34 ms`.
- Collect SASS opt reduces registers and improves issue/eligible metrics, but
  increases the profiled kernel duration and regresses 10000-step e2e.
- The experiment suggests the remaining count bottleneck is still latency and
  low eligible-warps, but this local SASS-guided cleanup does not move the key
  scheduler metrics.

Next count/collect work should target structural reduction in candidate/count
work or payload shape, not this local precompute/shared-atomic cleanup.

## 2026-07-03 Pop-And-Retest

After the rolling source cache instability check, the dirty experiment stash was
popped back into the worktree and rebuilt:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

The retest used the current 10000-step force-only case:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

Rolling source cache was kept disabled:

```sh
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0
```

Results:

| configuration | runs | stability | average speed | speed range |
|---|---:|---:|---:|---:|
| clean HEAD, rolling source cache off | 5 | 5 finite, 0 NaN | `107.217345 ns/day` | `105.671165-109.398247 ns/day` |
| dirty experiment code popped, new gates default off | 3 | 3 finite, 0 NaN | `106.095558 ns/day` | `104.869461-107.298065 ns/day` |
| dirty experiment code popped, new gates on | 2 | 2 finite, 0 NaN | `102.969200 ns/day` | `102.713722-103.224678 ns/day` |

The gate-on row enabled:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SASS_OPT=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_COLLECT_SASS_OPT=1
SPONGE_CLUSTERED_GMXPACKED_PAIR_SHIFT_SIMPLE_REFRESH=1
SPONGE_CLUSTERED_GMXPACKED_SORTED_CLUSTER_MAP=1
SPONGE_CLUSTERED_GMXPACKED_SCI_SHIFT_SPLIT_SKIP_EMPTY=1
```

Decision confirmed: keep these gates default-off and out of the peak env. They
are stable with rolling source cache disabled in this short matrix, but slower
than both clean rolling-off and dirty default-off runs.
