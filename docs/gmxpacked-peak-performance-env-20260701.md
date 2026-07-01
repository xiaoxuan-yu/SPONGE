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

## Peak env

This is the current peak-performance env for the force-only case:

```sh
SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1
SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1
SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=1
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

Do not add this flag to the peak env:

```sh
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_SOURCE_CACHE_PATCH=1
```

In the 2026-07-01 matrix it did not improve the peak path.

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
