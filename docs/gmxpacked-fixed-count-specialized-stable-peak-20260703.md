# Gmxpacked Fixed-Shift Count Specialization Stable-Peak Retest 2026-07-03

## Summary

This round fixed the fixed-shift/light count specialization dispatch so it can
run on the current stable peak path with rolling source cache disabled.

The old dispatch required:

```sh
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=1
```

That blocked the specialized count kernel on the current stable peak env, where
rolling source cache is intentionally disabled because repeated 10000-step
runs showed intermittent NaN.

## Code Changes

- `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED=1` now selects the
  fixed-shift/light count specialization without requiring rolling source cache.
- The gate is still limited to the current fixed-shift path:
  gmxpacked direct, outer lifecycle, active view, dense shift-partitioned
  fixed candidates, leaf screening, onepass candidate leaves, subgroup builder,
  parallel accum, parallel fragment emit, fill-prune-reuse-light, and non-null
  reach masks.
- The count path now emits a summary trace line when existing record-builder
  tracing is enabled:

```text
[clustered gmxpacked count variant] ... variant=subgroup-fixed-light-specialized ...
```

No candidate leaf list, source fragment ordering, fallback, or fill contract was
changed.

## Correctness

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

Result: passed.

Smoke:

```text
/tmp/sponge-fixed-count-specialized-20260703/smoke20.out
/tmp/sponge-fixed-count-specialized-20260703/smoke20.err
```

Result: exit 0, finite 20-step run. The trace confirmed the intended dispatch:

```text
variant=subgroup-fixed-light-specialized fixed_specialized=1 rolling_source_cache=0
```

2000-step verify:

```text
/tmp/sponge-fixed-count-specialized-20260703/verify2000.out
/tmp/sponge-fixed-count-specialized-20260703/verify2000.err
```

All verify launches reported zero mismatches:

```text
flag_mismatch=0 cj_mismatch=0 excl_mismatch=0
row_count_mismatch=0 field_mismatch=0
```

## Performance

Input:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

Stable peak env, rolling source cache disabled.

| configuration | runs | stability | average speed | speed range | average `Calculate_Force` |
|---|---:|---:|---:|---:|---:|
| specialized on | 3 | 3 finite, 0 NaN | `111.150754 ns/day` | `110.137749-111.908737 ns/day` | `7.107891 s` |
| specialized off | 1 | 1 finite, 0 NaN | `106.373154 ns/day` | n/a | `7.448277 s` |

The best observed stable speed is now:

```text
111.908737 ns/day
```

Nsys:

```text
/tmp/sponge-fixed-count-specialized-20260703/nsys_fixed_count_specialized_10000.nsys-rep
/tmp/sponge-fixed-count-specialized-20260703/nsys_fixed_count_specialized_10000_stats_cuda_gpu_kern_sum.csv
```

Nsys-wrapped speed:

```text
107.700607 ns/day
```

Key kernels:

| kernel group | total time |
|---|---:|
| force true variant | `2398.462 ms x9999` |
| force false variant | `371.537 ms x9999` |
| `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true,true,true,true,false>` | `1290.140 ms x24` |
| `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass` | `680.792 ms x24` |
| `Refresh_Gmxpacked_Pair_Shift_Bits` | `446.068 ms x10025` |
| `Fill_Gmxpacked_Record_Stream_Inner_Active_Sources` | `118.986 ms x24` |
| `Count_Gmxpacked_Record_Stream_Inner_Active_Sources` | `107.261 ms x24` |
| `Materialize_Gmxpacked_Record_Stream_Sources_From_Light_Count_Fragments` | `33.239 ms x24` |
| `Mark_Gmxpacked_Incremental_Dirty_J_Candidates_Parallel` | `46.147 ms x812` |

NCU same-input comparison used the first count launch from the 20-step input:

```text
/tmp/sponge-fixed-count-specialized-20260703/ncu_count_specialized.log
/tmp/sponge-fixed-count-specialized-20260703/ncu_count_specialized_off.log
```

| metric | specialized off `<true,true,false,false,false>` | specialized on `<true,true,true,true,false>` |
|---|---:|---:|
| duration | `62.878 ms` | `55.003 ms` |
| registers/thread | `150` | `71` |
| allocated registers/thread | `152` | `72` |
| theoretical occupancy | `25.00%` | `58.33%` |
| achieved occupancy | `11.30%` | `18.97%` |
| eligible warps/scheduler | `0.13` | `0.12` |
| issue active | `12.52%` | `11.37%` |
| SM throughput | `8.55%` | `9.34%` |
| memory throughput | `12.73%` | `13.39%` |
| DRAM throughput | `0.90%` | `0.97%` |
| active threads/warp | `4.30` | `4.06` |
| not-predicated threads/warp | `3.93` | `3.69` |

The win comes primarily from removing the register-heavy fallback variant from
the stable peak path. Scheduler eligibility is still low, so the next large
optimization should target structural work reduction rather than another local
SASS cleanup.

## Decision

Keep `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED=1` in the
stable peak env. Do not enable the previous SASS opt gates by default; the
documented 2026-07-03 retest still shows them slower.

Rolling source cache remains disabled in the peak env until its intermittent
NaN is fixed separately.
