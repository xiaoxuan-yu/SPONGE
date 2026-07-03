# Fixed-Shift Count Metadata Experiment - 2026-07-03

## Summary

This round implemented a gated metadata-assisted count path for the current
fixed-shift one-pass candidate leaf pipeline:

```sh
SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_COUNT_METADATA=1
```

The gate is default-off. It only feeds the current fixed-shift/light specialized
count path and keeps candidate leaf order, source fragment order, fill semantics,
and overflow fallback unchanged.

## Implementation

- One-pass candidate leaf collect now can carry one extra `int` per accepted
  candidate leaf: `prev_running_max_end`.
- The one-pass scatter copies that metadata into final candidate-leaf order.
- The specialized count variant
  `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true,true,true,true,false,true>`
  uses the metadata to replace its bounded backward dedup scan.
- Existing count variants receive `NULL` metadata and keep the old scan.
- One-pass scratch capacity remains byte-capped: metadata-on uses four scratch
  `int` streams instead of the default three.

This is intentionally not the removed three-int metadata design. It does not
store `cluster_j_start`, `cluster_j_end`, or `deduped_cluster_j_start`.

## Correctness

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

Result: passed.

20-step smoke:

```text
/tmp/sponge-count-metadata-20260703/smoke20.out
/tmp/sponge-count-metadata-20260703/smoke20.err
```

Result: finite. Trace confirmed the metadata count variant:

```text
variant=subgroup-fixed-light-metadata count_metadata=1 count_metadata_bytes=56571836
```

The reported metadata allocation is about `53.95 MiB`, including final
candidate-leaf metadata plus the extra one-pass scratch stream.

2000-step verify:

```text
/tmp/sponge-count-metadata-20260703/verify2000.out
/tmp/sponge-count-metadata-20260703/verify2000.err
```

All verify records were zero-mismatch:

```text
flag_mismatch=0 cj_mismatch=0 excl_mismatch=0
row_count_mismatch=0 field_mismatch=0
```

## Performance

Input:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

Peak env plus `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_COUNT_METADATA=1`.

One normal 10000-step run:

| case | speed | `Calculate_Force` | final temperature |
|---|---:|---:|---:|
| metadata on | `110.067223 ns/day` | `7.151694 s` | `293.93 K` |

Nsys:

```text
/tmp/sponge-count-metadata-20260703/nsys_metadata_on_10000.nsys-rep
/tmp/sponge-count-metadata-20260703/nsys_metadata_on_10000_stats_cuda_gpu_kern_sum.csv
```

Nsys-wrapped speed:

```text
105.334564 ns/day
```

Key kernel comparison against the fixed-count specialization baseline from
`docs/gmxpacked-fixed-count-specialized-stable-peak-20260703.md`:

| kernel group | baseline | metadata on | result |
|---|---:|---:|---|
| specialized payload count | `1290.140 ms x24` | `1298.789 ms x24` | worse by `8.649 ms` |
| one-pass collect | `680.792 ms x24` | `705.301 ms x24` | worse by `24.509 ms` |
| collect + count | `1970.932 ms` | `2004.089 ms` | worse by `33.157 ms` |
| one-pass scatter | not in baseline table | `7.787 ms x24` | added metadata scatter work |

NCU metadata count single launch:

```text
/tmp/sponge-count-metadata-20260703/ncu_count_metadata.log
```

| metric | metadata count |
|---|---:|
| duration | `60.214 ms` |
| registers/thread | `71` |
| theoretical occupancy | `58.33%` |
| achieved occupancy | `19.09%` |
| eligible warps/scheduler | `0.12` |
| issued warp/scheduler | `0.11` |
| SM throughput | `9.31%` |
| memory throughput | `13.43%` |
| DRAM throughput | `0.98%` |

## Decision

Do not add `SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_COUNT_METADATA=1` to the
peak env. The implementation is correct and stable in the tested verify path,
but it misses the acceptance gate: `Collect + Count` regressed by about `1.7%`
instead of improving by at least `5%`.

The result indicates that the bounded backward scan is not the dominant
remaining cost in the specialized count path, or that replacing it with one
extra metadata load is too small to repay the extra collect/scatter writes.
Next builder-side work should not be another per-candidate-leaf metadata field
unless it also removes a larger downstream loop or eliminates a launch.
