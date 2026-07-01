# gmxpacked count fragment parallel accumulator report - 2026-07-01

## Scope

This records the env-gated optimization that extends the existing
`SPONGE_CLUSTERED_GMXPACKED_COUNT_PARALLEL_ACCUM=1` path to the current
`fill-prune-reuse-light` source-fragment count kernel.

The optimized path is still default-off. It is active only when all of these are
true:

- `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1`
- `SPONGE_CLUSTERED_SHIFT_PARTITIONED_BUILDER=1`
- `SPONGE_CLUSTERED_FIXED_SHIFT_LEAF_SCREENING=1`
- `SPONGE_CLUSTERED_GMXPACKED_FILL_PRUNE_REUSE=1`
- `SPONGE_CLUSTERED_GMXPACKED_FILL_PRUNE_REUSE_LIGHT=1`
- `SPONGE_CLUSTERED_GMXPACKED_COUNT_PARALLEL_ACCUM=1`

## Implementation

File: `SPONGE/Lennard_Jones_force/clustered_lj.cpp`

The count-fragment path previously forced
`Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<false>` while emitting
source fragments, because the original parallel accumulator only handled the
non-fragment source-row count path.

This change enables `<true>` for the fragment path and keeps the source fragment
contract stable:

- subgroup lanes compute split-local source masks in parallel;
- active i sublanes compute their own exclusion masks in parallel;
- subgroup shared scratch stores split masks and exclusion masks;
- the leader lane still emits source fragments in the existing split order;
- fragment row count, row fields, source order, and overflow semantics are
  unchanged.

The first attempt used masked shuffle from a leader-only branch and hung. The
final version uses subgroup shared scratch plus `deviceSyncWarp(subgroup_mask)`
before leader consumption.

## Validation Commands

Common env:

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
```

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

NCU:

```sh
.pixi/envs/dev-cuda13/bin/ncu \
  --force-overwrite --target-processes all \
  --kernel-name regex:Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup \
  --launch-count 1 \
  --section SpeedOfLight --section SchedulerStats --section Occupancy \
  --section LaunchStats --section MemoryWorkloadAnalysis --section SourceCounters \
  /usr/bin/env ${COMMON_ENV} build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_2000_notimer.spg.toml
```

## Correctness

- 2000-step finite run:
  - `/tmp/sponge-count-frag-parallel-2000.out`
  - final `T=293.49 K`
  - `76.858192 ns/day`
- 2000-step verify:
  - `/tmp/sponge-count-frag-parallel-verify-2000.err`
  - count `flag_mismatch=0 cj_mismatch=0 excl_mismatch=0`
  - fill `row_count_mismatch=0 field_mismatch=0`
- flag-off regression smoke:
  - `/tmp/sponge-count-frag-parallel-off-2000.out`
  - final `T=294.79 K`
  - `59.478546 ns/day`

## Performance

NCU single launch:

| metric | fragment baseline `<false>` | optimized `<true>` |
|---|---:|---:|
| Duration | `197.31 ms` | `90.06 ms` |
| Compute throughput | `6.44%` | `8.35%` |
| Issued warp / scheduler | `0.09` | `0.12` |
| Eligible warps / scheduler | `0.09` | `0.13` |
| Registers / thread | `145` | `150` |
| Theoretical occupancy | `25%` | `25%` |
| Achieved occupancy | `11.58%` | `11.52%` |
| Local spilling requests | `0` | `0` |
| Excessive global sectors | `407.8M / 32%` | `632.7M / 53%` |

10000-step clean:

- `/tmp/sponge-count-frag-parallel-10000.out`
- final `T=293.66 K`
- `80.174301 ns/day`

10000-step nsys:

- report: `/tmp/sponge_trace_count_frag_parallel_10000.nsys-rep`
- kernel summary:
  `/tmp/sponge_trace_count_frag_parallel_10000_kern_sum_cuda_gpu_kern_sum.csv`
- final `T=294.77 K`
- `76.336830 ns/day`
- count kernel:
  `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true>`
  `1.565 s / 20 launches`, average `78.26 ms`

Previous fill-prune-reuse-light nsys anchor:

- `/tmp/sponge_trace_merged_fillprune_10000_kern_sum_cuda_gpu_kern_sum.csv`
- count kernel:
  `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<false>`
  `3.271 s / 19 launches`, average `172.14 ms`
- nsys speed: `67.701286 ns/day`

## Notes

The optimized path materially improves duration and scheduler eligibility, while
register pressure remains within the prior acceptance bound and local spilling
stays at zero. The uncoalesced global sector ratio worsens in NCU, so the next
profiling pass should check whether this is an attribution side effect of doing
more work in parallel or a real memory-layout regression worth addressing.

The path remains env-gated. It should not be made default until repeated
10000-step runs and a broader wall-time table confirm that the speedup is stable
outside this benchmark case.
