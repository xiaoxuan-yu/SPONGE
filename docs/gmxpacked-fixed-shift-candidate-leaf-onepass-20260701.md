# fixed-shift candidate leaf one-pass experiment - 2026-07-01

## Goal

Eliminate duplicate tree traversal between:

- `Count_Supercluster_Candidate_Leaves_Fixed_Shift`
- `Fill_Supercluster_Candidate_Leaves_Fixed_Shift`

and their subgroup-parallel variants.

This is not candidate-leaf deduplication. The candidate leaf list semantics,
ordering, and downstream `processed_cluster_end` cluster-span dedup remain
unchanged.

## Current decision

Use one-pass candidate leaf collection as the default path for the current
accepted peak-performance configuration.

This means the run harness / benchmark env should include:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
```

The code-level switch remains env-gated so broad fallback paths stay
unchanged. The accepted default here is the project performance default for the
fixed-shift + leaf-screened + gmxpacked fill-prune-reuse-light path.

## Gate

Code gate:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
```

The experiment only attempts the one-pass path when fixed-shift candidate leaves
and fixed-shift leaf screening are active. It can be combined with:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
```

to use the subgroup leaf-overlap test inside the single traversal.

## Implementation

File: `SPONGE/Lennard_Jones_force/clustered_lj.cpp`

The one-pass path performs one tree traversal per candidate SCI and writes:

- `d_sci_candidate_leaf_counts[sci]`
- temporary `onepass_sci_ids`
- temporary `onepass_ranks`
- temporary `onepass_leaf_ids`

After the traversal:

1. `d_sci_candidate_leaf_counts` is exclusive-scanned into
   `d_sci_candidate_leaf_offsets`.
2. `Scatter_Candidate_Leaves_From_Onepass` writes the final contiguous
   `d_sci_candidate_leaf_ids[offset[sci] + rank] = leaf`.

The per-SCI rank is assigned in traversal order, so the final candidate leaf list
preserves the original fill order.

## Current accepted peak path

Common env for the current best 160k force-only path:

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
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
```

Case:

```text
/tmp/sponge-onepass-capacity-check/mdin_dedupon_current_10000.spg.toml
```

The current best path is the one-pass-on row below. Run-to-run noise is visible,
so the decision is based on repeated same-binary comparisons, not one isolated
number.

| run | speed | core wall | `Calculate_Force` | final T |
|---|---:|---:|---:|---:|
| best observed one-pass on | `87.843224 ns/day` | `9.836689 s` | `9.175779 s` | `294.34 K` |
| same-binary clean one-pass on | `86.356201 ns/day` | `10.006073 s` | `9.351746 s` | `294.51 K` |
| latest post-cleanup one-pass on | `83.206787 ns/day` | `10.384807 s` | `9.559313 s` | `295.20 K` |
| latest post-cleanup one-pass off | `78.153923 ns/day` | `11.056213 s` | `10.244457 s` | `293.84 K` |

The latest clean post-cleanup pair gives:

- speed: one-pass on is `+6.07%` versus one-pass off;
- core wall: one-pass on is `-6.07%` versus one-pass off;
- force time: one-pass on is `-6.69%` versus one-pass off.

The nsys comparison confirms the intended kernel replacement:

| path | candidate leaf kernels | total over 10000 steps |
|---|---|---:|
| one-pass on | `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass` + `Scatter_Candidate_Leaves_From_Onepass` | `807.1 ms` |
| one-pass off | `Count_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup` + `Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup` | `1683.8 ms` |

So the candidate leaf stage itself is about `2.09x` faster with one-pass, saving
about `876.7 ms` over 10000 steps.

## Memory comparison

Process GPU memory was sampled with:

```sh
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
```

at roughly 50 ms intervals during each 10000-step run. This is process-level
resident GPU memory, not a per-buffer cudaMalloc breakdown.

| configuration | skin | peak GPU memory | speed | note |
|---|---:|---:|---:|---|
| clustered one-pass on | `10` | `3642 MiB` | `85.395683 ns/day` | accepted peak path |
| clustered one-pass off | `10` | `3634 MiB` | `78.863144 ns/day` | candidate leaf count+fill fallback |
| original SPONGE baseline | `2` | `1470 MiB` | `64.775650 ns/day` | original default cell-list skin |
| original SPONGE baseline | `10` | `1412 MiB` | `43.311314 ns/day` | diagnostic only; step-10000 temperature was NaN |

Observed one-pass memory cost in this case is only `+8 MiB` over one-pass off.
The multi-GiB memory gap is the clustered/gmxpacked layout itself, not the
candidate leaf one-pass scratch. Therefore one-pass should stay enabled in the
current peak path; disabling it would trade about 6-8% speed for only about
8 MiB of actual peak-memory reduction on this case.

## Capacity and fallback

The current implementation intentionally avoids allocating a worst-case
`candidate_sci_numbers * leaf_numbers` scratch buffer.

It uses a bounded dynamic record capacity:

- initial target: `128` temporary records per candidate SCI;
- slack from observed capacity / high-water mark;
- growth on overflow;
- hard scratch byte cap: `256 MiB` across the three temporary int arrays;
- overflow fallback: old count + scan + fill path.

Successful one-pass skips the duplicate fill traversal and only runs scan +
scatter. Overflow remains correctness-safe because it falls back to the old
path.

## Validation status

Completed:

- CUDA build:

  ```sh
  pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
  ```

- whitespace check:

  ```sh
  git diff --check
  ```

- 160k clustered forced-rebuild finite run:

  ```sh
  [neighbor_list]
  refresh_interval = 1
  ```

  with:

  ```sh
  SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
  SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
  ```

  Result: 3-step run completed, final reported `temperature = 295.40 K`.

- 160k clustered forced-rebuild verify run:

  ```sh
  SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1
  ```

  Result over steps 0-3:

  - `flag_mismatch=0`
  - `cj_mismatch=0`
  - `excl_mismatch=0`
  - `row_count_mismatch=0`
  - `field_mismatch=0`

## Short nsys check

Case: 160k clustered, 3 steps, `neighbor_list.refresh_interval = 1`, rolling
source cache disabled to force repeated full rebuilds.

Off:

- `Count_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup`:
  `4 launches, 101.057 ms`
- `Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup`:
  `4 launches, 94.797 ms`
- candidate count+fill total: `195.854 ms`

On:

- first build still falls back because no previous candidate capacity exists:
  - count: `1 launch, 24.447 ms`
  - fill: `1 launch, 25.083 ms`
- later rebuilds use one-pass:
  - `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass`:
    `3 launches, 77.836 ms`
  - `Scatter_Candidate_Leaves_From_Onepass`: `3 launches, 0.516 ms`
- candidate collection total: `127.882 ms`

The short forced-rebuild profile confirms the intended behavior: after the first
capacity-seeding build, the old count+fill duplicate tree traversal is replaced
by one collect traversal plus a cheap scatter.

## End-to-end result

The correct comparison case for the previous `~80 ns/day` result is the
force-only 160k case:

```text
/tmp/sponge-count-parallel-accum-20260630/mdin_forceonly_10000_notimer.spg.toml
```

That input has `[PM] MPI_size = 0`. A separate 160k PME benchmark only reached
about `56 ns/day` and should not be compared with the historical force-only
number.

Force-only 10000-step, current binary:

- historical count-fragment parallel result:
  - `/tmp/sponge-count-frag-parallel-10000.out`
  - `80.174301 ns/day`
  - core wall time `10.777599 s`
  - final `temperature = 293.66 K`
- one-pass off:
  - `/tmp/sponge_e2e_forceonly_onepass_off_10000.out`
  - `78.468384 ns/day`
  - core wall time `11.011906 s`
  - final `temperature = 294.72 K`
- one-pass on:
  - `/tmp/sponge_e2e_forceonly_onepass_on_10000.out`
  - `82.733131 ns/day`
  - core wall time `10.444262 s`
  - final `temperature = 295.71 K`

Relative to the current one-pass-off run:

- speed: `+5.43%`
- core wall time: `-5.15%`

Both current force-only 10000-step runs had empty stderr and no NaN, mismatch,
overflow, or error text.

## Status

Accepted for the current peak/default performance path.

Keep the env gate in code for now, but all 160k force-only peak comparisons
should include `SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1` unless
the explicit purpose is an ablation. The next optimization target is no longer
candidate leaf count/fill traversal; it is source fill / source fragment
emission and broader clustered/gmxpacked memory footprint.
