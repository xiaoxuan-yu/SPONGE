# fixed-shift candidate leaf one-pass experiment - 2026-07-01

## Goal

Eliminate duplicate tree traversal between:

- `Count_Supercluster_Candidate_Leaves_Fixed_Shift`
- `Fill_Supercluster_Candidate_Leaves_Fixed_Shift`

and their subgroup-parallel variants.

This is not candidate-leaf deduplication. The candidate leaf list semantics,
ordering, and downstream `processed_cluster_end` cluster-span dedup remain
unchanged.

## Gate

Default off:

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

## Capacity and fallback

The first implementation intentionally avoids allocating a worst-case
`candidate_sci_numbers * leaf_numbers` scratch buffer. It uses the previous
`candidate_leaf_capacity` as the temporary record capacity.

- First build, or no existing capacity: old count + scan + fill path.
- One-pass scratch overflow: old count + scan + fill path.
- Successful one-pass: skips the fill traversal and only runs scan + scatter.

This keeps memory bounded and correctness conservative while allowing repeated
rebuild benchmarks to measure whether eliminating the duplicate traversal is
worth a more aggressive capacity estimator.

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

## Required follow-up

Run with the existing stable performance flag set plus:

```sh
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
```

Correctness:

- 2000-step finite run.
- 2000-step
  `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1`.
- Confirm count/fill mismatches remain zero.

Performance:

- 10000-step nsys off/on comparison.
- Check whether `Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup`
  disappears on successful one-pass rebuilds.
- Compare replacement cost:
  `Collect_Supercluster_Candidate_Leaves_Fixed_Shift*_Onepass` +
  `Scatter_Candidate_Leaves_From_Onepass` versus old count + fill.
