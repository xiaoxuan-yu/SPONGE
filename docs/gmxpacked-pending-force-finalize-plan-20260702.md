# gmxpacked pending force finalize plan - 2026-07-02

This note records the design review for moving sorted-force scatter behind a
general pending-force finalize contract. It is a plan only; no implementation is
claimed here.

## Goal

The microbench showed that writing clustered LJ force into sorted scratch can be
faster than writing directly into the production force array, but the production
path currently pays a separate scatter/clear kernel:

- `Scatter_And_Clear_Sorted_Clustered_Force`
- `Scatter_And_Clear_Sorted_Clustered_Force_Float4`

The next low-risk architecture step is to make this scatter a generic
`Finalize_Pending_Force` operation instead of an unconditional tail of the LJ
direct function. This contract must apply to all force paths, not only
force-only.

## Current Code Boundary

Current immediate scatter is in
`SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`, near the sorted scratch
tail:

- sorted SoA scratch scatters through `Scatter_Sorted_Clustered_Force_SoA`;
- sorted `VECTOR` scratch scatters through `Scatter_And_Clear_Sorted_Clustered_Force`;
- sorted `float4` scratch scatters through
  `Scatter_And_Clear_Sorted_Clustered_Force_Float4`;
- clean-state flags are stored in `LJ_CLUSTERED_DIRECT_CACHE`.

The cache state lives in `SPONGE/Lennard_Jones_force/clustered_lj.h`:

- `d_sorted_atom_ids`
- `d_sorted_frc`
- `d_sorted_frc4`
- `d_sorted_frc_x/y/z`
- `gmxpacked_sorted_force_clean`
- `gmxpacked_sorted_force_clean_float4`
- `gmxpacked_sorted_force_clean_capacity`

`Main_Calculate_Force()` in `SPONGE/main.cpp` shows why this cannot be a
force-only local change: LJ direct is only one producer in a larger force
accumulation stage. After LJ, the code can still run solvent LJ, long-range
corrections, many-body/listed/bonded forces, plugins, PME reciprocal, CV forces,
SITS, virtual-atom redistribution, domain force exchange, and minimization
force scaling.

## Correct Contract

The contract is:

> A module may leave force contribution in a private pending buffer, but every
> consumer of total force must call the common finalizer before reading,
> copying, redistributing, exchanging, or integrating the force array.

This is a pending contribution flush. It is not a replacement of `dd.frc`.

The finalizer must be idempotent:

- if no pending contribution exists, it is a no-op;
- if a pending sorted-force contribution exists, it scatters exactly once,
  clears scratch if required, updates clean-state flags, and clears the pending
  state.

## Proposed Gates

Keep the first implementation default-off:

```sh
SPONGE_CLUSTERED_GMXPACKED_DEFER_FORCE_FINALIZE=1
```

Optional diagnostic gate:

```sh
SPONGE_CLUSTERED_FORCE_FINALIZE_TRACE=1
```

Do not add these to the peak env until a 10000-step nsys run shows a real
end-to-end win and all correctness gates pass.

## Proposed Data State

Add pending-state fields to `LJ_CLUSTERED_DIRECT_CACHE`:

```cpp
bool gmxpacked_sorted_force_pending = false;
bool gmxpacked_sorted_force_pending_float4 = false;
bool gmxpacked_sorted_force_pending_fused_clear = false;
int gmxpacked_sorted_force_pending_count = 0;
```

The first implementation should support the currently relevant sorted scratch
paths:

- `VECTOR` sorted scratch;
- `float4` sorted scratch;
- SoA sorted scratch only if the current dispatch can leave it pending. If not,
  keep SoA on the immediate-scatter path until it has a measured reason to join
  the pending contract.

## Proposed API

Expose the finalizer through the LJ public API so `main.cpp` does not reach into
`LJ_CLUSTERED_DIRECT_CACHE` internals:

```cpp
void LENNARD_JONES_INFORMATION::Finalize_Pending_Force(VECTOR* frc);
```

Then add a small main-level barrier:

```cpp
static void Main_Finalize_Pending_Forces()
{
    lj.Finalize_Pending_Force(dd.frc);
    lj_soft.Finalize_Pending_Force(dd.frc);
}
```

The soft-core call can be no-op initially if that path never marks pending
force, but the API should exist so all paths use the same barrier.

## Required Consumer Barriers

Insert `Main_Finalize_Pending_Forces()` before any path that can observe total
force:

- before `plugin.Calculate_Force()`, because plugin code can request the raw
  force pointer through `get_force_ptr()`;
- before `pm.Send_Recv_Force(...)`, because this exchanges local force data;
- before `selective_interaction.Update_And_Enhance(...)`, because SITS can copy
  `frc` into `select_force[0]` and then enhance `frc`;
- before `vatom.Force_Redistribute_CV(...)` and
  `vatom.Force_Redistribute(...)`, because these redistribute force;
- before `md_info.min.Scale_Force_For_Dynamic_Dt(...)`, because minimization
  scaling reads and writes force;
- at the end of `Main_Calculate_Force()` as a defensive no-op/fallback.

Producer-only force kernels between LJ and the first consumer can continue to
write directly to `dd.frc`; finalization just adds the pending LJ contribution
before any total-force consumer observes it.

## Why This Is The Low-Risk First Step

This first step preserves force-array semantics. It does not change atom order,
energy/virial output, source fragments, candidate lists, or integration math.
It only moves the existing scatter/clear operation behind a common flush point.

Expected performance impact by itself is likely neutral. It still launches a
scatter kernel. The value is architectural: after all force paths use the
finalize contract, later work can fuse finalization into real consumers.

## Follow-Up Fusion Options

Once the contract is validated, the measured fusion candidates are:

1. `Finalize_Into_LeapFrog` for pure NVE force-only cases.
2. `Finalize_Into_SITS_Copy_And_Enhance` for SITS paths that otherwise copy the
   full force array.
3. `Finalize_Into_SendRecvForce` for multi-rank force exchange.
4. `Finalize_Into_VirtualAtomRedistribute` if vatom redistribution is active.

These are separate optimizations. The first implementation should not attempt
all of them.

## Validation Matrix

Correctness:

- build `SPONGE`;
- 2000-step force-only finite run;
- 2000-step subgroup-builder verify with all count/fill mismatch lines at zero;
- 10000-step force-only finite run;
- repeat with energy, virial, and energy+virial enabled if those paths are
  routed through sorted-force scratch;
- SITS/plugin paths should be smoke-tested if the finalizer is placed before
  their consumer boundaries.

Performance:

- compare `SPONGE_CLUSTERED_GMXPACKED_DEFER_FORCE_FINALIZE=0/1` on the same
  10000-step peak env;
- inspect nsys for:
  - LJ force kernels;
  - sorted-force scatter/finalize kernels;
  - force exchange, SITS, vatom, and integrator consumers;
  - CUDA API synchronization/memcpy growth.

Acceptance:

- no NaN or mismatch regression;
- no extra scatter launches per step;
- no increase in total non-force GPU work above noise;
- keep default-off if the first contract-only version has no end-to-end win.

## Current Decision

Proceed with the finalize contract as the next code step only if the goal is to
enable later consumer fusion. Do not present the contract-only version as a
performance optimization until nsys shows that it either removes a launch or
enables a measured fused consumer.

## 2026-07-02 Nsys Hotspot Scan

After recording the finalize plan, a fresh 10000-step nsys run was taken on the
current checkout and current peak env.

Case:

```text
/tmp/sponge-onepass-capacity-check/mdin_onepass_10000.spg.toml
```

Important command detail: this binary requires `-mdin`. A first attempt without
`-mdin` ran the default task and failed during input loading with
`no atom_numbers found`; ignore that tiny `.nsys-rep`.

Valid artifacts:

```text
/tmp/sponge-finalize-hotspot-20260702/peak_10000.out
/tmp/sponge-finalize-hotspot-20260702/peak_10000.err
/tmp/sponge-finalize-hotspot-20260702/peak_10000.nsys-rep
/tmp/sponge-finalize-hotspot-20260702/peak_10000_kern_cuda_gpu_kern_sum.csv
```

Run result:

| metric | value |
|---|---:|
| `Core Run Speed` | `98.697578 ns/day` |
| `Core Run Wall Time` | `8.754890 s` |
| `Calculate_Force` | `7.998371 s` |
| total GPU kernel time | `7633.235 ms` |
| final temperature | `293.60 K` |
| stderr | empty |

Clean non-nsys check with the same env:

```text
/tmp/sponge-finalize-hotspot-20260702/peak_clean_10000.out
/tmp/sponge-finalize-hotspot-20260702/peak_clean_10000.err
```

| metric | value |
|---|---:|
| `Core Run Speed` | `104.880562 ns/day` |
| `Core Run Wall Time` | `8.238766 s` |
| `Calculate_Force` | `7.544070 s` |
| `Iteration` | `0.533657 s` |
| final temperature | `294.41 K` |
| stderr | empty |

This clean run is faster than the recorded July 1 clean repeat
`103.863182 ns/day` and much faster than the previous nsys-wrapped anchor
`95.765167 ns/day`. It does not reproduce a `107 ns/day` clean peak. If such a
run exists, keep it as a separate artifact with its exact env and output path.
A second clean repeat started after this run was interrupted because it exceeded
two minutes while still occupying the GPU; treat it as a polluted/invalid system
load sample, not as a performance data point.

Top kernel groups:

| kernel/group | total |
|---|---:|
| force kernels, all `ForceOnly_Warp_Record_Device` variants | `2927.770 ms` |
| main force true variant | `2407.011 ms x9999` |
| main force false/short variant | `520.114 ms x9999` |
| `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true,true,true,true>` | `1360.783 ms x20` |
| `Collect_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup_Onepass` | `868.460 ms x20` |
| `Refresh_Gmxpacked_Pair_Shift_Bits` | `449.046 ms x10021` |
| `MPI_PME_Excluded_Force_With_Atom_Energy_Correction` | `238.517 ms x10001` |
| `settle_triangle` | `223.215 ms x10001` |
| `Fill_Gmxpacked_Record_Stream_Inner_Active_Sources` | `162.977 ms x20` |
| `Count_Gmxpacked_Record_Stream_Inner_Active_Sources` | `156.506 ms x20` |
| `Gather_Sorted_LJ_Direct_Scratch_From_Plain` | `99.530 ms x10001` |
| `Refresh_Current_Cluster_Centers_From_Crd` | `74.712 ms x10001` |
| `Mark_Gmxpacked_Incremental_Dirty_J_Candidates_Parallel` | `57.581 ms x814` |
| `MD_Iteration_Leap_Frog` | `43.044 ms x10001` |
| sorted-force scatter | `0.026 ms x2` |

Interpretation:

- The pending-force finalize contract is still useful as an architecture step,
  but scatter itself is not a current hotspot in this peak run.
- The remaining large time is split between force kernels and full-rebuild
  builder work. Builder-related kernels sum to about `3343.923 ms`, dominated by
  count and one-pass collect.
- Dirty-J marking is already small in this peak env (`57.581 ms`) and should not
  be the next target unless a future env removes zero-dirty source reuse.
- `Refresh_Gmxpacked_Pair_Shift_Bits`, coordinate-center refresh, and sorted LJ
  scratch gather together cost about `623 ms` over 10000 steps. This is much
  smaller than count/collect, but lower risk because it does not alter candidate
  list ordering or source-fragment semantics.

## Low-Risk Next Targets

1. **Auxiliary coordinate refresh/gather consolidation.**
   `Refresh_Current_Cluster_Centers_From_Crd` and
   `Gather_Sorted_LJ_Direct_Scratch_From_Plain` both read coordinates every step.
   A gated experiment can combine sorted `xq` generation with cluster-center
   handling or avoid the per-atom binary search over `cluster_offsets` by using a
   precomputed sorted-atom-to-cluster map. Target saving is bounded by about
   `174 ms / 10000 steps`.

2. **Pair-shift refresh specialization.**
   `Refresh_Gmxpacked_Pair_Shift_Bits` is a steady `449 ms / 10000 steps`.
   A low-risk version should only specialize the fixed-shift peak path and keep
   exact fallback. Do not reuse the earlier skip-empty safe-count experiment as
   the default; that path introduced extra synchronization/memcpy risk. The
   safer first step is a fixed-path kernel variant with safe-flag logic stripped
   when no caller needs those flags.

3. **Per-step non-clustered small kernels only after clustered auxiliaries.**
   `MPI_PME_Excluded_Force_With_Atom_Energy_Correction`, `settle_triangle`, and
   `Bond_Force_With_Atom_Energy_And_Virial_Device` are each stable per-step
   costs, but they are broader SPONGE contracts. They are not the first low-risk
   gmxpacked target.

4. **Count/collect remains the high-impact path, not the low-risk path.**
   Count plus one-pass collect still costs about `2229 ms / 10000 steps`. Further
   work here can pay off, but it touches candidate traversal/count/fill
   semantics and should be treated as a separate higher-risk plan.

## Low-Risk Targets 1 And 2 Implementation Check

Two default-off gates were added for the first two low-risk ideas:

```text
SPONGE_CLUSTERED_GMXPACKED_SORTED_CLUSTER_MAP=1
SPONGE_CLUSTERED_GMXPACKED_PAIR_SHIFT_SIMPLE_REFRESH=1
```

The sorted-cluster-map gate adds a sorted-atom to cluster-id scratch map while
refreshing cluster centers, then uses that map in the plain-coordinate gather to
avoid the old per-atom binary search over `cluster_offsets`. The scratch map is
allocated only when the gate is enabled, so the default path does not reserve
extra memory for this experiment.

The pair-shift-simple gate adds a specialized refresh kernel for the non-exact
SCI flag path. It keeps the existing pair-shift bit output, SCI safe flags, and
safe-SCI count contract, but strips the exclusion/valid/local-mask exact checks
that are not needed for the non-exact path. The legacy refresh kernel remains
the fallback and is still used unless the gate is enabled.

Validation:

| check | result |
|---|---:|
| build, `pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4` | pass |
| 20-step smoke, both gates on | pass, stderr empty |
| 10000-step clean, both gates on | `100.176956 ns/day` |
| 10000-step clean, sorted-map only | `100.337715 ns/day` |
| 10000-step clean, pair-shift-simple only | `101.102509 ns/day` |
| reference clean peak before this change | `104.880562 ns/day` |

Nsys comparison against the same pre-change peak artifact:

| case | nsys speed | GPU kernel total | target effect |
|---|---:|---:|---|
| reference peak | `98.697578 ns/day` | `7633.235 ms` | baseline |
| sorted-map only | `97.085510 ns/day` | `7699.101 ms` | center+gather target regressed from `174.242 ms` to `189.193 ms` |
| pair-shift-simple only | `97.238701 ns/day` | `7665.668 ms` | pair-shift refresh improved from `449.046 ms` to `403.476 ms`, but total regressed |

Interpretation:

- Sorted-cluster-map is not worth enabling. The extra map writes in the center
  refresh cost more than the gather binary-search removal saves.
- Pair-shift-simple improves its target kernel by about `45.6 ms / 10000 steps`,
  but the full nsys and clean runs still regress. Do not default-enable it.
  If revisited, first compare safe/unsafe SCI counts and the downstream
  force-kernel split distribution, because the target-kernel win is not
  translating into end-to-end speed.
- Both experiments should remain default-off. They are useful as diagnostic
  variants, not as the current peak path.

## Pair-Shift-Simple Attribution

Follow-up attribution checked whether the pair-shift-simple regression came
from changed safe/unsafe SCI flag semantics.

Artifacts:

```text
/tmp/sponge-finalize-hotspot-20260702/shift_analyze_baseline_20.out
/tmp/sponge-finalize-hotspot-20260702/shift_analyze_baseline_20.err
/tmp/sponge-finalize-hotspot-20260702/shift_analyze_pair_simple_20.out
/tmp/sponge-finalize-hotspot-20260702/shift_analyze_pair_simple_20.err
/tmp/sponge-finalize-hotspot-20260702/current_baseline_clean_10000_r1.out
/tmp/sponge-finalize-hotspot-20260702/current_pair_simple_clean_10000_r1.out
```

The 20-step `SPONGE_CLUSTERED_GMXPACKED_SHIFT_ANALYZE=1` comparison showed
identical SCI shift classification:

| metric | baseline | pair-shift-simple |
|---|---:|---:|
| `sci` | `3896` | `3896` |
| `active_sci` | `3896` | `3896` |
| `safe_sci` | `3896` | `3896` |
| `unsafe_sci` | `0` | `0` |
| `flagged_safe_sci` | `3896` | `3896` |
| `flagged_unsafe_sci` | `0` | `0` |
| `active_slots` | `72549664` | `72549664` |
| `mismatch_sci_slots` | `0` | `0` |
| `central_slot_ratio` | `0.884445` | `0.884445` |
| `shifted_slot_ratio` | `0.115555` | `0.115555` |

This rules out the main suspected failure mode: the simple refresh does not
change the safe-SCI flags or the force split distribution on this case.

A fresh clean 10000-step repeat on the current binary still shows the gate as a
real end-to-end regression:

| case | speed | wall | `Calculate_Force` | stderr |
|---|---:|---:|---:|---|
| peak env, no new gate | `104.915108 ns/day` | `8.236053 s` | `7.562046 s` | empty |
| + `SPONGE_CLUSTERED_GMXPACKED_PAIR_SHIFT_SIMPLE_REFRESH=1` | `102.597862 ns/day` | `8.422070 s` | `7.743130 s` | empty |

Conclusion: do not spend more implementation effort on pair-shift-simple now.
It improves the isolated refresh kernel but does not produce a stable
end-to-end win, and the regression is not explained by SCI safe-flag semantics.
The next low-risk performance work should move back to the force kernel
production variant or to broader per-step small kernels rather than this gate.
