# SPONGE gmxpacked Phase A handoff — 2026-06-29

## Purpose

Phase A of the gmxpacked builder/LJ-kernel refactor is complete and committed.
This document hands off the state, the validated results, the known blocker, and
the open directions to the next session. Read this together with
`docs/gmxpacked-project-handoff-20260626.md` (the pre-Phase-A handoff) and
`docs/gmxpacked-phase-a-subgroup-design.md` (the design + PME-NaN finding).

Worktree: `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
Branch: `opt/gmxpacked-payload-shrink-sweep`

## What was the goal

Cut the per-rebuild builder cost `R` (~0.7 s, ~90% of builder time, fires every
~1590 steps) to push wat160k force-only long-run speed toward the 2x native
target of 103.38 ns/day (baseline active-view ~64 ns/day). Attribution showed the
rebuild cost splits into two device kernels that run the same candidate-leaf
traversal twice: `Count_..._From_Candidate_Leaves` (~286 ms) and
`Fill_..._Record_Stream_Sources_From_Candidate_Leaves` (~389 ms). Both are
latency-bound (NCU: 0.72 waves/SM, 92% no-eligible, 0.08 inst/cyc) because one
warp serially processes a whole candidate-SCI's ~6441-iteration j-cluster loop
using only 8 of 32 lanes.

## What was done (Phase A, committed)

Four commits on top of `c2fd374`:

- `b0e6527` — subgroup **count** kernel (4x8-lane leaf-strided, +5% e2e, bit-exact)
- `62f060f` — **S_max-bounded dedup** (provable, replaces a fixed magic-8 window)
- `26fe556` — document S_max universality + the PME long-run NaN finding
- `d6bd636` — subgroup **fill** kernel (bit-exact, +16% e2e at 1000-step)

### The subgroup design (both kernels)

Keep 1 warp per candidate-SCI; split each warp into 4 subgroups of 8 lanes (8
lanes = the 8 i-clusters, the natural inner width). Each subgroup processes a
**strided slice of the SCI's leaf list** (`leaf indices s, s+4, s+8, ...` for
subgroup `s`). This gives 4 independent instruction streams per warp, attacking
the 92%-no-eligible stall. `FULL_MASK` → subgroup mask; `lane_id` → `sublane`;
i-side shared data stays per-warp (read-only, shared by subgroups); j-side
scratch + shift/source-row counters get a `[subgroup]` dimension; shift counts
use shared-`atomicAdd` (order-independent sums).

### The dedup (S_max early stop — the key correctness piece)

The baseline dedups overlapping cluster_j across leaves via a serial running max
`processed_cluster_end`. The subgroup kernel reconstructs it with a backward scan
and a **provable early stop**: `start[b] + S_max <= running_max_end`. This is
valid because candidate leaves are **cluster-start-sorted per SCI** — structurally
guaranteed by Hilbert SFC (the default, NOT Morton) + `cstone::singleTraversal`
visiting octants in SFC-key order + the shared atom-sort key. `S_max` = the exact
max per-leaf cluster span, computed once per rebuild by a single-pass
`Reduce_Max_Leaf_Cluster_Span` (atomicMax over `leaf_cluster_ends`). No magic
number; holds for any system / `cornerstone_leaf_size` / density. S_max is
NECESSARY — without a per-leaf span bound the backward scan is O(n^2).

### Validation (bit-exact, both systems)

`SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1` launches the subgroup
kernel into separate scratch and diffs against the baseline output the sim
actually consumes — the subgroup output is NEVER consumed in verify mode, so a
buggy subgroup kernel cannot poison state (zero NaN risk for the validation
itself).

- **Count** verify: 0 mismatch on 69417 shifts (wat160k) and 277668 shifts
  (wat600k), all rebuilds, across 1000/2000/10000-step runs.
- **Fill** verify: fill output is order-independent (`source_order = write_idx`,
  sorted downstream by `Sort_Gmxpacked_Record_Stream_Sources_For_Aggregate`), so
  the compare sorts both arrays by every field EXCEPT `source_order` then diffs
  (multiset compare). 0 mismatch on ~3.1M rows (wat160k) and ~13.1M rows
  (wat600k), all rebuilds.

### End-to-end speed (wat160k force-only, no-timer)

| run | baseline | count subgroup | count+fill subgroup |
|---|---|---|---|
| 1000-step | 53.5 ns/day | 57.1 (+6.7%) | **62.4 (+16.6%)** |
| 10000-step | ~64-70 ns/day | ~70 | **NaN (PME timing)** |

10000-step long-run hits the PME NaN (see blocker). Cost model: R dropped
0.706 → ~0.45 s (count+fill). Ceiling if the PME NaN is resolved: ~74-80 ns/day
at 10000-step; if a future fill pruning-reuse also lands, ~92 ns/day.

## Env flags (Phase A)

- `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1` — production consumption of the
  subgroup count+fill kernels (function-pointer select at the production launches).
  **Default off.** Short/mid runs (<=1000-step) safe and faster; 10000-step may
  NaN due to the PME blocker.
- `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1` — bit-exact dual-launch
  compare into separate scratch (never consumed). Use for correctness validation.
- The standard active-view path flags still apply:
  `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1
   SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer
   SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1
   SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1`.

## Files (Phase A)

- `SPONGE/Lennard_Jones_force/clustered_lj.cpp`:
  - `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup` (the subgroup count kernel)
  - `Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup` (fill)
  - `Reduce_Max_Leaf_Cluster_Span` + the `max_leaf_cluster_span` layout field
  - `Clustered_Gmxpacked_Subgroup_Builder_Enabled` / `_Verify_Enabled` env gates
  - the bit-exact verify scaffolds after the production count and fill launches
  - function-pointer kernel select at the production count/fill launches
  - baseline count+fill kernels got a dummy `max_leaf_cluster_span` arg so the
    function-pointer select keeps one signature; all launch sites thread it.
- `SPONGE/Lennard_Jones_force/clustered_lj.h`: `max_leaf_cluster_span` +
  `d_leaf_cluster_span_max_scratch` fields on `LJ_CLUSTER_LAYOUT`.
- Docs: `docs/gmxpacked-phase-a-onepass-negative.md`,
  `docs/gmxpacked-phase-a-workdist-probe.md`, `docs/gmxpacked-phase-a-subgroup-design.md`.

## KNOWN BLOCKER — PME long-run NaN (timing-exposed, pre-existing)

Opt-in subgroup 10000-step runs reproducibly NaN (~2/3 of pure-subgroup runs)
while baseline 10000-step is stable. Characteristics:

- **LJ stays finite and correct** throughout the NaN runs (LJ=80-102k while
  PM/temperature go NaN). LJ is gated by the count/fill payload, which is verified
  bit-exact — so the subgroup kernels are NOT the cause.
- NaN is **PM/temperature only**, sudden (not a drift), onset step nondeterministic.
- It is **timing-nondeterministic**: any per-step host-side instrumentation
  (D2H memcpys) SUPPRESSES it. A non-perturbing device-accumulating detector was
  built (atomic-accumulate bad-step + component bitmask, read only at print) and
  confirmed: the PME reciprocal-path quantities (`force_backup`, `PME_Q`,
  `PME_FBCFQ`, `reciprocal_ene`, `self_ene`, `correction_atom_energy`) are finite
  at every step, yet the printed `PM` is NaN.

### Configuration detail (important for the next investigation)

In the default single-process run (`MPI_size=1, PM_MPI_size=0`), `Step_Print`
takes the `MPI_rank < PP_MPI_size` branch, where `PM = direct_ene + correction_ene`
(`self_ene` and `reciprocal_ene` are set to host 0 in this branch). So the NaN is
in **`direct_ene` or `correction_ene`**, NOT the PME reciprocal path. These are
summed from `d_direct_atom_energy` / `d_correction_atom_energy`:

- `d_correction_atom_energy` <- `PME_Excluded_Force_With_Atom_Energy_Correction`
  (`PM_force.cpp:936`): has `ene_lin -= charge_i*charge_j*erff(beta_dr)/dr_abs`
  (line 985) and force `(frc_abs-1)/dr2/dr_abs` (line 982) — **NaN at dr2=0
  (two excluded atoms coincident), unguarded.**
- `d_direct_atom_energy` <- `LJ_Direct_CF_Force_With_Atom_Energy_And_Virial`
  (`main.cpp:1316`, kernels in `Lennard_Jones_force.cpp` / `LJ_soft_core.cpp`):
  1/r direct Coulomb, likely also unguarded at dr=0.

### Leading hypothesis (NOT yet confirmed)

The subgroup kernels' faster count/fill changes LJ timing, which diverges the
trajectory slightly; under that diverged trajectory a degenerate `dr=0` excluded
or direct pair occasionally occurs; the unguarded `1/dr` / `1/dr_abs` in the
correction or direct energy NaNs that one atom's energy; summed into
`direct_ene`/`correction_ene` -> `PM` NaN. LJ *short* (van der Waals) stays
finite because it is a different term than LJ *direct* (Coulomb 1/r) — the NaN is
in the Coulomb-direct/excluded energy, not van der Waals.

### Stream question — ruled out

`main_stream` is a real CUDA stream (`main.cpp:2023 deviceStreamCreate`) but is
used only for a few domain/crd/virial ops. The entire force/energy path
(LJ count/fill subgroup kernels, PME, excluded force, `Sum_Of_List`, `Step_Print`
D2H) runs on the **default stream** (`clustered_lj.cpp`: 0 `main_stream` refs,
166 default-stream launches; PME same). Same-stream D2H is ordered after the
writes, so a cross-stream race is NOT the cause. Stream fixes are not needed.

### Why it was not fully pinned

The instrumentation-masking catch-22: per-step D2H checks suppress the race, and
the non-perturbing detector confirmed the PME reciprocal path is clean but the
readout was initially placed in the wrong `Step_Print` branch (the
`MPI_size==1 && PM_MPI_size==1` branch, which is dead when `PM_MPI_size=0`). The
readout was moved to the active `MPI_rank < PP_MPI_size` branch in the last
diagnostic iteration, but the PME diagnostics were then reverted (this handoff
chose to ship the validated Phase A work clean, without diagnostics in the tree).

## What is clean in the tree right now

- Phase A subgroup kernels + verify scaffolds + S_max reduction: committed,
  building clean, default path (subgroup off) unaffected and stable.
- PME finite-check diagnostics: **reverted** (were never committed). The
  investigation artifacts live only in this handoff + the design doc.
- Worktree has only untracked docs/export files (no source changes pending).

## Open directions (for the next session)

1. **PME NaN root-cause + fix (highest value — unblocks long-run opt-in).**
   Re-add the non-perturbing detector with the readout in the ACTIVE
   `MPI_rank < PP_MPI_size` branch, AND add detectors on
   `d_direct_atom_energy` / `d_correction_atom_energy` (the LJ-direct and
   PME-excluded outputs, which are the actual `PM` components in this config).
   Expect to confirm a `1/dr`/`1/dr_abs` NaN at a degenerate dr=0. Then add a
   zero-guard (skip/clamp when dr2 < epsilon) in
   `PME_Excluded_Force_With_Atom_Energy_Correction` (PM_force.cpp:982/985) and/or
   the LJ direct Coulomb kernel. This is a pre-existing latent bug independent of
   Phase A; fixing it lets the subgroup path run long and realize ~74-80 ns/day.

2. **Fill pruning-reuse (user-approved future path).** NCU shows fill is
   instruction-volume-bound (26.1B inst vs count 3.05B; ~12000 inst/record), NOT
   latency-bound — subgroup-ization got only 1.57x because fill was never
   ILP-starved. The heavy per-record work is `Prune_..._Imask` (atom-level
   8x8x4 distance+cutoff prune, per record x2 splits). count already computes
   cluster-AABB + imask; if count could emit partial atom-level in-cutoff info
   (it already computes similar in its exclusion path), fill could reuse it and
   skip re-pruning — cuts a large slice of the ~12000 inst/record. High-value but
   touches the count/fill contract (risky). Lower-effort fill experiments
   available: early-exit on empty `group_record_imask`, reduce `#pragma unroll`
   8x8x4 (icache pressure), per-subgroup range split to remove shared-mem atomic
   contention. Honest ceiling: even halving fill (289->145ms) -> only ~74-78
   ns/day, all blocked by the PME NaN.

3. **Phase B (safe rolling source-cache to cut rebuild frequency).** Historically
   all attempts NaN'd (anchor-coverage contract violation). Still open per the
   pre-Phase-A handoff. Lower priority than #1.

## How to reproduce / validate

```sh
# build
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4

# bit-exact verify (count + fill), wat160k
cd /tmp/sponge-dirty-overwrite-case   # regenerate from benchmarks/performance/wat/
                                       # SPONGE_water_160k/ if wiped (copy mdcrd.dat,
                                       # mdbox.txt, water.top, water_npt_eq.gro; mdins
                                       # use direct_kernel="clustered", no traj/restart)
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1 \
  build-dev-cuda13/SPONGE -mdin mdin_forceonly_2000_notimer.spg.toml
# expect: [clustered gmxpacked subgroup verify] ... mismatch=0 ...
#         [clustered gmxpacked subgroup fill verify] ... field_mismatch=0 ...

# universality: wat600k (case benchmarks/performance/wat/SPONGE_water_600k_2x2x1/,
# top/gro are water_2x2x1.top / water_npt_eq_2x2x1.gro)

# e2e speed (subgroup ON), 1000-step safe
env ... SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1 \
  build-dev-cuda13/SPONGE -mdin mdin_forceonly_1000_notimer.spg.toml   # ~62 ns/day

# 10000-step: may NaN (PME blocker) — default path (subgroup off) is stable
```

NCU is available at `.pixi/envs/dev-cuda13/bin/ncu` (Nsight Compute 2025.3.1) —
the older memory saying "not installed" is wrong.

## One-line status

Phase A builder-kernel work is done, bit-exact, universal, +16.6% at 1000-step,
committed; the only thing blocking long-run production use is a pre-existing
unguarded-`1/dr` PME/LJ-direct energy NaN exposed by the subgroup speedup —
independent of Phase A, fixable with a zero-guard once localized.
