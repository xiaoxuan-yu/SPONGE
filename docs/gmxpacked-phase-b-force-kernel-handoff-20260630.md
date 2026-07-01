# gmxpacked Phase B force-kernel investigation handoff — 2026-06-30

This hands off the **force-kernel performance** investigation (the gate to the 2x
target) to a separate agent. The rebuild line is being handled in parallel by the
originating agent. Read this together with
`docs/gmxpacked-phase-a-followup-handoff-20260629.md` (the Phase A follow-up),
`docs/gmxpacked-phase-a-handoff-20260629.md` (Phase A), and the SPONGE-dev
reference `docs/nbnxm_optimization_report_2026-06-04.md`.

Worktree: `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
Original branch: `opt/gmxpacked-payload-shrink-sweep`
Phase B branch: `opt/gmxpacked-phase-b-force-kernel`
Sibling reference: `/home/youmans/sidereus/SPONGE-dev` (microbench + tuned kernel + optimization report)

## 2026-07-01 current peak-path update

The older `BASE0 ~51.5 ns/day` locked config below is no longer the current
best end-to-end path. For the 160k force-only case, the accepted peak path now
uses the gmxpacked active-view / rolling-source-cache / fill-prune-reuse-light
stack plus:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_PARALLEL_ACCUM=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
```

Latest decision: keep candidate leaf one-pass enabled in the performance-default
env. Same-binary comparisons show about 6-8% end-to-end speedup over returning
to `Count/Fill_Supercluster_Candidate_Leaves_Fixed_Shift_Subgroup`, while the
observed process GPU-memory delta was only `+8 MiB` on this case.

Best observed 10000-step force-only run on this path:

- `87.843224 ns/day`
- core wall `9.836689 s`
- `Calculate_Force = 9.175779 s`
- final temperature `294.34 K`

For the full env, memory comparison, and nsys candidate-leaf kernel split, use
`docs/gmxpacked-fixed-shift-candidate-leaf-onepass-20260701.md`. Treat the older
locked config and target math in this file as historical context unless a future
run explicitly disables the July 1 peak-path flags for an ablation.

## 2026-06-30 Phase B correction

The first Phase B check found that one premise below was stale for the current
source after a fresh rebuild:

- BASE0 active-view gmxpacked force-only does **not** fall back to the plain
  `Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device`.
- With BASE0 env, production dispatch reaches
  `Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device`
  in the full-local dense gmxpacked path.
- The sorted/float4 force target is the part still default-off. It is controlled
  by `SPONGE_CLUSTERED_GMXPACKED_FORCE_SORTED_SCRATCH=1`,
  `SPONGE_CLUSTERED_GMXPACKED_FUSED_SORTED_FORCE=1`, and
  `SPONGE_CLUSTERED_GMXPACKED_FLOAT4_SORTED_FORCE=1`.

Evidence from `opt/gmxpacked-phase-b-force-kernel`:

- Gate trace with BASE0 plus the three sorted-force envs:
  `has_payload=1`, `frc4=1`, `full_local_dense=1`,
  `use direct=1`, `fast=1`, `compact=1`, `fused=1`, `float4=1`.
- NCU force-only launch after skipping the two initialization energy kernels:
  `ForceOnly_Warp_Record_Device<0,0,0,1,1,1,1,1,float4,0>`, duration
  `277.47 us`.
- NCU BASE0 default force-only launch:
  `ForceOnly_Warp_Record_Device<0,0,0,0,1,1,1,1,VECTOR,0>`, duration
  `294.50 us`.
- 2000-step skin=12 BASE0: `48.76 ns/day`.
- 2000-step skin=12 with sorted/fused/float4 force target: `50.39 ns/day`.
- 10000-step skin=12 with sorted/fused/float4 force target: finite at step
  10000, `49.73 ns/day`, temperature `294.97 K`, `LJ=78668.27`,
  `PM=3131485.50`.

Interpretation:

- The deep gate (`d_sorted_frc4` allocation or full-local dense layout) is not
  the blocker in the current source. It succeeds when the explicit sorted-force
  envs are set.
- `force-only`, `energy-only`, and `full energy+virial` gmxpacked total-output
  variants are connected in production dispatch.
- The missing production contract is per-atom energy/virial for the gmxpacked
  fast path. The microbench has per-atom/full-output variants, and this file's
  native warp-record path has per-atom writeback, but the gmxpacked
  `ForceOnly_Warp_Record_Device` currently enforces
  `total_output || (!need_energy && !need_virial)`.
- Do not default-enable sorted/float4 from the single-kernel NCU win alone. The
  extra sorted-force memset/scatter path needs end-to-end and correctness gates;
  the first 2000-step run showed a small speed win, but the 10000-step run was
  below the locked best and showed different long-run trajectory values from
  changed atomic accumulation order.

Updated next step: map and, if needed, port the microbench per-atom gmxpacked
writeback path into production behind an explicit per-atom-output gate. Keep
total-output as the production default for ordinary energy/pressure reductions
unless the caller has a real per-atom contract.

## 2026-06-30 current cost decomposition

This supersedes the older "force kernel is 62%" Phase 0 attribution below for
the current `opt/gmxpacked-phase-b-force-kernel` build. The production force-only
path now dispatches the gmxpacked `ForceOnly_Warp_Record_Device`, so the next
bottleneck is the gmxpacked record-builder/rebuild path, not the LJ force kernel
alone.

Measurement setup:

- Case: `/tmp/sponge-dirty-overwrite-case`, `mdin_skin12_10000.spg.toml` for the
  real baseline and `mdin_skin12_2000.spg.toml` for profiler samples.
- Env: BASE0 =
  `SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1`,
  `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1`,
  `SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer`,
  `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1`,
  `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1`,
  `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0`.
- Evidence files from this run:
  `/tmp/sponge_profile_base10000.out`,
  `/tmp/sponge_profile_trace10000.err`,
  `/tmp/sponge_profile_stage2000.err`,
  `/tmp/sponge_profile_ncu_all_steady_norebuild.txt`.

Real 10000-step baseline:

| component | total | per step | share of core wall |
|---|---:|---:|---:|
| Core Run Wall Time | 17.162429 s | 1.716 ms | 100% |
| `Calculate_Force` | 16.382205 s | 1.638 ms | 95.5% |
| outer `Iteration` work | 0.603176 s | 0.060 ms | 3.5% |
| residual / print / accounting | 0.177048 s | 0.018 ms | 1.0% |

Trace-only 10000-step run:

- Full record-builder rebuilds: 20, at steps
  `0, 588, 1120, 1642, 2216, 2701, 3350, 3840, 4332, 4774, ... 9820`.
- Active-view inner refreshes: 797.
- No active-view reuse/cache-hit lines in this BASE0 run.

Stage-timer 2000-step sample:

| record-builder stage | count | avg per full rebuild | amortized per step |
|---|---:|---:|---:|
| all full-rebuild stages | 4 | 405.47 ms | 0.811 ms |
| `record-stream-source-row-generation` | 4 | 218.96 ms | 0.438 ms |
| `primary-source-offset-count-scan` | 4 | 150.67 ms | 0.301 ms |
| `record-stream-inner-active-flag-scan` | 4 | 8.53 ms | 0.017 ms |
| `record-stream-device-compact-pack` | 4 | 6.48 ms | 0.013 ms |
| `record-stream-inner-active-fill` | 4 | 6.19 ms | 0.012 ms |
| small active-view dirty-source checks | 157 | 0.107 ms | 0.008 ms |

Steady non-full-rebuild NCU window (`--launch-skip 320 --launch-count 500`)
covered 32 steps. Per-step kernel-duration sum was about 0.794 ms:

| steady component | per step |
|---|---:|
| gmxpacked LJ direct `ForceOnly_Warp_Record_Device` | 0.303 ms |
| active-view dirty detection (`Mark_Gmxpacked_Incremental_Dirty_*`) | 0.274 ms |
| other clustered bookkeeping/reductions | 0.058 ms |
| pair-shift metadata refresh | 0.053 ms |
| SETTLE | 0.039 ms |
| PM excluded correction | 0.034 ms |
| bond | 0.023 ms |
| integrator kernel | 0.010 ms |

Current target math:

- Current speed: `50.35 ns/day` = `1.716 ms/step`.
- Old 2x target from the native `~41.7 ns/day` baseline is about
  `83.4 ns/day` = `1.036 ms/step`.
- The required saving is therefore about `0.68 ms/step`.
- Perfectly eliminating the LJ force kernel would save only `0.303 ms/step`,
  so force-kernel tuning alone cannot reach the target.
- The two dominant full-rebuild stages alone amortize to about
  `0.739 ms/step`, which is enough to explain the current gap.

Next optimization target:

1. First target `record-stream-source-row-generation` and
   `primary-source-offset-count-scan`: reduce full rebuild cost, reduce full
   rebuild frequency, or avoid rebuilding the full outer source rows when the
   active-view contract can be repaired incrementally.
2. Second target the steady active-view dirty path, especially
   `Mark_Gmxpacked_Incremental_Dirty_J_Candidates`: reduce scan frequency,
   restrict the candidate set, or replace the broad scan with a cheaper
   displacement/index guard.
3. Only after the record-builder/dirty path is materially smaller should the LJ
   force kernel be tuned further. It is now about 18% of wall, useful but no
   longer the primary gate.

## Historical Phase 0 force-kernel rationale

The following section is preserved for context from the earlier plain-kernel
regime. Do not use it as the current bottleneck attribution after the
2026-06-30 correction above.

Phase 0 cost decomposition (wall-only, reliable; see "NCU methodology" caveat below)
on the corrected regime (GRO velocities loaded + rolling-source coverage fix; real
300 K motion) shows the **force kernel dominates**:

| component | per step | share of wall (skin=12) |
|---|---|---|
| **LJ force kernel** `Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device` | **1.08 ms** | **~62%** |
| rebuild (R=174 ms × ~17) | — | ~17% |
| per-step host gap | 0.22 ms | ~13% |
| other per-step GPU | 0.13 ms | ~8% |

Corrected-regime baselines (10000-step, no-timer, RTX 4090, wat160k force-only):
- NATIVE ≈ 41.7 ns/day  ⇒ **2x target ≈ 83.4 ns/day**
- BASE (active-view + subgroup + rolling) ≈ 42.1 (≈ native — gmxpacked earns nothing yet)
- **BASE + `ROLLING_SOURCE_CACHE=0` + skin=12 = ~51.5 ns/day** (the current locked best; see "Locked config" below)

**Ceiling:** even with zero rebuild + zero host overhead, the per-step GPU floor
(1.21 ms/step) caps at ~71 ns/day — **below 2x**. So the force kernel MUST be cut
to reach 2x. The rebuild line (being handled separately) caps at ~71.

The force kernel is **identical in NATIVE and BASE** (NCU-confirmed): the gmxpacked
active-view path only changes the builder payload, not the force kernel. So gmxpacked
currently has zero leverage on the 62% force kernel — that is the gap to close.

## The plain production force kernel

`SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp:4275`,
`Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device`, templated
`<bool need_force, bool need_energy, bool need_virial, bool need_coulomb>`.
Block `(cluster_size, cluster_size, 1)` = `(8,8,1)` = 64 threads = 2 warps.
Grid = `sci_numbers` (~5017 for wat160k). `max_block_warps = 2`.

Two variants fire in the force-only benchmark (both profiled):
- `<1,0,0,1>` force+coulomb (no energy/virial): 1.08 ms, **theoretical occ 41.67%, 88 reg**, achieved 38.9%, latency-bound (49% no-eligible, 0.76 eligible warps/scheduler), 3.92 waves (GPU filled), DRAM 8%, Compute 47%. Register-limited.
- `<1,1,0,1>` force+energy+coulomb: ~0.91 ms, **theoretical occ 58.33%** (14 blocks/SM, naturally high), achieved ~53%.

No `__launch_bounds__` on this plain kernel.

## The tuned near-GROMACS kernel EXISTS in this branch but is gated OFF

- `Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device` — `Lennard_Jones_force.cpp:2658`
- `Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Virial_Warp_Record_Device` — `:3229`
- Both have `__launch_bounds__(kClusteredClusterSize * kClusteredSuperClusterClusters, 13)` = **`__launch_bounds__(64, 13)`** → ~78 reg budget (achieves ~64 reg, user recalls ~64). This is the tuning to match/reference.
- Per SPONGE-dev `docs/nbnxm_optimization_report_2026-06-04.md`: SPONGE nbnxm went from 4.5x → **~10–13% residual gap vs GROMACS 2026.2** in standalone microbench; best variants = "comb-dense / comb-gmxpacked force-only" = these Warp_Record kernels.
- **NCU confirmed the Warp_Record kernel does NOT fire in production** (only the plain kernel fires). It is gated OFF.

### The production gate (why it's off)

`Lennard_Jones_force.cpp:6993`:
```cpp
const bool use_gmxpacked_float4_sorted_force =
    use_gmxpacked_fused_sorted_force &&
    requested_gmxpacked_float4_sorted_force &&
    gmxpacked_fast_full_local_dense_compatible &&
    has_sorted_force_float4_scratch;
```
The gmxpacked-path Warp_Record dispatch at `:7164` fires only when
`use_gmxpacked_float4_sorted_force` is true. Sub-conditions:
- `requested_gmxpacked_float4_sorted_force` = `Clustered_Gmxpacked_Float4_Sorted_Force_Enabled()` — env `SPONGE_CLUSTERED_GMXPACKED_FLOAT4_SORTED_FORCE` (default OFF).
- `use_gmxpacked_fused_sorted_force` needs `Clustered_Gmxpacked_Fused_Sorted_Force_Enabled()` (env `SPONGE_CLUSTERED_GMXPACKED_FUSED_SORTED_FORCE`, OFF) + `use_gmxpacked_compact_force_scratch` + `gmxpacked_forceonly_sorted_scratch` (needs `Clustered_Gmxpacked_Force_Sorted_Scratch_Enabled()` — env `SPONGE_CLUSTERED_GMXPACKED_FORCE_SORTED_SCRATCH`, OFF).
- `gmxpacked_fast_full_local_dense_compatible` = `gmxpacked_fast_layout_compatible && clustered_layout.ghost_numbers == 0 && ...` (`:6912`).
- `has_sorted_force_float4_scratch` = `clustered_direct_cache->d_sorted_frc4 != NULL` (`:6870`) — **the float4 sorted-force scratch buffer must be allocated**.

Env-gate functions cluster at `Lennard_Jones_force.cpp:148–168`:
`USE_FAST_KERNEL` (defaults ON via active-view), `FORCE_SORTED_SCRATCH`, `FUSED_SORTED_FORCE`, `FLOAT4_SORTED_FORCE` (all default OFF). Another gate `SPONGE_CLUSTERED_USE_WARP_RECORD_FULL` enables a *non-gmxpacked* Warp_Record path but it requires `!use_gmxpacked_direct && need_virial` (`:7006`) — not the active-view gmxpacked path.

**Tried and FAILED to activate:** setting `FLOAT4_SORTED_FORCE=1 FUSED_SORTED_FORCE=1 FORCE_SORTED_SCRATCH=1` together still left the plain kernel firing (NCU-confirmed). So a **deeper gate fails** — prime suspects: `has_sorted_force_float4_scratch` (d_sorted_frc4 not allocated — there may be a setup/allocation path that isn't triggered) or `gmxpacked_fast_full_local_dense_compatible` (ghost_numbers or layout). **This is the first thing to diagnose.**

## Naive launch_bounds REGRESSED (do not repeat blindly)

Adding `__launch_bounds__(64, 12)` to the plain kernel → e2e **49.9 → 41.4 ns/day (−17%)**, reverted. Reason: a single fixed launch_bounds sets one reg budget (85 for 12 blocks) that **caps the high-occupancy energy variant `<1,1,0,1>`** (was 58.33%, pushed down) while barely helping the force-only variant (41.67%→50%, 1.08→1.02 ms). The energy-variant loss outweighed the force-only gain.

**Lesson (the user's autotune point):** launch params must be tuned **per variant × per payload-size**, not one fixed value. SPONGE-dev achieves this by having **separate Warp_Record kernels per output mode** (ForceOnly vs Virial), each with its own `__launch_bounds__(64, 13)`. **Skin changes the payload size** (bigger skin → bigger build_cutoff → more candidate leaves/SCI work → different optimal launch config), so the tune is payload-dependent.

## NCU methodology (critical — avoid my mistake)

`/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/.pixi/envs/dev-cuda13/bin/ncu` = Nsight Compute 2025.3.1. nsys is **absent**.

- **RELiable:** `--kernel-name regex:"..." --launch-count N --section SpeedOfLight` and read the per-launch **`Duration`** line (single launch, real kernel duration). Aggregate by kernel name in Python (`csv` module — awk breaks on templated names with commas).
- **UNreliable (do NOT use for totals):** `--csv --metrics gpu__time_duration.sum` summed across launches — NCU **replays kernels under instrumentation**, so the sum EXCEEDS wall (I saw 891 ms GPU "total" for a 429 ms-wall run). Use it only for relative per-kernel comparison, never for GPU-utilization accounting.
- Stage-timer env `SPONGE_CLUSTERED_GMXPACKED_RECORD_BUILDER_STAGE_TIMERS=1` produced NO output on this path (gated by `need_gmxpacked_payload && runtime_gmxpacked_direct_requested && Record_Builder_Enabled()` at `clustered_lj.cpp:24711` — one is false). Don't rely on it.
- Force-kernel full NCU (occupancy, scheduler, memory): `--launch-count 5 --section SpeedOfLight --section SchedulerStats --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --kernel-name regex:"Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb"`.

## The microbench (the agreed reference/autotune tool)

In SPONGE-dev (and the infra ports to this branch — verify):
- `tools/nbnxm_microbench/nbnxm_microbench.cu`, `nbnxm_microbench_snapshot.h`, `gromacs_forceonly_replay.cu`
- `cmake/targets/NBNXM_MICROBENCH.cmake` → `NBNXM_MICROBENCH` binary
- NCU reports: `SPONGE-dev/ncu_reports/microbench_20260602/`
- Evidence: `SPONGE-dev/.omo/evidence/mainline-nbnxm-gmxpacked-20260612/microbench-baseline.txt`, `gmxpacked-kernel-dispatch.txt`, `performance-validation.txt`

Dump a production payload from this branch's SPONGE:
```sh
SPONGE_CLUSTERED_DUMP_MICROBENCH=/tmp/nbnxm_payload_current \
  build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml   # from benchmarks/performance/wat/SPONGE_water_160k/
```
Then replay/compare (comb-gmxpacked = Warp_Record; the 4-way compare isolates kernel vs payload):
```sh
build-dev-cuda13/NBNXM_MICROBENCH --kernel sponge --sponge-lj-mode comb-gmxpacked \
  --snapshot /tmp/nbnxm_payload_current.sponge_forceonly.bin --warmup 200 --iters 2000
build-dev-cuda13/NBNXM_MICROBENCH --kernel gmx \
  --snapshot /tmp/nbnxm_payload_current.sponge_forceonly.bin --warmup 200 --iters 2000
```
**skin changes payload size** — to autotune, dump payloads at skin=10 and skin=12 and compare the Warp_Record kernel's per-payload timing/occupancy. The microbench isolates the kernel from rebuild/PME/scheduling noise (the optimization report emphasizes this was the key methodological win).

## Suggested next steps for the force-kernel agent

1. **Run the microbench** to confirm the Warp_Record (comb-gmxpacked) kernel's actual speed and occupancy on the current skin-dependent payload, vs the plain kernel and vs GROMACS replay. This is the cheapest, non-perturbing measurement and grounds everything.
2. **Diagnose the deep gate**: why `has_sorted_force_float4_scratch` (`d_sorted_frc4 == NULL`?) and/or `gmxpacked_fast_full_local_dense_compatible` are false in production. Find the allocation/setup path that populates `d_sorted_frc4` (grep `d_sorted_frc4`, `sorted_force_float4`, `Force_Sorted_Scratch` in `clustered_lj.cpp`/`clustered_lj.h`). The fast path may be intentionally off due to a correctness caveat (the dispatch evidence `gmxpacked-kernel-dispatch.txt` mentions a prior "divergent warp-synchronous subgroup control flow" bug fixed for the compact path — check whether the float4 fast path has a similar open issue).
3. **Wire the Warp_Record kernel into production** (if the gate is just missing allocation/setup, enable it; if there's a correctness caveat, fix or scope it). Validate bit-exact (the `Reference_Nbnxm_..._Device` kernel at `:5751` exists for comparison; also `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1` and the physics finite gate) + finite 10000-step + coverage gate.
4. **Per-variant × per-payload autotune** of `__launch_bounds__(64, K)` for K ∈ {10,12,13,14,16} on each variant, using the microbench (not e2e, to avoid rebuild noise). The SPONGE-dev value `13` is the reference起点.
5. If the Warp_Record kernel can't be made production-safe, **tune the plain kernel per-variant** (different launch_bounds per template instantiation via a template-dependent `constexpr int min_blocks`).

## Locked config + run harness (use for any e2e validation)

- Build: `pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4`
- Case dir: `/tmp/sponge-dirty-overwrite-case` (regenerate from `benchmarks/performance/wat/SPONGE_water_160k/` if wiped: copy mdcrd.dat/mdbox.txt/water.top/water_npt_eq.gro; mdins have `direct_kernel="clustered"`, no traj/restart).
- **BASE0 env** (locked best so far, ~51.5 ns/day): `SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1 SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0` + mdin `[LJ] clustered_rebuild_skin = 12`.
- mdins: `mdin_forceonly_{1000,2000,10000}_notimer.spg.toml`; per-skin copies `mdin_skin{10,12,14,16,20}_{2000,10000}.spg.toml` (add `[LJ] clustered_rebuild_skin = V`).
- **Gates (cheapest-first, always):**
  1. coverage: `SPONGE_CLUSTERED_TRACE_WARP_RECORDS=1` short run, grep `source_limit.*active_cutoff` (expect 0).
  2. finite: BASE0 on `_10000_notimer`, must reach step 10000 finite (temp≈294-300, LJ≈78k, PM finite). NOTE: skin≥14 NaNs (physics, coverage clean — likely the pre-existing unguarded-1/dr at dr=0 exposed by larger drift; separate from this work).
  3. bit-exact: `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1` on `_2000_notimer` → mismatch=0 / field_mismatch=0.
  4. speed: BASE0 on `_10000_notimer` ×2, compare to 51.5.

## What NOT to do

- Do NOT slap a single fixed `__launch_bounds__(64, K)` on the plain templated kernel — it regressed (capped the energy variant). Tune per-variant.
- Do NOT trust NCU `gpu__time_duration.sum` totals (replay-inflated). Use single-launch SpeedOfLight `Duration`.
- Do NOT reintroduce undercovered rolling-source reuse (the coverage invariant `active_target_cutoff + 2·anchor_max_displacement ≤ outer_source_cutoff` must hold) — that was the Phase A "PME NaN" root cause, already fixed.
- Do NOT change the force kernel's arithmetic/contract by intuition — the handoffs' rule is "NCU first, one targeted change, re-profile." The microbench is the safe place to experiment.

## Files to read first

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp` — force kernels (plain `:4275`, Warp_Record ForceOnly `:2658`, Warp_Record Virial `:3229`, Reference `:5751`), env gates `:148–168`, dispatch `:5559–7400` (gates around `:6700–7010`, `:7128–7316`).
- `SPONGE/Lennard_Jones_force/clustered_lj.cpp` / `clustered_lj.h` — sorted-force scratch allocation (`d_sorted_frc4`, `d_sorted_frc`, `Force_Sorted_Scratch`), the gate buffer setup.
- `/home/youmans/sidereus/SPONGE-dev/docs/nbnxm_optimization_report_2026-06-04.md` — the GROMACS comparison + tuning history.
- `/home/youmans/sidereus/SPONGE-dev/.omo/evidence/mainline-nbnxm-gmxpacked-20260612/gmxpacked-kernel-dispatch.txt` — production dispatch validation history (incl. the prior subgroup-control-flow bug).
- `docs/gmxpacked-phase-b-force-kernel.md` (memory) — short-form of this handoff.

## One-line status

Force kernel is 62% of wall and the gate to 2x; a tuned near-GROMACS Warp_Record kernel (`__launch_bounds__(64,13)`, ~64 reg) exists in-branch but is gated OFF in production by a deep (non-env) condition (`has_sorted_force_float4_scratch` / `gmxpacked_fast_full_local_dense_compatible`); naive single launch_bounds regressed; next is to run the microbench to measure the Warp_Record kernel's actual perf and diagnose the deep gate before wiring it into production.
