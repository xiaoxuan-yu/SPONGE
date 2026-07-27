# NBNXM Clustered Direct Dispatch Policy

## Worktree / Branch

- Worktree: `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
- Branch: `opt/gmxpacked-phase-b-force-kernel`
- Base commit: `a19bf5b` (master, `Merge pull request #1 from yuhaosimba/master`)

## Current Policy (2026-07-25 cleanup)

On GPU, regular and soft-core clustered PME direct LJ both use the compact
gmxpacked payload. Selecting `[LJ] direct_kernel = "clustered"` dispatches
gmxpacked without an environment opt-in. Missing payload or force scratch is a
hard runtime error; there is no GPU native clustered fallback.

CPU builds accept the same `direct_kernel = "clustered"` setting. Regular LJ
uses the CPU clustered executor, and soft-core LJ uses its CPU clustered
executor over the same host-built pair ownership and exclusion metadata.
`legacy` remains selectable only as a temporary comparison oracle while the
remaining migration gates below are closed.

Excluded-list connectivity is authoritative for clustered molecule ownership.
The builder forms connected components directly from the exclusion graph
instead of assuming that residue/molecule ranges contain every excluded pair.
This is required for inputs whose residue metadata is absent or finer grained
than their bonded/excluded topology.

### Production Env Vars

| Variable | Behavior |
|---|---|
| *(none)* | Clustered regular and soft-core LJ dispatch gmxpacked on GPU and the clustered CPU executors on CPU. |
| `SPONGE_CLUSTERED_DUMP_MICROBENCH=<prefix>` | Writes `.sponge_fulloutput.bin` for retained `NBNXM_MICROBENCH --kernel sponge --sponge-lj-mode comb-gmxpacked` replay. It does not change production dispatch. |

`SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT` and
`SPONGE_CLUSTERED_GMXPACKED_FALLBACK_NATIVE` are removed. Historical experiment
notes may still show them in recorded command lines.

### Kernel variants

Each retained regular-LJ and soft-LJ dispatch family exposes two output
variants:

- force-only;
- full energy+virial.

Energy-only and virial-only requests reuse the full kernel with runtime output
masks. The former grouped-virial, warp-record virial-only, and total-output
dispatch experiments have been removed.

## Historical Performance Numbers

The T4/T11 tables below describe the earlier native-default/opt-in phase. They
are retained as experiment history, not as the current dispatch policy.

### T4 Pre-Kernel Baseline

| System | Core Wall | Speed | Microbench Replay |
|---|---|---|---|
| wat160k | 0.132146 s | 1.307640 ns/day | 0.498278 ms |
| wat600k | 0.240456 s | 0.718635 ns/day | 2.031821 ms |

### T11 Default (Native Clustered Direct, No Gmxpacked)

| System | Core Wall | Speed | vs T4 |
|---|---|---|---|
| wat160k | 0.101413 s | 1.703921 ns/day | 23.3% faster |
| wat600k | 0.115704 s | 1.493467 ns/day | 51.9% faster |

### T11 Opt-In (Gmxpacked Direct, With Host-Side Conversion Overhead)

| System | Core Wall | Primary Payload | Direct Kernel |
|---|---|---|---|
| wat160k | 0.449605 s | 0.324847 s | 0.007874 s |
| wat600k | *(not measured; similar host-conversion regression)* | | |

### T11 Opt-In Microbench Replay (Kernel-Only)

| System | avg_ms | vs T4 |
|---|---|---|
| wat160k full | 0.228147 ms | 54.2% faster |
| wat600k full | 1.005773 ms | 50.5% faster |

### Historical Observation

The gmxpacked direct kernel itself is fast (30-50% kernel-level improvement),
but end-to-end wall regresses because compact primary payload is still built via
host-side conversion/upload from the finalized native payload. Subsequent
device-side builder work removed this result as a reason to keep a production
native fallback.

## Soft-Core GPU Result (RTX 4090, wat160k)

The soft-core gmxpacked full kernel was optimized with an NCU-first loop. The
final launch bound retains 96 registers per thread and 41.67% theoretical
occupancy.

| Variant | NCU duration | Registers/thread | Theoretical occupancy | Local spill requests |
|---|---:|---:|---:|---:|
| unconstrained | 4.00 ms | 155 | 25.00% | 0 |
| 7 blocks/SM | 3.64 ms | 128 | 33.33% | 5,199,612 |
| 10 blocks/SM (retained) | 3.44 ms | 96 | 41.67% | 17,152,810 |

The retained version improves the NCU kernel duration by about 14%. A separate
20-step run without the profiler measured 54.08 ms total kernel-launch time
versus 64.55 ms for the 128-register version. The extra spill traffic remains
cache-resident enough to win on this workload (`97.58%` L2 hit rate).

## T9 Blocker: Derived Compact `ATOM_GROUP`

**Status: `[~]` (disabled).**

The legacy `ATOM_GROUP` compatibility view derived from the clustered compact
payload is intentionally disabled. The adapter hook
(`Ensure_Legacy_Neighbor_View_From_Clustered_Payload`) rejects all derivation
requests with `"half-list pair-set proof is missing; grid legacy build remains required"`.

Root cause: legacy local-local half-list ownership is assigned by global atom
order (`global_j > global_i` in `neighbor_list.cpp`), while clustered compact
records own pairs by spatial supercluster traversal. A future derivation must
reassign every candidate pair to the legacy owner and pass a direct pair-set
comparison before the adapter can be safely enabled.

Consequence: non-LJ half-list consumers (custom pairwise Morse), all full-list
consumers (SW, EDIP, EAM, TERSOFF, ReaxFF), and any consumer needing legacy
`ATOM_GROUP` continue to use the grid-based legacy neighbor-list build. The
clustered PME water fast path remains the only consumer that skips the legacy
build entirely.

## Cleanup Policy

- Regular clustered native/reference kernels and production opt-in/fallback
  flags are removed.
- GPU soft-core clustered execution uses gmxpacked; its native clustered
  executor is retained only for the CPU backend.
- CPU regular and soft-core clustered executors are production alternatives to
  the legacy non-clustered path.
- Legacy non-clustered LJ remains temporarily available only for uncovered
  features. It is not the correctness oracle at periodic boundaries and is not
  the intended post-backport architecture.
- Diagnostic snapshot export is kept.
- Peak force-only and full-output specializations are kept.
- Non-peak dispatch probes and standalone neighbor API scaffolding are removed.
- `NBNXM_MICROBENCH` and its snapshot types remain isolated under `tools/`.

## Correctness Evidence

- A 256-atom soft-core fixture with exclusions crossing residue/molecule
  metadata boundaries produces `potential = -634.03 kcal/mol` on clustered GPU
  and legacy GPU after deriving ownership from the exclusion graph.
- The regular clustered GPU path passes the analogous fixture at
  `-633.72 kcal/mol`.
- TIP3P clustered CPU and GPU agree (`-4501.68` and `-4501.75 kcal/mol`,
  respectively); the remaining small difference from legacy (`-4499.57`) is
  accumulation/ownership convention, not a missing large electrostatic term.
- For 164,544 water atoms, clustered CPU and GPU produce
  `LJ_soft_short = 17846.07 kcal/mol`. An independent periodic cell-list oracle
  gives `17846.0747267`; legacy gives `19498.88` because it overcounts periodic
  boundary pairs. The analogous regular-LJ legacy discrepancy is also present.

Consequently, legacy equality is no longer a removal gate for periodic systems.
The replacement validation gates are clustered CPU/GPU agreement plus an
independent periodic pair oracle.

## Current Removal Gates

- The soft-core lambda-derivative path still calls the legacy neighbor-list
  implementation. A gmxpacked/CPU-clustered dU/dlambda output must be added
  before that consumer can be removed.
- `LENNARD_JONES_NO_PBC_INFORMATION` still provides the force and optional
  per-atom-energy fallback for non-periodic runs. It needs a clustered
  replacement or an explicitly retained standalone implementation.
- Other non-LJ `ATOM_GROUP` consumers listed in the T9 section still require
  the grid neighbor list; removing LJ legacy must not remove their provider.
- A one-thread 4000-step bad-coordinate minimization is finite with clustered
  direct LJ and reaches below `-4100 kcal/mol`; legacy enters
  NaN by step 100. Parallel CPU atomics change floating-point accumulation
  order enough to move this deliberately pathological Adam trajectory between
  basins, so parallel determinism remains a gate before making the CPU path the
  unconditional default.
- General defaults use `skin = 2.0`. Peak water benchmark inputs opt in to
  `skin = 10.0` and `clustered_rebuild_skin = 10.0`.

## Commands

### Build SPONGE

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 8
```

### Build NBNXM_MICROBENCH

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target NBNXM_MICROBENCH --parallel 8
```

### Clustered Gmxpacked Run (wat160k)

```sh
cd benchmarks/performance/wat/SPONGE_water_160k
../../../../build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml
```

### Diagnostic Snapshot Dump

```sh
SPONGE_CLUSTERED_DUMP_MICROBENCH=/tmp/snapshot \
  ../../../../build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml
```

### NCU Soft-Core Full Kernel

```sh
pixi run -e dev-cuda13 ncu \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*Nbnxm_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core.*' \
  --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --section LaunchStats --section Occupancy \
  build-dev-cuda13/SPONGE -mdin mdin.clustered.spg.toml
```

### Microbench Replay

```sh
../../../../build-dev-cuda13/NBNXM_MICROBENCH \
  --kernel sponge \
  --snapshot /tmp/snapshot.sponge_fulloutput.bin \
  --sponge-lj-mode comb-gmxpacked \
  --warmup 1 --iters 5
```
