# NBNXM Clustered Direct Dispatch Policy

## Worktree / Branch

- Worktree: `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
- Branch: `integrate-nbnxm-gmxpacked`
- Base commit: `a19bf5b` (master, `Merge pull request #1 from yuhaosimba/master`)

## Policy (T11 Final, 2026-06-13)

The clustered PME direct path operates under a safe default / explicit opt-in policy.
The **default production path** uses the base native clustered direct kernel
(`Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_Device`). No compact
gmxpacked payload is built or dispatched by default.

### Production Env Vars

| Variable | Behavior |
|---|---|
| *(none)* | Default native clustered direct. No host-side payload conversion, no gmxpacked dispatch. |
| `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` | Explicit opt-in for gmxpacked production dispatch. Requires compact payload availability and is still suppressed by `SPONGE_CLUSTERED_GMXPACKED_FALLBACK_NATIVE=1`. |
| `SPONGE_CLUSTERED_GMXPACKED_FALLBACK_NATIVE=1` | Forces native clustered dispatch even when `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` is set. |
| `SPONGE_CLUSTERED_DUMP_MICROBENCH=<prefix>` | Diagnostic/debug snapshot path. Builds compact payload and writes `.sponge_fulloutput.bin` for `NBNXM_MICROBENCH --kernel sponge --sponge-lj-mode comb-gmxpacked` replay. Does **not** enable gmxpacked production dispatch unless `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` is also set. |

### Experimental Env Vars (Development Only)

These control alternate native-clustered kernel dispatch paths and are **not production defaults**:

| Variable | Effect |
|---|---|
| `SPONGE_CLUSTERED_USE_WARP_RECORD_FULL` | Selects the warp-record virial kernel variant within the native dispatch chain. |
| `SPONGE_CLUSTERED_USE_WARP_RECORD_TOTAL_ONLY` | Selects the total-output variant of the warp-record kernel (requires `USE_WARP_RECORD_FULL`). |
| `SPONGE_CLUSTERED_USE_GROUPED_VIRIAL` | Selects the grouped-clustered virial kernel (higher dispatch priority than warp-record). |

All experimental flags are marked `[EXPERIMENTAL]` in `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`.

## Performance Numbers

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

### Key Observation

The gmxpacked direct kernel itself is fast (30-50% kernel-level improvement),
but end-to-end wall regresses because compact primary payload is still built via
host-side conversion/upload from the finalized native payload. The
`SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` opt-in preserves this path for
microbench analysis and future device-native compact builder integration, but
the safe default avoids the host-conversion regression.

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

## Cleanup Policy (T12, 2026-06-13)

- Experimental dispatch flags (`USE_WARP_RECORD_FULL`, `USE_WARP_RECORD_TOTAL_ONLY`, `USE_GROUPED_VIRIAL`) are marked `[EXPERIMENTAL]` in source with a policy comment block at the flag-definition site.
- No code paths were removed; only documentation comments were added.
- Production flags (`USE_GMXPACKED_DIRECT`, `GMXPACKED_FALLBACK_NATIVE`, `DUMP_MICROBENCH`) are kept and documented.
- `nbnxm_microbench_snapshot.h` include in `Lennard_Jones_force.cpp` is marked `[DIAGNOSTIC DUMP ONLY]`.
- No `Gromacs*POD` types are used outside `tools/nbnxm_microbench/`.
- `clustered_lj.{h,cpp}` have zero dependency on `nbnxm_microbench_snapshot.h` or `Gromacs*POD` types.

## Commands

### Build SPONGE

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 8
```

### Build NBNXM_MICROBENCH

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target NBNXM_MICROBENCH --parallel 8
```

### Default Run (wat160k)

```sh
cd benchmarks/performance/wat/SPONGE_water_160k
../../../../build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml
```

### Opt-In Gmxpacked Run

```sh
SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  ../../../../build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml
```

### Diagnostic Snapshot Dump

```sh
SPONGE_CLUSTERED_DUMP_MICROBENCH=/tmp/snapshot \
  ../../../../build-dev-cuda13/SPONGE -mdin mdin_pme_nve.clustered_dump.spg.toml
```

### Microbench Replay

```sh
../../../../build-dev-cuda13/NBNXM_MICROBENCH \
  --kernel sponge \
  --snapshot /tmp/snapshot.sponge_fulloutput.bin \
  --sponge-lj-mode comb-gmxpacked \
  --warmup 1 --iters 5
```
