# SPONGE NBNXM Gmxpacked Project Handoff, 2026-06-26

## Purpose

This document is a project-level handoff for the gmxpacked NBNXM optimization
line in:

```text
/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked
```

It is intentionally broader than the latest payload-shrink experiment notes. A
new agent should be able to read this first, then inspect the source and run the
validation commands listed below before making any further change.

## Current Source State

- Branch: `opt/gmxpacked-payload-shrink-sweep`
- Current HEAD: `c2fd374 Add gmxpacked source-cache refresh probe`
- Parent integration branch visible in history: `integrate-nbnxm-gmxpacked`
- Relevant recent commits:
  - `8393ea0 Add NBNXM gmxpacked microbench and kernel`
  - `26ef7e5 Clean gmxpacked builder baseline`
  - `5f7ee8f Add gmxpacked active-view refresh`
  - `f22baf5 Enable active-view pair-shift metadata cache`
  - `6d03630 Default active-view to primary record builder`
  - `665e23b Default active-view to fast split gmxpacked kernel`
  - `436ad53 Default active-view rolling source cache`
  - `2059e83 Clean dead gmxpacked source-cache prototypes`
  - `378b942 Add gmxpacked source-cache patch scaffolding`
  - `228acaa Expose gmxpacked incremental refill probe toggle`
  - `c2fd374 Add gmxpacked source-cache refresh probe`

At the time this handoff was written, the only uncommitted changes were
documentation files:

```text
docs/gmxpacked-payload-shrink-status-20260626.md
docs/gmxpacked-project-handoff-20260626.md
```

No uncommitted source-code experiment is intentionally left in the worktree.

## Objective And Success Criteria

The performance target is wat160k force-only long-run speed:

- Native baseline from the current optimization plan: `51.688854 ns/day`
- 2x target: `>=103.38 ns/day`
- Current validated active-view long-run baseline in this branch:
  `64.029243 ns/day`

Do not treat 200-step stage-timer runs as speed conclusions. Use them only for
attribution. Real speed conclusions should come from no-timer 1000/10000-step
runs.

## High-Level Architecture

The optimization line is for the clustered LJ/PME direct path, specifically the
gmxpacked compact payload and direct kernel used by:

```text
SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp
SPONGE/Lennard_Jones_force/clustered_lj.cpp
SPONGE/Lennard_Jones_force/clustered_lj.h
```

The main runtime chain is:

1. `LENNARD_JONES_INFORMATION::LJ_PME_Direct_Force_With_Atom_Energy_And_Virial`
   decides whether clustered direct and gmxpacked direct are requested.
2. `LJ_CLUSTER_DIRECT::Build` / `LJ_CLUSTER_LAYOUT::Build` prepares clustered
   metadata and payloads.
3. `clustered_lj.cpp` builds one or more payload representations:
   - native clustered SCI/CJ payload;
   - record-stream source rows;
   - record-stream aggregate rows;
   - compact gmxpacked SCI/CJ/exclusion payload;
   - optional pair-shift metadata;
   - optional active-view caches and masks.
4. `Lennard_Jones_force.cpp` dispatches either the native clustered direct
   kernel or the gmxpacked direct kernel family.

The end-to-end bottleneck is currently builder/payload lifecycle, not the
gmxpacked direct kernel body itself.

## Important Data Concepts

### Native Clustered Payload

The native clustered payload is the older clustered direct representation:

- `d_nbnxm_sci`
- `d_nbnxm_cjpacked`
- `d_pair_shift_bits`

It remains important as a fallback and as a correctness reference.

### Gmxpacked Compact Payload

The gmxpacked direct kernel consumes:

- `d_gmxpacked_sci`
- `d_gmxpacked_cjpacked`
- `d_gmxpacked_exclusions`
- pair-shift metadata, either per SCI or per pair depending on compatibility.

The force-only fast path can use the split/fast gmxpacked kernel when the compact
payload and LJ comb-table assumptions are compatible.

### Record-Stream Source Rows

The record-stream source rows are an intermediate builder representation:

```text
LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE
```

Each source row represents a `(candidate SCI, shift, supercluster i,
j-cluster, j split, i-local mask, pair words)` unit. It is not an individual
atom-pair row. This matters for source-cache updates: appending a source row can
duplicate all pairs represented by that row unless pair-word differences are
handled explicitly.

### Source Cache And Anchors

The active-view path uses an outer source cache:

- `d_gmxpacked_incremental_record_stream_sources`
- `d_gmxpacked_incremental_source_offsets_by_candidate`
- `gmxpacked_incremental_source_cutoff`
- `d_gmxpacked_outer_source_anchor_crd`

The source cache is valid relative to its outer source anchor. Moving that
anchor without rebuilding or repairing the required source coverage is unsafe.

### Active View

Active view keeps the outer source coverage large enough for safety, then builds
a smaller active payload for the current cutoff:

- outer source cache cutoff is based on `cutoff + source_skin`;
- active payload is pruned by current cutoff/guard;
- active masks are tracked per source row;
- active compact payload is routed to the direct kernel.

The active-view path is the current production optimization line.

## Important Runtime Flags

Validated main path:

```sh
SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1
SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1
SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1
```

Key behavior:

- `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` requests gmxpacked dispatch.
- `SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer` enables the outer-source
  lifecycle.
- `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1` enables record-builder active view.
- `SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1` should be used for speed benchmarks.

Diagnostics / attribution:

- `SPONGE_CLUSTERED_GMXPACKED_RECORD_BUILDER_STAGE_TIMERS=1`
  prints builder stage timings. Use only on short runs.
- `SPONGE_CLUSTERED_GMXPACKED_SOURCE_CACHE_REFRESH_PROBE=1`
  probes source-cache refresh pressure using the committed read-only probe.
- `SPONGE_CLUSTERED_GMXPACKED_INCREMENTAL_REFILL=1`
  exposes a refill diagnostic path, not a production default.
- `SPONGE_CLUSTERED_GMXPACKED_SHIFT_ANALYZE=1`
  analyzes pair-shift metadata.
- `SPONGE_CLUSTERED_DUMP_MICROBENCH=<prefix>`
  dumps a microbench snapshot.

Currently disabled or not-default paths:

- current-inner lifecycle;
- adaptive-current lifecycle;
- stable target layout / stable source contract;
- active compact delta;
- incremental source merge;
- coverage repair / append-style active repair;
- mixed-source build;
- stable-source reuse.

Do not enable those as a default path without fresh validation.

## Validated Benchmark Commands

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

1000-step sanity:

```sh
cd /tmp/sponge-dirty-overwrite-case
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  /home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_1000_notimer.spg.toml
```

Latest observed result:

```text
Core Run Speed: 54.712196 ns/day
```

10000-step long-run check:

```sh
cd /tmp/sponge-dirty-overwrite-case
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  /home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_10000_notimer.spg.toml
```

Latest observed result:

```text
Core Run Speed: 64.029243 ns/day
```

Both latest validation runs completed without NaN.

## Current Performance Interpretation

The active-view path is faster than the earlier outer-only safe baseline and is
currently the path to preserve. It is still below the 2x native target.

Known payload-shape signal:

- cutoff-style payload experiment previously produced about
  `source_rows ~= 399746` and `compact_cj ~= 58011`;
- the active-view/outer payload can still reach much larger source/compact
  counts during runtime;
- the cutoff-only payload is a diagnostic shape, not a production contract.

The important target is not "use cutoff-only source cache"; it is:

1. keep outer coverage safety;
2. reduce runtime active payload toward cutoff-scale size;
3. avoid repeated full source-row generation/count-scan;
4. avoid changing the force kernel contract until profiler evidence requires it.

## Known Negative Results

### Current-Inner / Adaptive-Current / Guard Tuning

These directions were already explored before this handoff and should remain
dead ends unless a new design changes the underlying contract. They were either
unsafe, unstable, or much slower than the current active-view baseline.

### Stable Source Reuse / Mixed Source Build

These are not production defaults. Earlier experiments showed rebuild storms or
long-run instability. Keep them as debug opt-in or historical context only.

### Source-Cache Patch Prototype

Latest attempted design:

- reuse old source cache;
- replace dirty candidate SCI source rows;
- recompute offsets;
- refresh outer source anchor.

Result:

- 2000-step guard=1 opt-in run ended with NaN;
- patch `used=1` 57 times and `used=0` 14 times;
- full `primary-source-offset-count-scan` and
  `record-stream-source-row-generation` still each appeared 81 times.

Saved diff:

```text
/tmp/gmxpacked_source_cache_patch_negative.diff
```

Reason to reject:

The patch refreshed the source anchor while keeping old clean source rows. That
advertises old coverage as valid for a new anchor and can miss newly entering
source-shell pairs.

### Shell Probe / Min-Cutoff Source-Row Prototype

Latest attempted design:

- add source-row shell/min-cutoff support;
- use it as an opt-in diagnostic;
- do not modify production payload.

Result:

- narrow trigger produced no useful probe lines in 2000-step runs;
- broader trigger made a 2000-step opt-in run end with NaN;
- no trustworthy shell-row measurements were retained.

Saved diff:

```text
/tmp/gmxpacked_shell_probe_negative.diff
```

Reason to reject:

The prototype was not clean enough. Even diagnostic kernel calls must avoid
reusing or perturbing builder scratch state. It was rolled back.

## Why Source-Raw Updating Is Hard

The source cache is not just a list of pairs. It is a coverage contract tied to:

- candidate SCI layout;
- candidate leaf coverage;
- source row offsets by candidate;
- source row `imask` and pair words;
- outer source cutoff;
- outer source anchor coordinates.

A safe source-cache rolling update must prove all of the following:

1. Clean source rows are still semantically mapped to the same candidate SCI.
2. The old source anchor is either preserved, or all newly required source shell
   rows are added before refreshing the anchor.
3. Appended rows do not duplicate pair work already represented in old rows.
4. Dirty rows and shell rows are small enough that patching is cheaper than full
   rebuild.
5. Any fallback path returns to the known-safe full builder without poisoning
   active masks, offsets, anchors, or compact payload state.

The most recent NaN came from violating point 2.

## Suggested Next Work

The next agent should not start by writing another source-row patch. Start with
attribution that cannot perturb device-builder state.

Recommended sequence:

1. Keep the worktree source-clean except for documentation or instrumentation.
2. Add a host-side or strictly read-only diagnostic for every full primary
   builder call in the clean default 10000-step active-view run.
3. Log why the full path was chosen:
   - source-cache coverage miss;
   - active payload guard undercoverage;
   - compact reuse failure;
   - candidate layout/candidate SCI count change;
   - pair-shift metadata incompatibility;
   - fallback from another disabled/experimental path.
4. Log source-cache metadata at each full builder:
   - `cached_build_step`;
   - `previous_gmxpacked_incremental_source_numbers`;
   - `previous_gmxpacked_incremental_source_cutoff`;
   - source anchor displacement;
   - requested active cutoff;
   - target guard cutoff;
   - candidate SCI count.
5. Re-run:
   - 200-step stage-timer for attribution;
   - 1000-step no-timer for sanity;
   - 10000-step no-timer for real speed.

Only after this attribution is stable should source-cache patching be redesigned.

## Acceptable Future Design Shapes

The likely correct families are:

### A. Anchor-Preserving Dirty Refresh

Do not refresh the outer source anchor. Only refresh active masks/compact payload
inside the existing source-cache coverage window. When coverage expires, fall
back to full rebuild.

This is safe but may not remove the expensive later full rebuilds.

### B. Full Rolling Source-Cache Repair

Before refreshing the outer source anchor:

- identify dirty candidate SCI rows;
- identify newly required shell rows;
- append or replace rows with pair-level duplication prevention;
- update source offsets and active masks;
- then refresh anchor only if coverage is complete.

This is the correct direction if the goal is to remove later full
source-row-generation/count-scan events.

### C. Compact-Level Local Patch

Leave source cache alone, but patch compact SCI/CJ/exclusion ranges for dirty
regions. This may reduce compact rebuild cost but will not solve source-cache
coverage expiry by itself.

### D. Builder Kernel Optimization

Only pursue direct count/fill kernel tuning after NCU evidence. Existing evidence
suggests the bottleneck is low-parallelism traversal/control flow and payload
scale, not a simple DRAM bandwidth wall.

## NCU / Kernel Rule

Do not change CUDA kernel launch parameters, register pressure, memory layout, or
template dispatch based on intuition. For kernel-level work:

1. Run NCU first.
2. Record roofline, memory hierarchy, warp stalls, occupancy, and instruction
   mix.
3. Make one targeted kernel change.
4. Re-profile and compare.

This rule is especially important because a seemingly small launch-path change
can alter payload consumption order or target buffer semantics and produce NaN.

## Files To Read First

Start here:

```text
docs/gmxpacked-project-handoff-20260626.md
docs/gmxpacked-payload-shrink-status-20260626.md
docs/nbnxm-clustered-direct-policy.md
SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp
SPONGE/Lennard_Jones_force/clustered_lj.cpp
SPONGE/Lennard_Jones_force/clustered_lj.h
```

Useful code regions:

- `Lennard_Jones_force.cpp`: gmxpacked env flags and direct dispatch selection.
- `Lennard_Jones_force.cpp`: gmxpacked force-only kernel dispatch.
- `clustered_lj.cpp`: `Clustered_Gmxpacked_*` policy functions.
- `clustered_lj.cpp`: `try_inner_active_payload_refresh_from_outer_cache`.
- `clustered_lj.cpp`: record-stream source count/fill and compact build stages.
- `clustered_lj.h`: `LJ_CLUSTER_LAYOUT` gmxpacked cache fields.

## Handoff Status

Validated:

- Active-view default path builds and runs.
- 1000-step active-view no-timer sanity: `54.712196 ns/day`.
- 10000-step active-view no-timer: `64.029243 ns/day`.
- Latest source-code experiments were rolled back.

Partial:

- A source-cache refresh probe is committed, but it is diagnostic only.
- Payload-shrink design is not complete.
- Later full source-row generation/count-scan events still need exact
  attribution before implementation.

Do not commit:

- `/tmp/gmxpacked_source_cache_patch_negative.diff`
- `/tmp/gmxpacked_shell_probe_negative.diff`

They are preserved only as negative evidence.

