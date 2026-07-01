# SPONGE gmxpacked Phase A follow-up handoff - 2026-06-29

This document follows `docs/gmxpacked-phase-a-handoff-20260629.md`.
It records the PME/active-view follow-up investigation, the source cleanup, the
two fixes that remain in the worktree, and the paused performance state.

Worktree: `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
Branch: `opt/gmxpacked-payload-shrink-sweep`

## Current status

The earlier "PME NaN" diagnosis was too shallow. Adding `dr2 != 0` guards in
PME/direct energy code is not the correct fix for this path. The NaN/huge-force
symptom is downstream of an invalid direct-space trajectory caused by active-view
gmxpacked payload reuse, not by PME itself.

Current source cleanup is complete: the temporary PME root diagnostics, pair-force
diagnostics, pair trace, and fast-kernel compare instrumentation were removed.
Only two source changes remain:

- `SPONGE/xponge/load/gromacs.hpp`
- `SPONGE/Lennard_Jones_force/clustered_lj.cpp`

Build after cleanup:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

Result: passed. The build emitted existing CUDA warnings in unrelated soft-core
code, then linked `SPONGE`.

## Fix 1: read GRO velocities

File: `SPONGE/xponge/load/gromacs.hpp`

`Gromacs_Load_Gro` now reads optional velocity fields from `.gro` atom lines when
`line.size() >= 68`.

Conversion used:

```cpp
Gromacs_To_Angstrom(std::stof(line.substr(...))) /
    CONSTANT_TIME_CONVERTION;
```

Reason: GROMACS stores velocities in nm/ps; SPONGE coordinates use Angstrom and
its internal time unit. This makes the wat160k case start at the expected
temperature:

- after fix, step 0 temperature is about `299.85 K`.
- an earlier incorrect `* 10` attempt produced an unphysical `125461 K`.

This is a real input-loader fix, independent of gmxpacked.

## Fix 2: active-view rolling source coverage

File: `SPONGE/Lennard_Jones_force/clustered_lj.cpp`

Root cause: rolling outer source reuse checked whether the cached outer source
covered only:

```text
cutoff + 2 * anchor_max_displacement
```

But active-view constructs/reuses an inner active payload with a larger target
cutoff, commonly `active_cutoff = 9.85 A` for the wat160k case. The correct
coverage requirement must include that target cutoff:

```text
active_target_cutoff + 2 * anchor_max_displacement <= outer_source_cutoff
```

The failing trace showed the old logic allowing reuse even when coverage was
insufficient:

```text
source_limit=8.625809 active_cutoff=9.850000
```

That means some pairs required by the active payload could be missing from the
cached outer source. The direct force then evolved an invalid trajectory, which
later appeared as close contacts, huge pair forces, and PME/direct-energy NaNs.

The fix computes `rolling_source_target_cutoff` from the actual active/current
inner target before testing source coverage:

- current-inner path: use `current_inner_target_guard_cutoff`.
- active-view path: use `cutoff + active_view_payload_guard_margin` for fixed
  cutoff mode, otherwise `cutoff + guard_margin`.
- if a cached active guard cutoff is larger, keep covering that larger cutoff.

When the old outer source cannot cover that target, rolling reuse is rejected and
the code falls back to the normal rebuild path. This is not a branch bypass; it
is the missing validity condition for reusing the branch.

## What was ruled out

The following were checked during diagnosis and are not the root fix:

- `dr2 != 0` guard in PME excluded correction: PME had no original guard; adding
  one only masks a downstream symptom.
- pair-shift metadata cache: disabling it did not fix the failure and sometimes
  failed earlier.
- fast dense offset assumptions: dense diagnostics showed full masks and dense
  offsets were consistent.
- fast-vs-reference pair force formula: energy-enabled fast/ref compare showed
  force differences only around `1e-5`, not enough to explain the blow-up.
- disabling fast kernel: useful as a diagnostic because it passed, but it is not
  a fix.

## Validation already run

Important 10000-step validations on `/tmp/sponge-dirty-overwrite-case`:

```sh
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1 \
  build-dev-cuda13/SPONGE -mdin mdin_forceonly_10000_notimer.spg.toml
```

After the rolling coverage fix, this default active-view fast path passed 10000
steps with finite values. One representative run:

```text
step=10000 temperature=294.73
potential=3210603.75
LJ=78177.96
PM=3132425.50
```

Additional diagnostic A/B results before cleanup:

- active-view fast with `ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0`: passed 10000.
- active-view fast disabled: passed 10000.
- native clustered: passed 10000.
- original default active-view fast before the fix: failed after a few thousand
  steps via real close contacts / huge pair forces.

Post-cleanup status:

- full build passed.
- a fresh 10000-step runtime validation after cleanup has not yet been rerun.
  The cleanup restored only diagnostic files and kept the two actual fixes above,
  so this should be the first command to run next if final validation is needed.

## Performance follow-up plan

Performance tuning was paused by request before a full optimization pass. The
numbers below are therefore diagnostic orientation, not final benchmark results.
They should be interpreted together with the original Phase A handoff goal:
recover the active-view gmxpacked long-run speedup after fixing the correctness
bug, without reintroducing undercovered rolling-source reuse.

Observed after GRO velocity loading became correct:

- default active-view + subgroup + rolling source cache: about `41-42 ns/day`.
- same but `ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0`: about `45 ns/day`.
- active-view with subgroup off: about `34 ns/day`.
- non-active-view gmxpacked: about `29 ns/day`.

Important interpretation:

- The old `>70 ns/day` number is not yet comparable to the corrected path.
  It may have depended on two invalid conditions: the previous undercovered
  rolling-source reuse and/or the old GRO loader not reading velocities, which
  reduced displacement and active-refresh pressure.
- The corrected path is slower because it now rejects unsafe outer-source reuse
  and because the 300 K trajectory triggers frequent active-view refreshes.
- Do not restore the old coverage check to recover speed. The invariant must
  remain:

```text
active_target_cutoff + 2 * anchor_max_displacement <= outer_source_cutoff
```

The last trace with correct velocities showed the active-view cost pattern:

```text
2000-step trace:
full builds = 4
inner active reuse = 1837
inner active refresh = 160
```

`inner active refresh` fires roughly every 12-13 steps with default
`ACTIVE_VIEW_REFRESH_FRACTION=0.4`. Many refresh lines had:

```text
dirty_source_rows=0
changed_source_rows=0
compact_patch=1
active_cutoff=9.850000
```

That means a high-priority target is avoiding unnecessary active compact rebuild
work when the active source mask did not change.

### Performance Work Order

1. Establish a clean correctness/performance baseline.

   - Use the cleaned tree, no diagnostics, correct GRO velocities.
   - Run the default active-view + subgroup command at 10000 steps.
   - Record both finite physics values and `Core Run Speed`.
   - Repeat once to separate normal run-to-run jitter from real changes.

2. Profile active-view refresh cost, not the already validated subgroup kernels.

   - Use `SPONGE_CLUSTERED_TRACE_WARP_RECORDS=1` only for short counting runs.
   - For profiling, use NCU on the active refresh build/compact kernels called
     after `inner active refresh`, especially aggregate build and compact pack.
   - Compare a 2000-step trace with `ACTIVE_VIEW_ROLLING_SOURCE_CACHE=0` against
     default rolling cache to quantify whether rolling is currently net negative.

3. First optimization target: zero-dirty active refresh.

   If `dirty_source_rows=0` and `changed_source_rows=0`, the source set is
   unchanged. Current code can still rebuild aggregates/compact payload and
   refresh active anchor. Investigate whether this can become a pure reuse path:

   - keep existing `gmxpacked_sci/cj/excl` compact payload;
   - refresh pair-shift metadata only if needed;
   - refresh active anchor only if required by the reuse policy;
   - preserve the corrected coverage invariant before returning.

   This is the safest likely win because it removes work from a very frequent
   path without changing pair enumeration.

4. Second optimization target: active refresh threshold/guard tuning.

   Test `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_REFRESH_FRACTION` and guard
   margin only after the zero-dirty reuse path is understood. A run with
   `ACTIVE_VIEW_REFRESH_FRACTION=0.5` was started but intentionally interrupted
   when performance work was paused, so there is no result for it.

   Any tuning must validate:

   - 10000-step finite run;
   - no `source_limit < active_cutoff` condition in trace;
   - no hidden reintroduction of the old undercoverage.

5. Third optimization target: reduce active compact cost.

   If zero-dirty reuse is not enough, profile and optimize:

   - `Build_Gmxpacked_Record_Stream_Aggregates`;
   - `Build_Gmxpacked_Record_Stream_Compact_Payload`;
   - active source-mask generation and dirty-source detection.

   Only touch subgroup count/fill again if NCU shows they remain a dominant cost
   after the active refresh frequency/cost problem is solved.

### Performance Guardrails

- Do not add PME/direct `dr2 != 0` guards as a performance or correctness fix for
  this issue.
- Do not disable active-view fast kernel as the final answer; it is only a
  diagnostic control.
- Do not compare corrected-velocity runs against old no-velocity runs as if they
  were the same benchmark.
- Keep performance and correctness validation separate: short traces for counts,
  clean no-diagnostic runs for ns/day.

## Current worktree

Expected tracked modifications:

```text
M SPONGE/Lennard_Jones_force/clustered_lj.cpp
M SPONGE/xponge/load/gromacs.hpp
```

Known untracked files that predate this cleanup/follow-up and were not touched:

```text
SPONGE/2026-06-28-174033-builderlj-kernelgmxp.txt
docs/gmxpacked-payload-shrink-status-20260626.md
docs/gmxpacked-project-handoff-20260626.md
```

This new handoff file is also untracked until added/committed.

## Suggested next steps

1. Run one clean 10000-step validation after this cleanup:

```sh
cd /tmp/sponge-dirty-overwrite-case
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1 \
  /home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_10000_notimer.spg.toml
```

2. Review the two source diffs.
3. Commit the two fixes plus this handoff if accepted.
4. When performance work resumes, start with the zero-dirty active refresh reuse
   path described above, then remeasure the clean 10000-step baseline.
