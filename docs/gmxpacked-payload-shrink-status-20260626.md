# Gmxpacked Payload Shrink Status, 2026-06-26

## Scope

本文件记录 `/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked`
在 `opt/gmxpacked-payload-shrink-sweep` 分支上完成本轮 source-raw /
source-cache refresh 探索后的状态。

目标仍是 wat160k force-only 长跑下，将 gmxpacked active-view 路径推进到
`>=103.38 ns/day`。当前可保留的状态是：默认 active-view 路径稳定，上一轮两个
source-raw 原型均为负结果，未进入代码基线。

## Current Baseline

### Source State

- Branch: `opt/gmxpacked-payload-shrink-sweep`
- HEAD at validation time: `c2fd374 Add gmxpacked source-cache refresh probe`
- Before writing this document, code worktree was clean.
- Negative source-code experiments were rolled back and not committed.

### Validated Runtime Baseline

Build:

```sh
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
```

Default active-view 1000-step sanity:

```sh
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  /home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_1000_notimer.spg.toml
```

Result:

- `Core Run Speed: 54.712196 ns/day`
- No NaN observed.

Default active-view 10000-step no-timer:

```sh
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 \
  SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer \
  SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  /home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/build-dev-cuda13/SPONGE \
  -mdin mdin_forceonly_10000_notimer.spg.toml
```

Result:

- `Core Run Speed: 64.029243 ns/day`
- No NaN observed.
- This preserves the previous `~60 ns/day` active-view baseline.

## Negative Results From This Round

### 1. Source-Cache Patch Prototype

Attempted direction:

- Keep the previous source cache.
- Replace source rows for dirty candidate SCI.
- Recompute source offsets.
- Refresh the outer source anchor after the patch.

Observed result:

- 2000-step guard=1 opt-in run ended with NaN.
- The patch reported `used=1` 57 times and `used=0` 14 times.
- Full `primary-source-offset-count-scan` still appeared 81 times.
- Full `record-stream-source-row-generation` still appeared 81 times.
- It did not eliminate the later full source-row generation/count-scan work.

Saved negative diff:

```text
/tmp/gmxpacked_source_cache_patch_negative.diff
```

Conclusion:

This implementation is unsafe. It can mark an incomplete source cache as valid
for a refreshed outer source anchor.

### 2. Shell Probe / Min-Cutoff Source-Row Prototype

Attempted direction:

- Add a source-row min-cutoff / shell-count capability.
- Use it as an opt-in diagnostic for source-cache miss / primary builder points.
- Do not modify the production payload.

Observed result:

- The first narrow trigger did not produce useful probe output in 2000-step runs.
- A broader trigger caused a 2000-step opt-in run to end with NaN.
- No trustworthy shell-row measurements were produced.

Saved negative diff:

```text
/tmp/gmxpacked_shell_probe_negative.diff
```

Conclusion:

This prototype is not clean enough to keep. Even a diagnostic-only source-row
kernel call can perturb state if scratch flags/counters or builder side effects
are not isolated perfectly.

## Root Cause Learned

The rejected `source-cache patch` failed because source-cache validity is tied to
the outer source anchor.

Old clean source rows are complete only for the old anchor and old coverage
window. If the code refreshes the outer source anchor after copying old clean
rows, the cache is now advertised as complete for the new anchor even though
newly entering shell pairs may be missing. Later active-view refreshes then
operate on an under-covered source cache and can miss force contributions.

This is different from an ordinary dirty-SCI refresh:

- Dirty-SCI replacement can update changed rows within an already-covered source
  cache.
- Rolling the source anchor changes the coverage contract itself.
- Safe rolling update must either preserve the old anchor until a full rebuild,
  or explicitly add/repair the newly required source shell before moving the
  anchor.

## Design Boundary Going Forward

Do not merge or revive a source-cache refresh that:

- Reuses clean old source rows after refreshing the source anchor without a
  coverage proof.
- Replaces only dirty SCI rows while ignoring newly required outer shell rows.
- Appends source rows at cluster/split granularity without preventing duplicate
  pair work.
- Uses `stable-source-reuse` or `mixed-source-build` as a default path.

Any source-raw payload shrink needs a stronger contract:

1. Source-cache anchor and source-row coverage must be explicit metadata.
2. A rolling update must identify both dirty SCI rows and newly required shell
   rows.
3. If shell rows are appended, pair-level duplication must be prevented. A source
   row is not an atom-pair row; appending a whole row can duplicate work already
   present in the old cache.
4. The update must fall back to full rebuild when dirty/shell work approaches
   full payload scale.

## Recommended Next Step

The next useful step is attribution, not another source-row patch:

1. Capture when clean default 10000-step still enters full
   `primary-source-offset-count-scan` and `record-stream-source-row-generation`.
2. For each full rebuild event, log:
   - reason / caller path;
   - cached source anchor displacement;
   - previous source cutoff;
   - requested active cutoff;
   - whether the miss is source-cache coverage, active-mask guard, compact reuse,
     candidate layout change, or another fallback.
3. Keep the diagnostic strictly host-side or read-only with no device-builder
   scratch reuse until it is proven not to perturb the simulation.
4. Only after that attribution should the source-raw update be redesigned.

The promising production direction remains:

- preserve outer coverage safety;
- shrink runtime active payload toward the cutoff-scale shape;
- avoid repeated full source-row generation/count-scan;
- keep the direct kernel payload layout stable unless NCU evidence indicates a
  kernel-level bottleneck.

