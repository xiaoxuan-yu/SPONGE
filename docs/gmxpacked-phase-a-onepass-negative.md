# Phase A: one-pass count+fill fusion — negative result (2026-06-26)

## Goal of Phase A

Reduce the per-rebuild builder cost `R` (~0.706 s, measured). Attribution
established that the full builder rebuild fires only every ~1590 steps but
accounts for ~90% of builder time, and that if `R` were zero the 10000-step
wat160k force-only run would reach ~101 ns/day (≈ the 2× native target of
103.38). Current baseline: 1000-step 55.3 ns/day, 10000-step 64–65 ns/day.

The rebuild cost splits into two device kernels that run the **same**
candidate-leaf traversal twice:

- `Count_Nbnxm_Payload_From_Candidate_Leaves` — `primary-source-offset-count-scan`, ~286 ms
- `Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves` — `record-stream-source-row-generation`, ~389 ms

## Sub-approach tried: enable the existing one-pass source cache

`Clustered_Gmxpacked_Record_Builder_One_Pass_Source_Cache_Enabled()` was
hard-coded `false` since the baseline-cleanup commit `26ef7e5`. It gates an
existing (scaffolded) path that:

- skips the count kernel entirely (`defer_source_count` → `skip-native-count`,
  `cjpacked_numbers = 0` at that point);
- fills source rows in one pass using an estimated capacity + overflow retry
  (`Estimate_Gmxpacked_Primary_One_Pass_Source_Capacity`);
- derives source offsets *after* the fill by sorting
  (`record-stream-one-pass-source-{low,high}-sort`, `-gather`);
- builds the compact payload from the filled/sorted rows.

It was temporarily re-enabled behind an env gate
(`SPONGE_CLUSTERED_GMXPACKED_ONE_PASS_SOURCE_CACHE`) and measured.

## Result: net regression + NaN — rejected

| run | baseline (count+fill) | one-pass | verdict |
|---|---|---|---|
| 200-step stage-timer | count-scan present | **count-scan removed** ✓ | fusion itself works |
| 1000-step no-timer | 55.3 ns/day | **31.99 ns/day** | regression |
| 10000-step no-timer | 64–65 ns/day | **2.84 ns/day, 36 NaN lines** | unsafe |

### Why it regresses

The one-pass source layout is sorted *after* the fill, so it does not match the
counted/offset layout that the per-step **inner-active payload reuse** relies on.
Evidence from the 200-step stage timers:

- `record-stream-device-compact-pack` and `record-stream-inner-active-fill`
  fired **10×** (steps 0, 25, 46, 67, 91, 109, 129, 152, 172, 193) under
  one-pass, versus **1×** in the baseline window.
- i.e. the expensive compact rebuild now recurs every ~21 steps (the anchor
  reuse-limit cadence) instead of once per ~1590-step source rebuild.

The accumulated cost of those repeated compact rebuilds far exceeds the single
286 ms count-scan that was removed. On the 10000-step run the same contract
mismatch additionally produces NaN around rebuild boundaries — the same failure
family documented for the rolling source-cache prototypes.

### Why it could not have reached the target anyway

Even a *perfect* one-pass (count-scan removed, reuse preserved) only removes the
286 ms count-scan, not the 389 ms `record-stream-source-row-generation`
traversal (which still fires at step 0). Ceiling estimate:

```
10000 steps, 7 rebuilds:
  baseline       wall 13.49 s -> 64.0 ns/day
  one-pass IDEAL wall 11.49 s -> 75.2 ns/day   (+~15%, reuse NOT broken)
```

So the fusion sub-approach has a low ceiling (~75 ns/day) and a fragile contract.

## Action taken

- Reverted `One_Pass_Source_Cache_Enabled()` to hard `false` with an explanatory
  comment. No env gate is left reachable (matches the repo convention of rolling
  back NaN-producing prototypes rather than leaving them opt-in).
- Source worktree restored to baseline behavior.

## Conclusion / next lever

The real lever for `R` is **not** fusion but making the shared count/fill
candidate-leaf traversal kernel itself faster. NCU (now confirmed available at
`.pixi/envs/dev-cuda13/bin/ncu`, Nsight Compute 2025.3.1) shows both kernels are
**latency-bound and under-subscribed**: Compute(SM) 6.5%/7.5%, DRAM 0.4%/0.5%,
0.72 waves/SM, 33% achieved occupancy (72 reg/thread), 91.8% of cycles with no
eligible warp; dominant stall is instruction-fetch/branch-resolve in the large
branchy warp-per-candidate-SCI traversal. Speeding up that one kernel cuts
**both** the 286 ms and 389 ms passes and does not touch the source/active
payload contract (no reuse breakage, no NaN risk). That is the next Phase A
target.
