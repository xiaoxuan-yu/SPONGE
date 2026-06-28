# Phase A work-distribution probe — diagnosis (2026-06-26)

## Why this probe

The Phase A occupancy experiment (`__launch_bounds__(128,8)`) proved achieved
occupancy is *not* register-limited: cutting regs 72→64 raised theoretical
occupancy 58%→67% but left achieved occupancy byte-identical at 32.96%. The two
candidate explanations left were:

- **(H1) work imbalance** — a long tail of heavy candidate-SCIs serializes the
  grid (warp-per-SCI, so one fat SCI stalls its scheduler);
- **(H2) coarse/insufficient parallelism** — the grid is too few, too-long warps.

This probe measures the per-candidate-SCI work distribution to decide between
them. It is **read-only and host-side**: it piggybacks on the already-safe,
already-gated `Trace_Clustered_Builder_Stats` (env `SPONGE_CLUSTERED_TRACE_WARP_RECORDS`),
which already host-replicates the count traversal and copies the same buffers.
No device kernel, no builder scratch reuse, no payload mutation.

Run:

```sh
env SPONGE_CLUSTERED_DISABLE_FINE_TIMERS=1 SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1 \
  SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 \
  SPONGE_CLUSTERED_TRACE_WARP_RECORDS=1 \
  SPONGE -mdin mdin_active_view_20.spg.toml
```

## Measured distribution (step-0 full build, wat160k, 2571 candidate SCIs)

| proxy | mean | p50 | p90 | p99 | p99.9 | max | max/mean | top-1% of work | SCIs holding 50% |
|---|---|---|---|---|---|---|---|---|---|
| raw_leaf_clusters (inner traversal iters) | 6441 | 6418 | 6854 | 7641 | 8119 | 8554 | **1.33** | 1.2% | 1227 (47.7%) |
| accepted_records (output work) | 845 | 846 | 1335 | 1581 | 1681 | 1695 | **2.01** | 1.9% | 871 (33.9%) |

## Verdict: H1 rejected, H2 confirmed

**The work is near-uniform, not tail-heavy.** max/mean is only 1.33 (traversal)
and 2.01 (output); ~half of all SCIs are needed to hold 50% of the work; the
busiest 1% hold <2%. So achieved occupancy is **not** capped by a heavy-SCI tail.
Imbalance-targeted designs (work-stealing, per-SCI binning, dynamic scheduling)
would not help.

**The grid is too few, too-long warps (H2).** Reconciled against RTX 4090
(CC 8.9, 48 warps/SM max, 4 schedulers):

- 2571 candidate SCIs = 2571 warps (1 warp per SCI), 643 blocks.
- Theoretical occupancy 58% (register-limited, 72 reg/thr) → GPU can hold ~3582
  warps resident → the **entire grid is 0.72 waves** (matches NCU "Waves Per SM
  0.72"). The GPU never fills once.
- Each warp runs a **long serial inner loop**: ~6441 j-cluster iterations on
  average. NCU's dominant stall is instruction-fetch / branch-resolve in exactly
  this big branchy loop, with 91.8% of cycles having no eligible warp.

## Implication for the kernel refactor

The lever is **finer-grain parallelism per candidate-SCI**: split each SCI's
inner j-cluster traversal across more execution resources (e.g. across the 32
lanes, or across multiple warps/blocks per SCI), turning 2571 long-serial warps
into many shorter ones. This simultaneously:

1. raises the wave count well above 1 (fills the GPU, hides latency), and
2. shortens the per-warp serial inner loop that produces the fetch/branch stalls.

Because the work is uniform (max/mean 1.33), a *static* even split is sufficient
— no dynamic load balancing needed. This applies to **both** the count and fill
kernels (same traversal), so it attacks the full ~0.7 s/rebuild `R`, and it does
**not** change the source/active-payload contract (unlike the rejected one-pass
fusion), so it carries no NaN/coverage risk.

This is the recommended next implementation step and overlaps the in-progress
manual builder/LJ-kernel refactor.

## Probe status

The probe code is an additive block inside `Trace_Clustered_Builder_Stats`
(emits `[clustered builder work-dist]` lines). It was reverted from the worktree
after measurement to keep the source clean; re-apply from this doc if the
distribution needs re-checking on a different system (wat600k) or after the
refactor.
