# Phase A subgroup kernel — design notes (2026-06-26)

## Goal

Cut the per-rebuild builder cost `R` (~0.7 s, ~90% of builder time, fires every
~1590 steps) by speeding up the shared candidate-leaf traversal used by both the
count kernel (`Count_Nbnxm_Payload_From_Candidate_Leaves`, ~286 ms) and the fill
kernel (`Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves`, ~389 ms).

## Why the current kernel is slow (NCU, measured)

Both kernels are **latency-bound and under-subscribed**, not compute/bandwidth
bound:

- Compute(SM) 6.5%/7.5%, DRAM 0.4%/0.5%.
- 0.72 occupancy-waves: 2571 candidate SCIs = 2571 warps (1 warp/SCI) on 128 SMs.
- 33% achieved occupancy; 91.8% of cycles have no eligible warp; 0.08 inst/cyc.
- 2.12/32 active lanes per executed instruction (only ~8 i-cluster lanes used).
- Each warp runs a ~6441-iteration serial inner j-cluster loop.

`__launch_bounds__` to raise occupancy did nothing (achieved occupancy stayed
32.96%) — the limit is too-few, too-long, low-ILP warps, not registers.

## Work-distribution facts (read-only probes)

- Per-SCI work is **near-uniform**, not tail-heavy: raw_leaf_clusters max/mean
  1.33, accepted_records max/mean 2.01. So no dynamic load-balancing is needed;
  a static even split is correct.
- leaves/SCI ≈ 4241; **cluster_j/leaf ≈ 1.52** → cannot split a leaf across
  subgroups; the parallel dimension is **leaves**, not cluster_j within a leaf.

## The dedup is NOT serial (key enabling discovery)

The inner loop dedups overlapping cluster_j across a candidate-SCI's leaves via a
serial running max `processed_cluster_end`, then
`deduped_start = max(cluster_j_start, processed_cluster_end)`. This removes 52.4%
of cluster_j visits, so it cannot be dropped.

**Leaf-order probe result: candidate leaves are PERFECTLY cluster-start-sorted
within each SCI — 0 non-monotonic out of 10,903,362 leaves.** Therefore the
running max equals *the previous leaf's* `cluster_j_end`:

```
deduped_start[k] = max(cluster_j_start[k], cluster_j_end[k-1])
```

This is a **pure local function of leaf k and leaf k-1** — no scan over all prior
leaves. So the dedup can be parallelized bit-exactly: any subgroup/lane assigned
leaf k computes its own deduped_start from leaf k-1's end, with zero cross-leaf
serial state and zero double-counting. Counts are order-independent sums, so a
correct reorder is provably bit-identical to the baseline.

## Chosen design

- Keep **1 warp per candidate-SCI** grid mapping (preserves the shared i-cluster
  data load, which is reused by all of the SCI's leaves).
- Split the warp into **4× 8-lane subgroups** (8 lanes = the 8 i-clusters, the
  natural inner width). Each subgroup processes a **distinct stride of the SCI's
  leaf list** (leaf indices `s, s+4, s+8, …` for subgroup `s`).
- Each subgroup computes its own `deduped_start[k] = max(start[k], end[k-1])`
  using the immediately preceding leaf in the *full* list (read directly from
  `leaf_cluster_ends[candidate_leaf_ids[idx-1]]`), so dedup stays bit-exact
  without serial warp state.
- Per-subgroup shift-count / exclusion-count / source-count accumulators in
  shared memory, reduced across the 4 subgroups at the end before the single
  per-(sci,shift) write.
- i-cluster shared data (`shared_i_*`) stays per-warp (read-only, shared by all 4
  subgroups). Only j-side scratch + counters get a `[subgroup]` dimension
  (+~4.5 KB shared/block, within Ada's budget).

This adds 4 independent instruction streams per warp → directly attacks the
91.8% no-eligible stall, and raises active-lane utilization, while leaving the
source/active-payload contract untouched (no NaN/coverage risk).

## Validation protocol (mandatory, in order)

1. **Bit-exact compare** (`SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1`):
   subgroup count kernel writes separate scratch; diff the 3 output arrays
   (sci_shift_flags, cjpacked_counts, exclusion_counts) against baseline. MUST be
   0 mismatches before the kernel is ever consumed. The scaffold is validated
   (verbatim clone → 0 mismatches on 69417 shifts).
2. NCU reprofile: expect inst/cyc up, no-eligible down, waves > 1, duration down.
3. Only if exact AND faster: enable consumption
   (`SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER=1`), run 120-step physics gate,
   1000-step + 10000-step no-timer (no NaN, speed > 64 ns/day baseline).
4. Mirror the identical transformation into the fill kernel (its writes are
   position-based, so it additionally needs the per-subgroup write cursors to
   land in the same order — verified by the existing source-row/payload compare).

## Status

- Env gates added: `SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER[_VERIFY]`.
- Subgroup count kernel exists as a verbatim clone (compile + compare scaffold
  validated). The inner-loop subgroup transform is the next edit.
- Probes (work-dist, leaf-order) are read-only and will be reverted before final.
