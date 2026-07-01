# gmxpacked source fragment/source fill optimization plan - 2026-07-01

This plan is for the next optimization step after the env-gated count-kernel
parallel accumulator. It is intentionally separate from the rolling source cache
correctness fix.

## Current evidence

- Count kernel optimization is effective on the fixed-shift candidate-leaf path:
  `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true>` dropped from about
  `118 ms/launch` to about `49 ms/launch` in the stable 10000-step nsys run.
- After that change, the dominant full-rebuild stage is source fill:
  `Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup` costs
  about `164 ms/launch`.
- Stable cache-off profiling showed 30-31 full rebuilds over 10000 steps, so the
  source fill stage can exceed 5 seconds total even after the count kernel is
  reduced.
- The count path still leaves source fragment/source row emission in a mostly
  leader-lane serialized form. That was kept deliberately to preserve fragment
  order and overflow behavior for the first count-kernel change.

## Goal

Reduce the `Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup`
cost without changing:

- `LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE` row count.
- Source row fields.
- Per-candidate source row order.
- Overflow semantics.
- Fill-prune-reuse-light consumer behavior.
- Verify path count/fill mismatch guarantees.

## Proposed gated experiment

Add a new default-off gate:

```sh
SPONGE_CLUSTERED_GMXPACKED_SOURCE_FILL_PARALLEL=1
```

Only enable it for the same narrow path as count parallel accum:

- subgroup builder enabled;
- fixed-shift candidate path;
- candidate leaf reach masks available;
- source offsets by candidate available.

Fallback to the existing fill kernel for every other path.

## Implementation sketch

1. Keep the existing per-candidate source offsets as the ordering contract.
2. During fill, let sublanes compute split-local `source_imask` in parallel.
3. Use subgroup ballot/popc to derive each split's local emission rank.
4. Write source rows directly to:

   ```text
   record_stream_source_offsets_by_candidate[candidate_sci] + local_rank
   ```

5. Preserve split order by ranking over split ids, not over arbitrary active
   lanes.
6. Keep the existing overflow counter and debug trace path in the fallback
   kernel until the parallel writer is proven equivalent.

## Correctness gates

Run all gates with and without `SPONGE_CLUSTERED_GMXPACKED_COUNT_PARALLEL_ACCUM=1`.

1. Build:

   ```sh
   pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE --parallel 4
   ```

2. 2000-step finite run on the fixed-shift path.
3. 2000-step verify run:

   ```sh
   SPONGE_CLUSTERED_GMXPACKED_SUBGROUP_BUILDER_VERIFY=1
   ```

   Required result:

   - count `flag/cj/excl mismatch=0`;
   - fill `row_count_mismatch=0 field_mismatch=0`;
   - no source overflow.

4. 10000-step finite run for the accepted rolling-source-cache configuration.

## Performance gates

Use nsys from the pixi Nsight Compute tree:

```sh
.pixi/envs/dev-cuda13/nsight-compute-2025.3.1/host/target-linux-x64/nsys
```

Compare source-fill flag off/on on the same 10000-step case and same rolling
source cache setting.

Acceptance:

- `Fill_Gmxpacked_Record_Stream_Sources_From_Candidate_Leaves_Subgroup` drops at
  least 20%.
- `Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup` does not regress more
  than 5%.
- Other non-source-fill kernels do not increase by more than 1% total.
- End-to-end `Core Run Speed` improves on a finite 10000-step clean run.

## If this fails

Do not broaden the experiment by default. First inspect whether source fill is
still dominated by per-source distance pruning or by serialized writeback. If
writeback is not the dominant cost, prefer fixed-path specialization of count and
fill kernels before changing the source row format.
