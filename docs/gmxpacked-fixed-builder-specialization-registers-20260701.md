# gmxpacked fixed-builder specialization register review - 2026-07-01

This note records why the first fixed-shift builder specialization raised
register pressure, and what to change next.

## Scope

The current specialized path is gated by:

```sh
SPONGE_CLUSTERED_GMXPACKED_FIXED_SHIFT_BUILDER_SPECIALIZED=1
```

It is only active on the current peak fixed-shift path:

- gmxpacked direct;
- outer lifecycle;
- active view;
- rolling source cache;
- fixed-shift leaf screening;
- one-pass candidate leaves;
- subgroup builder;
- fill-prune-reuse-light;
- count parallel accum;
- count fragment parallel emit.

## Resource evidence

`cuobjdump --dump-resource-usage build-dev-cuda13/SPONGE` showed the runtime
architecture instance used by the 4090:

| count subgroup instance | registers/thread | stack | shared |
|---|---:|---:|---:|
| `<true,true,false>` | `150` | `64 B` | `6384 B` |
| `<true,true,true>` | `160` | `0 B` | `6000 B` |

NCU single-launch data matched the same direction:

| metric | `<true,true,false>` | `<true,true,true>` |
|---|---:|---:|
| duration | `87.52 ms` | `80.06 ms` |
| registers/thread | `150` | `160` |
| active threads/warp | `4.37` | `4.62` |
| not-predicated threads/warp | `4.00` | `4.22` |
| eligible warps/scheduler | `0.13` | `0.13` |
| local spilling | `0` | `0` |

The register increase is therefore real, but it is not a local-spilling
regression. The specialized instance trades the old `64 B` stack frame for more
scalar registers.

## Interpretation

The first specialization only added a third template boolean to the existing
generic count kernel. That removed some runtime branches, but it did not remove
the generic kernel shape.

Two parts still keep unnecessary state live in the specialized instance:

1. **Generic shift-group loop.**

   On the fixed-shift + reach-mask path all active i lanes for a candidate group
   share the same `fixed_shift_id`, but the kernel still builds:

   - `pair_shift_id`;
   - `remaining_lane_mask`;
   - `leader_sublane`;
   - `group_shift_id`;
   - `group_lane_mask_local`;
   - `group_record_imask`;
   - the `while (remaining_lane_mask != 0)` grouping loop.

   In the specialized path this loop should execute at most once, so the generic
   grouping machinery is mostly register pressure and control-flow overhead.

2. **Runtime fragment-mode fallback paths.**

   The gate guarantees fill-prune-reuse-light and light count fragments, but the
   templated kernel still compiles full-source-fragment and no-fragment paths:

   - `emit_count_source_fragments`;
   - `emit_count_light_source_fragments`;
   - `emit_any_count_source_fragments`;
   - `!use_parallel_fragment_emit`;
   - full `LJ_CLUSTERED_GMXPACKED_RECORD_STREAM_SOURCE` emission.

   ptxas cannot infer those runtime pointer/capacity invariants from the call
   site, so it preserves more live state than this path needs.

## Next implementation steps

Do these in order and keep the env gate default-off:

1. **Single fixed-shift group path.**

   For `kFixedShiftLeafScreenedSpecialized`, bypass the generic shift-group
   `while` and process one group directly:

   ```text
   group_record_imask = precomputed_i_mask
   source_shift_id = fixed_shift_id
   output_shift_idx = 0
   ```

   This should shorten the live ranges of `pair_shift_id`, `group_shift_id`,
   and `remaining_lane_mask`.

2. **Compile-time light-fragment-only assumption.**

   For the specialized path, compile only the light count-fragment emission
   mode. Leave full-source-fragment and no-fragment behavior in the generic
   fallback kernel.

3. **Narrow dedicated specialized kernel signature.**

   If step 1 and 2 are still not enough, split a dedicated fixed-shift screened
   light-fragment count kernel with a smaller parameter list. It can drop
   generic-only inputs such as `candidate_shift_ids`, `fixed_shift_candidates`,
   `cluster_extents`, `cluster_radii`, and `super_cluster_centers`, subject to
   compile verification.

Acceptance for this round:

- build succeeds;
- 2000-step verify still reports zero count/fill mismatches;
- specialized resource usage should move toward `<=150` registers/thread;
- no local spilling;
- count-kernel duration should not regress beyond the previous specialized
  `80.06 ms` NCU anchor unless the full peak path improves end-to-end.
