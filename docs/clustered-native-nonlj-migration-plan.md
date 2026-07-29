# Clustered-Native Non-LJ Consumer Migration Plan

## Status and decision rule

**The regular-LJ Phase 0 performance baseline is passed.** The cleanup
regressions were repaired and the final alternating benchmark/NCU matrix
qualifies the cleaned regular-LJ production path as the frozen comparison
reference. See
`clustered-cleanup-performance-recheck-20260725.md`.

This does not declare the complete LJ/soft-LJ migration reference ready.
Phase 1A remains blocked by the pair-image ownership gate and the dedicated
soft-LJ full/output and end-to-end performance gate described below. No
non-LJ clustered path becomes the production default, and no legacy-list
reason is removed, until both close. An explicit opt-in implementation may be
developed and differentially validated meanwhile; it must fail closed rather
than fall back silently.

The first implementation unit is deliberately narrow:

1. extract a read-only, LJ-independent clustered spatial view without changing
   the builder or pair set;
2. add a canonical pair-set oracle before changing any consumer;
3. rewire regular LJ and soft-LJ through that view as the no-behavior-change
   proof;
4. migrate custom/JIT pairwise as the first non-LJ consumer, on GPU and CPU;
5. remove only the custom-pairwise reason for building the legacy half list
   after its remaining correctness, DD and performance gates pass.

GPU minimization, full-output numerical comparison and the retained regular-LJ
microbench have passed. They do not waive the remaining pair-image ownership
and soft-LJ gates or reopen the frozen regular-LJ performance decision.

This plan covers algorithm-level migration of non-LJ spatial-neighbor
consumers to the clustered neighbor representation. It does not make
`ATOM_GROUP`, a derived half list, or a derived full adjacency list the final
architecture.

The migration must not reintroduce a production adapter that materializes
`ATOM_GROUP` from clustered records. Temporary legacy execution may remain as
an explicit differential oracle until each consumer passes its removal gate,
but it must never be an automatic success path for a failed clustered-native
implementation.

### Decision checkpoint - derive only structure or operator state

The production migration is **direct clustered algorithm refactoring**, not a
clustered-to-legacy table conversion. This is not a requirement for every
kernel to scan unindexed builder arrays. The allowed derivation boundary is:

1. the shared spatial service may derive generation-matched **O(cluster)**,
   **O(SCI)** or **O(CJ)** representation sidecars, such as fractional
   cluster geometry, grouped SCI IDs, SCI-safety flags and packed pair-shift
   words, provided they do not expand masks into accepted atom pairs or expose
   a second neighbor-list API;
2. pair and many-body operators consume the authoritative SCI/CJ payload
   through pair-tile or grouped-center traversal;
3. an operator may derive an **operator-owned accepted relation** only after
   spatial candidates acquire a new mathematical identity and lifetime, such
   as ReaxFF reaction edges or the EEQ sparse matrix;
4. half/full `ATOM_GROUP`, generic pair CSR and per-atom spatial adjacency are
   test-only materializations, never a production migration stage.

This gives the implementation decision for the current consumers:

| consumer class | first production path | permitted derived data |
|---|---|---|
| custom/JIT pairwise | direct pair-tile traversal | sorted consumer fields and JIT constants |
| SITS | first non-LJ mainline-backport consumer; direct pair-tile traversal | marks, scaling fields and operator accumulators |
| EAM pair stages | direct pair-tile traversal | mathematical per-atom intermediates |
| SW, EDIP and Tersoff center-neighbor stages | wait for the grouped-center cursor, then refactor directly | O(SCI) grouping; operator scratch only |
| ReaxFF spatial stages | direct candidate traversal followed by operator compaction | canonical reaction edges, bond-order graph and EEQ CSR |
| Plugin/PRIPS compatibility | versioned clustered capability or explicit retirement | plugin-owned state; no SPONGE compatibility list |

REST/REST2 are explicitly out of scope for this mainline backport. Their
legacy implementation must not gate, dispatch or alter SITS validation.

Consequently, a production derived full list is not an acceptable temporary
bridge for SW, EDIP or Tersoff. Those migrations remain disabled until the
grouped-center contract and their pair/triplet oracles exist. A test-only
materialized reference is still allowed for differential correctness and for
measuring the complete build-plus-consume cost against the direct algorithm.

### Consumer decision audit - direct refactor versus derived table

The decision is made from the lifetime and mathematical meaning of the
relation, not from how the legacy kernel happens to receive it:

1. if every accepted spatial pair can be evaluated independently, refactor the
   operator to consume pair tiles directly;
2. if an operator repeatedly enumerates neighbors of one center but the
   relation is still only geometric, add a center-complete structural cursor
   and refactor the algorithm around tiled scans;
3. if a candidate becomes a bond, matrix entry or other operator-defined
   relation used by later stages, compact that relation into operator-owned
   state;
4. if an external ABI exposes `ATOM_GROUP`, either add a versioned clustered
   capability or retire that ABI. Do not keep a hidden compatibility list
   indefinitely.

#### Audited choice - keep the old implementation or refactor directly

The source audit makes the transition choice more precise. There are three
different actions that must not be conflated:

1. an unmigrated consumer may continue to run its existing grid/full-list
   implementation as an explicitly selected legacy implementation;
2. a migrated consumer refactors its algorithm to consume the authoritative
   clustered payload;
3. the shared provider may derive bounded structural sidecars, but it does not
   manufacture a replacement `ATOM_GROUP`.

There is therefore no production stage in which clustered records are
expanded into a generic half/full table and then passed to an unchanged
consumer. The current
`Ensure_Legacy_Neighbor_View_From_Clustered_Payload` path does not provide
such a bridge: after validating the request it still rejects the conversion
because the legacy half-list pair-set ownership proof is missing, and
`Main_Update_Legacy_Neighbor_List_If_Needed` falls back to the independent
grid builder. Treating that route as a migration step would preserve both
builders and both lifecycles while proving neither representation removable.

The production choice by access pattern is:

| access pattern | transition while unmigrated | first migrated implementation |
|---|---|---|
| independent pair evaluation | retain the explicitly selected legacy loop as a differential oracle | direct SCI/CJ pair-tile traversal |
| repeated pair-decomposable passes | retain the existing full-list implementation until all passes are ported | replay pair tiles and retain only mathematical per-atom intermediates |
| center-complete J/K or edge/K traversal | retain the existing full-list implementation until a center-complete cursor exists | direct grouped traversal over authoritative tiles |
| candidates becoming bonds or sparse-matrix entries | retain the existing operator graph builder | direct candidate traversal followed by operator-owned semantic compaction |
| external per-atom neighbor ABI | keep the ABI decision open explicitly | versioned clustered capability or explicit retirement |

This is a larger algorithm change for the center-neighbor consumers than a
table adapter, but it is the only route that removes the representation fork.
A temporary derived table remains useful only inside a test binary for
canonical pair/triplet comparison and for measuring the complete
materialize-plus-consume cost. It is not an intermediate production API.

This produces the following concrete decisions:

| consumer | current access pattern | production decision | allowed state |
|---|---|---|---|
| regular LJ and soft-LJ | independent accepted pairs | already direct SCI/CJ consumers | sorted atom fields and output scratch |
| custom/JIT pairwise | independent accepted pairs with runtime pair expression | direct SCI/CJ; the current clustered force-only/full kernels are the reference implementation | sorted custom fields and JIT constants |
| SITS | independent pairs plus per-atom marks and weighted outputs | first non-LJ mainline-backport consumer; direct pair-tile refactor | marks, scaling fields and operator accumulators; no REST/REST2 dependency |
| REST/REST2 | legacy implementation retained; excluded from this mainline backport | no clustered migration or dispatch coupling | legacy state must not affect SITS dispatch or acceptance |
| EAM density pass | pair contribution to per-atom density | direct pair-tile refactor | per-atom density |
| EAM embedding/force pass | pair contribution using both endpoint embedding derivatives | second direct pair-tile pass | embedding value/derivative and force/energy scratch |
| SW/EDIP | repeated `(j,k)` enumeration for a fixed center | direct grouped-center tiled refactor; do not publish a full atom adjacency | bounded CTA scratch or repeated J-tile scans |
| Tersoff | for each directed `(i,j)`, repeatedly scans `k != j` to form and differentiate zeta | direct grouped-center multi-pass refactor; do not publish a full atom adjacency | bounded CTA scratch and per-directed-pair zeta/intermediates |
| ReaxFF candidate stages | geometry first, then reaction-specific multi-stage graph | direct clustered candidate scan followed by semantic compaction | EEQ matrix CSR, canonical reaction/bond-order graph and its derivatives |
| solvent LJ fast path | legacy memory-locality specialization | retire when clustered regular LJ is active; do not port it | none |
| Plugin/PRIPS | external API exposes legacy neighbor counts and atom indices | versioned clustered capability or explicit retirement | plugin-owned state only |

For SW, EDIP and Tersoff, "grouped-center" must mean **center-complete**, not
merely the current native-i SCI grouping. The compact half-pair payload may
store a local-local interaction in only one orientation, while these operators
need every neighbor of a local center. The shared service may therefore build
an endpoint-incidence index over SCI/CJ tiles so a tile can be replayed in its
native or transposed orientation. That index is structural, generation keyed
and O(SCI/CJ); it must not expand cluster masks into accepted atom pairs.

The existing grouped-SCI offsets are only the native-i half of that contract.
They are sufficient for consumers whose ownership is already expressed by the
stored SCI orientation, but they are not by themselves a full-neighborhood
cursor. The center-complete index may contain bounded references of the form
`{sci_id, cj_id, orientation}` for the native and transposed tile endpoints.
It may index active i-cluster lanes within a tile with a fixed-size mask, but
it may not emit one row per accepted atom pair. Its key is the exact provider
incarnation, representation generation and grouping policy; geometry-derived
shift state additionally pins the geometry generation.

The corresponding algorithms are:

- SW/EDIP: select a local center lane, stream its J tiles, and evaluate the
  tiled Cartesian product of neighbor blocks. A block may be replayed instead
  of retaining an unbounded neighbor span.
- Tersoff: stream directed `(i,j)` tiles, rescan the same center-complete K
  cursor to obtain zeta, then rescan or retain bounded intermediates for the
  derivative pass.
- ReaxFF: use the same center-complete cursor only to discover candidates.
  Once a candidate passes bond-order or EEQ semantics, write the canonical
  operator relation because later stages consume that mathematical graph, not
  a spatial-neighbor cache.

Thus a temporary derived `ATOM_GROUP` can remain only as a differential oracle
while a consumer is being migrated. It is not an implementation milestone and
its build time must be included in every comparison. The production escape
hatch is operator-local tiled replay, not a generic half/full adjacency.

The audited implementation order is:

1. finish the custom-pairwise correctness/DD/full-performance gates using its
   direct two-variant kernel as the template;
2. migrate SITS first, then EAM, because they require no center-complete
   structural API; REST/REST2 remain out of scope and are neither a
   prerequisite nor an acceptance gate for SITS;
3. add and independently test the generation-keyed endpoint-incidence and
   center-complete cursor contract;
4. migrate SW/EDIP and then Tersoff with pair/triplet-set oracles and bounded
   scratch;
5. migrate ReaxFF candidate discovery while preserving its operator-owned EEQ
   and bond-order graphs;
6. version or retire Plugin/PRIPS neighbor access, then remove the final
   legacy-list build reasons.

The current Plugin/PRIPS API is a removal blocker: it exposes per-atom neighbor
counts and an atom-index pointer but is not represented in
`Main_Get_Legacy_Neighbor_List_Need`. Until the API is made explicit, a loaded
plugin can observe a skipped or stale legacy list. This must be fixed before
claiming that all legacy consumers have been retired.

#### Source-audit checkpoint: an input type is not an algorithm requirement

The 2026-07-27 source audit confirms that the migration must classify the
mathematics performed by each consumer, not copy the container in its current
kernel signature. `const ATOM_GROUP *nl` proves only that the legacy
implementation is table-shaped. It is not evidence that the production
replacement should derive the same table.

The audited decisions are:

| consumer | source evidence | clustered-native decision |
|---|---|---|
| SITS | `Selective_Interaction/SITS.cpp:5-95` applies pair-local marks, scaling, local/ghost ownership and output reductions inside one J loop | first consumer: lift those policies into the regular/soft pair-tile body; do not build a half `ATOM_GROUP` |
| REST/REST2 | `Selective_Interaction/REST2.cpp:38-127` remains a legacy path | excluded from this mainline backport; it must not participate in SITS dispatch or acceptance |
| EAM | `manybody/eam.cpp:34-86` computes density, `:89-110` computes the embedding derivative, and `:113-175` replays the same pairs for force | one physical clustered pair supplies both directed density contributions, followed by the embedding pass and one two-endpoint force pass; retain only `rho` and `dF/drho` |
| SW | `manybody/sw.cpp:217-324` evaluates one pair term and the upper triangle `K > J` for a center | enumerate the upper triangle of a resettable center cursor or J-tile/K-tile product |
| EDIP | its coordination and redistribution loops are pair-decomposable, while its three-body loop is center-complete | use direct pair passes for coordination/redistribution and the same center cursor as SW for triplets |
| Tersoff | `manybody/tersoff.cpp:97-269` scans every directed `(i,j)` and replays K twice, first for zeta and then its derivative | retain scalar zeta per active edge lane and replay the center cursor; do not prebuild raw edge/K rows |
| ReaxFF | `manybody/reaxff/reaxff.cpp:120-167` currently mixes half/full spatial lists with bond-order consumers; bond order and EEQ then build their own CSR relations | discover raw candidates from clustered tiles, then compact only accepted reaction edges and EEQ matrix entries |
| Plugin/PRIPS | `plugin/plugin.cpp:79-107` and API version 2 expose host neighbor counts and raw atom-index pointers | add a versioned immutable clustered capability or reject initialization; do not synthesize compatibility rows |

This audit also fixes two design traps:

1. the legacy full list is atomically filled and is not a meaningful sorted
   neighbor order. SW's `k = j + 1` uses an arbitrary but complete ordering to
   enumerate each unordered J/K pair once. A stable clustered cursor order may
   replace it without reproducing the legacy insertion order;
2. EAM's current directed full-list implementation stores `rho` and
   `dF/drho` for local atoms but dereferences endpoint state for every listed
   J. The clustered port must define local/local and local/ghost endpoint
   updates and the required halo exchange explicitly before claiming DD
   support. It must not preserve a possible out-of-bounds legacy assumption
   by hiding it behind a derived full list.

##### What may be derived

The center-neighbor consumers do require an inverse structural lookup. That
lookup is a **tile-incidence index**, not a neighbor table. For each valid
packed CJ/JM record it emits at most one native-I and one transposed-J
reference:

```text
{center_supercluster, sci_id, cjpacked_id, jm,
 orientation, fixed_width_i_cluster_mask}
```

The record points back to the authoritative tile. It does not contain an atom
pair, an accepted endpoint row or a pre-applied exclusion result. Its total
cardinality is bounded by eight references per packed CJ record, plus
supercluster offsets. Provider incarnation and gmxpacked payload generation
are part of its identity; geometry-dependent shift replay additionally pins
the geometry generation.

Production generation is on demand only when a center-neighbor consumer is
initialized. The CPU builder uses deterministic count/prefix/fill. The GPU
builder uses a bounded reference stream and stable grouping by
`(center_supercluster, source_tile_ordinal, orientation)`; it must not copy
SCI/CJ payloads to the host or synchronize the LJ hot path. The existing host
builder remains the contract oracle until both production builders pass the
same byte-level reference test.

The cursor resolves one center lane, replays the referenced native or
transposed tile, reapplies lane validity, exclusion, cutoff and explicit image
shift, and yields neighbor lanes in stable structural order. Resetting a
cursor or holding two cursors is sufficient for SW/EDIP J/K products and
Tersoff K replay. Transposed replay must reverse force ownership and negate or
reorient the stored pair shift; it may not recompute a minimum image that
loses the payload's image identity.

##### Next migration slices and gates

The implementation sequence after the LJ/soft-LJ prerequisite gates is:

1. **Publish endpoint incidence as a structural capability.** Add
   CPU/GPU-owned storage, exact generation keys and fail-closed view
   validation. Test reset/rebuild, stale provider and payload generations,
   native/transposed replay, two simultaneous cursors, partial masks, mixed
   SCI shifts and the fixed `<= 8 * cjpacked_count` size bound.
2. **Finish pair-decomposable consumers without the incidence index.**
   Custom/JIT remains the template. Port SITS force-only and full first,
   keeping REST/REST2 legacy outside this mainline backport and independent of
   SITS dispatch and acceptance; then port EAM density/embedding/force.
   Compare canonical pair sets, force, energy and virial on CPU/GPU and
   local/ghost DD. EAM additionally compares per-atom `rho` and `dF/drho` and
   validates halo ownership.
3. **Port center-complete consumers directly.** Port SW, EDIP and Tersoff in
   that order. Before force comparison, compare the center-neighbor set and
   canonical unordered triplets or directed edge/K tuples. Exercise neighbors
   from different shift SCI records, exclusions, cutoff boundaries and
   transposed local/local ownership.
4. **Port ReaxFF by semantic boundary.** Migrate VDW and raw candidate scans,
   then count/scan/fill bond-order edges and EEQ CSR directly from clustered
   candidates. Assert capacity/overflow before consumption, stable edge IDs,
   CSR terminal offsets, EEQ convergence and local/ghost DD behavior. Remove
   the unused angle/torsion neighbor parameters rather than providing them a
   full-list adapter.
5. **Remove one legacy reason at a time.** A reason bit is removed only after
   its clustered path has no silent fallback and passes correctness, CPU/DD
   and paired performance gates. The grid/full builders are removed only
   after Plugin/PRIPS is versioned or explicitly retired.

For all migrated kernels, the production variant set remains exactly
force-only and full. Energy-only or virial-only requests use the full kernel
with runtime store suppression; a third virial specialization is not added.

The first implementation of every consumer is direct. An operator-private
cache may be proposed only after NCU shows a greater-than-3% failure caused by
repeated clustered decoding/replay, and the comparison includes cache
count/scan/fill, memory and rebuild cadence. Even then, the cached entries
must be accepted operator relations or bounded scheduling descriptors, never
a shared derived half/full spatial adjacency.

### Phase 1A checkpoint - 2026-07-26

The neutral structural types, validated cache-to-view factory and retained
microbench canonical-pair oracle now exist in the worktree. The oracle builds
an independent CPU cell-list pair set from coordinates, cell and exclusions,
then compares it with tuples decoded from the production gmxpacked payload.

The wat160k snapshot generated with the complete locked peak environment
passes exactly:

```text
metadata_ready=1 matched=1
payload=17131674 oracle=17131674
duplicates=0 missing=0 extra=0
```

The non-active compatibility builder also passes the same exact count. A
deliberately incomplete environment that set only
`SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1`, without the locked outer lifecycle
and builder contract, produced one missing cutoff-boundary pair:

```text
missing=(88966,111256,shift=14)
payload=17131673 oracle=17131674
```

The pair is not excluded, has no duplicate/alternate-shift entry and is inside
the exact force cutoff in the gathered coordinate frame. The complete peak
configuration contains it. Therefore:

- the locked peak path remains the valid migration reference;
- a partial active-view environment is not a supported fallback or a second
  production mode;
- cleanup must make the active-view lifecycle atomic: select the complete
  outer-source/inner-active contract or fail configuration early;
- migration tests must always record the full environment and must not infer
  correctness from `ACTIVE_VIEW` alone.

This result is also the first concrete use of the oracle: it distinguishes a
valid clustered payload from a superficially similar experimental route
without using a derived legacy list as the authority.

The regular-LJ and soft-LJ launch boundaries have subsequently been rewired to
obtain structural counts and pointers from `CLUSTERED_SPATIAL_VIEW`. Their
coordinates, parameter fields, padded sorted atom IDs and force scratch remain
consumer-owned. CUDA kernel templates, pair bodies and launch geometry were
not changed.

Post-rewire checks:

- CUDA `SPONGE` and `NBNXM_MICROBENCH` build;
- CPU `SPONGE` build;
- locked peak wat160k one-step run is finite;
- a newly generated locked-peak snapshot again matches all 17,131,674
  canonical pairs exactly;
- the 133,266-atom softcore fixture passes one step on the standard gmxpacked
  route with finite `LJ_soft = 57368.00`.

The water-specific peak builder environment is not a valid softcore
performance environment: applying it verbatim to the softcore fixture
exhausted candidate-builder memory before force launch. Soft-LJ performance
and NCU acceptance must use its dedicated fixture/environment rather than
reusing water queue geometry. Full-output comparison, grouped-SCI fixtures and
the frozen performance/NCU gate remain open, so Phase 1A is not yet complete.

The structural contract is now independently executable rather than being
covered only through LJ smoke runs. `CLUSTERED_SPATIAL_VIEW_TEST` validates:

- payload/source generation, backend, cutoff and local/ghost domain mismatch;
- missing structural pointers;
- a valid native payload with an empty exclusion pool;
- native-grouped, gmxpacked-grouped and pair-shift capability requirements;
- a grouped supercluster range containing multiple SCI shifts;
- a single gmxpacked SCI/CJ record whose i-clusters require different explicit
  pair shifts;
- replay of the same grouped range without materializing atom adjacency.

The test is registered with CTest and passes in the CPU build. CPU `SPONGE`,
CUDA `SPONGE` and CUDA `NBNXM_MICROBENCH` also build after the contract change.

The grouped index is no longer ambiguous. Existing
`d_grouped_sci_offsets/ids` remain explicitly native and index
`d_nbnxm_sci`. A separate, on-demand
`d_gmxpacked_grouped_sci_offsets/ids` index is built from
`d_gmxpacked_sci` only when an auxiliary clustered consumer requests it.
Regular and soft LJ do not request that metadata, so the peak path gains no
extra grouping scan. Both indices are O(SCI), generation-local structural
state; neither duplicates accepted atom pairs.

The partial active-view configuration is now rejected before a payload build.
The wat160k negative smoke with `ACTIVE_VIEW=1` and no lifecycle terminates
with:

```text
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1 requires
SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer (or outer-source);
partial active-view configuration is unsupported
```

The known-unstable rolling source cache is no longer enabled implicitly and an
explicit value of one is rejected. The complete locked outer lifecycle still
passes the wat160k one-step CUDA run with finite energies and forces after this
change. Full-output differential checks and the frozen performance/NCU matrix
remain open, so Phase 1A is still not complete.

The one-step result was not sufficient for the lifecycle gate. A subsequent
three-run, 10,000-step wat160k NVT/NPT matrix completed with process exit code
zero but produced NaN thermodynamics in every run. The first diagnostic
locked-path NVT run was finite through step 1,000 and printed NaN state after
that point.

This was initially suspected to be active-view reuse, but controlled evidence
rejected that attribution:

- disabling zero-dirty source reuse, cached fill or dirty-J scan independently
  still failed, with the first printed NaN moving between steps 1,300 and
  1,700;
- `ACTIVE_VIEW=0` also failed by step 1,800;
- the independently rebuilt pre-change binary at `02272de` remained finite for
  the same 5,000-step input.

The source differential then exposed the actual correctness regression.
`Refresh_Nbnxm_Pair_Shift_Bits`,
`Refresh_Gmxpacked_Pair_Shift_Bits`,
`Refresh_Gmxpacked_Pair_Shift_Bits_Simple_Flags` and
`Prune_Clustered_Inner_Imask` had been changed from a pair-specific
`Determine_Clustered_Pair_Shift_Id(center_i, center_j, rcell)` calculation to
the single `sci_entry.shift_id`. That made the SCI-safe test tautological and
violated the explicit view contract in which different i-clusters in one SCI
may require different shifts.

The pair-specific geometric shift calculation has been restored in all four
paths. Post-fix evidence:

- `ACTIVE_VIEW=0` and the complete locked active view both remain finite for
  5,000 steps;
- the locked 5,000-step run reaches 139.729 ns/day;
- the independent snapshot oracle matches exactly:
  `payload=17131673`, `oracle=17131673`, with zero duplicates, missing or
  extra tuples;
- the repaired three-run, 10,000-step wat160k matrix is finite in all six
  NVT/NPT runs; NVT averages 149.283 ns/day with 4.917 s force time, and NPT
  averages 142.229 ns/day with 5.081 s force time;
- the production force-only NCU duration is 303.744 us versus the frozen
  296.992 us reference (+2.27%, inside the 3% gate), with the same 69
  registers/thread and no spills;
- the full energy+virial NCU duration is 417.056 us versus 415.392 us
  (+0.40%), with the same 96 registers/thread and essentially unchanged spill
  count;
- the corrected refresh kernel itself increases from 7.136 us to 16.576 us
  because it performs the required geometry and uses 34 rather than 25
  registers/thread, but its complete end-to-end cost remains inside the
  accepted envelope.

Raw artifacts:

```text
/tmp/sponge-phase1a-shiftfix-wat160k-3cycle-20260726/results.tsv
/tmp/wat160k-shift-ncu-TLScqx/current_pair_shift.ncu-rep
/tmp/wat160k-shift-ncu-TLScqx/fixed_pair_shift.ncu-rep
/tmp/wat160k-shift-ncu-TLScqx/shiftfix_forceonly.ncu-rep
/tmp/wat160k-shiftfix-full-ncu-fhSo8X/shiftfix_full.ncu-rep
```

The long-running numerical blocker is therefore repaired. Phase 1A still
requires the remaining full-output differential, GPU minimization, retained
microbench tolerance and dedicated soft-LJ performance gates before non-LJ
production dispatch is enabled.

### Phase 1A closure plan - native verification, not legacy derivation

The remaining verification tooling must follow the same representation rule
as production. In particular, a correctness oracle is not a reason to rebuild
or retain a production half/full neighbor table.

The current full-output snapshot path is not yet suitable for this gate.
`Capture_Clustered_Microbench_Full_Output_Diagnostic_View` requires
`forceonly_warp_record_offsets` and `forceonly_warp_j_records`, and
`RunSpongeFullOutput` replays those legacy native records. Consequently,
requesting a full-output dump currently forces auxiliary/native payload
construction even though the production regular-LJ and soft-LJ paths consume
gmxpacked SCI/CJ records directly.

Close this gap as a test-tooling change:

1. extend the versioned full-output snapshot schema with the existing
   gmxpacked payload fields plus reference force, energy and virial;
2. capture those fields directly from the generation-matched clustered cache;
3. replay the production gmxpacked full kernel and compare the three reference
   outputs;
4. keep the old record reader only for existing snapshot compatibility in the
   microbench, never as a production builder request;
5. remove the microbench-dump reason from
   `cached_native_payload_required` after the new writer is active.

This makes the snapshot an immutable copy of the authoritative clustered
payload, not a derived spatial representation. Temporary tuple expansion
inside the offline oracle remains acceptable because it is test-only and is
not consumed by a production force operator.

The remaining Phase 1A gates are executed in this order:

1. **Full-output differential.** Generate force-only and gmxpacked full-output
   snapshots from the same wat160k state. Require exact canonical pair-set
   equality and tolerance-qualified force, total energy and virial replay.
2. **GPU minimization.** Reuse the 1,011-atom TIP3P bad-coordinate fixture in
   `benchmarks/validation/misc/tests/test_min.py`, but lock clustered
   production dispatch explicitly and assert the selected payload/backend in
   addition to the existing 4,000-step finite/final-potential checks.
3. **Retained microbench.** Run force-only and full replay with interleaved
   repeats; reject a kernel-duration regression greater than 3% and record
   variance, registers, spills and occupancy.
4. **Dedicated soft-LJ gate.** Use the 133,266-atom softcore fixture and its
   own builder limits rather than the water peak environment. Preserve the
   existing one-step energy oracle (`LJ_soft = 57368.02`, absolute tolerance
   2.0), add a multi-step finite run, and establish isolated full-kernel
   timing plus NCU data for
   `Nbnxm_Gmxpacked_Lennard_Jones_And_Direct_Coulomb_Soft_Core`.

No kernel or launch tuning is performed while closing these gates unless a
fresh NCU report first identifies the limiting roofline, memory hierarchy,
warp stalls, instruction mix, occupancy, register count and spills. Phase 1B
starts only after all four gates pass.

### Phase 1A gate results - 2026-07-26

Three of the four closure gates now pass on the production gmxpacked
representation. The dedicated soft-LJ gate remains a measured blocker, so
Phase 1B must not start yet.

#### Gmxpacked full-output differential - passed

The full-output snapshot is now versioned as kind 5 and embeds the complete
kind-4 gmxpacked production payload plus reference force, energy, direct-PME
energy, LJ energy and virial arrays. Capturing it no longer requests native
SCI/warp records and no longer changes the production builder route. The old
kind-4 reader remains usable for frozen snapshots.

For the wat160k full fixture:

```text
sci=3900 cjpacked=90005 exclusions=38018 atoms=164544
payload_pairs=17131674 oracle_pairs=17131674
duplicates=0 missing=0 extra=0

force   max_scaled=7.554855e-06
energy  max_scaled=9.505848e-06
direct  max_scaled=1.539589e-06
LJ      max_scaled=1.136959e-07
virial  max_scaled=7.564748e-06
```

All fields pass the `2e-5` scaled tolerance. Three 2,000-iteration replays are
`0.469446`, `0.473307` and `0.473932 ms`; their mean is `0.472228 ms` with
less than one-percent spread.

#### GPU minimization - target path passed

The 1,011-atom bad-coordinate minimization passes 4,000 steps when the complete
locked active-view lifecycle is selected, both with and without snapshot
capture:

```text
with dump: final potential = -4586.82
no dump:   final potential = -4444.45
```

The old non-active compatibility route is not a valid replacement:

```text
default with dump: final potential = +146488.48
default no dump:   final potential = -1154.60
```

This does not justify preserving or repairing a second production mode. The
removal target is the complete active-view route. The pathological fixture is
also unsuitable as a canonical-pair oracle: both builders miss the same 674
wrapped oracle tuples after the fixture deliberately overlaps clusters and
changes the box. Canonical-pair acceptance remains the independent wat160k
fixture, which matches exactly.

#### Retained regular-LJ microbench - passed

On the frozen wat160k force-only payload, three alternating 2,000-iteration
runs give:

```text
comb kernel:         0.234382 0.239184 0.248617 ms
sorted-force kernel: 0.232145 0.228785 0.235117 ms
```

The means, `0.240728` and `0.232016 ms`, are respectively 13.2% and 14.2%
faster than the documented `0.277435` and `0.270448 ms` anchors. The existing
post-shift-fix production force-only/full NCU reports remain within the
three-percent gate, and the full-output replay above also passes.

#### Dedicated soft-LJ - correctness passed, performance failed

The 133,266-atom fixture passes the existing energy oracle on clustered
gmxpacked:

```text
clustered LJ_soft = 57368.00
legacy    LJ_soft = 57368.02
reference LJ_soft = 57368.02
absolute tolerance = 2.0
```

The benchmark must set top-level `skin = 2.0`; placing only
`clustered_rebuild_skin = 2.0` under `[LJ]` leaves the global clustered direct
default at 10 Angstrom and does not produce a matched protocol.

With matched 2-Angstrom skin, the 1,000-step steady-state measurement is:

| path | `Calculate_Force` | payload build | gather | direct kernel | scatter |
|---|---:|---:|---:|---:|---:|
| clustered gmxpacked | 1.386780 s | 0.138839 s | 0.052328 s | 0.777350 s | 0.033069 s |
| legacy | 1.198766 s | neighbor search 0.058356 s | - | included | - |

The clustered end-to-end force time is 15.68% slower, outside the three-percent
gate. The isolated force-only NCU result shows that the pair kernel itself is
not slower:

| metric | clustered force-only | legacy force-only |
|---|---:|---:|
| duration | 912.064 us | 922.400 us |
| registers/thread | 91 | 40 |
| achieved occupancy | 30.56% | 53.71% |
| DRAM throughput | 2.25% | 15.48% |
| L1 hit rate | 78.35% | 73.98% |
| L2 hit rate | 92.97% | 91.16% |
| long-scoreboard stall/issue | 0.798 | 20.050 |
| local spill requests | 0 | 0 |

Thus the steady-state regression is dominated by the first payload build and
per-step gather/scatter around a kernel that is 1.12% faster than legacy.
Removing the legacy half-list reason alone cannot remove the required soft
coordinate gather.

The actual full energy+virial call must be profiled after skipping the
initialization force-only invocation. Its mangled template IDs are `Lb1` for
gmxpacked and `Lb1ELb1ELb1ELb1ELb0` for legacy:

| metric | clustered full | legacy full |
|---|---:|---:|
| duration | 1.190592 ms | 0.941024 ms |
| registers/thread | 96 | 56 |
| achieved occupancy | 30.79% | 53.49% |
| shared memory/block | 9.024 KB | 1.024 KB |
| local spill requests | 11,795,054 | 0 |
| DRAM throughput | 2.83% | 15.56% |
| long-scoreboard stall/issue | 1.806 | 14.129 |

Full is 26.52% slower and its per-i energy/virial arrays spill heavily. This
is the primary kernel blocker. A first NCU-driven attempt to compute A/B
soft-core states sequentially reduced force-only registers from 91 to 90 but
regressed force-only duration by 2.65%, left full spills unchanged and
regressed full duration by 2.07%; the change was rejected and reverted.
A second experiment moved only the 48 long-lived per-thread virial
accumulators into shared memory. With the GPU otherwise available, the
one-step full fixture did not complete within 70 seconds instead of the normal
single-digit seconds; the pair-loop shared read/modify/write traffic was
therefore rejected and reverted before collecting an NCU report.

The next NCU iteration separated the two retained kernel variants instead of
forcing one shared-cache layout on both:

- force-only keeps the 32-byte `VECTOR_LJ_SOFT_TYPE` AoS cache;
- full uses a field-wise SoA view over the same 2-KB shared allocation;
- `__launch_bounds__` remains 10 blocks/SM for force-only and 8 blocks/SM for
  full.

This hybrid layout is retained. It measures 900.064 us for force-only with 91
registers/thread and no spills, versus the original 912.064 us. Full measures
1.100320 ms with 128 registers/thread and 1,968,072 local spill requests,
versus the original 1.190592--1.211744 ms and 11,795,054 requests. A clean
rebuild and independent exact-`<1>` NCU capture measures 1.128672 ms with the
same 128 registers, 8-KB shared allocation and 1,968,072 requests. The
one-step full fixture remains finite and reports `LJ_soft = 57368.00`.

A 24-byte packed shared atom experiment nearly eliminated shared-memory
conflicts, but regressed force-only to 924.608 us and full to 1.125792 ms
relative to the hybrid reports. It is rejected and is not present in the
source.

The clean 1,000-step matched-skin rerun is:

| path | `Calculate_Force` | payload/search build | gather | direct kernel | scatter |
|---|---:|---:|---:|---:|---:|
| clustered gmxpacked hybrid | 1.371290 s | 0.136706 s | 0.043405 s | 0.773079 s | 0.030227 s |
| legacy | 1.193617 s | 0.055327 s | - | included | - |

The remaining end-to-end regression is 14.89%. Therefore the hybrid kernel is
a cleaner performance sample, but it still does not close the soft-LJ gate.
The force-only steady-state path continues to be dominated by the initial
payload build plus per-step gather/scatter rather than by the isolated pair
kernel.

The next soft-LJ work must therefore:

1. retain the current parallel A/B pair math;
2. redesign full output accumulation to eliminate the per-thread
   `energy_lj_i[8]`, `energy_coulomb_i[8]` and `virial_i[8]` spill pattern
   without materializing atom-pair state;
3. separately reduce or amortize gather/scatter and first-build costs;
4. re-profile force-only and full after every kernel/launch change;
5. run the existing correctness fixture plus a multi-step finite NPT pressure
   comparison after every retained change.

The clustered CPU dispatch already instantiates exactly two soft-LJ paths:
force-only and full. It has no CPU energy-only or virial-only specialization.
Both paths now execute on the 133,266-atom fixture. The force-only path
completes with the default dense shift-partitioned candidate builder, the
sparse-shift builder and the unpartitioned builder. The full NPT path reports
`LJ_soft = 57368.00`, matching the clustered GPU result above, and finite
pressure/virial output.

The first CPU execution exposed a producer/consumer encoding bug rather than
a kernel failure. The host payload builder copied and indexed
`candidate_sci_numbers` supercluster IDs even when the dense builder had
expanded each base SCI into 27 shift-partitioned candidates; sparse candidates
also copied the base-ID buffer instead of their compacted ID buffer. At
candidate 2086 this read a stale supercluster ID 7281 against a valid range of
2083 and crashed. The host input now records the candidate encoding, copies
the actual `candidate_sci_supercluster_ids` source, copies sparse shift IDs,
maps dense candidates through `candidate_sci / 27`, and evaluates only the
candidate's encoded shift. The temporary rebuild-time structural diagnostics
used to isolate the stale ID were removed after the dense, sparse and
unpartitioned fixtures passed; they are not retained in the production path.

This closes CPU path existence and one-step correctness, but it does not close
the soft-LJ gate: the GPU full-kernel spill and end-to-end performance
regressions remain blockers.

The latest clean CUDA source build, one-step full runtime check, exact full NCU
capture and 1,000-step clustered/legacy rerun have completed. The GPU
availability pause is closed; the measured performance failure above is the
remaining gate, not an execution-environment waiver.

Raw artifacts:

```text
/tmp/sponge-gmxpacked-full-phjkXF/wat160k_full.sponge_fulloutput.bin
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_gmxpacked_full.ncu-rep
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_legacy_full.ncu-rep
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_gmxpacked_fulloutput.ncu-rep
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_legacy_fulloutput.ncu-rep
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_gmxpacked_force_state_reuse.ncu-rep
/tmp/sponge-softlj-phase1a-sWYZxg/softlj_gmxpacked_full_state_reuse.ncu-rep
/tmp/softlj-hybrid-force-20260726.ncu-rep
/tmp/softlj-hybrid-full-20260726.ncu-rep
/tmp/softlj-hybrid-restored-full-20260726.ncu-rep
/tmp/softlj-packed24-force-20260726.ncu-rep
/tmp/softlj-packed24-full-20260726.ncu-rep
```

### Pair-image ownership and lifecycle audit - final design

The later aggregate-ownership experiment is rejected. It mixed two different
lifecycles: shift-partitioned traversal/source rows are structural payload,
while the image used by the force kernel is current-coordinate metadata. Moving
an entire `i_local` lane and its exclusion words to one SCI image made a
step-zero snapshot look less duplicated, but it did not preserve the
time-dependent image contract and it could not prove ownership across retained
base and independently compacted delta payloads.

The stable pre-cleanup design at `02272de` and the lifecycle work in
`5f2643b`, `2d5cbb4`, `ee302ec` and `5888c4a` establish the intended split:

1. topology, ordering, source membership and compact SCI/CJ/exclusion payload
   are rebuilt only when their anchor/source generation changes;
2. current cluster centers are refreshed by coordinate gather;
3. `pair_shift_bits` are replayed from current `center_i`, `center_j` and
   `rcell`, independently of whether the structural payload was reused;
4. SCI-safe flags are an optimization result of that replay, not the
   definition of the image;
5. a published force payload is a single generation-matched SCI/CJ/exclusion
   set. Source changes either leave that payload byte-identical or republish a
   complete replacement.

This is also the only lifecycle consistent with the representation. One SCI
can cover several i-clusters, and the view contract explicitly permits their
current nearest images to differ from `sci.shift_id`. Therefore
`sci.shift_id` cannot replace the per-CJ/per-i-cluster sidecar during payload
reuse.

Two experiments are retained as negative evidence:

```text
fixed SCI shift:
  small step zero  payload=106328 oracle=106328 duplicates=0 missing=0
  wat160k step zero payload=17131673 oracle=17131673 duplicates=0 missing=0
  wat160k dynamics finite at step 500, NaN by step 1000

cross-shift aggregate lane transfer plus center refresh:
  small step zero payload=105274 oracle=106328
  duplicates=0 missing=1054 extra=0
```

Disabling active view did not cure the fixed-SCI-shift failure, while the
legacy control remained finite at step 1000. The failure is therefore a shift
metadata lifetime error, not evidence that steady-state active-view reuse
should be removed. Disabling active view also reduced the diagnostic wat160k
run from about 40-52 ns/day to 17.5 ns/day. The retained route therefore keeps
zero-dirty compact-payload reuse and incremental source discovery; it performs
a full compact replacement only when the active source set actually changes.

The 1,011-atom bad-coordinate fixture is not promoted to a canonical pair-set
oracle. It deliberately overlaps clusters and changes the box during
minimization; the earlier Phase 1A evidence already showed both builders miss
the same wrapped tuples there. It remains a minimization, finite-force and
selected-route stress test. The canonical pair authority is the matched
wat160k snapshot, for which the repaired pair-specific refresh has already
passed with zero duplicate, missing or extra tuples.

The provider changes before Phase 1B are consequently limited to lifecycle
hardening, not a new derived ownership table:

- restore and retain pair-specific geometric refresh in native, gmxpacked
  simple/exact and prune paths;
- separate three identities instead of overloading one counter:
  source generation identifies candidate/source rows, the active anchor/source
  key proves that a compact payload was built from the requested active view,
  and a monotonic compact-payload generation identifies the published
  SCI/CJ/exclusion bytes;
- bind pair-shift readiness to the compact-payload generation, exact
  SCI/CJ/exclusion counts and `rcell`. A source-only patch does not invalidate
  the sidecar until it actually republishes compact rows;
- retain unconditional pair-shift replay after every coordinate gather. The
  build-time cache predicate only removes duplicate replay before the next
  gather; it is not a substitute for a coordinate/geometry lifetime;
- do not retain a second base/delta force-payload lifecycle. The audited delta
  helpers were hard-coded off, `CLUSTERED_SPATIAL_VIEW` exposed no delta
  fields, and regular/soft LJ launched only the primary payload. Publishing
  such a delta would therefore have reported success without consuming its
  pairs;
- remove that unreachable delta compact, pair-shift, baseline-imask and
  publication scaffolding. A changed active source set rebuilds and publishes
  one complete compact payload; an unchanged source set reuses the existing
  generation;
- keep all shared indices O(SCI), O(CJ) or O(source/aggregate). Do not add an
  atom-pair adjacency or a second neighbor-list lifecycle.

Implementation checkpoint:

- the main compact payload owns one monotonic generation. Clearing or
  successfully publishing the payload advances that generation;
- the active anchor/source generations remain a reuse proof and are no longer
  used as the pair-shift cache identity;
- the pair-shift metadata cache records the main compact-payload generation,
  SCI/CJ/exclusion counts and exact `rcell`; successful replay commits that
  complete key;
- `CLUSTERED_SPATIAL_VIEW::payload_generation` now exposes the actual compact
  payload generation instead of the build-attempt counter, so same-count
  in-place replacement is visible to future non-LJ consumers;
- the pure cache-key predicate is covered by
  `ClusteredSpatialViewContract`, including same-count generation changes,
  independent count changes, `rcell`, reset/storage readiness and cache-disable
  cases;
- the unreachable delta SCI/CJ/exclusion buffers, delta pair-shift replay,
  delta source kernels and the per-build baseline-imask copy have been removed;
- a clean CUDA build and a clean CPU build pass for this exact source state;
  `ClusteredSpatialViewContract` also passes;
- the clean CUDA binary completes the locked one-step wat160k run and its new
  snapshot matches the independent pair oracle exactly:

  ```text
  metadata_ready=1 matched=1
  payload=17131674 oracle=17131674
  duplicates=0 missing=0 extra=0
  ```

- deleting fields from `LJ_CLUSTER_LAYOUT` changes translation-unit ABI.
  Incremental builds that retain an object compiled against the old member
  offsets can report valid builder counts but null payload pointers in
  `CLUSTERED_SPATIAL_VIEW`. Cleanup validation must therefore use
  `--clean-first` (or an otherwise proven complete rebuild); the clean rebuild
  restores the route without a source workaround;
- long-run NVT/NPT and NCU/performance revalidation for this exact source
  state are recorded below.

This closes the publication-generation and pair-shift cache-key model without
adding an unconsumed representation. Base/delta overlap is no longer a
runtime state because only one force payload can be published.

A build-time baseline-shift sidecar is not selected. The proposed arithmetic
has an unproven sign under the force kernel's subtract-shift convention, can
produce components outside the encodable `[-1, 1]` range and adds another
generation-bound buffer. The pre-cleanup dynamic refresh already
solves the lifecycle at lower state complexity.

Acceptance uses one fully rebuilt binary and the exact production protocol:

- wat160k locked snapshot matches the independent oracle exactly;
- the performance gate is conjunctive, not a single-system sampling rule:
  wat160k, wat600k and DNA_COU must each independently pass. Results may not
  be averaged across systems, and a fast water result cannot compensate for a
  slow DNA result (or vice versa). Both the water LJ-combination path and DNA
  packed-AB path must be observed;
- each required system runs three 10,000-step NVT and NPT cases and remains
  finite; force time and end-to-end speed must remain within 3% of the
  matching frozen reference. Each of the three systems also supplies both a
  force-only and a full kernel replay, and every replay must independently
  remain within 3%;
- a run records its actual parameter path. A water-only result, a matrix in
  which DNA silently executes the combination path, or a matrix with even one
  required system outside the threshold cannot qualify a migration
  checkpoint;
- the pair-shift refresh remains within the measured corrected envelope
  (`16.576 us` in the accepted report) unless a fresh NCU analysis justifies a
  change;
- the official performance run selects the retained route atomically with
  `SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1` and
  `SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY=outer`, plus the complete
  locked environment. `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT` has no
  production reader and must not appear in a qualifying manifest; neither may
  any `*_PROBE` variable. Diagnostic runs with frequent output are not
  performance comparisons;
- the 1,011-atom minimization remains finite and selects the locked clustered
  route, but its tuple count is not a migration gate.

Post-cleanup acceptance checkpoint, 2026-07-27:

- a telemetry-controlled three-cycle matrix completed all six 10,000-step
  runs without NaN:

  | ensemble | mean force | mean wall | mean speed | repaired reference | result |
  |---|---:|---:|---:|---:|---|
  | NVT | 4.868072 s | 5.724372 s | 150.948914 ns/day | 149.283 ns/day | +1.12%, pass |
  | NPT | 4.980668 s | 5.941232 s | 145.445251 ns/day | 142.229 ns/day | +2.26%, pass |

  The raw rows are
  `/tmp/sponge-delta-cleanup-wat160k-3cycle-telemetry-20260727/results.tsv`.
  During measured force intervals the RTX 4090 held 2,715-2,730 MHz,
  approximately 299-306 W and 92-96% SM utilization.
- two earlier matrices contained isolated 83-113 ns/day samples while their
  other rows remained at 145-153 ns/day. In those slow samples both
  `Iteration` and `Calculate_Force` grew together; no clustered fallback was
  reported. They are retained as device-state/desktop-contention evidence and
  are not silently mixed with the telemetry-controlled result.
- NCU `--set full` on the actual 3,901-block production symbols gives:

  | kernel | current | frozen | delta | registers | occupancy | local spill requests |
  |---|---:|---:|---:|---:|---:|---:|
  | force-only | 299.936 us | 296.992 us | +0.99% | 72 | 41.06% | 0 |
  | full template, pressure/virial runtime branch | 417.856 us | 415.392 us | +0.59% | 96 | 31.95% | 3,211,863 |
  | pair-shift refresh | 16.544 us | 16.576 us | -0.19% | 37 | 63.64% | 0 |

  The production reports are
  `/tmp/sponge-delta-cleanup-live-pair-20260727.ncu-rep`,
  `/tmp/sponge-delta-cleanup-live-full-exact-20260727.ncu-rep` and
  `/tmp/sponge-delta-cleanup-live-pair-shift-refresh-20260727.ncu-rep`.
  Force-only is latency/issue limited rather than bandwidth limited
  (SM 44.08%, memory 36.32%, DRAM 5.31%, 1.10 eligible warps/scheduler).
  The `417.856 us` launch is the pressure-only runtime branch of the shared
  full template, not a simultaneous energy-plus-virial launch. Full remains
  register/spill and issue limited (SM/memory 41.14%, DRAM
  5.20%, 0.76 eligible warps/scheduler); its dominant stall ratios per issued
  instruction are short-scoreboard 2.040, wait 1.946 and long-scoreboard
  1.609. Cleanup did not worsen those known full-output characteristics.

This closes the wat160k-specific pair-set, stability and NCU checks for the
single-payload cleanup. It does not authorize the next consumer slice by
itself: wat160k, wat600k and DNA_COU must still close the conjunctive gate
below on one uncontended source state. It also does not close the separate
soft-LJ production gate.

These lifecycle and image semantics belong to the shared spatial provider.
Custom, EAM, SW, EDIP, Tersoff and ReaxFF consume the generation-matched view;
they must not invent a private shift repair or derive a legacy table to hide a
provider defect.

#### Boundary-image oracle correction and refresh performance checkpoint - 2026-07-27

A new regular `9 x 9 x 9` lattice fixture exposed a correctness hole that the
earlier large-system snapshot did not isolate. A cluster AABB can straddle the
half-box image boundary even when its wrapped center is not exactly on the
tie. Center-only refresh can then normalize records from distinct fixed-shift
SCI buckets to the same pair shift, producing both duplicate canonical tuples
and a missing image.

The failing and repaired exact-oracle results are:

```text
center-only refresh:
  payload=52673 oracle=53217
  duplicates=738 missing=544 extra=0

extent-aware image ownership:
  payload=53217 oracle=53217
  duplicates=0 missing=0 extra=0

wat160k extent-aware snapshot:
  payload=17131673 oracle=17131673
  duplicates=0 missing=0 extra=0
```

The retained end-to-end oracle command is:

```text
pixi run -e dev-cpu python \
  tools/custom_pairwise_benchmark/run_pair_oracle.py \
  /tmp/custom-pair-oracle-side9-v2-20260727 \
  --sponge build-dev-cuda13-lineinfo/SPONGE \
  --microbench build-dev-cuda13-lineinfo/NBNXM_MICROBENCH
```

`prepare_case.py --pair-oracle` creates the zero-LJ snapshot producer input,
and the runner requires `metadata_ready=1`, equal payload/oracle counts and
zero duplicate, missing and extra tuples. The canonical replay now uses
`super_cluster_offsets`; it no longer assumes
`supercluster_id * kClusteredSuperClusterClusters`, so partial or
non-uniformly grouped superclusters are represented correctly.

Correctness alone does not qualify the current extent-aware implementation.
Full NCU on the same wat160k input shows:

| refresh implementation | duration | registers/thread | executed instructions | result |
|---|---:|---:|---:|---|
| frozen center-only reference | 17.152 us | 37 | 7.74 M | performance reference, boundary incorrect |
| extent computed per active pair | 64.928 us | 44 | 44.51 M | reject |
| shared i-cluster fractional cache | 50.528 us | 36 | 33.35 M | reject |
| center-first, lazy extent fallback | 45.344 us | 40 | 24.31 M | reject |

All variants have zero register spills. The last variant reaches 70.08% of
the compute-memory throughput roof and 389.36 GB/s DRAM traffic, but remains
instruction- and dependency-latency heavy; long-scoreboard stall rises to
5.233 issued-instruction equivalents. The reports are:

```text
/tmp/sponge-geometry-epoch-pair-shift-sm89-20260727.ncu-rep
/tmp/sponge-extentfix-pair-shift-sm89-20260727.ncu-rep
/tmp/sponge-extentfix-pair-shift-shared-sm89-20260727.ncu-rep
/tmp/sponge-extentfix-pair-shift-fastpath-sm89-20260727.ncu-rep
```

Protocol correction and precomputed-geometry result:

The four exploratory reports above were launched manually with abbreviated
environment names such as `ACTIVE_VIEW`, `QUEUE2_COUNT` and
`DISABLE_FINE_TIMERS`. SPONGE reads only the full
`SPONGE_CLUSTERED_*` names. Those reports therefore remain useful for
comparing rejected source shapes under one diagnostic route, but they are not
production locked-path evidence and their absolute durations do not qualify
or reject a migration checkpoint.

The extent-aware implementation now computes wrapped fractional cluster
centers and extents once during the geometry gather and lets the pair-shift
refresh consume those O(cluster) sidecars. A new full-name locked NCU run uses
the same `3901 x 128` launch as the accepted production reference:

| metric | accepted locked reference | precomputed extent-aware | result |
|---|---:|---:|---:|
| duration | `16.544 us` | `15.424 us` | `-6.77%`, pass |
| registers/thread | `37` | `39` | +2, no occupancy-limit change |
| executed instructions | `7,737,812` | `6,955,966` | `-10.10%` |
| global-load instructions | `504,801` | `260,085` | `-48.48%` |
| shared-load instructions | `0` | `93,755` | intentional i-geometry reuse |
| achieved active occupancy | `63.64%` | `65.16%` | improved |
| register spills | `0` | `0` | unchanged |

The multidimensional profile shows that this is not a hidden bandwidth win:
SM throughput is `43.41%`, memory throughput `26.83%`, DRAM throughput
`38.88%`, L1 hit rate `73.28%` and L2 hit rate `84.91%`. Eligible warps per
scheduler fall from `1.66` to `1.29`, and long-scoreboard stall rises from
`5.02` to `5.58` issue equivalents, but the lower instruction and global-load
counts dominate. Branch-target uniformity improves from `78.27%` to `82.00%`.
The full locked report is:

```text
/tmp/sponge-precomputed-locked-pair-shift-sm89-20260727.ncu-rep
```

The same source passes the exact side9 and wat160k pair oracles with zero
duplicates, missing or extra tuples. A full-name locked wat160k diagnostic
run reaches `149.812820 ns/day`; it is consistent with the accepted envelope
but is only a single end-to-end sample, not the required three-system
acceptance matrix.

This evidence refines, rather than reverses, the no-derived-list decision:

- **Do not force every record to use `sci.shift_id`.** Fixed-shift builder and
  inner prune process shift buckets independently, and one SCI covers several
  i-clusters. SCI shift is valid only for a proven-safe SCI.
- **Do not derive a half/full/CSR atom adjacency.** It would neither encode
  the current-coordinate image lifetime correctly nor remove the need for
  explicit shift metadata.
- **Do derive representation sidecars at their natural cardinality.**
  Wrapped fractional center/extent is O(cluster), safe/unsafe routing and an
  optional compact unsafe-SCI work list are O(SCI), and `pair_shift_bits`
  remains O(CJ). None expands cluster masks into atom-pair rows.
- **Refactor each operator to consume the clustered payload directly.**
  Pairwise operators use pair tiles; center-neighbor operators use the
  center-complete grouped cursor; ReaxFF alone compacts candidates after they
  become reaction/EEQ relations.

The staged refresh optimization now has the following status:

1. **implemented:** compute wrapped fractional cluster centers and fractional
   AABB extents once per geometry epoch, adjacent to coordinate
   gather/publication;
2. **not needed for the current result:** use a center-only common-case pass
   and identify SCI records whose center image differs from their SCI image;
3. **contingency only:** correct only those records with the extent-aware rule
   using the precomputed O(cluster) geometry;
4. **contingency only:** compact unsafe SCI IDs with the existing
   generation-matched safety flags and run a second correction pass on that
   O(SCI) work list;
5. **partially complete:** side9 and wat160k exact oracles plus full locked NCU
   pass. The complete wat160k/wat600k/DNA_COU matrix remains required before
   the implementation becomes the migration reference.

The refresh microkernel and the two retained exact-oracle fixtures now pass.
The pair-image ownership gate nevertheless remains open until the full
wat160k/wat600k/DNA_COU force-only/full and NVT/NPT matrix passes on an
uncontended GPU, together with the remaining soft-LJ gate. The precomputed
extent-aware path is the current migration candidate, not yet the backport
reference.

### Publication state machine and view lifetime

The pre-refactor lifecycle is useful as a control because its ownership was
simple even though its representation is now obsolete. `NEIGHBOR_LIST` owns
the half-list/grid buffers from `Initial` through `Clear`; `UPDATOR::Check`
uses cached coordinates to decide when `Update` may replace their contents
(`neighbor_list.h:17-26,43-89,91-115` and
`neighbor_list.cpp:182-192,667-727`). `FULL_NEIGHBOR_LIST` separately owns
its expanded storage, rebuilds it from the half list, and releases it in one
place (`full_neighbor_list.cpp:3-25,62-79,149-157`). A consumer could not
retain contents across `Update` or `Clear`, but the owner and mutation
boundary were unambiguous.

The clustered replacement must preserve that clarity without preserving the
old table. The shared `LJ_CLUSTERED_DIRECT_CACHE` is the storage owner;
`CLUSTERED_SPATIAL_VIEW` is only a borrowed descriptor. Native publish,
gmxpacked publish, geometry gather and provider retirement advance distinct
generations/epochs (`clustered_lj.cpp:13045-13116`,
`clustered_lj.h:410-417`). `Release_Shared_LJ_Clustered_Direct_Cache` retires
the publication through `Clear` before storage is freed
(`clustered_lj.cpp:32065-32067,32818-32834`).

The existing legacy adapter cannot be treated as a lifecycle bridge. It
rejects full-list and non-LJ requests and, after every other precondition has
passed, still returns false because half-list pair ownership has not been
proved (`clustered_lj.cpp:32386-32488`). Therefore the replacement contract is
not “derive `ATOM_GROUP` and borrow it until the next step”; it is:

1. acquire a fresh representation-specific view after structural publication
   and geometry gather;
2. pin
   `{provider incarnation, lease epoch, representation generation, geometry
   generation}` plus the exact sidecar key while enqueueing one call;
3. run on the producer stream and discard the raw view before `Build`, gather,
   DD metadata refresh, `Reset` or `Clear`;
4. let a consumer retain only its own derived buffers, keyed by the complete
   generation tuple and derivation parameters, never provider pointers;
5. reject cross-stream use until a ready event and deferred-reclamation rule
   are implemented. A host generation proves enqueue order, not device
   completion.

This call-scoped lease is the direct replacement for the old owner/update
boundary. It keeps the regular-LJ fast path synchronization-free, while making
future grouped-center and operator-owned caches fail closed on DD, reorder,
same-count republish or provider reinitialization.

The pre-refactor implementation did not have a complete publication identity.
It used positive counts and non-null pointers as readiness, keyed pair-shift
reuse by counts plus `rcell`, and exposed the payload build-attempt counter as
the view generation. This admitted three ambiguous cases:

1. a same-count in-place compact replacement was indistinguishable from reuse;
2. source/anchor changes could invalidate a sidecar even when the published
   compact bytes were unchanged, while a same-count byte replacement could
   leave that sidecar apparently ready;
3. `pair_shift_bits` changes after every coordinate gather even when compact
   generation, counts and `rcell` are unchanged, so its content does not have
   the same lifetime as the structural payload.

The provider contract must consequently distinguish four identities:

| identity | changes when | consumers use it for |
|---|---|---|
| source generation | candidate/source membership or ordering changes | source-cache and rebuild decisions only |
| active anchor/source key | the requested active-mask anchor changes | proof that compact rows represent that active view |
| compact payload generation | a complete SCI/CJ/exclusion payload is published or explicitly withdrawn | structural view freshness and all derived O(SCI) indices |
| geometry epoch | gathered coordinates/current cluster centers change | freshness of pair shifts and any other coordinate-derived sidecar |

`rcell` remains part of the pair-shift key, but it is not a substitute for the
geometry epoch. Two gathers can have identical `rcell` and different centers.
The current LJ call order is safe because both regular and soft LJ execute
`Build -> Gather -> unconditional pair-shift replay -> View -> Launch` in one
call. The shared non-LJ API must make that ordering a checked contract rather
than an LJ-specific convention.

The target publication state machine is:

```text
Empty
  -> Building(structural working state is provider-private)
  -> Published{payload generation, counts, immutable pointers, active key}
  -> Building
  -> Published(new generation) or Withdrawn(new generation)

Published
  -> Gather(new geometry epoch)
  -> PairShiftReady{payload generation, geometry epoch, counts, rcell}
  -> consumer view/launch
```

Publication is one host-side commit point after all three compact arrays are
allocated and filled in stream order. Counts, pointers, generation and active
key become externally valid together. An internal count/scan/fill attempt is
not a publication and must not advance the public generation. A failed
replacement may either leave the previous immutable publication intact or
withdraw it explicitly; it must not expose new counts with old/null storage.
The current single-buffer builder has no concurrent readers, so in-place fill
is acceptable during Phase 1, but `ready` remains false until the commit.
When ownership moves into the shared neighbor service, use a small published
descriptor (or front/back descriptors if asynchronous build is introduced)
rather than letting consumers infer readiness independently from mutable
builder fields.

`CLUSTERED_SPATIAL_VIEW` is a borrowed, non-owning lease, not a persistent
neighbor-list object. Its structural pointers are valid only until the next
provider `Build`, `Reset`, `Clear`, domain repartition or capacity-changing
publication. A consumer that stores work across those boundaries stores the
payload generation and reacquires/revalidates the view before use. A consumer
of `pair_shift_bits` additionally requires:

```text
sidecar.payload_generation == view.payload_generation
sidecar.geometry_epoch == view.geometry_epoch
sidecar SCI/CJ/exclusion counts == view counts
sidecar.rcell == requested rcell
```

The existing build-time pair-shift cache may continue to avoid a duplicate
replay before the next gather by keying compact generation/counts/`rcell`.
That optimization is provider-private. Exported sidecar readiness must also
match the geometry epoch, because the gather replay mutates the sidecar in
place.

The implementation sequence for this hardening is deliberately separated from
kernel tuning:

1. keep the current compact generation and exact count/`rcell` cache key;
2. add a monotonic geometry epoch committed after current cluster centers are
   gathered, and bind successful pair-shift replay to it;
3. make the view factory publish `pair_shift_metadata_ready` only when the
   complete structural-plus-geometry key matches;
4. add stale-same-count, stale-geometry, changed-`rcell`, reset, failed-build
   and domain-repartition contract tests;
5. keep regular/soft LJ launch code and CUDA templates unchanged, then rerun
   the oracle, long-run and performance gates.

The 2026-07-27 one-step failure also exposed a build-system lifecycle hazard.
After the layout header changed, Ninja recorded zero dependencies for an NVCC
compiled `.cpp` object: `clustered_lj.cpp.o` was rebuilt against the new
layout, while `Lennard_Jones_force.cpp.o` retained the old field offsets.
The builder printed valid counts and pointers, but the consumer read those
pointers as null. A target clean rebuild restored the payload and the exact
wat160k oracle:

```text
payload=17131674 oracle=17131674
duplicates=0 missing=0 extra=0
```

Therefore any CUDA ABI/layout-header change requires a target clean rebuild
before correctness, benchmark or NCU evidence is accepted. The validation log
must record object/header timestamps or an equivalent clean-build proof; an
incremental `ninja: no work to do` result is not sufficient until the NVCC
dependency-recording defect is fixed.

The clean-build revalidation for this checkpoint now has the following
evidence:

- the exported wat160k compact payload matches the independent oracle exactly:
  `payload=17131674`, `oracle=17131674`, with zero duplicate, missing or extra
  tuples;
- three native-input 10,000-step force-only peak runs are finite at
  `145.937195`, `144.605240` and `146.913315 ns/day`; their
  `145.818583 ns/day` mean is 1.94% above the frozen `143.048642 ns/day`
  reference;
- the matched `CUDA_ARCH=all-major` force-only NCU report is `303.78 us`,
  69 registers/thread, zero spills and 41.21% achieved occupancy, versus
  `303.74 us`, 69 registers/thread, zero spills and 41.17% in the accepted
  post-shift-fix report;
- the matched full energy+virial report is `420.38 us`, 96 registers/thread,
  3,211,725 local spill requests and 31.87% achieved occupancy, versus
  `417.06 us` with the same register/spill envelope in the accepted report.

An initial `sm_89` report measured `307.65 us` and 72 registers/thread. It is
not a source-regression comparison because the frozen reports execute the
`sm_80` cubin from `CUDA_ARCH=all-major`; rebuilding the current source with
that exact target restored the accepted 69-register code shape. NCU evidence
must therefore record both runtime CC and compiled cubin target. The
force-only/full isolated gates pass for this checkpoint.

Geometry-epoch implementation and clean-source revalidation, 2026-07-27:

- `Publish_Gathered_Cluster_Geometry` now advances a monotonic geometry
  generation after coordinate/center work has been enqueued and invalidates
  the old gmxpacked pair-shift publication;
- pair-shift refresh invalidates readiness before any empty-buffer/failure
  return and commits
  `{gmxpacked payload generation, geometry generation, SCI/CJ/exclusion
  counts, exact rcell}` only after replacement work has been enqueued;
- the view factory exports the complete sidecar key and does not report
  pair-shift readiness unless every field matches the current structural and
  geometry publication;
- regular and soft LJ pin the current geometry generation and exact requested
  `rcell` at their immediate launch boundary;
- the contract test now rejects stale geometry, stale sidecar payload,
  stale sidecar geometry, stale counts, changed `rcell`, failed publication
  and domain-invalidated cache states.

No CUDA kernel body, launch shape or synchronization was changed. The current
implementation is deliberately stream ordered: gather, geometry publication,
pair-shift replay and the consuming force launch are enqueued on the same
stream. The host generation means "work for this geometry has been enqueued on
the producer stream", not "all device writes are globally complete".

Clean ABI evidence uses both `build-dev-cuda13` and a fresh exact-sm89 build.
The one-step wat160k snapshot again matches the independent oracle exactly:

```text
metadata_ready=1 matched=1
payload=17131674 oracle=17131674
duplicates=0 missing=0 extra=0
```

Its full-output replay also matches at the established `2e-5` tolerance
(`force max abs = 1.907349e-05`, `energy max abs = 2.110004e-05`).

The exact-sm89 three-cycle 10,000-step matrix is finite:

| ensemble | mean force | mean speed | frozen speed reference | result |
|---|---:|---:|---:|---|
| NVT | 5.110301 s | 145.757456 ns/day | 149.283 ns/day | -2.36%, pass |
| NPT | 5.202434 s | 141.033015 ns/day | 142.229 ns/day | -0.84%, pass |

The same current source built as `CUDA_ARCH=all-major` completed a second
matrix at `144.615494 ns/day` NVT and `141.680903 ns/day` NPT. NCU comparison
uses exact sm89 on both sides:

| kernel/branch | current | accepted comparison | delta | code-shape result |
|---|---:|---:|---:|---|
| force-only hot launch | 303.52 us | 299.94 us | +1.19% | 3901x64, 72 registers, 0 spill unchanged |
| full template, pressure/virial runtime branch | 414.34 us | 417.86 us | -0.84% | 3901x64, 96 registers, 3,211,756 spill requests unchanged |
| pair-shift refresh, three-run mean | 16.64 us | 16.54 us | +0.61% | 3901x128, 37 registers, 0 spill unchanged |

The pair-shift samples are `17.15`, `16.26` and `16.51 us`. An initial
`619.71 us` full measurement is retained as a protocol warning, not a
regression: setting `print_zeroth_frame=1` selected simultaneous
energy-plus-virial work and executed 273.6 million instructions, whereas the
accepted `417.86 us` report selected the pressure/virial runtime branch of the
same template and executed 182.6 million. Repeating the latter condition
restored the same instruction, spill and occupancy envelope.

Raw current reports:

```text
/tmp/sponge-geometry-epoch-forceonly-sm89-20260727.ncu-rep
/tmp/sponge-geometry-epoch-full-sm89-20260727.ncu-rep
/tmp/sponge-geometry-epoch-pair-shift-sm89-20260727.ncu-rep
/tmp/sponge-geometry-epoch-pair-shift-sm89-r2-20260727.ncu-rep
/tmp/sponge-geometry-epoch-pair-shift-sm89-r3-20260727.ncu-rep
```

The geometry-sidecar hardening therefore passes correctness, long-run and
performance gates.

### Lifecycle design review before widening the shared API

The geometry fix closes one real defect, but the audit found that the current
public generation model is still too narrow for general non-LJ consumers.
`CLUSTERED_SPATIAL_VIEW::payload_generation` is copied from
`gmxpacked_compact_payload_generation` for every backend. A CPU native SCI/CJ
rebuild, clear or allocation replacement can therefore leave that value
unchanged while replacing the pointers and bytes that a CPU consumer uses.
Similarly, `LJ_CLUSTER_LAYOUT::Clear` frees storage and resets both payload and
geometry counters to zero, so an old borrowed view can collide with a later
publication after reinitialization. The present immediate LJ and custom
pairwise calls do not retain a view across either boundary, but this is not a
sound reusable cache identity.

The shared provider must use the following public identities before another
consumer is allowed to retain derived work:

| public identity | scope | publication rule |
|---|---|---|
| provider incarnation | lifetime of the owning spatial service | advances on `Clear`/reinitialization and is never reset within the object lifetime |
| lease epoch | all externally visible descriptor/pointer state | advances when entering a mutating build/withdrawal and on successful descriptor commit |
| native payload generation | native SCI/CJ/exclusion representation | advances on every native publish, clear or replacement, including same-count replacement |
| gmxpacked payload generation | compact SCI/CJ/exclusion representation | retains the existing complete-publication semantics |
| geometry generation | gathered coordinates and current centers | advances for every enqueued gather, even when cell and structural payload are unchanged |
| derived-index key | grouped or operator-specific O(SCI)/O(CJ) state | exact tuple of representation generation plus derivation parameters; a bare `ready` flag is insufficient |

The ambiguous generic `payload_generation` should be removed or made
non-authoritative. A consumer declares whether it uses native or gmxpacked
rows, and validation checks the corresponding generation. A consumer that can
select either representation must carry both generations and record which one
its derived state used.

The provider state machine is refined to:

```text
Empty/Withdrawn{incarnation, lease epoch}
  -> Building{new lease epoch; no public view}
  -> Published{
       incarnation, lease epoch,
       native descriptor + generation if present,
       gmxpacked descriptor + generation if present
     }
  -> GeometryEnqueued{geometry generation, producer stream}
  -> PairShiftEnqueued{
       gmxpacked generation, geometry generation, counts, rcell,
       producer stream
     }
  -> consumer launch on the producer stream
```

With the current single-buffer builder, entering a build that may mutate
published storage must withdraw the old lease first. Keeping the previous
publication is legal only if construction uses separate immutable storage.
Counts, pointers and generations are committed as one host descriptor; a
count/scan/fill attempt is not a publication. Failed work remains withdrawn
and cannot expose old pointers under new counts.

`CLUSTERED_SPATIAL_VIEW` remains a non-owning, call-scoped lease. The default
rule for migrated consumers is:

1. acquire after the provider has built and gathered;
2. validate the exact representation and optional sidecar key;
3. enqueue derived work and the consumer on the same producer stream;
4. discard the view before the next Build, gather, DD update, Reset or Clear.

Long-lived consumers may cache their own derived buffers, but not raw view
pointers. They key those buffers by
`{provider incarnation, representation generation, derivation parameters}`
and reacquire a fresh view at every invocation.

The current implementation is safe only on its existing single-stream path.
Before the provider accepts an arbitrary consumer stream, the view must expose
a producer-stream readiness token. The zero-overhead fast path requires the
same stream; a cross-stream caller must request a lazily recorded event and
insert `StreamWaitEvent` before dereferencing geometry or sidecar storage.
`geometry_generation` alone is an enqueue-order token and cannot prove
cross-stream completion. `Clear` must synchronize or defer reclamation until
outstanding readiness events complete.

This leads to two implementation slices before further shared-cache reuse:

1. split native and gmxpacked structural generations, add a non-resetting
   provider incarnation/lease epoch, and make validation representation
   specific;
2. encode the current same-stream restriction explicitly; add a lazy event
   handoff only when a future consumer actually requests another stream.

Neither slice changes a CUDA kernel or adds synchronization to the current LJ
fast path. Both require a clean CUDA rebuild because they change descriptor
layout. The current custom pairwise implementation remains valid because it
acquires and consumes its view immediately in the same call and stream, but it
must not become the precedent for caching `payload_generation` as a generic
native/GPU identity.

Representation-specific lifecycle implementation checkpoint, 2026-07-27:

- the generic public `payload_generation` has been replaced by
  `native_payload_generation` and `gmxpacked_payload_generation`;
- `provider_incarnation` and `lease_epoch` are monotonic within the owning
  layout. `Clear` retires the provider identity instead of resetting it to a
  collidable zero value;
- native publish/withdrawal and compact gmxpacked publish/withdrawal advance
  only their representation generation and the common lease epoch;
- a view declares `HOST_COMPLETE` or `PRODUCER_STREAM_ORDERED`. CUDA/HIP
  consumers currently require the same default producer stream explicitly;
  a different stream fails validation because no event handoff exists yet;
- regular LJ, soft-LJ and custom pairwise pin the provider, lease, exact
  representation generation and geometry generation at their immediate
  launch boundary. CPU additionally requires the native payload; CUDA/HIP
  requires the gmxpacked payload and pair-shift sidecar;
- `CLUSTERED_SPATIAL_VIEW_TEST` rejects stale provider, lease, native
  generation, gmxpacked generation and producer-stream identity. It also
  exercises the exact provider-retirement transition.

This slice deliberately does not make a bare grouped-index `ready` flag a
long-lived cache identity. Before SW, EDIP or Tersoff may retain a
grouped-center cursor, its key must be the exact tuple
`{provider incarnation, representation generation, derivation parameters}`.
Until then, grouped metadata remains call-scoped and is withdrawn with its
representation.

Clean-build evidence:

- `build-dev-cpu` builds `SPONGE` and
  `CLUSTERED_SPATIAL_VIEW_TEST`; CTest passes 1/1;
- `build-dev-cuda13` was rebuilt with `SPONGE --clean-first` and linked all
  123 compilation units; the retained `NBNXM_MICROBENCH` was then rebuilt;
- `git diff --check` passes. No CUDA kernel body, launch shape or
  synchronization changed in this lifecycle slice.

The expanded two-water-plus-DNA performance gate caught a measurement issue
that the wat160k-only gate could not expose. The first all-system run also
contained the one-time CUDA module/JIT cost of the freshly linked all-major
binary: its first DNA NVT row was `40.963978 ns/day`, while an immediate warm
repeat was `375.917419 ns/day`. The cold row is retained but excluded from
steady-state means.

The subsequent 18-row warm matrix was finite and confirmed the actual
parameter paths:

| system | ensemble | mean force | mean speed | historical speed anchor |
|---|---|---:|---:|---:|
| DNA_COU | NVT | 2.853164 s | 381.295837 ns/day | 368.115509 ns/day |
| DNA_COU | NPT | 2.992764 s | 362.587138 ns/day | 349.552999 ns/day |
| wat160k | NVT | 4.947289 s | 149.660904 ns/day | 148.528737 ns/day |
| wat160k | NPT | 5.057001 s | 144.467875 ns/day | 144.324870 ns/day |
| wat600k | NVT | 16.047052 s | 44.795453 ns/day | 47.187855 ns/day |
| wat600k | NPT | 16.111748 s | 44.143195 ns/day | 45.501160 ns/day |

DNA prints the packed-AB fallback on all six rows. Fresh wat600 replay reports
`lj_mode=comb`; the earlier wat160 payload uses the same combination path.
The wat600 absolute rows were polluted by an unrelated Chrome GPU process
that varied between roughly 14% and 30% idle SM use. They are not accepted as
evidence of a source regression or silently counted as passing rows.

Attribution used paired binaries and telemetry rather than selecting fast
outliers:

- three alternating wat600 NVT old/current pairs changed
  `Calculate_Force` by `+0.002%`, `+0.304%` and `+0.338%`;
- two absolute wat600 NVT samples with 96--99% measured SM activity and
  2,700--2,715 MHz measured `47.106762` and `46.657387 ns/day`; a later sample
  with a 2,235 MHz excursion is recorded as invalid telemetry, not averaged
  into the result;
- three telemetry-qualified wat600 NPT samples measured
  `15.829688`, `15.966656` and `15.812972 s` force time, mean
  `15.869772 s`, which is +2.71% versus the frozen `15.451507 s`;
  their mean speed is `45.120725 ns/day`, -0.84% versus the historical
  anchor;
- the fresh wat600 force-only isolated replay is `0.778995 ms` versus the
  documented `0.779630 ms` (-0.08%);
- production wat600 pressure/virial NCU is `1.36 ms` versus `1.33 ms`
  (+2.3%), with the same 13.4k-by-64 launch, 96 registers/thread and 41.67%
  theoretical occupancy. The larger spill-request count is the already
  accepted unified-full-template shape, not a new lifecycle change.

Fresh correctness evidence covers all parameter-path classes:

```text
wat600k metadata_ready=1 matched=1
payload=68526788 oracle=68526788
duplicates=0 missing=0 extra=0

DNA_COU metadata_ready=1 matched=1
payload=3479268 oracle=3479268
duplicates=0 missing=0 extra=0
```

The retained replay mismatch is now repaired without relaxing `2e-5`.
`SpongeProductionGmxpackedReplayKernel` compiles the same unified
`<energy=true, virial=true>` full entry, accepts the runtime store flags and
uses the production packed-AB two-part strided SCI traversal. Each work-part
CTA first merges the two internal warp virials through shared memory, then
performs one atom-virial atomic add, matching the production reduction order.
Its launch bounds now also reproduce the production combination and packed-AB
resource shapes.

Fresh strict full-output checks pass:

| fixture/path | requested output | force max scaled | energy max scaled | virial max scaled |
|---|---|---:|---:|---:|
| wat160k / combination | energy | `9.209522e-6` | `1.257829e-5` | n/a |
| wat600k / combination | energy + virial | `9.428581e-6` | `1.063029e-5` | `5.241917e-6` |
| wat600k / combination | virial | `1.144191e-5` | n/a | `5.710298e-6` |
| DNA_COU / packed-AB split2 | virial | `5.325069e-6` | n/a | `5.431496e-6` |

The post-change wat600 virial NCU replay is `1.27 ms`, 96 registers/thread,
41.67% theoretical and 36.60% achieved occupancy, with 13.32 million local
spill requests. This now matches the production unified-full resource class
(96 registers, 41.67% theoretical occupancy and 13.40 million requests);
the old incorrect replay was a 120-register virial-only specialization with
no spills. The DNA packed-AB report confirms a `996 * 2` grid, 96 registers,
4.35 KiB static shared memory and `120.77 us`.

The retained force-only entry is also narrowed to the actual production
variant: water combination payloads use one SCI work part, while packed-AB
payloads use contiguous split3. DNA now reports `0.078746 ms` in an unprofiled
50-iteration replay. Force-only NCU confirms a `996 * 3` grid, `84.42 us`, 70
registers/thread, no spills and 58.33%/36.06% theoretical/achieved occupancy.
The matching production atom-order split3 anchor is `89.12 us`; the retained
path is inside the gate and no longer benchmarks the obsolete one-part
algorithm.

The retained microbench now reports the selected route explicitly. A short
post-build DNA check prints
`output_mode=force-only lj_mode=packed-ab sci_work_parts=3
contiguous_sci_work=1 sanity=ok`; the strict full replay prints
`output_mode=full lj_mode=packed-ab sci_work_parts=2
contiguous_sci_work=0` and `matched=1` at the unchanged `2e-5` tolerance.
These fields are acceptance inputs, not informational decoration.

The replay-tool blocker is closed. The lifecycle implementation passes build,
contract, pair-set and code-specific performance checks. A fresh absolute
three-system end-to-end acceptance matrix still requires an uncontended GPU:
the desktop Chrome/Sunshine workload has repeatedly occupied roughly 25--46%
SM at the attempted idle windows (latest check: 32%), so rows collected under
that load are invalid rather than an acceptance waiver. Existing
matched/telemetry-qualified evidence above remains the source-change
attribution result.

The existing scratch matrix driver and analyzer do not mechanically enforce
this gate: they emit and summarize TSV rows but never fail on an incomplete
Cartesian product or a greater-than-3% cell. Their output alone therefore
cannot qualify a migration checkpoint.

The retained checker is
`benchmarks/performance/clustered_lj/check_migration_gate.py`, exposed as:

```text
pixi run -e dev-cuda13 clustered-lj-gate MATRIX.tsv REPLAYS.tsv
```

It returns nonzero unless it sees exactly wat160k, wat600k and DNA_COU, both
NVT and NPT, at least three valid 10,000-step baseline/current samples per
cell, plus force-only and full baseline/current replays for every system. It
checks force, wall time and speed independently for each end-to-end cell and
kernel duration for each replay cell. Replay requires at least three matched
cycles per cell and applies the 3% threshold to the median of the same-cycle
current/baseline ratios; improvements do not fail, but a greater-than-3%
regression in any cell does. It rejects invalid/non-finite rows, mismatched
cycles, short replay measurements and cross-system averaging. Route fields
must report `comb` for both water systems and
`packed-ab` for DNA; replay layout must be water split1, DNA force-only
contiguous split3 and DNA full split2. Strict full rows additionally require
`sanity=ok` and `matched=1`.

This is a fixed minimum acceptance set, not a menu of representative cases:
wat160k, wat600k and DNA_COU must all pass independently. Each must contribute
both NVT and NPT end-to-end cells and both force-only and full replay cells.
No mean across systems, faster water result or force-only fast path may hide
a regression in another system or in the full-output path.

CLI overrides may only tighten the protocol: the checker rejects a threshold
above 3%, fewer than three matrix runs, fewer than 10,000 steps, fewer than
three replay cycles, fewer than 2,000 replay iterations, and non-finite
thresholds such as `NaN` or infinity.

The positive and negative fixtures in
`benchmarks/performance/clustered_lj/tests/test_migration_gate.py` prove that
missing DNA, one regressing DNA/NPT cell, the wrong parameter route, a missing
full replay, a missing replay cycle, an invalid/short row, a failed strict
match, the wrong split layout or a paired-median replay regression all fail
the command. This checker is mandatory before the next consumer is enabled;
the scratch analyzer remains diagnostic only.

Matrix cases are staged from tracked immutable inputs, not copied from
historical `.tmp` run directories:

| gate system | tracked source |
|---|---|
| wat160k | `benchmarks/performance/wat/SPONGE_water_160k/water.top` and `water_npt_eq.gro` |
| wat600k | `benchmarks/performance/wat/SPONGE_water_600k_2x2x1/water_2x2x1.top` and `water_npt_eq_2x2x1.gro` |
| DNA_COU | `benchmarks/performance/sinkmeta/statics/dna_cou_sinkmeta/` with `Pmin_coordinate.txt`, `Pmin_velocity.txt` and the `2m2c_*` topology |

The retained producer is
`benchmarks/performance/clustered_lj/run_migration_matrix.py`, exposed as:

```text
pixi run -e dev-cuda13 clustered-lj-matrix \
  BASELINE_SPONGE CURRENT_SPONGE \
  CURRENT_ENVIRONMENT.json BASELINE_ENVIRONMENT.json OUTPUT_ROOT
```

It refuses fewer than three cycles or 10,000 steps, stages all 36
baseline/current cases with alternating implementation and ensemble order,
and refuses to reuse a non-empty output root. Before and after every launch
it queries GPU idle utilization; the maximum qualifying idle SM threshold is
5% and cannot be weakened by CLI. A contended, failed, timed-out, non-finite
or wrong-route attempt is written as `valid=0` with its reason and the
producer exits nonzero instead of silently dropping the row. `--dry-run`
stages the complete plan and manifest without touching the GPU.

The two environment files are deliberately distinct. The current environment
must be probe-free and exercises only promoted production dispatch. A
historical cleanup baseline may select the frozen DNA peak implementation only
with the complete split3/split2 probe set, dense padding and the historical
`SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT=1` opt-in. The opt-in remains active for
all historical baseline systems because that binary otherwise falls back to
native LJ; it is forbidden in current because the promoted production path no
longer has that reader. The producer removes the DNA-only keys from both water
systems. Unknown or partial probe sets fail before any GPU launch. Manifest
schema 2 records configured and per-system resolved environments, so a
comparison cannot silently run the baseline probes in current, the DNA probe
path in water or native LJ as the historical reference.

The end-to-end producer, replay producer and gate tests total 56 passing
cases. A repository-input dry-run with split current/historical environments
staged all 36 case directories and matched baseline/current mdin hashes in:

```text
/tmp/sponge-clustered-lj-matrix-dryrun-v4-20260727-WkKujX/
```

Its schema-2 manifest has zero probes in both baseline water systems, the
complete three-probe frozen set only in baseline DNA, zero probes in every
current system, and the historical direct opt-in in all baseline systems
only.

The first real idle precheck correctly refused the contended desktop without
launching SPONGE. It wrote
`valid=0 reason="pre-run idle SM 26.0% exceeds 5.0%"` and exited nonzero:

```text
/tmp/sponge-clustered-lj-matrix-idle-precheck-20260727/
```

A later acceptance attempt first rebuilt the current CUDA `SPONGE` target with
`--clean-first` (all 123 units) and rebuilt `NBNXM_MICROBENCH`, because the
existing binaries predated the latest clustered-layout header. The clean
microbench then reconfirmed DNA force-only as packed-AB contiguous split3 and
DNA full as packed-AB split2 with `matched=1`. The 36-cell matrix again failed
closed before its first launch when desktop utilization rose to 25%:

```text
/tmp/sponge-clustered-lj-matrix-real-20260727-clean/
```

That directory contains the invalid attempt row and staged input only; it has
no SPONGE stdout/stderr because the binary was not launched. The absolute
three-system gate therefore remains pending an uncontended run.

The retained replay producer is
`benchmarks/performance/clustered_lj/run_replay_matrix.py`, exposed as:

```text
pixi run -e dev-cuda13 clustered-lj-replays REPLAY_SPEC.json OUTPUT_ROOT
```

Legacy specs name exactly the baseline/current microbench binaries and one
shared force-only/full snapshot set for all three systems. Schema-2 specs may
instead resolve binary, snapshots, `sponge_lj_mode`, additional arguments and
environment independently for baseline and current. Protocol arguments
(`kernel`, snapshot, mode, warmup and iterations) cannot be overridden by the
free argument list, replay environments cannot use `_PROBE` keys as fake path
selectors, and only accepted production modes are allowed for each
system/output cell. The manifest records every resolved command, environment,
binary hash and snapshot hash.

Both end-to-end and replay launchers sanitize inherited state before starting
a child process: all parent `SPONGE_*` variables are removed, then the
validated per-case overlay is applied. Non-SPONGE variables such as CUDA
runtime/library configuration remain inherited. Manifest schema 2 records
this policy. Therefore a probe or legacy opt-in left in the interactive shell
cannot silently change a qualifying current or baseline row.

This distinction is required for the historical HEAD baseline. Its DNA
force-only row explicitly selects
`production-gmxpacked-sorted-force-sci-split3`; current selects the promoted
default `production-gmxpacked`. Full rows use the unified
`production-gmxpacked` entry. Paired baseline/current rows use byte-identical
snapshots even though the schema permits implementation-specific captures.
The producer runs three cycles of the 12 system/mode/implementation
combinations, for 36 required rows. Baseline/current remain adjacent within
each pair; implementation order is
baseline-current/current-baseline/baseline-current across the three cycles,
and the even cycle reverses system/output traversal to reduce warmup and order
bias. Every row has at least 2,000 measured iterations. The gate computes each
same-cycle current/baseline ratio and applies the 3% threshold to their median,
never to a cross-system aggregate. The producer parses the production
route/split fields and requires strict `matched=1` for every full row. It uses
the same non-weakenable 5% idle-SM pre/post checks and invalid-row behavior as
the end-to-end producer.

The historical `02272de` replay harness was rebuilt with only the exact
production full-output backport: unified energy+virial template, packed-AB
split2 launch, shared two-work-part virial merge and runtime store
suppression. Its existing dedicated force-only sorted split3 launch was kept;
the current generic force split3 and refresh changes were deliberately not
backported. The resulting baseline microbench SHA-256 is
`33f3ef89402b9edb65f4477f6d07af6b8dcbd7e2a70213260047f0b08ed90aac`.
One-iteration correctness checks (not performance samples) now report:

| baseline cell | route/layout | strict result |
|---|---|---|
| wat160k force-only | comb split1 | `sanity=ok` |
| wat160k full | comb split1 | `matched=1`, virial max scaled `7.564748e-6` |
| wat600k force-only | comb split1 | `sanity=ok` |
| wat600k full | comb split1 | `matched=1`, virial max scaled `5.218873e-6` |
| DNA_COU force-only | packed-AB contiguous split3 | `sanity=ok`, force max scaled `9.914495e-6` |
| DNA_COU full | packed-AB split2 | `matched=1`, virial max scaled `5.431496e-6` |

No tolerance was relaxed. The schema-2 dry-run proves all 36 rows resolve,
the three-cycle implementation order alternates, and every paired snapshot is
byte-identical:

```text
/tmp/sponge-clustered-lj-replay-dryrun-v4-20260727-DUQs7o/
/tmp/sponge-clustered-lj-replay-idle-precheck-20260727/
```

The latter observed 92% pre-run SM from an already-running external SPONGE
process, wrote the wat160k force-only baseline row as invalid and did not
launch the microbench.

The generated NVT/NPT mdins, both executable hashes, every tracked-input hash
and the exact environment are archived beside the raw TSV. Baseline and
candidate use byte-identical case inputs and matched alternating cycles.
Qualifying environment manifests contain the active-view/lifecycle pair
above and reject stale `SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT`, every
`*_PROBE` key and unsupported rolling-source-cache requests.

`lj_mode` is not inferred from the system name or from the absence of a
warning. It is copied from the retained replay snapshot/header for that exact
case generation. DNA additionally requires the production incompatibility
message selecting the AB-table path. Thus the two water rows prove
`use_lj_comb=1`, while DNA proves `use_lj_comb=0`; a hand-authored route label
cannot close the gate.

Raw artifacts:

```text
/tmp/sponge-representation-lifecycle-3system-warm-20260727/results.tsv
/tmp/sponge-lifecycle-wat600-snapshot-20260727/
/tmp/sponge-lifecycle-wat600-virial-snapshot-20260727/
/tmp/sponge-lifecycle-wat600-forceonly-20260727.ncu-rep
/tmp/sponge-lifecycle-wat600-virial-full-20260727.ncu-rep
/tmp/sponge-lifecycle-wat600-production-full-20260727.ncu-rep
/tmp/sponge-unified-replay-wat600-virial-20260727.ncu-rep
/tmp/sponge-unified-replay-dna-ab-split2-20260727.ncu-rep
/tmp/sponge-production-replay-dna-ab-force-split3-20260727.ncu-rep
/tmp/sponge-lifecycle-dna-snapshot-20260727/
/tmp/sponge-pre-lifecycle-dna-snapshot-20260727/
```

### Phase 1B checkpoint - custom/JIT pairwise opt-in - 2026-07-27

The first non-LJ implementation slice now proves the intended architecture
without deriving a half or full table:

- `PAIRWISE_FORCE` owns sorted coordinate, charge and pairwise-type channels
  and a fixed clustered traversal driver; the generated JIT fragment is the
  potential evaluator rather than an `ATOM_GROUP` loop;
- GPU consumes the authoritative gmxpacked SCI/CJ payload and explicit
  per-pair shift sidecar; CPU consumes the native SCI/CJ payload and its
  `shift_id`;
- the shared provider can be requested by a non-LJ consumer without creating
  fake LJ parameters, while regular LJ and soft-LJ retain independent
  consumer-selection flags;
- `SPONGE_CUSTOM_PAIRWISE_CLUSTERED_NATIVE=1` is an explicit opt-in. A stale,
  absent or incompatible view raises an error; it never selects the legacy
  pairwise kernel as a success path;
- the grid half-list reason is suppressed only for that explicit run. It is
  not removed from the default configuration.

Clean CPU and CUDA builds pass. A fresh 12-atom Morse differential fixture,
created without inherited output files, gives:

| comparison | maximum absolute force difference | RMS force difference |
|---|---:|---:|
| CPU legacy vs CPU clustered | `1.5258789e-5` | `3.9430543e-6` |
| GPU legacy vs GPU clustered | `1.5258789e-5` | `2.7500437e-6` |
| CPU clustered vs GPU clustered | `3.0517578e-5` | `7.4656100e-6` |

The clustered CPU/GPU runs both report `-1.760562e+02` effective potential
and `-537.24` pressure at the current output precision. The fixture also
passes a force-only dispatch with zeroth-frame and pressure output disabled.
This is an ABI/traversal correctness smoke, not a performance gate: its GPU
grid contains only one 64-thread block.

The first NCU-before-tuning reports are:

| mode | report | duration | registers/thread | spills | theoretical / achieved occupancy |
|---|---|---:|---:|---:|---:|
| force-only | `/tmp/custom-pairwise-clustered-forceonly-20260727.ncu-rep` | `5.66 us` | 48 | 0 | 83.33% / 3.72% |
| full | `/tmp/custom-pairwise-clustered-full-20260727.ncu-rep` | `6.94 us` | 48 | 0 | 83.33% / 3.71% |

The tiny launch is latency and underfill dominated, so its 0.02% SM
throughput and low achieved occupancy are not actionable tuning evidence.
It does establish that full adds instructions and output traffic without a
new virial-only kernel: executed instructions rise from 1,099 to 1,231 and
DRAM writes from 512 bytes to 4.608 KiB. There are no local-memory spills.
The largest sampled stall is long scoreboard (8.39 instructions per issue in
force-only, 7.62 in full), followed by wait (2.36 and 2.21), but a
representative multi-block custom-pairwise workload is required before launch
shape, atomics or data layout are changed.

#### Representative force-only tuning checkpoint

The representative workload is an 8,000-atom, 310-SCI synthetic Morse case
with a `310 x 8` grid, `8 x 8` blocks, 10,000 zero-timestep force evaluations,
8-Angstrom exact cutoff and 2-Angstrom skin. It is large enough to exercise
2,480 blocks rather than the single-block correctness fixture. Every retained
kernel change began from a fresh `--set full` NCU report and was re-profiled
before the next change:

| force-only implementation | duration | registers/thread | static shared/block | spills |
|---|---:|---:|---:|---:|
| shared-i per-pair atomics | `98.27 us` | 48 | 768 B | 0 |
| scalar i registers + warp-over-J reduction | `73.63 us` | 80 | 0 | 0 |
| plus shared i coordinate/charge/type/atom cache | `70.05 us` | 72 | 1.28 KB | 0 |
| plus 8-lane j-force subgroup reduction | `40.70 us` | 72 | 1.28 KB | 0 |

The retained final report is
`/tmp/custom-pairwise-scalarfci-icache-jreduce-20260727.ncu-rep`.
Relative to the immediately preceding i-cache report, the j reduction lowers
hot-kernel duration by 41.90% without changing the register, shared-memory or
spill envelope. Eligible warps per scheduler improve from 0.47 to 1.17 and
warp cycles per issued instruction fall from 19.74 to 8.98. The final kernel
executes 15,570,672 instructions, reaches 34.75% achieved occupancy, and its
largest stall ratios per issued instruction are wait 2.246, long scoreboard
1.839 and not-selected 1.223.

The first subgroup implementation is retained as negative evidence. Using a
full-warp CUDA synchronization mask while some 8-lane groups had skipped the
operation was invalid. Making every lane scan every otherwise empty `jm`
record fixed synchronization but regressed the 1,000-step force timer to
1.224281 seconds. The retained implementation first uses a subgroup ballot to
form a uniform eight-lane work branch, then uses the current active mask for
the width-eight shuffle.

The generated source has a measurable cold-JIT effect. A new source shape can
place about one second of compilation in the first `Calculate_Force` sample,
so a source-changing one-shot run is not a steady-state comparison. After both
sources were compiled, the matched 10,000-step measurements are:

| path | `Calculate_Force` samples | mean | core-wall samples | mean |
|---|---|---:|---|---:|
| clustered final | `0.992236`, `0.995912`, `0.990203` s | `0.992784 s` | `1.184088`, `1.192242`, `1.180502` s | `1.185611 s` |
| legacy | `1.085845`, `1.082870`, `1.082421` s | `1.083712 s` | `1.307717`, `1.306797`, `1.302789` s | `1.305768 s` |

The clustered force timer is 8.39% faster and core wall is 9.20% faster than
legacy on this force-only fixture. Two legacy runs under desktop GPU
contention took 12.27--13.75 seconds in the force timer while neighbor search
simultaneously grew from about 0.15 to 4.21--4.64 seconds. They are recorded
as device-contention diagnostics and are not mixed into either mean.

#### Final two-variant integration and representative full checkpoint

Adding runtime full-output control flow to the same generated kernel changed
the force-only schedule even when all output flags were false. Two final-source
NCU captures took `42.40 us` and `44.77 us`; at essentially identical SM
frequency, the latter executed 97,171 elapsed cycles compared with 88,231 in
the earlier `40.70 us` force-only checkpoint. This was a real compile-shape
effect rather than a dynamic-shared allocation: force-only still requested
zero dynamic shared memory.

The retained fix compiles exactly two clustered GPU variants from the same
evaluator and traversal source:

- **force-only** defines full output off at compile time, so energy, PME-direct
  energy and virial shared accumulation/reduction are absent;
- **full** defines full output on and uses runtime masks for listed energy,
  atom energy, PME-direct energy and all six virial components;
- there is no energy-only or virial-only kernel. Any requested output selects
  the one full variant.

The final force-only report is
`/tmp/custom-pairwise-forceonly-split-final-20260727.ncu-rep`: `42.30 us`,
69 registers/thread, 1.28 KiB static shared, zero dynamic shared and zero
spills. Relative to the unsplit final-source recheck, elapsed cycles fall from
97,171 to 92,221, registers from 72 to 69 and executed instructions from
15,513,513 to 14,443,189. The final 10,000-step samples are:

| path | `Calculate_Force` samples | mean | core-wall samples | mean |
|---|---|---:|---|---:|
| clustered two-variant final | `0.997728`, `0.992479`, `0.997589` s | `0.995932 s` | `1.197307`, `1.190009`, `1.194162` s | `1.193826 s` |
| previous clustered checkpoint | `0.992236`, `0.995912`, `0.990203` s | `0.992784 s` | `1.184088`, `1.192242`, `1.180502` s | `1.185611 s` |
| legacy | `1.085845`, `1.082870`, `1.082421` s | `1.083712 s` | `1.307717`, `1.306797`, `1.302789` s | `1.305768 s` |

The two clustered means differ by 0.32% in the force timer and 0.69% in core
wall, while the final variant remains 8.10% and 8.57% faster than legacy.
Thus full integration has not created an end-to-end force-only regression.

The representative full workload uses the same 8,000 atoms and launch shape,
but requests energy and pressure every step so every launch executes the full
variant. Each experiment was preceded by NCU and re-profiled:

| full output strategy | duration | registers/thread | dynamic shared/block | result |
|---|---:|---:|---:|---|
| per-pair global output atomics | `141.95 us` | 72 | 0 | baseline |
| contended shared per-i aggregation | `100.64 us` | 72 | 0 | rejected |
| private shared slots, partition 8 | `67.81 us` | 72 | 16.38 KiB | retained |
| energy/coulomb static + virial private | `69.38 us` | 71 | 12.29 KiB | rejected |
| private shared, partition 4 | `74.66 us` | 72 | 16.38 KiB | rejected |
| private shared, partition 1 | `201.06 us` | 72 | 16.38 KiB | rejected |
| separate full kernel, 64 output registers | `79.94 us` | 142 | 0 | rejected and removed |

The exact retained two-variant full report is
`/tmp/custom-pairwise-full-split-final-20260727.ncu-rep`: `68.54 us`,
18,509,512 executed instructions, 72 registers/thread, 1.28 KiB static plus
16.38 KiB dynamic shared, zero spills and 20.83%/18.31%
theoretical/achieved occupancy. It matches the pre-split `67.81 us` report
within 1.1%. Shared loads still average a 1.8-way bank conflict; 43.50% of
their wavefronts are affected. The rejected register variant removed that
traffic but rose to 142 registers and 23.32 million instructions, so it was
17.9% slower and left no experimental code in the branch.

The final full end-to-end samples are:

| path | `Calculate_Force` samples | mean | comparison |
|---|---|---:|---:|
| clustered two-variant final | `0.247887`, `0.249120`, `0.242784` s | `0.246597 s` | retained |
| clustered pre-split | `0.254511`, `0.253228`, `0.250702` s | `0.252814 s` | final is 2.46% faster |
| legacy | `0.204918 s` | `0.204918 s` | clustered is 20.34% slower |

The full kernel improves 51.71% over the original global-atomic clustered
kernel, but it still fails the migration gate: isolated kernel time is 25.85%
above the `54.46 us` legacy report and the end-to-end force timer is 20.34%
above the legacy sample. This remains an explicit Phase 1B performance
blocker. Default dispatch and deletion of the pairwise legacy path are not
authorized by this checkpoint.

Post-tuning validation passes:

- three GPU clustered Morse/LAMMPS perturbations;
- three CPU clustered Morse/LAMMPS perturbations;
- `ClusteredSpatialViewContract`;
- all four fresh-output runner tests;
- CUDA and CPU `SPONGE` builds, with no diff whitespace errors.

This closes the representative **force-only** kernel and end-to-end
performance item. It does not close Phase 1B. A representative full
energy-plus-virial report, the independent custom pair/shift/exclusion oracle,
two-rank ownership/reverse-force coverage, CPU/GPU full-output comparison and
NVE drift remain required before default dispatch or legacy removal.

Immediate closure order:

1. retain the completed fresh-output harness that treats `spongeError`,
   missing stop markers and absent outputs as failures;
2. add the independent canonical pair/shift/exclusion comparison for the custom
   driver, including non-central triclinic shifts and padded clusters;
3. add local/ghost ownership and two-rank reverse-force coverage;
4. retain the completed representative force-only result and add the matching
   full energy-plus-virial isolated timing and NCU report;
5. retain the passing CPU/GPU LAMMPS perturbations, then add CPU/GPU
   full-output comparison and NVE drift checks;
6. only after these gates and the upstream Phase 1A gates pass, enable
   clustered pairwise by default and delete its generated `ATOM_GROUP` loop
   and legacy half-list reason.

The next consumer work does not introduce a derived compatibility table.
After custom/JIT closure, implement SITS as a policy on the existing
regular/soft clustered evaluators, then migrate EAM as two direct pair-tile
passes around its per-atom embedding stage. REST/REST2 are not migrated in
this backport and must remain independent of SITS dispatch. In parallel with those
pair-decomposable consumers, add the generation-matched O(SCI) native
grouping plus O(CJ) endpoint-incidence cursor and its pair/triplet oracle.
Only after that center-complete cursor is validated do SW, EDIP and Tersoff
start, in that order. ReaxFF follows last: clustered tiles produce candidates,
while only accepted reaction edges and EEQ entries are compacted into
operator-owned structures.

## Scope

In scope:

- a reusable clustered traversal contract independent of LJ parameters;
- custom pairwise JIT forces;
- the SITS selective direct policy, as an extension of regular/soft LJ
  evaluators rather than an independent spatial-neighbor consumer;
- EAM;
- Stillinger-Weber;
- EDIP;
- Tersoff;
- the spatial-neighbor stages of ReaxFF;
- GPU and CPU implementations;
- PBC shifts, exclusions, local/ghost ownership, domain decomposition and
  reverse-force communication;
- force-only and full energy+virial output variants.

Explicitly out of scope:

- `SOLVENT_LENNARD_JONES`: clustered regular LJ already covers all local atoms,
  so the former solvent memory-access specialization is obsolete rather than a
  migration target;
- SPONGE Manager and Manager-specific orchestration;
- REST/REST2 migration or cleanup; its legacy path is retained outside the
  backport and must not gate SITS dispatch or acceptance;
- listed/bonded forces that use topology-local `Get_Local` metadata but do not
  traverse a spatial neighbor list;
- retaining a derived legacy `ATOM_GROUP` view as the final API.

## Consumer audit update - 2026-07-26

The live code narrows and corrects the earlier conceptual inventory:

| consumer | current spatial input | actual role | migration target |
|---|---|---|---|
| custom/JIT pairwise | half `neighbor_list.d_nl` | true pairwise spatial consumer | pair-tile evaluator; first migrated consumer |
| Stillinger-Weber | derived full `ATOM_GROUP` | two- and three-body center traversal | re-iterable center-tile algorithm |
| EDIP | derived full `ATOM_GROUP` | coordination, triplet and redistribution stages | pair pass + center range + pair redistribution |
| EAM | derived full `ATOM_GROUP` | density, embedding derivative and force stages | two pair passes with per-atom intermediates |
| Tersoff | derived full `ATOM_GROUP` | per-edge bond-order reduction over center neighbors | edge tile plus repeated K-tile traversal |
| ReaxFF | half and full `ATOM_GROUP` | mixed spatial candidates plus intrinsic reaction graph | clustered spatial front end; retain reaction graph/EEQ data |
| SITS | legacy direct LJ/Coulomb wrapper | selection/scaling policy over LJ/soft-LJ and listed terms | first non-LJ clustered evaluator/output policy; no independent neighbor API |
| REST/REST2 | legacy direct LJ/Coulomb wrapper | out-of-scope legacy behavior | no clustered migration; no SITS dispatch or acceptance coupling |
| Plugin/PRIPS | host `h_nl` count/index ABI | external compatibility surface, not an in-tree algorithm | versioned clustered-view ABI or explicit ABI retirement |

Evidence:

- `main.cpp:738-766` records the remaining legacy-list reasons:
  `selective_direct`, native LJ/soft-LJ, solvent LJ, custom pairwise, ReaxFF
  and full-list consumers.
- `main.cpp:830-875` shows that the current clustered-to-legacy route is an
  explicit compatibility adapter, not a target traversal API.
- `main.cpp:1774,1883-1911` dispatches ReaxFF, passes the derived full list to
  SW/EDIP/EAM/Tersoff and passes the half list to custom pairwise.
- `custom_force/pairwise_force.cpp:247-330,490-547` embeds the
  `ATOM_GROUP` loop in the generated JIT kernel and implements local/local
  versus local/ghost energy ownership.
- `manybody/eam.cpp:178-231`, `manybody/edip.cpp:444-500`,
  `manybody/sw.cpp:326-360` and `manybody/tersoff.cpp:409-435` launch the
  current full-list CUDA algorithms.
- `manybody/reaxff/reaxff.cpp:119-170` uses both half and full lists across
  EEQ, bond-order, VDW/bond, angle, torsion and hydrogen-bond stages.
- `Selective_Interaction/Selective_Interaction.cpp:50-149` dispatches SITS or
  REST2 through the direct LJ/soft-LJ entry points. Only SITS direct
  LJ/soft-LJ is migrated here; REST2 and both modes' listed interactions are
  outside this spatial migration.
- `plugin/plugin.cpp:79-104,170-172` exports legacy host-list capacity, count
  and index functions. Removing the grid list without an ABI decision would
  silently break external plugins.

Listed/bonded forces, walls and collective-variable modules were rechecked and
remain outside this plan unless an external plugin obtains the legacy neighbor
pointer and uses it.

### Consumer disposition ledger

The choice is made per mathematical stage, not per source file and not by
whether the old entry point happened to accept `ATOM_GROUP`.

| consumer/stage | clustered action | allowed derived state | legacy table disposition |
|---|---|---|---|
| regular LJ / soft-LJ | direct pair-tile traversal through the neutral view | sorted coordinates, parameters and output scratch | native/legacy LJ path removed after the frozen gate |
| solvent LJ | no migration; regular clustered LJ already covers the atoms | none | delete the obsolete specialization and its grid-list reason |
| custom/JIT pairwise | direct pair-tile driver with a generated potential evaluator | sorted pairwise type and dynamic charge channels | remove its half-list reason after GPU/CPU/DD gates |
| SITS direct | policy on regular/soft clustered evaluators | selection/scaling channels | remove the SITS portion of `selective_direct` only after SITS gates pass |
| REST/REST2 direct | retain legacy dispatch outside this backport | existing legacy state | retain its legacy-list requirement; never use it as a SITS fallback |
| EAM density/force | replay direct pair tiles | `rho`, `F(rho)`, `dF/drho` | remove EAM full-list reason after halo and CPU support |
| SW | grouped center cursor and J/K tile product | optional tile scheduling descriptor only after profiling | remove SW full-list reason |
| EDIP | direct coordination pass, grouped J/K pass, direct redistribution pass | `z`, `dE/dz` | remove EDIP full-list reason |
| Tersoff | direct i-j edge traversal with repeated grouped K traversal | accepted-edge/zeta scratch only if NCU justifies it | remove Tersoff full-list reason |
| ReaxFF candidate discovery | direct pair-tile candidate generation | canonical reaction edges, bond-order graph and EEQ sparse matrix | remove half/full inputs stage by stage |
| PME excluded, GB and 1-4 | no clustered migration; they already use operator-owned exclusion/topology relations | their existing private relations | no neighbor-list dependency to remove |
| bonded, listed, SHAKE and SETTLE | no clustered migration; topology determines the tuples | topology tuples and constraint state | no neighbor-list dependency to remove |
| plugin/PRIPS host neighbor ABI | versioned clustered capability or explicit retirement | plugin-owned data only | never synthesize a SPONGE-owned compatibility list |

The bonded and constraint exclusion is semantic rather than merely a scope
choice. Spatial proximity cannot recover ordered bond/angle/dihedral tuples,
force parameters, target constraint distances or SHAKE/SETTLE iteration state.
Likewise, PME exclusions, GB and 1-4 interactions already consume explicit
operator-owned relations and gain nothing from clustered traversal.

The full-list many-body consumers do not receive a permanent private full
adjacency as an intermediate migration step. Their first production
implementation is the direct grouped clustered algorithm described below.
An atom-pair-sized scheduling cache is considered only after that
implementation is correct, profiled and fails the locked performance gate for
a diagnosed repeated-traversal reason.

## Architecture decision: refactor consumers, do not derive spatial lists

The production default is **direct clustered-native algorithm refactoring**.
Do not derive a generic half list, full adjacency, `ATOM_GROUP`, CSR neighbor
list or per-atom vector merely to preserve the old loop shape.

This decision follows from the current code and performance target:

- `Build_From_Half` expands every half-list edge into a symmetric full list
  using atomics and fixed per-atom capacity
  (`neighbor_list/full_neighbor_list.cpp:27-79`). Keeping that representation
  preserves its build traffic, memory footprint and overflow model.
- The disabled compatibility adapter cannot prove the legacy global-atom-order
  half-list ownership from spatially owned clustered records
  (`clustered_lj.cpp:32764` and its caller at `main.cpp:820-887`).
- Pairwise consumers can evaluate SCI/CJ tiles directly; a derived list adds no
  mathematical information.
- Angular/bond-order consumers need a re-iterable grouped center range across
  every periodic shift. That is better represented by grouped SCI ranges than
  by rebuilding a generic atom-centric full list.
- Eliminating the grid/full-list builders is an end goal. Making a clustered
  adapter their permanent producer would preserve the same split architecture
  and prevent removal.

The per-consumer decision is:

| consumer/stage | direct clustered traversal | permitted derived state |
|---|---|---|
| custom/JIT pairwise | pair tiles | sorted consumer fields only |
| SITS direct | regular/soft LJ pair evaluator policy | selection/scaling outputs |
| REST/REST2 direct | no migration in this backport | existing legacy state only |
| EAM | density pair pass, embedding pass, force pair pass | `rho`, `F(rho)`, `dF/drho` |
| Stillinger-Weber | center J range and J/K tile product | optional active-lane/tile descriptor after profiling |
| EDIP | coordination pair pass, center J/K pass, redistribution pair pass | `z`, `dE/dz` |
| Tersoff | i-j edge tile with repeated K range | zeta or per-edge derivative scratch only if NCU justifies it |
| ReaxFF spatial front end | pair tiles | canonical reaction edges, reaction graph and EEQ matrix |
| Plugin | versioned clustered view | plugin-owned algorithm data, not SPONGE `ATOM_GROUP` |
| NO_PBC | central-shift clustered traversal if selected | none beyond consumer intermediates |

“Direct clustered traversal” does not require every consumer to scan the raw
builder arrays without an index. It has a precise two-level meaning:

- the shared spatial service may maintain generation-matched, read-only
  **structural indices** whose size is proportional to SCI/supercluster count,
  such as center-range offsets and an ordering of SCI IDs that groups every
  periodic-shift record for the same center;
- no production path may materialize an **atom-pair adjacency** whose size is
  proportional to accepted pairs, including half/full lists, CSR, per-center
  neighbor vectors or `ATOM_GROUP`.

The structural index does not duplicate pair identities or become a second
neighbor-list lifecycle. Consumers still decode the authoritative SCI/CJ
records, masks, exclusions and shifts in their tile driver. This distinction
is required for SW, EDIP and Tersoff: they need a re-iterable center range
across shifts, but they do not need the old atom-centric adjacency.

A derived operator relation is allowed in production only when all of the
following hold:

1. it represents an algorithmic relation beyond spatial proximity, such as an
   accepted reaction bond, sparse EEQ entry or accumulated intermediate;
2. it has canonical operator-specific identity and lifetime;
3. its existence is required by the downstream algorithm rather than by the
   shape of the legacy loop or API;
4. it is not exposed as a compatibility `ATOM_GROUP` or generic neighbor-list
   API.

An optional scheduling/cache structure has the stricter exception gate: the
correct direct implementation must miss the locked end-to-end gate by more
than 3%, NCU must attribute the loss to repeated clustered decoding or
traversal, and the complete build, memory and rebuild-cadence cost must be
included. This performance exception does not apply to intrinsically required
relations such as ReaxFF reaction edges or EEQ CSR, whose necessity follows
from the downstream graph/solver algebra.

Temporary half/full materialization is restricted to test binaries for
canonical pair-set differential checks. It must be compile-time isolated from
production dispatch and deleted when its oracle role is replaced by the
independent tuple comparison.

### Concrete routing rule: derive state or refactor the operator

Apply this decision in order for every old neighbor-list loop:

0. Treat an existing `ATOM_GROUP` parameter, generated JIT loop or
   `full_neighbor_list.d_nl` call site only as evidence of the legacy
   implementation. It is not evidence that the algorithm requires a derived
   table. The decision is made from the mathematical access pattern and
   lifecycle below.
1. If the loop consumes independent pairs, refactor it directly onto pair
   tiles. Do not derive a table.
2. If the loop needs to revisit all neighbors of a center, add or reuse the
   generation-matched grouped-SCI structural index and refactor the J/K
   traversal. Do not derive per-atom adjacency.
3. If the algorithm first filters spatial candidates into a new physical
   relation, derive only that operator-owned relation. ReaxFF reaction edges
   and EEQ entries qualify; raw spatial candidates do not.
4. If the sole reason is an external or legacy ABI, provide a versioned
   clustered-view capability or reject the ABI explicitly. Do not make SPONGE
   materialize a generic compatibility list.
5. If direct traversal is measurably worse after a correct implementation,
   profile first. The only permitted production cache is the smallest
   operator-specific descriptor justified by NCU and end-to-end data; it must
   not expose atom-pair adjacency as a shared neighbor API.

Consequently, the planned routing is:

| old consumption shape | migration form | derived data decision |
|---|---|---|
| one half/full pair loop | direct SCI/CJ pair-tile evaluator | none |
| multiple pair-decomposable passes | repeat pair-tile traversal | mathematical per-atom intermediates only |
| center J/K or edge K loop | grouped center range over authoritative tiles | SCI ordering/offsets only |
| spatial candidates becoming chemical edges | clustered candidate pass followed by operator compaction | canonical operator edges allowed |
| host/plugin `ATOM_GROUP` ABI | versioned clustered view or explicit retirement | no SPONGE-owned compatibility list |

This makes derivation a result of operator mathematics, not a mechanism for
preserving the legacy loop shape.

The implementation decision can therefore be made without ambiguity:

| question | yes | no |
|---|---|---|
| Does the stage consume independent spatial pairs? | direct pair-tile traversal | continue |
| Does it only need to reset/revisit a center neighborhood? | direct grouped-SCI cursor; derive only O(SCI) ordering/offsets | continue |
| Does candidate acceptance create a new physical/algebraic identity and lifetime? | derive the operator-private accepted relation | continue |
| Is the table needed only to preserve an old function/JIT/plugin ABI? | refactor or version/retire the ABI; do not derive | continue |
| Has a correct direct implementation missed the locked end-to-end gate for a profiled repeated-traversal reason? | consider the smallest private scheduling cache under the exception gate | no pair-sized materialization |

This is intentionally asymmetric: an O(SCI) grouping index is a shared
structural view of the authoritative payload, while an O(pair) **spatial
adjacency** is presumed forbidden. An operator-private accepted relation or
the smallest scheduling descriptor may itself be pair-sized only under the
semantic or measured exception gates above. Reaching the last row never
authorizes a generic half/full/CSR spatial table.

The practical ownership rule is:

```text
shared spatial service
  = authoritative SCI/CJ payload
  + generation-matched O(SCI) grouping/shift indices

consumer
  = direct clustered traversal
  + consumer-owned sorted fields
  + mathematical intermediates

operator-private relation
  = allowed only after candidate acceptance changes the identity/lifetime
    of the data (for example ReaxFF reaction edges or EEQ entries)
```

This explicitly rejects both shortcuts that would recreate the fork:

- deriving a shared half/full/CSR/`ATOM_GROUP` table from clustered records;
- giving each many-body consumer its own raw spatial adjacency merely to keep
  the old atom-centric loop shape.

### Implementation audit: derive or refactor - 2026-07-26

The live implementations confirm that the default must be an algorithm
refactor, not a clustered-to-legacy table adapter. An `ATOM_GROUP*` argument
describes the current storage shape; it does not establish a mathematical need
for that shape.

Use the following review rule for every stage:

```text
raw distance/cutoff eligibility
  -> traverse authoritative clustered tiles directly

repeat all neighbors of one center
  -> reset/replay a grouped-SCI center cursor
  -> share only O(SCI) offsets and SCI IDs

candidate acceptance creates a new operator identity
  -> compact the accepted operator-private relation

old ABI expects ATOM_GROUP/count/index
  -> version or retire the ABI; never synthesize a compatibility table
```

The resulting source-level disposition is:

| implementation stage | evidence in the current loop | first clustered implementation | retained/derived state |
|---|---|---|---|
| custom/JIT pairwise | generated loop at `custom_force/pairwise_force.cpp:490-574` evaluates independent half-list pairs and implements local/local versus local/ghost ownership | fixed clustered pair-tile driver plus generated potential evaluator | sorted pairwise type and dynamic charge channels only |
| SITS/REST2 direct | `Selective_Interaction.cpp:50-149` selects regular/soft direct evaluators; REST2's direct loop applies selection/lambda and ownership policy | regular/soft clustered evaluator and output policies | selection masks, lambda/scaling channels and selected outputs |
| EAM density | `manybody/eam.cpp:34-88` sums a scalar contribution for every eligible center-neighbor pair | direct center/pair-tile pass | per-atom `rho` |
| EAM force | `manybody/eam.cpp:114-176` replays the same spatial pairs using `dF/drho` from both endpoints | second direct pair-tile pass | per-atom `dF/drho`; no pair table |
| SW | `manybody/sw.cpp:217-324` scans J and the upper triangle `K > J` for one center | grouped center cursor and direct J-tile/K-tile product | no raw adjacency; an optional scheduling descriptor only after the exception gate |
| EDIP coordination/redistribution | `manybody/edip.cpp:339-442` performs two independent pair-decomposable passes | direct pair-tile passes | per-atom `z` and `dE/dz` |
| EDIP triplets | `manybody/edip.cpp:220-337` revisits K for each center/J lane | grouped center cursor and direct J/K product | no raw adjacency |
| Tersoff | `manybody/tersoff.cpp:97-269` traverses every K twice for each accepted i-j lane | direct i-j edge lane plus two grouped-K cursor passes | no initial pair table; optional accepted-edge/zeta scratch only after profiling |
| ReaxFF raw bond-order candidates | `manybody/reaxff/bond_order.cpp:4-86` filters spatial pairs and applies `bo_cut` | direct clustered candidate pass followed by deterministic compaction | accepted reaction-edge IDs and bond-order arrays |
| ReaxFF VDW | `manybody/reaxff/vdw.cpp:47-70` is an independent spatial pair loop | direct clustered pair-tile pass | none |
| ReaxFF EEQ entries | `manybody/reaxff/eeq.cpp:248-310` filters spatial candidates into H-matrix entries | count/scan/fill the EEQ matrix directly from clustered candidates | EEQ CSR, because it is solver algebra rather than a spatial cache |
| ReaxFF bond/graph stages | `manybody/reaxff/reaxff.cpp:141-169` currently mixes half/full list arguments with bond-order CSR | consume canonical reaction edges/bond CSR directly; hydrogen-bond spatial candidates use clustered tiles | reaction graph, corrected bond order and graph-derived tuples |
| Plugin/PRIPS | `plugin/plugin.cpp:79-104` exports host `ATOM_GROUP` count/index pointers | versioned immutable clustered-view capability or explicit initialization failure | plugin-owned data only |

This audit also exposes two removable false dependencies in ReaxFF.
`Calculate_Valence_Angle_Energy_And_Force` and
`Calculate_Torsion_Energy_And_Force` still accept `const ATOM_GROUP*`, but
their implementations do not dereference it; their topology traversal is
already based on the bond-order graph. Remove those unused parameters in the
ReaxFF API-cleanup commit rather than deriving a full list for them.
Hydrogen-bond code does still scan a spatial neighborhood and therefore needs
an explicit clustered candidate cursor before the final full-list reason can
be removed.

The many-body J/K products enumerate O(degree-squared) mathematical work, but
that is not permission to store an O(pair) spatial table. The first
implementation forms the product while replaying grouped tiles. It may emit
forces or an operator relation, but it must not first emit atom-neighbor rows.

#### Per-stage implementation order

The detailed order is intentionally chosen so each commit removes one
specific legacy dependency without introducing a new representation:

1. **Pair traversal driver.** Add common CPU and GPU pair-tile cursors over
   `CLUSTERED_SPATIAL_VIEW`, with consumer-owned fields and force-only/full
   output policies. They emit no pair array.
2. **Custom/JIT pairwise.** Split evaluator from traversal, port GPU
   force-only then full, add scalar/OpenMP parity, validate ownership and
   shifts, then remove only `pairwise_legacy`.
3. **SITS/REST2 policies.** Reuse the regular/soft pair driver, preserve the
   half-weight local/ghost energy rule and selected-output reductions, then
   remove only `selective_direct`.
4. **EAM.** Port density and force as two direct traversals around the
   embedding pass. Add typed `rho` and `dF/drho` halo exchange and CPU
   variants before removing EAM's contribution to `full_legacy`.
5. **Grouped center cursor.** Make gmxpacked grouped SCI metadata and the
   endpoint-incidence index generation-matched, deterministic and available on
   CPU/GPU. Test reset, two simultaneous cursors, transposed tile replay and
   J/K atoms from different periodic-shift SCI records. The native grouping
   size gate is O(SCI); center-complete endpoint incidence is O(CJ) with a
   fixed number of tile references and masks, never O(accepted atom pairs).
6. **SW, EDIP and Tersoff.** Port in that order. Each starts with direct
   grouped traversal, canonical triplet/edge-K oracles, CPU/GPU
   force-only/full and DD ownership. No scheduling cache is introduced before
   the direct path is correct and profiled.
7. **ReaxFF spatial front end.** Port EEQ counting/fill, raw bond-order
   discovery, VDW and hydrogen-bond candidate discovery separately. Compact
   accepted bonds into canonical stable edge IDs, then remove unused
   angle/torsion neighbor parameters and retire half/full list inputs stage by
   stage.
8. **ABI and provider removal.** Version or reject the plugin ABI, resolve and
   verify the independent NO_PBC policy, and only then remove the
   grid/full-list builders. The obsolete solvent-LJ specialization and its
   reason bit are deleted earlier, immediately after Phase 1A proves the
   regular clustered LJ replacement; other full-list consumers do not justify
   retaining the solvent specialization.

At every step, the legacy implementation may remain as an explicitly selected
test oracle. It must not be a production fallback from a failed clustered
selection. The corresponding reason bit in
`Main_Get_Legacy_Neighbor_List_Need` is removed only after the operator's
correctness, CPU/DD and performance gates pass.

### Decision record: structural grouping versus pair-table derivation

The migration does not make a single “derive or refactor” choice for every
array. It uses three deliberately different layers:

1. derive the generation-matched **O(SCI)** grouped structural index once;
2. refactor spatial consumers to traverse the authoritative clustered tiles
   through that index;
3. derive an **O(accepted relation)** table only after an operator has changed
   spatial candidates into a different mathematical object.

The grouped metadata demonstrates the first layer, with payload identity made
explicit.
`Count_Final_Sci_Per_Supercluster` and
`Fill_Final_Sci_Groups_By_Supercluster`
group every final SCI ID by `supercluster_id`. Native and gmxpacked payloads
have separate offsets/ID arrays because their SCI indices are not
interchangeable. `Build_Grouped_Sci_Metadata` retains the native diagnostic
index; `Build_Gmxpacked_Grouped_Sci_Metadata` constructs the production
gmxpacked index on demand. Because distinct periodic shifts are separate SCI
records in the same group, a center atom can iterate the complete J range,
reset the cursor and iterate it again, or form a J-tile/K-tile Cartesian
product without materializing atom-pair adjacency. Every cursor must load the
shift from each SCI and, when required by the compact active payload, the
explicit per-CJ/per-i-cluster shift metadata.

This directly resolves the apparent conflict exposed by the legacy many-body
kernels:

- SW's `nl_i` J loop and `k = j + 1` loop (`manybody/sw.cpp:189-317`);
- EDIP's reusable neighbor loops in force, `Get_Z` and redistribution
  (`manybody/edip.cpp:191-425`);
- Tersoff's J edge loop followed by two K scans
  (`manybody/tersoff.cpp:97-230`)

prove that each algorithm needs a re-iterable center neighborhood. They do
**not** prove that the neighborhood must be stored as `ATOM_GROUP`. The
clustered-native replacement is a center-lane cursor over all grouped SCI
records, CJ entries, lane masks, exclusions and explicit pair shifts. A reset
or a second cursor replays tiles; it does not rebuild a per-atom neighbor
vector.

In fact, the legacy `ATOM_GROUP` contains only a count and atom IDs. Those
kernels recompute a minimum-image displacement from coordinates and cannot
represent the clustered payload's explicit shift identity. A compatibility
table would therefore be both more expensive and semantically weaker unless
it invented a new pair-plus-shift ABI. That would become another neighbor-list
representation and is not the migration target.

The initial production choice per consumer is therefore:

| consumer | initial clustered implementation | pair-sized materialization |
|---|---|---|
| custom/JIT pairwise | direct eligible pair-tile pass | forbidden |
| EAM | replay direct pair tiles for density and force | forbidden; retain only `rho` and `dF/drho` |
| SW | grouped center cursor and upper-triangular J/K tile product | forbidden |
| EDIP | direct pair pass, grouped J/K pass, direct redistribution pass | forbidden; retain only `z` and `dE/dz` |
| Tersoff | direct i-j lanes with repeated grouped K traversal | no raw neighbor table; optional accepted-edge/zeta scratch only after profiling |
| ReaxFF | direct clustered candidate front end | required reaction-edge, bond-order and EEQ sparse structures are retained |
| plugin/PRIPS | versioned clustered-view ABI or explicit rejection | compatibility `ATOM_GROUP` forbidden |

ReaxFF is the important non-exception exception. Its EEQ CSR matrix and
bond-order/reaction graph arrays are operator-owned sparse algebra and
chemical topology, not cached spatial adjacency. They should be generated
directly from clustered candidates and retained for the solver and graph
stages. Raw candidates must not first be expanded into a half/full
`ATOM_GROUP` merely to feed those builders.

An optional pair-sized operator cache may be considered only after the direct
implementation is correct and profiled. The comparison must include
materialization time, kernel time, rebuild cadence, peak memory and
end-to-end speed; a faster consumer kernel alone is insufficient. It is
admissible only when:

- the direct path misses the locked performance gate by more than 3%;
- NCU attributes the loss to repeated clustered decoding/traversal rather than
  unrelated kernel structure;
- the cached rows represent accepted operator edges or the smallest
  scheduling descriptor, never a shared half/full spatial list;
- the cache is private to one operator and invalidated by the clustered view
  generation;
- the direct path remains the correctness oracle during introduction.

A single faster consumer kernel or a single NCU capture is insufficient. The
exception report must publish, for the same generation/rebuild cadence:

- structural traversal time;
- cache count/scan/fill time;
- consumer kernel time;
- cache bytes and total peak device memory;
- end-to-end time from three paired/interleaved runs;
- the NCU metric that attributes the direct-path loss to repeated decoding.

The shared structural indices have separate, mechanical size gates for each
backend:

- native-i grouping contains exactly one ID per SCI plus
  `supercluster_count + 1` offsets;
- center-complete endpoint incidence is bounded by a fixed number of
  `{sci_id, cj_id, orientation, cluster_mask}` references per authoritative CJ
  tile plus `cluster_count + 1` or `supercluster_count + 1` offsets;
- neither index may contain an accepted atom-pair ID, a variable per-atom
  neighbor row or an expanded exclusion result.

Thus an endpoint-incidence entry proportional to CJ count is structural and
permitted, while an entry proportional to accepted atom pairs is not.
Per-lane state is permitted only as a fixed-size bit mask stored in the tile
reference; expanding the set bits into rows fails the structural-index gate
and must be reviewed as an operator-private cache instead.

Thus the implementation order is not “derive first, refactor later.” The
first production implementation is direct clustered-native. Pair-table
materialization is test-only unless the operator creates a new physical
relation or later profiling satisfies the narrow exception above.

## Current reference semantics

The clustered payload supplies:

- sorted atoms grouped into 8-atom clusters and 8-cluster superclusters;
- SCI ranges over packed CJ tiles;
- valid and local lane masks;
- exclusion masks;
- periodic SCI and per-pair shift metadata;
- sorted-slot to local/ghost atom mapping;
- cache/rebuild state.

The current payload types live under the LJ module. The first refactoring step
must separate spatial traversal state from LJ parameter and output state
without changing the existing LJ kernel or builder behavior.

The compatibility hook
`Ensure_Legacy_Neighbor_View_From_Clustered_Payload` is intentionally disabled.
It remains useful only as a differential-test hook. It must not become the
production route for the migrated operators.

## Shared clustered traversal contract

The final interface has a structural view and consumer-owned sorted fields:

```text
ClusteredSpatialView
  `- CenterSuperclusterRange
       |- center clusters and lane masks
       `- re-iterable neighbor-tile range
            `- grouped SCI records from every periodic shift

ClusteredParticleFields<Consumer>
  |- sorted coordinates
  `- consumer parameters/channels
```

`ClusteredSpatialView` is immutable topology and eligibility state. It does not
own charge, LJ type, pairwise type, force scratch or energy/virial buffers.
Each consumer gathers only the fields it needs using the common sorted-atom
mapping. This prevents the shared API from becoming an LJ payload with a new
name and permits pairwise JIT, EAM and many-body consumers to use different
per-atom channels.

### Pair-tile traversal

Used by two-body and pair-decomposable stages:

```cpp
template<class PairEvaluator, class OutputPolicy>
void Clustered_For_Each_Pair_Tile(const ClusteredSpatialView& view,
                                  const ConsumerFields& fields,
                                  PairEvaluator evaluator,
                                  OutputPolicy output);
```

The traversal owns only geometry and pair eligibility:

- SCI/CJ decoding;
- valid/exclusion masks;
- sorted atom IDs and periodic shift selection;
- cutoff checks;
- local/ghost classification;
- view generation validation.

The consumer fields own coordinates and parameters. The evaluator owns
potential mathematics. The output policy owns force writeback, ghost handling,
energy partition and virial semantics. LJ ownership and local-ghost weighting
must not be hard-coded into the common traversal.

### Re-iterable center-tile traversal

Used by angular and bond-order potentials:

```cpp
template<class CenterEvaluator>
void Clustered_For_Each_Center_Tile_Range(
    const ClusteredSpatialView& view,
    const ConsumerFields& fields,
    CenterEvaluator evaluator);
```

For one center supercluster, the neighbor-tile range must group all SCI
records, including distinct periodic shifts. It must be safe to traverse the
range repeatedly or as a J-tile/K-tile Cartesian product.

This requirement is stronger than the current LJ pair loop. Processing each
SCI independently would miss triplets whose J and K atoms occur in different
periodic-shift SCI records.

### Required view metadata

The public traversal view should expose immutable counts, spans and generation
IDs. It must not expose builder capacities, dirty queues, incremental caches,
debug traces or LJ parameter arrays.

At minimum:

- local and ghost atom counts;
- sorted-slot to local/ghost atom ID;
- cluster and supercluster offsets;
- valid/local masks;
- grouped SCI indices;
- packed CJ records;
- exclusion records;
- per-pair shift metadata;
- cached cutoff/skin superset;
- layout/domain generation;
- backend/device kind.

The first concrete files should be
`SPONGE/neighbor_list/clustered_spatial_view.h` for POD types and
host/device decode helpers, plus
`SPONGE/neighbor_list/clustered_spatial_view.cpp` for cache validation and the
temporary adapter from `LJ_CLUSTER_LAYOUT`. Existing hot payload storage and
builder code stay in `clustered_lj.cpp` during Phase 1. Moving allocation and
construction ownership is a later commit after the view is proven.

## Operator algorithms

### Custom pairwise

This is the first direct pair-tile consumer.

- Replace the generated atom-centric `ATOM_GROUP` loop with a generated
  pair evaluator embedded in a fixed clustered tile driver.
- Gather `gpu_pairwise_types_local` into sorted order when the
  layout/domain generation changes. Gather coordinates every step through the
  shared sorted mapping. Pair coefficient tables remain consumer-owned.
- Evaluate only enabled pairs after the common mask and cutoff.
- Preserve current ownership exactly: local/local pairs are evaluated once and
  update both endpoints; local/ghost pairs update the local endpoint and
  contribute one-half energy/virial on each domain.
- Compile only force-only and full variants. Full uses runtime store masks for
  atom energy, virial, PME direct energy and listed-item energy.
- Provide a scalar/OpenMP driver over the same view and generated evaluator;
  the JIT potential expression must be shared with the GPU path.

No spatial-neighbor materialization is required.

### SITS and REST2

These are not separate neighbor-list migrations. They select or scale the
regular/soft LJ direct evaluator and listed topology terms.

- Add selection masks, lambda scaling and selected-output channels as
  evaluator/output policies on the clustered regular/soft LJ path.
- Do not create a SITS/REST2 clustered neighbor list.
- Backport only SPONGE runtime support and tests; the Manager smoke test remains
  out of scope with SPONGE Manager.
- When both policies use clustered direct, remove `selective_direct` as a
  reason for the grid builder in `Main_Get_Legacy_Neighbor_List_Need`.

### EAM

Use three stages over the same clustered view:

1. pair-tile traversal accumulates electron density;
2. per-atom embedding evaluates `F(rho)` and `dF/drho`;
3. pair-tile traversal evaluates pair and embedding force contributions.

`rho` and `dF/drho` are mathematical intermediates, not derived neighbor
lists. Domain-decomposed execution must synchronize the ghost values required
by the third stage.

### Stillinger-Weber

For each local center atom:

1. traverse J tiles for the two-body term;
2. traverse the upper triangle of J-tile x K-tile;
3. evaluate valid lane pairs with `j != k`, applying pair cutoffs and the
   canonical no-duplicate rule;
4. accumulate the center force locally and J/K forces with the
   operator-specific local/ghost policy.

An optional active-lane/tile descriptor cache may be added after profiling, but
an atom-centric adjacency list is not part of the target algorithm.

### EDIP

Use the existing mathematical phases, replacing every neighbor loop with
clustered traversal:

1. pair-tile pass computes coordination `z`;
2. center J-tile/K-tile traversal computes pair/triplet forces and `dE/dz`;
3. pair-tile pass redistributes the `dE/dz` contribution.

Only `z` and `dE/dz` scratch are required. Ghost coordination and derivative
state must be synchronized consistently with domain decomposition.

### Tersoff

For each valid i-j edge tile:

1. traverse every K tile for center i and reduce zeta;
2. evaluate bond order and direct pair terms;
3. traverse K tiles again to propagate zeta derivatives to i, j and k.

The first implementation should recompute K traversal without storing a
spatial adjacency or per-edge neighbor list. A per-edge zeta scratch is an
allowed later optimization only if NCU shows that kernel splitting reduces
register or dependency stalls.

### ReaxFF

Clustered traversal replaces the spatial candidate list, not the reaction
graph:

1. pair tiles generate raw bond-order candidates, VDW/Bond terms and EEQ
   matrix entries;
2. accepted bond-order candidates are compacted into canonical reaction-edge
   IDs;
3. corrected bond orders are evaluated;
4. a reaction-bond graph is built for over/under, angle, torsion and hydrogen
   bond stages;
5. derivative forces are propagated through reaction edges and clustered
   spatial pairs.

The reaction graph and the EEQ sparse matrix are intrinsic algorithm data
structures. They are not legacy spatial neighbor lists and are not removal
targets.

## Output variants

Every externally visible migrated force family exposes only:

- force-only;
- full energy+virial.

Energy-only and virial-only requests reuse full with runtime store masks.
Internal dependency stages such as EAM density or EDIP coordination are shared
and do not multiply output variants.

## CPU and GPU policy

The logical tile traversal and pair eligibility rules must be common.

- GPU uses the packed warp-oriented representation.
- CPU uses scalar/OpenMP loops over a normalized clustered view; it must not
  depend on CUDA warp ABI details.
- CPU/GPU pair-set tests must compare canonical
  `(global_i, global_j, shift_id)` tuples.
- Floating-point reduction order may differ, but pair ownership, exclusions
  and requested outputs must not.

## Domain decomposition and lifecycle

The traversal view is invalidated when any of the following changes:

- local/ghost atom membership or ordering;
- sorted permutation;
- cluster/supercluster topology;
- exclusion metadata;
- cell/periodic shift metadata;
- maximum requested cutoff or skin.

A shared service builds a candidate superset at the maximum active consumer
`cutoff + skin`. Each evaluator applies its own exact cutoff. Per-operator
exclusion policies must be explicit; the current global nonbonded exclusion
policy may be retained only where it matches existing behavior.

Force writeback is operator-specific:

- pairwise may write only locally owned partner forces;
- SW/EDIP/Tersoff may accumulate ghost J/K forces followed by reverse-force
  communication;
- EAM and EDIP require intermediate halo synchronization between stages.

The present DD implementation exchanges coordinates and performs reverse
ghost-force accumulation
(`Domain_decomposition.cpp:877-927,977-1028`), but it has no generic halo
exchange for consumer intermediates. Current SW/EDIP/EAM/Tersoff entry points
also receive only `dd.atom_numbers`, not explicit local/ghost ownership.
ReaxFF synchronizes charge only in the one-PP-process case. Therefore:

- custom pairwise may be enabled for DD only after its existing local/ghost
  ownership is reproduced and tested;
- EAM must add halo exchange for density/embedding derivative state;
- EDIP must add halo exchange for `z` and `dE/dz`;
- SW/Tersoff must center work on local atoms, accumulate J/K ghost forces and
  use reverse-force communication;
- ReaxFF needs a separate typed halo/edge-state design before clustered
  dispatch can be enabled in multi-rank runs.

SW, EDIP, EAM, Tersoff and most ReaxFF stages are currently CUDA-only. A
clustered GPU result must not be presented as removal-ready until the
corresponding scalar/OpenMP path exists, or the configuration is rejected
explicitly instead of silently falling back to the legacy list.

## Pre-migration performance baseline

**Performance status: complete.** The locked matrix and post-change NCU reports
are recorded in `clustered-cleanup-performance-recheck-20260725.md`. The
protocol below remains the per-commit regression gate.

### Locked hardware/protocol

- RTX 4090;
- one process, no profiler wrapper for end-to-end acceptance;
- CUDA 13 pixi environment;
- 10,000 steps for accepted end-to-end cases;
- three paired/interleaved runs for an acceptance decision;
- exact system, ensemble, cutoff, skin and output mode matching the historical
  comparison.

### Required systems

1. water 160k force-only NVE peak;
2. water 600k force-only NVE peak;
3. DNA_COU force-only;
4. DNA_COU full energy+virial/NPT;
5. water 160k and 600k NVT/NPT public-comparison cases;
6. regular-LJ force-only and full-output microbench replay;
7. soft-LJ full kernel timing and its existing correctness fixtures.

DNA is mandatory. The cleanup removed the former default-off
`FORCE_SCI_SPLIT3_CONTIGUOUS_PROBE` and
`ENERGY_VIRIAL_SCI_SPLIT2_PROBE`. If those implementations contained accepted
peak performance rather than disposable experiments, the regression must be
fixed or the winning implementation promoted under production names before
the baseline can be frozen.

### Historical anchors

The first cleaned-branch run is compared with the documented same-machine
anchors:

| system/mode | historical current mean |
|---|---:|
| wat160k NVT | 148.528737 ns/day |
| wat160k NPT | 144.324870 ns/day |
| wat600k NVT | 47.187855 ns/day |
| wat600k NPT | 45.501160 ns/day |
| DNA_COU NVT | 368.115509 ns/day |
| DNA_COU NPT | 349.552999 ns/day |
| wat160k NVE peak envelope | 143.048642 ns/day |
| wat600k NVE peak envelope | 47.274160 ns/day |
| DNA_COU NVE peak envelope | 299.265869 ns/day |

Historical values are diagnostics, not a reason to accept a broken current
binary. This is an all-system AND gate: wat160k, wat600k and DNA_COU must each
pass their own matching rows. A better result in one system cannot offset a
regression in another. The cleaned implementation becomes the migration
reference only when:

- all runs are finite and pass numerical checks;
- every required system's valid three-run mean is within 3% of its matching
  historical current mean. A protocol/input mismatch or external GPU
  contention invalidates that row and requires a matched rerun; it is not an
  acceptance waiver;
- `Calculate_Force` does not regress by more than 3%;
- kernel-only replay does not regress by more than 3%;
- the current implementation remains faster than public SPONGE in every
  locked same-system/same-ensemble comparison.

Any larger regression blocks migration and triggers attribution:

1. end-to-end paired repeat;
2. kernel-only snapshot replay;
3. NCU profile covering roofline, memory hierarchy, warp stalls, instruction
   mix and occupancy;
4. restore/promote only the proven winning implementation;
5. rebuild, re-run correctness, re-profile and repeat the paired benchmark.

No kernel or launch change is allowed without a preceding NCU profile.

### Baseline artifacts

Store cleaned-branch results outside source directories during execution, then
commit a concise result note containing:

- exact binary commit/worktree diff state;
- GPU and CUDA versions;
- complete environment;
- input hashes or canonical paths;
- raw per-run speed, wall and `Calculate_Force`;
- snapshot payload counts;
- microbench average and variance;
- NCU report paths for any diagnosed regression;
- pass/fail decision.

## Validation gates per migrated operator

Before enabling a clustered-native consumer:

- canonical pair/edge set matches an independent oracle;
- random PBC boxes cover central and non-central shifts;
- exclusions, empty/padded clusters and local/ghost boundaries are covered;
- DD repartition and rebuild-generation changes invalidate stale views;
- force, total energy, per-atom energy and virial match the existing
  implementation within an operator-specific tolerance;
- NVE energy drift is not worse;
- CPU and GPU correctness agree;
- force-only and full are the only instantiated output variants;
- end-to-end and isolated kernel performance pass the locked baseline.

Triplet potentials additionally compare canonical `(i,j,k,shift_j,shift_k)`
sets on small systems. ReaxFF additionally validates reaction-edge IDs, bond
orders, EEQ convergence and graph-derived angle/torsion/HB counts.

Existing validation starting points:

| consumer | existing test | required addition |
|---|---|---|
| custom/JIT pairwise | `benchmarks/comparison/tests/lammps/tests/comp_custom_force.py:209-344` Morse energy/force/pressure comparison | clustered/legacy differential, CPU/GPU, exclusions, non-central shifts and DD ownership |
| SITS | `benchmarks/validation/rest2/tests/test_selective_interaction_sits.py` and `benchmarks/performance/sits/tests/test_sits.py` | clustered regular/soft direct, force-only/full |
| REST2 | `benchmarks/validation/rest2/tests/test_rest2.py:24-58` | exclude Manager smoke from backport; add clustered direct output comparison |
| EAM | `benchmarks/comparison/tests/lammps/tests/comp_cuni_eam_alloy.py` | intermediate-state and CPU/GPU checks |
| EDIP | `benchmarks/comparison/tests/lammps/tests/comp_edip.py` | canonical triplets and `z/dE_dz` checks |
| Tersoff | `benchmarks/comparison/tests/lammps/tests/comp_tersoff.py` | canonical edge/K sets and CPU/GPU checks |
| ReaxFF | `benchmarks/performance/reaxff/tests/test_petn_lammps_frame.py` and `test_petn_nve_perf.py` | reaction-edge/EEQ/graph-count checks |
| Stillinger-Weber | no dedicated validation found | add an independent energy/force/virial and canonical-triplet fixture before implementation |

The retained microbench now contains the independent canonical pair oracle,
and `CLUSTERED_SPATIAL_VIEW_TEST` covers the structural-view contract. Each
new consumer must extend those fixtures with its operator-specific
pair/triplet/edge relation before production dispatch; using the legacy
`ATOM_GROUP` output itself as the oracle is not sufficient.

### Per-consumer production/removal gate

The target representation and the legacy-removal decision are coupled. A new
clustered kernel may exist behind a test dispatch before these gates pass, but
its old grid/full-list reason remains enabled and it may not silently fall
back from a failed production selection.

| consumer | first production representation | correctness gate | CPU/DD gate | performance evidence | legacy reason removed |
|---|---|---|---|---|---|
| custom/JIT pairwise | direct pair tiles | exact canonical pairs; Morse force/energy/virial; exclusions and shifts | scalar/OpenMP parity; two-rank local/ghost ownership and reverse force | isolated plus end-to-end within 3%; NCU before launch tuning | remove `pairwise_legacy` |
| SITS/REST2 direct | clustered LJ/soft evaluator policy | existing SITS/REST2 outputs; force-only/full | implement scalar/OpenMP for every supported CPU mode; reproduce clustered LJ/soft DD ownership or reject unsupported CPU/DD configurations explicitly | no greater-than-3% end-to-end regression | remove `selective_direct` |
| EAM | two direct pair passes plus `rho`/`dF_drho` | pair oracle, forces/energy/virial and intermediate-state comparison | scalar/OpenMP; typed `rho`/`dF_drho` halo test | count both traversals and halo cost; within 3% | remove EAM contribution to `full_legacy` |
| SW | grouped center J/K tiles | independent force/energy/virial fixture and exact canonical triplets | scalar/OpenMP; two-rank J/K ghost-force reversal | direct implementation first; NCU plus complete test-only table build/memory comparison | remove SW contribution to `full_legacy` |
| EDIP | direct coordination, grouped J/K and direct redistribution | exact pairs/triplets and `z`/`dE_dz` comparison | scalar/OpenMP; typed intermediate halos and reverse force | all three stages plus halo within 3% | remove EDIP contribution to `full_legacy` |
| Tersoff | direct edge tile with repeated grouped K traversal | exact edge/K tuples, zeta, forces/energy/virial | scalar/OpenMP; two-rank edge/K force ownership | direct path first; private accepted-edge/zeta cache only under exception report | remove Tersoff contribution to `full_legacy` |
| ReaxFF | direct candidate tiles plus intrinsic reaction/EEQ graphs | stable canonical edge IDs, bond orders, graph counts, EEQ convergence and outputs | CPU support or explicit rejection; typed charge/edge/derivative halo gate | candidate build + graph/EEQ + end-to-end within 3% | remove `reaxff_legacy` and its full-list request stage by stage |
| plugin/PRIPS | unresolved Phase 5 decision: versioned clustered capability or explicit rejection | ABI version/error text and capability validation | host/device backend behavior is explicit | initialization and any plugin traversal cost documented | decide before provider deletion; retire host `h_nl` API and never emulate it |

## Migration order and stop conditions

The dependency order is:

```text
regular-LJ Phase 0 performance baseline (complete)
  -> neutral spatial view + pair oracle
       -> pair-image ownership gate
            -> regular-LJ no-change proof
                 -> soft-LJ correctness + performance gate
                      -> custom pairwise GPU + CPU
                           -> SITS clustered evaluator policy
                                -> EAM pair-stage migration
                                     -> reusable center-range traversal
                                          -> SW -> EDIP -> Tersoff
                                               -> ReaxFF spatial front end
                                                    -> Plugin/NO_PBC decision
                                                         -> grid/full-list removal
```

### Phase 0: freeze the regular-LJ performance baseline - complete

- Alternating water/DNA NVT/NPT matrix and wat600k 30,000-step guardrail pass.
- Retained force-only and full kernels have post-change NCU reports.
- Inputs, result tables and acceptance thresholds are frozen in the cleanup
  performance note.
- GPU minimization, full-output comparison and the retained regular-LJ
  microbench subsequently passed in Phase 1A. Pair-image ownership and
  dedicated soft-LJ performance remain independent blockers before Phase 1B
  or backport.

### Phase 1A: neutral view and pair oracle

Implement this as small reviewable commits:

1. **Canonical oracle.** Extend the retained microbench/test support to emit and
   compare sorted canonical `(global_i, global_j, shift_id)` tuples. Cover
   exclusions, padded lanes, central/non-central shifts, local/local and
   local/ghost pairs. The independent oracle reads coordinates, box and
   exclusions; it must not decode the clustered payload using the same helper
   under test.
2. **Neutral POD types.** Add
   `neighbor_list/clustered_spatial_view.h`; move or alias only SCI/CJ,
   exclusion, shift and decode types. Keep ABI/static-size assertions. Do not
   move builder queues or LJ parameter/output arrays.
3. **Validated view factory.** Add
   `neighbor_list/clustered_spatial_view.cpp` and a factory from the current
   `LJ_CLUSTER_LAYOUT`. Reject stale generation, insufficient
   `cached_cutoff + skin`, mismatched local/ghost domain, missing grouped SCI
   ranges or backend mismatch.
4. **Re-iterable grouping contract.** Expose grouped SCI offsets/IDs so a
   center range covers every periodic-shift record. Add a test where J and K
   occur in different shift records; a per-SCI-only implementation must fail
   this fixture.
5. **LJ no-change proof.** Change regular LJ and soft-LJ launch signatures to
   accept the neutral view while retaining the same pointers, launch geometry,
   pair body and output buffers. No builder ownership move is allowed in this
   commit.
6. **Phase gate.** Build CUDA and CPU, run the pair oracle, regular/soft
   force-only/full correctness, the frozen end-to-end matrix and microbench.
   Compare registers, spills, occupancy and kernel duration against the frozen
   NCU reports. Any kernel/code-size or greater-than-3% regression stops Phase
   1B.

Phase 1A is complete only when all closure conditions hold together:

- canonical pair/oracle and pair-image ownership gates pass on the required
  small-box and wat160k fixtures, including exclusions, same-count compact
  replacement generations and geometry-epoch invalidation;
- regular-LJ and soft-LJ force-only/full correctness and the frozen
  performance/NCU gates pass;
- the long-run wat160k finite-state gate passes;
- production LJ no longer includes structural types from an LJ-owned header.

`LJ_CLUSTER_LAYOUT` may still own allocation and build implementation
temporarily; that dependency is one-way through the view factory. Structural
header decoupling alone is not Phase 1A completion.

### Phase 1B: custom/JIT pairwise

1. Split the generated JIT source into a potential evaluator and fixed
   traversal driver. Preserve the existing SAD derivative expression and
   coefficient table ABI.
2. Add consumer-owned sorted buffers for pairwise type and any required charge
   channel. Pairwise type is refreshed on layout/domain generation; coordinates
   and dynamic charge are gathered at the required step cadence.
3. Implement GPU force-only first. Validate canonical pair ownership and force
   against legacy and LAMMPS before adding outputs.
4. Profile the force-only kernel with NCU before changing launch layout. Analyze
   roofline, cache traffic, scheduler/warp stalls, instruction mix, occupancy,
   registers and spills. Tune only from that evidence and re-profile every
   retained change.
5. Add the single full variant with runtime masks for atom energy, virial, PME
   direct energy and listed-item energy. Do not instantiate energy-only or
   virial-only kernels.
6. Implement the scalar/OpenMP driver over the same view/evaluator and compare
   canonical pairs plus force/energy/virial with GPU.
7. Run the three-perturbation Morse/LAMMPS test, randomized PBC/exclusion
   fixtures, CPU/GPU comparison, two-rank ownership/reverse-force test, NVE
   drift and isolated/end-to-end benchmarks.
8. Switch production dispatch to clustered pairwise. Remove
   `pairwise_legacy` from `Main_Legacy_Neighbor_List_Need` only after all gates
   pass; keep an explicit test-only legacy differential command until the next
   cleanup checkpoint.
9. Delete the generated `ATOM_GROUP` loop and its pairwise-only legacy
   plumbing. Preserve only the explicitly selected, test-only legacy
   differential oracle; it must not be reachable as production fallback. Do
   not delete the grid builder yet because full-list and ReaxFF consumers still
   require it.

The acceptance target is numerical parity within the existing LAMMPS
tolerances, exact canonical pair ownership, no stale-view access, and no
greater-than-3% isolated or end-to-end regression. A faster kernel does not
waive a failed ownership or DD test.

### Phase 2: remaining pair-decomposable policies and EAM

1. Implement SITS as a selection/scaling and output policy on regular and
   soft clustered LJ. Its topology-listed terms stay unchanged. Add
   scalar/OpenMP execution for supported CPU configurations and reproduce the
   clustered LJ/soft local/ghost ownership for supported DD; unsupported
   CPU/DD combinations must fail explicitly rather than select legacy.
   Develop and time experimental scheduling only in the retained microbench.
   Once the fused implementation passes correctness and performance there,
   replace the production SITS direct path in one step and remove the SITS
   portion of `selective_direct`. Do not add a production probe, environment
   opt-in, dual-path dispatch or fallback gate. REST/REST2 remain legacy and
   outside this backport; they must not alter SITS dispatch or acceptance. Do
   not migrate SPONGE Manager.
2. Migrate EAM in density, embedding and force commits. Add typed halo exchange
   for density and `dF/drho`, plus scalar/OpenMP implementations, before
   enabling DD or removing its full-list dependency.
3. After EAM passes, move allocation/build ownership of the structural payload
   from `clustered_lj.cpp` into the neighbor-list service. LJ becomes an
   ordinary consumer. Re-run the frozen LJ performance matrix because this
   changes lifecycle ownership.

### Phase 2 SITS implementation checkpoint - 2026-07-27

The first SITS clustered implementation now consumes
`CLUSTERED_SPATIAL_VIEW` directly on CUDA and CPU. It uses the gmxpacked
SCI/CJ/exclusion/explicit-pair-shift payload on CUDA and the native payload on
CPU, implements only force-only and full templates, and is selected only by
`SPONGE_SITS_CLUSTERED_NATIVE=1`. REST/REST2 and SPONGE Manager do not
participate in this dispatch. The CUDA kernel covers selected and
non-selected pairs in one traversal, so the clustered route does not launch
the old solvent-LJ fast path. This environment-selected route is an existing
experimental checkpoint, not the intended production shape: the fused
replacement must delete the opt-in and obsolete standalone dispatch rather
than add another probe or gate.

The ALA2 SITS differential test passes on both CPU and CUDA:

```text
2 passed (dev-cpu)
2 passed (dev-cuda13)
```

NCU-guided retained changes:

- partitioning each SCI over up to eight cjpacked ranges reduced the original
  full kernel from 311.360 us to 70.752 us;
- reducing J-end force and enhancing-force writes over the eight I lanes
  reduced it further to 59.520 us, with global reductions falling from
  100,040 to 74,552 and no spills.
- loading `atom_sys_mark` and calculating the SITS factor only for selective
  owners is retained as semantic cleanup; two NCU repeats measured 60.320 and
  59.808 us with the same 64 registers/thread, inside the retained path's
  run-to-run band.
- promoting the four independent `jm` records from a serial loop to
  `grid.z=4` is retained. Two full NCU runs measured 40.960 and 40.192 us.
  The grid grew from `(132,8,1)` to `(132,8,4)`, waves/SM from 0.52 to 2.06,
  and achieved occupancy from 11.13% to 32.76--33.03%, with 64
  registers/thread and no spills.

The implementation is therefore a valid correctness path but not yet a
migration-qualified peak path. The same full ALA2 sample now measures about
40.2--41.0 us, while the legacy selective plus solvent kernels total about
33.376 us (`15.616 + 17.760 us`). The remaining gap is about 20--23%. The
following alternatives were measured and rejected rather than retained:

- regular clustered full base plus a SITS correction kernel: the regular base
  was 402.080 us on the generic route, 166.560 us with active view, and
  79.552 us even with dense AB split2, before adding the correction;
- sparse correction partition collapse: 41.664 to 253.056 us;
- warp/block aggregation of energy outputs: about 60.6--61.6 us versus the
  59.520 us reference;
- shared-atomic I-force aggregation: 83.840 us;
- register plus two-warp I-force aggregation: 66.400 us;
- sorted-force scratch: 58.432 us for the main kernel plus 3.200 us scatter
  and an additional memset, for no end-to-end win.
- raising the partition cap from eight to sixteen did not change this
  workload's `(132,8)` grid and measured 61.824 us;
- replacing the complete pair math locally with regular-clustered scalar
  helpers measured 64.064 us; sharing only LJ `inv_r2/inv_r6` while retaining
  the existing Coulomb path measured 42.080 us versus the 40.192 us retained
  `jm`-grid result;
- splitting the I-cluster loop into two additional work parts raised achieved
  occupancy to 44.34%, but duplicated control and J writeback, increased
  issued instructions from about 6.386M to 7.837M, and regressed to
  43.104 us.

Reports are retained under `ncu_reports/sits_*.ncu-rep`. These results rule
out adding more local reductions or a second regular-base traversal. The next
SITS performance step is the fused all-pairs-plus-mask design: make SITS a
compile-time policy inside the retained warp-record evaluator so every pair's
distance, LJ, PME and ordinary force are computed once, then accumulate the
enhancing force/energy/virial only when the selected-owner mask is active.
This is not a regular-LJ launch followed by a correction launch. It must share
the evaluator's existing force writeback, arrange selected/non-selected
output ownership without per-pair atomics where possible, and must not
materialize an `ATOM_GROUP`-like derived neighbor table. The existing regular
full-output path is itself 79.552 us on this small sample, so the policy must
preserve the force-only evaluator's register footprint and use
selective-specific output buffering rather than blindly instantiating that
full-output specialization. Prototype and compare this policy inside the
existing microbench only. If it is not within 3% of the legacy total, do not
introduce it into production. Once it passes, directly replace the production
SITS path, remove `SPONGE_SITS_CLUSTERED_NATIVE`, delete the standalone
clustered SITS evaluator and remove the SITS legacy-list reason. There must be
no new runtime A/B switch or compatibility fallback.

The mask is an additional-output policy, not an ordinary-force acceptance
mask. Every accepted clustered pair must still contribute once to the normal
LJ/direct-Coulomb force. For the SITS enhancing outputs, the pair weight is
derived from the two atom marks: mark sum zero maps to `1`, mark sum one maps
to `pwwp_enhance_factor`, and mark sum two maps to `0`. Force-only therefore
still needs the enhancing-force accumulator but must not instantiate energy
or virial state. Full additionally accumulates the weighted enhancing
energy/virial while reusing the already-live distance powers, LJ coefficients,
charge product and PME intermediates. The fused dispatch replaces the
standalone clustered SITS evaluator; it must never launch regular LJ first
and then SITS, because that would traverse and count the ordinary interaction
twice.

#### Fused SITS microbench checkpoint - 2026-07-27

The first force-only fused policy was implemented only in the retained
`NBNXM_MICROBENCH`; no production probe, environment gate or dispatch was
added. It leaves ordinary force on the existing warp-record reduction and
adds enhancing force only for atoms below the configured SITS boundary. Its
built-in checks cover two mask invariants: with every atom selected,
enhancing force matches ordinary force within `2e-5` scaled tolerance; atoms
outside the selected range remain exactly zero.

On the ALA2 AB-table snapshot, the microbench replay measured 35.207 us for
ordinary force-only and 41.724 us for the sparse 22-atom fused policy. NCU
measured 41.54 and 49.09 us respectively under replay: the fused policy added
about 25% executed instructions, 30% global-load sectors and 21.5% global
reduction sectors. A single follow-up replaced per-pair enhancing atomics with
eight I-cluster register accumulators plus final reductions. It regressed to
46.806 us in the event benchmark and introduced 27,058 local loads plus
23,077 local stores, so it was reverted after profiling.

The replay result is not yet an acceptance result because the retained
microbench clone has diverged materially from the production evaluator. On
the same 100-SCI ALA2 workload, the clone's ordinary force-only kernel took
41.54 us under NCU, whereas the production
`Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device`
launch took 8.32 us. Their kernel templates and scheduling are different.
Before another fused-policy iteration, the microbench must call a shared
production evaluator/launch wrapper instead of maintaining another copy of
the pair loop. This is code consolidation, not a new probe or gate. Once that
single implementation passes the force-only and full microbench contracts,
it can directly replace the standalone SITS evaluator and the temporary
microbench-only selection mode can be removed.

#### Microbench production-evaluator repair - 2026-07-27

The apparent `8.32 us` production baseline above was not an active evaluator
launch. A same-metric NCU audit showed only `200` global-load sectors and zero
global-reduction sectors: it was the slow SCI class returning immediately
against the safe-class flags. The original replay comparison was therefore
invalid in both directions: the microbench used a divergent pair-loop clone
and the production profile selected an empty launch.

The warp-record evaluator is now sourced from
`clustered_lj_warp_record_kernel.cuh` by both the production LJ translation
unit and `NBNXM_MICROBENCH`. The default `production-gmxpacked` mode dispatches
that shared evaluator according to the snapshot's safe/unsafe SCI flags.
Microbench-only experimental variants remain available for retained
experiments, but are identified separately in benchmark output and are not
used as the production baseline. No production probe, environment gate, or
alternate executable was introduced.

An unfiltered ALA2 production NCU run captured the active and empty launches:

| launch | duration (us) | global load sectors | global RED sectors | registers |
|---|---:|---:|---:|---:|
| force-only, active safe SCI | 108.608 | 68,716 | 27,186 | 67 |
| force-only, empty slow SCI | 2.336 | 200 | 0 | 69 |
| full, active safe SCI | 166.400 | 64,267 | 33,106 | 72 |
| full, empty slow SCI | 2.304 | 200 | 0 | 72 |

The repaired shared-evaluator microbench matches the active production
instances:

| mode | duration (us) | global load sectors | global RED sectors | registers |
|---|---:|---:|---:|---:|
| force-only | 108.224 | 68,516 | 27,126 | 67 |
| full | 166.464 | 64,067 | 33,084 | 72 |

CUDA-event rechecks are `95.876-97.545 us` for force-only and
`142.093-153.268 us` for full.
The full-output snapshot remains within the `2e-5` contract (force maximum
scaled error `2.81e-7`; energy maximum scaled error `5.47e-6`). The reports are
`production_ala2_all_lj_20260727.ncu-rep`,
`microbench_shared_forceonly_ala2_20260727.ncu-rep`, and
`microbench_shared_full_ala2_20260727.ncu-rep`.

This closes the baseline-infrastructure blocker. Subsequent fused SITS work
must modify the shared evaluator policy and compare against these active
force-only/full instances; the old `8.32 us` row and the replay-clone
`35.207/41.54 us` numbers are retained only as evidence of the measurement
failure.

#### Unified sparse SITS production checkpoint - 2026-07-27

The fused all-pairs policy above was not promoted. For the real SITS workload,
the selected region is normally much smaller than the complete system, so
charging every accepted pair for the selection policy is the wrong production
tradeoff. Production now uses one unconditional sparse design for eligible
SITS-only runs:

1. the regular clustered gmxpacked LJ/direct-Coulomb kernel computes the
   ordinary force and full outputs once;
2. SITS builds a generation-keyed compact SCI/CJ view containing only records
   incident to selected endpoints;
3. a correction-only kernel evaluates that compact view and accumulates only
   the SITS enhancing force and, in the full variant, its energy/virial
   correction.

This is the sole production SITS direct route when SITS direct LJ/Coulomb is
active and REST2 direct LJ/Coulomb is not. The
`SPONGE_SITS_CLUSTERED_NATIVE` opt-in and the legacy/clustered A/B dispatch
have been removed. There is no selection-ratio threshold, system-size gate,
runtime probe, or compatibility fallback. REST/REST2 and SPONGE Manager
remain outside the migration scope.

The retained full-output microbench selected a fixed four-way SCI partition
for the ordinary clustered base. ALA2 measured `0.162310`, `0.064379`,
`0.047292`, and `0.031493 ms` for one, two, four, and eight work parts,
respectively. Wat160k measured `2.856753`, `1.974328`, `1.916698`, and
`2.169176 ms`; four work parts is the large-system peak and avoids introducing
a runtime size split. The split variants matched the reference globally; the
wat160k per-atom differences were limited to floating-point atomic ordering
(split-four maxima: force `2.67e-5`, energy `3.03e-5`, virial `6.87e-5`).

NCU verification on the ALA2 production path changed the ordinary full
clustered launch from one work part, grid `(132,1,1)`, `404.70 us`, and
`4.33%` achieved occupancy to four work parts, grid `(528,1,1)`,
`82.53 us`, and `11.70%` achieved occupancy. The full sparse correction keeps
the `(SCI,8,4)` schedule because flattening `jm` into a loop regressed from
about `12.6-12.8 us` to `28.58 us`; the restored full variant measured
`13.06 us`, 64 registers/thread, with grid size 1728. Force-only and full are
the only correction variants. This fixed scheduling is intentionally shared
by light and heavy systems: when sparse correction is a visible fraction of a
light workload, adding a heavier routing policy would cost more complexity
than it saves.

Before backport, acceptance still requires the final CPU/CUDA SITS validation
suite, end-to-end ALA2 timing, and the frozen wat160k LJ/full-output replay.
The old standalone SITS evaluator, solvent-LJ fast route, and stale
environment-switch test scaffolding can then be removed; the retained
microbench remains the only place for further scheduling experiments.

The first end-to-end ALA2 check exposed a builder-wide small-box limitation,
not a SITS correction-kernel problem. With the correctness-preserving
small-box fallback from clustered skin 10 Angstrom to the ordinary 2 Angstrom
skin, three 10,000-step unified-sparse runs measured about
`161.9-168.7 ns/day`; the pre-migration legacy lineinfo binary measured
`451.5 ns/day`. The sparse correction itself remains about `13 us`; the
regression is dominated by clustered payload rebuild/lifecycle cost.

The skin contract was audited before changing dispatch:

- the minimum triclinic face-height calculation and the default
  `skin_permit=0.5` dual-endpoint Verlet rule are correct;
- using the simple geometric bound
  `skin < 0.5 * minimum_face_height - cutoff` is not sufficient for the
  current cluster candidate builder. On ALA2 it selected 6.25 Angstrom,
  changed the zeroth-frame potential by roughly 45 kcal/mol and became
  non-finite during the 10,000-step run, so that experiment was reverted;
- an explicit retained benchmark sweep found skins 2, 3, 4, and 5 Angstrom
  identical through step 20 at printed precision. Skin 5 remained finite for
  10,000 steps and measured `287.69 ns/day`, but skin 6 already differed at
  step zero (`potential -36.45`, `PM -38.28 kcal/mol` relative to skin 2);
- canonical pair-oracle replays explain the boundary. The skin-5 snapshot
  matched exactly (`payload=138084`, `oracle=138084`, `duplicates=0`,
  `missing=0`, `extra=0`). Skin 6 had the same unique pair/oracle counts but
  35 duplicate payload pairs, all first examples using the central image,
  such as `(100,104,shift=13)`.

Therefore the production fallback remains 2 Angstrom. Five Angstrom is only
an experiment, not a safe general default. Raising the small-box skin is
blocked on a builder-owned image-dedup contract keyed by canonical global atom
pair and per-pair periodic image; SCI shift alone is not a valid identity.
This issue affects every clustered consumer and must not be hidden behind a
SITS size/selection gate.

The default production registration of clustered fine-grained time recorders
was also removed. It was diagnostic scaffolding and forced synchronization
around build/gather/kernel sub-stages. Removing it preserved both CPU/CUDA
SITS validation (`2 passed` each), but the 10,000-step rate remained in the
same band, so timer synchronization was not the main regression. The next
performance work is canonical payload dedup plus reducing the count/fill
builder cost at the correctness-preserving skin; further SITS kernel
partitioning is not justified.

#### DNA default-path regression repair - 2026-07-27

The apparent ALA2-versus-DNA anomaly was not evidence that the widened ALA2
skin or SITS scheduling should become a size-based policy. A fresh DNA_COU
force-only run exposed a separate default-path regression: the current
10,000-step rate was only `172.07 ns/day`, while the qualified historical
envelope was about `300 ns/day`.

NCU isolated the mismatch between the retained builder and dispatch policies.
The regressed default payload had 1544 SCI and launched the generic pair-shift
AB split-four kernel as 6176 blocks. It took `435.55 us`, used 70
registers/thread and achieved `44.58%` occupancy. The historical qualified
payload had 996 SCI. Replaying the current binary with the qualified outer
active-view builder restored 996 SCI and the SCI-shift split-four launch
without changing the kernel schedule. The post-repair default profile is 3984
blocks, `85.95 us`, 67 registers/thread and `42.19%` achieved occupancy:
`5.07x` faster (`-80.3%`) despite slightly lower occupancy. L1 hit rate also
improved from `44.0%` to `47.7%`. The regression was excess/generic payload
work, not insufficient occupancy and not a split-three versus split-four
problem.

The qualified outer active-view builder decisions are now the sole production
path: shift-partitioned/fixed-leaf discovery, queue2 fused counting,
fixed-light dedicated cooperative count, parallel dirty-J scan, zero-dirty
source reuse, cached inner-active fill, compact-payload metadata caching and
SCI-safe flag publication. Their old environment switches are no longer
consulted. The unstable rolling source cache stays hard-off. The historical
DNA-only full-dense padding was deliberately not promoted: the same optimized
builder measured `362.06 ns/day` without it versus `358.96 ns/day` with it, so
the clean general path needs no DNA-specific padding branch.

Post-repair, no-environment 10,000-step checks measured:

- DNA_COU force-only: `371.88 ns/day`, `Calculate_Force=2.914 s`;
- wat160k force-only: `149.85 ns/day`, `Calculate_Force=5.093 s`;
- DNA canonical pair oracle: `payload=3479268`, `oracle=3479268`,
  `duplicates=0`, `missing=0`, `extra=0`;
- the clustered-LJ/microbench Python suite: `60 passed`;
- CUDA SITS validation: `2 passed`.

This restores a clean performance reference for subsequent consumer migration.
It also changes the remaining cleanup order: remove the now-unreachable
builder-policy environment parsing and dead alternative branches before the
backport, but retain the microbench scheduling variants and canonical
pair-oracle.

#### Unreachable builder cleanup and SITS recheck - 2026-07-27

The promoted production policy has now been made structural in the builder
instead of remaining behind compile-time-constant helpers. The cleanup removes
the unreachable non-queue2 one-pass candidate builder and its production probe
block, the sparse/non-fixed and serial fixed-shift count/fill alternatives,
the non-cooperative dedicated-count launch, the rolling-source-cache leg and
definition-only policy/probe helpers. The queue2 fused path still retains its
runtime overflow fallback into the subgroup count/fill implementation; pointer,
capacity, cache-generation and current-mask guards remain runtime checks.

The CUDA 13 build of both `SPONGE` and `NBNXM_MICROBENCH` succeeds after the
cleanup. A same-case, same-filter NCU recheck of DNA_COU force-only preserved
the payload and launch shape exactly: 996 SCI, 3984 blocks, 64 threads/block
and 67 registers/thread. Kernel duration changed from `85.95 us` to
`86.34 us` (`+0.4%`), while achieved occupancy changed from `42.19%` to
`42.6%`. This is an unchanged result at the observed run-to-run scale, not a
return of the prior 1544-SCI/`435.55 us` regression. The post-cleanup report is
`/tmp/dna-deadbranch-cleanup-20260727.ncu-rep`.

ALA2 SITS was then repeated with the correctness-preserving production
configuration: RTX 4090, 1840 atoms, 10,000 steps, cutoff 8 Angstrom, effective
clustered rebuild skin 2 Angstrom, unified sparse SITS and no experimental
environment gate. Three runs measured `241.13`, `239.83` and `226.84 ns/day`.
The latter two were created without a pre-existing `SITS_nk_rest.txt`; the
last run used the final binary after definition-only dead symbols were
removed. This `226.84-241.13 ns/day` band is materially above the earlier
correctness-preserving `161.9-168.7 ns/day` band without widening the skin.
Final CUDA SITS validation passes `2/2`.

The cleaned source is therefore the migration sample: one production queue2
fused/subgroup path plus a real overflow fallback, rather than a collection of
inactive experiment routes. Further cleanup must not remove the runtime
overflow/capacity and generation guards, and microbench-only scheduling
variants remain available for isolated experiments.

#### Dynamic image dedup and small-box skin guard - 2026-07-27

The cleanup baseline was committed as
`5842a28 feat: consolidate clustered LJ and SITS paths`. The subsequent
pair-oracle audit identified the skin-6 duplicates precisely: fixed SCI image
records such as shift 10 can normalize to the same current per-pair shift as
the central SCI. Build-time SCI aggregation cannot remove these records
because their fixed images are needed across coordinate updates.

The production payload now stores two eight-bit current-image ownership masks
plus a format marker in the previously unused high bits of each 64-bit
per-CJ/JM pair-shift word. Old snapshots without the marker retain all lanes.
During pair-shift refresh, only lanes whose current shift differs from their
fixed SCI shift enter the dedup proof. A deterministic competing SCI wins only
when it resolves to the same current shift and its valid/local/exclusion
coverage is a superset of the lane being removed. Force-only/full LJ,
soft-LJ, fused SITS and custom pairwise consumers all apply the same mask;
SITS sparse compaction preserves the word unchanged. The obsolete
simple-refresh environment branch was removed rather than extended with a
second implementation.

The final skin-6 ALA2 snapshot matches the independent canonical oracle
exactly:

```text
payload=138084 oracle=138084 duplicates=0 missing=0 extra=0
```

The pre-change full NCU profile of pair-shift refresh was `4.10 us`,
39 registers/thread. The final noinline cold proof path measures `4.19 us`;
the `0.09 us` delta is negligible at the end-to-end scale. Both CUDA targets
and the CPU `SPONGE` target link successfully.

This closes current-image duplication, but it does not qualify an arbitrarily
wide small-box rebuild horizon. With the half-box fallback temporarily
removed, default skin 10 had an exact zeroth-frame oracle but 10,000-step
stability depended on synchronization/output cadence: one run was finite at
`417.61 ns/day`, while unsynchronized repeats became non-finite. Therefore
the geometric small-box guard remains production policy and reduces the
28.50-Angstrom ALA2 box from clustered skin 10 to the independently qualified
global skin 2. The final guarded 10,000-step run was finite at
`247.48 ns/day` with `Calculate_Force=5.215 s`, within or above the previous
`226.84-241.13 ns/day` band. Widening that horizon remains blocked on a
separate rebuild-lifecycle proof; it must not be inferred from a step-zero
pair oracle.

The Python SITS validation command could not be rerun in this workspace
because neither the pixi environment nor system Python contains `pytest`.
This is an infrastructure gap, not a test pass; the final CUDA acceptance run
must repeat the existing two-test SITS suite in the qualified test
environment.

#### Phase 2 EAM execution checkpoint - 2026-07-29

SITS production convergence is complete in commit `2df2c62`. Regular and
soft-LJ SITS now use the clustered base plus one generation-keyed sparse
correction, consume `atom_sys_mark_local` directly, and expose only force-only
and full variants. The unreachable standalone regular/soft SITS kernels and
their solvent-prefix API have been removed. REST/REST2 and SPONGE Manager were
not changed.

EAM is the next legacy full-list consumer. It is pair-decomposable and does
not require endpoint incidence or a center-complete cursor. The production
replacement is fixed as three mathematical stages:

1. one authoritative clustered half-pair traversal atomically contributes
   `rho(type_j, r)` to endpoint `i` and `rho(type_i, r)` to endpoint `j`;
2. one per-local-atom embedding pass evaluates `F(rho)` and `dF/drho`;
3. one authoritative clustered half-pair traversal evaluates `phi`,
   `dF_i * rho_j`, and `dF_j * rho_i`, then writes equal and opposite endpoint
   forces. Full output assigns half of pair energy and pair virial to each
   local endpoint while embedding energy remains per atom.

This is algebraically equivalent to the current two directed full-list
passes, but it evaluates each physical pair once. `rho` and `dF/drho` remain
EAM-owned per-atom intermediates; no full adjacency, CSR spatial table, or
`ATOM_GROUP` compatibility view is permitted.

The unchanged full-list CUDA baseline was captured before any EAM kernel
modification on the tracked 10,976-atom Cu/LAMMPS fixture:

| legacy kernel | duration | grid | registers/thread | spills | achieved occupancy | no eligible |
|---|---:|---:|---:|---:|---:|---:|
| density | `64.16 us` | `43 x 256` | 31 | 0 | 15.18% | 75.65% |
| embedding + energy | `19.17 us` | `43 x 256` | 21 | 0 | 6.59% | 97.18% |
| force + energy + virial | `725.02 us` | `43 x 256` | 56 | 0 | 15.55% | 96.17% |

The complete report is
`/tmp/eam-legacy-baseline-20260729.ncu-rep`. The force kernel executed
9,884,785 instructions but exposed only 43 CTAs; its dominant first-order
problem is the atom-centric serial neighbor loop and lack of eligible warps,
not spilling. The first clustered implementation therefore uses SCI/CJ
partitioning to expose pair-tile work. Launch-shape or accumulator tuning
beyond that structural change requires a new full NCU report first.

The implementation boundary is:

1. EAM requests the shared clustered spatial service directly and pins the
   exact native or gmxpacked generation at every invocation. It does not use
   LJ parameters and does not request endpoint incidence.
2. EAM owns sorted coordinates and atom types. Atom types are gathered when
   local/domain ordering changes; coordinates are gathered for the current
   geometry using the provider permutation and cluster centers.
3. CUDA/HIP consumes gmxpacked SCI/CJ/exclusion/current-image masks. CPU
   consumes native SCI/CJ/exclusion rows. Both reapply the exact EAM cutoff.
4. Density and force each traverse the same half-pair ownership once. There is
   no separate local-local orientation table and no minimum-image
   recomputation that discards the provider's explicit pair image.
5. The externally visible variants remain force-only and full. Density and
   embedding are dependency stages, not additional output variants. The full
   pair kernel uses runtime energy/virial store masks.
6. Once CPU/GPU correctness and performance pass, EAM no longer sets
   `neighbor_list.is_needed_full`; SW, EDIP, Tersoff, and ReaxFF retain their
   independent full-list reasons.

The current SPONGE domain layer does not expose the typed two-way intermediate
exchange required by EAM. Its reusable halo primitive performs forward
local-to-ghost overwrite, while reverse communication is force-specific; no
generic additive reverse exchange exists. Multi-PP-rank clustered EAM must
therefore remain unsupported until a typed exchange supplies:

- forward owner `rho` values when the embedding stage needs complete owner
  density;
- forward owner `dF/drho` values before the second pair traversal;
- reverse-add of ghost density contributions to owners after the first pair
  traversal;
- ordinary reverse ghost-force accumulation after the force traversal.

The first production change must reject multi-PP-rank EAM explicitly. It may
not silently dispatch the legacy full list or claim DD support.

Acceptance uses the existing Cu and Cu/Ni LAMMPS fixtures and adds direct
intermediate comparison:

- Cu 10,976 atoms and Cu/Ni 864 atoms, including all three perturbations;
- CPU clustered versus GPU clustered force, energy, pressure, and virial;
- per-atom `rho` and `dF/drho` against a test-only independent directed
  evaluator;
- force-only and full NCU after every retained kernel/launch modification;
- isolated density + embedding + force total and end-to-end force time within
  3% of the matching legacy run;
- no runtime probe, environment opt-in, size gate, or compatibility fallback.

##### EAM clustered implementation result - 2026-07-29

The first production slice is now implemented on both backends. CUDA consumes
the gmxpacked SCI/CJ/exclusion/current-image payload with an `8 x 8` pair tile;
CPU consumes the native SCI/CJ/exclusion payload. Both use the provider's
explicit pair shift, accumulate density into both endpoints, run one
per-atom embedding pass, then accumulate equal-and-opposite force and split
pair energy/virial between endpoints. Cubic table interpolation now returns
value and analytic derivative together, so the force stage no longer rebuilds
the same interpolation polynomial through three automatic-differentiation
objects.

The old directed `ATOM_GROUP` density and force kernels and their public EAM
dispatch were removed after the clustered paths passed. Periodic single-rank
EAM no longer contributes a full-list reason: the existing fixture reports
`is_needed_full: false`. Non-periodic EAM is rejected explicitly because this
consumer has no supported non-PBC implementation; multi-PP-rank EAM is
rejected with the typed rho/df halo requirement. Neither case falls back to
the deleted full-list path.

Existing LAMMPS comparison coverage passes on both CUDA 13 and CPU:

```text
Cu funcfl, perturbations 0/0.1/0.2:       3 passed
Cu/Ni setfl, perturbations 0/0.1/0.2:     3 passed
```

The post-change full NCU report is
`/tmp/eam-clustered-after-20260729.ncu-rep`:

| clustered kernel | duration | grid/block | registers/thread | achieved occupancy | eligible warps/scheduler |
|---|---:|---:|---:|---:|---:|
| density | `37.73 us` | `358 x 8` / `8 x 8` | 64 | 37.94% | 1.03 |
| embedding + energy | `19.30 us` | `43` / `256` | 19 | 7.12% | 0.03 |
| force + energy + virial | `402.98 us` | `358 x 8` / `8 x 8` | 72 | 45.19% | 0.07 |

The three profiled stages fell from `808.35 us` to `460.00 us`, a `43.1%`
reduction. Density improved `41.2%`; force improved `44.4%`. The structural
goal is therefore met: the pair stages expose 2864 CTAs instead of 43
atom-centric CTAs and do each physical pair once. NCU still identifies the
full force path as latency-heavy with uncoalesced endpoint/table accesses and
high scoreboard/LG-throttle stalls. That is the next EAM tuning target; it is
not a reason to restore a derived adjacency or directed full-list kernel.

Per-atom test-only rho/df differential coverage and the typed DD halo remain
open. They do not block the single-rank production convergence above, but DD
must not be enabled until reverse-add density exchange and forward df exchange
exist.

### Phase 3: center-neighbor many-body consumers

#### SW execution checkpoint - 2026-07-29

The existing independent SW validation fixture is sufficient for migration:
the LAMMPS comparison builds a 10,648-atom two-type diamond system and checks
the unperturbed, 0.1-Angstrom and 0.2-Angstrom perturbations. The unchanged
legacy full-list implementation passes all three CUDA comparisons before any
SW source modification.

The pre-change full NCU report is
`/tmp/sw-legacy-baseline-20260729.ncu-rep`:

| kernel | duration | grid/block | registers/thread | achieved occupancy | active threads/warp | local-spill requests |
|---|---:|---:|---:|---:|---:|---:|
| SW force + energy + virial | `144.99 us` | `333` / `32 x 32` | 64 | 64.08% | 8.61 | 1,724,976 |

The native-CUDA kernel is cache/local-memory bound rather than DRAM bound:
SM throughput is `38.60%`, memory throughput is `61.06%`, DRAM throughput is
only `4.89%`, L1 hit rate is `68.11%`, and L2 hit rate is `98.13%`. NCU reports
100% local-spill overhead and 351,384 local-store sectors. LG-throttle stalls
consume about 5.7 of the 16.35 warp cycles per issued instruction (`34.8%`).
The existing kernel also averages only 8.61 active threads per warp because
one warp owns each atom-centric neighbor loop and neighbor counts are small.

The first retained change is therefore structural only: replace the derived
directed `ATOM_GROUP` with a center cursor over endpoint incidence, and form
the J/K upper triangle directly from deterministic incident clustered tiles.
The SW formulas and automatic-differentiation arithmetic remain unchanged for
that iteration. This isolates traversal effects in the first post-change NCU
diff; analytic three-body derivatives or launch changes require that report
as evidence.

The first direct cursor implementation was functionally correct but
unacceptable for migration: it kept `is_needed_full=false` and passed all three
SW comparison cases, but `SW_Clustered_Center_Direct` took `382.70 ms` in
`/tmp/sw-clustered-after-20260729.ncu-rep` because each ordinal lookup
rescanned the endpoint cursor. A resumable cursor reduced that to `28.03 ms`
(`/tmp/sw-clustered-after-cursor-20260729.ncu-rep`), still far above the
legacy `144.99 us` baseline.

The next two fixes address the actual candidate explosion. Non-LJ clustered
spatial-service consumers now keep the global `skin` instead of being widened
to a default `10.00 A` reuse skin; the SW fixture now reports
`reuse_skin=0.40`. SW also filters center neighbors by their pair-specific
`a*sigma` cutoff before forming the J/K upper triangle. That version passes
all three comparison cases and profiles at `793.63 us` in
`/tmp/sw-clustered-cutfilter-20260729.ncu-rep` with `is_needed_full=false`.
The retest reports `/tmp/sw-clustered-cutfilter-retest-20260729.ncu-rep` and
`/tmp/sw-clustered-cutfilter-clean-20260729.ncu-rep` match that result at
`793.47 us` and `796.10 us`, respectively.

The one-thread-per-center scheduling variant was tested and rejected. It
passed the three comparison cases, but
`/tmp/sw-clustered-onethread-20260729.ncu-rep` measured `15.17 ms`,
SM throughput `7.68%`, achieved occupancy `14.91%`, and only `42` blocks for
`128` SMs. The current retained implementation is therefore the warp-owned
center kernel with resumable cursor traversal, pair-cut filtering, and the
global-skin clustered spatial-service fix. Remaining work must reduce the
`472M` executed instructions and `6.06M` local-spill requests before this SW
path is suitable as the clean migration sample.

The first arithmetic cleanup replaces the six-variable SAD three-body
derivative with its equivalent analytic gradient. It retains the same
clustered traversal and passes all three SW comparison cases. NCU report
`/tmp/sw-clustered-analytic-20260729.ncu-rep` measures `749.02 us`, reduces
local-spill requests from `6,063,346` to `4,391,616`, and reduces executed
instructions from `472,397,136` to `464,095,630`.

The decisive direct-consumption fix is warp-cooperative neighbor-cache fill.
Lane zero still decodes each endpoint reference once, but all warp lanes test
the reference's cluster/lane candidates and compact accepted neighbors into
shared memory. This is not a derived neighbor table and introduces no runtime
probe or gate. It passes all three comparison cases and report
`/tmp/sw-clustered-cooperative-fill-20260729.ncu-rep` measures `391.46 us`.
Relative to the analytic serial-fill version, executed instructions fall to
`233,629,980`, local-spill requests fall to `956,176`, and average active
threads per warp rise from `1.92` to `12.46`. The analytic two-body derivative
then gives a smaller retained improvement to `384.19 us` in
`/tmp/sw-clustered-analytic-two-body-20260729.ncu-rep`; it leaves instruction
and spill counts essentially unchanged.

The direct clustered force is therefore about `2.65x` slower than the
`144.99 us` legacy full-list kernel, although it has recovered more than half
of the initial direct-path gap. The remaining difference is structural:
`233.45M` executed instructions versus `62.12M` for the legacy kernel. SW
requires an atom-centric J/K sequence, so the next migration iteration should
derive a compact per-center atom-id adjacency from endpoint incidence and
stamp it with provider incarnation plus compact-payload generation. The force
kernel should recompute minimum-image displacement from current coordinates,
cell, and reciprocal cell, as the legacy arithmetic does; therefore the
derived topology need not be rebuilt for every gathered-coordinate geometry
generation. This is the evidence-based exception to direct tile consumption,
not a return to the global legacy full neighbor list.

A usable source-line build now exists at `build-dev-cuda13-sw-lineinfo`. It is
configured with the pixi GCC 11.4 host compiler and `-lineinfo`, avoiding the
CUDA 13/system-GCC incompatibility. Report
`/tmp/sw-clustered-analytic-lineinfo-20260729.ncu-rep` is available for GUI
source-counter inspection; the NCU CLI source page emits annotated source but
does not export its per-line counter columns.

The direct path then satisfied the documented scheduling-cache exception:
even after cooperative decode and analytic arithmetic it remained `2.65x`
slower than legacy, and NCU attributed the loss to repeated endpoint decoding
before the mathematical J/K work. SW now owns a generation-keyed compact
center-to-atom relation derived from endpoint incidence. This is not exposed
by the spatial service, is not an `ATOM_GROUP` compatibility API, and is never
consumed by another operator. Its identity is exactly
`{provider incarnation, gmxpacked payload generation}`. It stores atom IDs
only; current displacement and minimum image are recomputed from `crd`,
`cell`, and `rcell` on every force invocation, so geometry changes do not
silently reuse stale distances.

The rejected and retained derivation steps are:

| implementation | force duration | decision |
|---|---:|---|
| unfiltered derived adjacency | `11.01 ms` | reject: conservative cluster candidates explode the J/K product |
| pair-active filtering only | `10.63 ms` | reject: image ownership alone does not control the candidate radius |
| pair cutoff plus rebuild-skin filtering | `275.49 us` | retain derivation rule; force still global-access limited |
| shared strict-neighbor cache, 32 warps/block | `176.29 us` | retain cache shape, continue launch tuning |
| shared strict-neighbor cache, 16 warps/block | `168.38 us` | retain |
| shared strict-neighbor cache, 8 warps/block | `180.99 us` | reject and revert |

The corresponding reports are
`/tmp/sw-clustered-derived-20260729.ncu-rep`,
`/tmp/sw-clustered-derived-active-20260729.ncu-rep`,
`/tmp/sw-clustered-derived-skin-20260729.ncu-rep`,
`/tmp/sw-clustered-derived-shared-20260729.ncu-rep`,
`/tmp/sw-clustered-derived-shared-16warp-20260729.ncu-rep`, and
`/tmp/sw-clustered-derived-shared-8warp-20260729.ncu-rep`.
Two clean 16-warp force retests measured `173.79` and `170.21 us`;
the post-builder retest measured `174.94 us`, proving that builder tuning did
not change the steady-state force code shape.

The first derived-relation builder used one thread per center and took about
`8.9 ms` in each of count and fill. The retained builder assigns one warp per
center and cooperatively tests the fixed 64 tile candidates. Count now takes
`203.14 us` and fill `204.06 us`, reducing generation cost from about
`17.8 ms` to `0.41 ms`. The two passes run only when the provider incarnation
or compact-payload generation changes. Their reports are
`/tmp/sw-clustered-builder-20260729/report.ncu-rep` and
`/tmp/sw-clustered-coop-builder-20260729.ncu-rep`.

Fresh full NCU then identified the remaining force bottleneck: `54%` of
global sectors and `15%` of shared wavefronts were excessive, with only
`24.12%` of scheduler cycles having an eligible warp. The retained
NCU-driven changes are:

1. ballot/prefix compaction replaces the per-neighbor shared atomic counter.
   Shared excessive wavefronts fall from `191,664` to zero without a
   performance regression (`170.69 us`);
2. each lane owns one cached neighbor while J is replayed in structural
   order. Three-body J forces are warp-reduced and every neighbor performs one
   final global force update instead of O(degree) scattered updates. Duration
   falls to `104.06 us` and excessive global sectors fall from `5,133,287` to
   `1,782,255`;
3. matching `__launch_bounds__` to the actual 512-thread block reduces local
   spill requests from `1,778,216` to `1,000,912` and duration to
   `99.49 us`;
4. caching neighbor types beside geometry removes repeated random type loads.
   The final full kernel measures `96.61 us`, 64 registers/thread,
   24.58 KiB static shared memory, 53.68% achieved occupancy, 57.16M executed
   instructions, `1,178,624` excessive global sectors, zero excessive shared
   wavefronts, and `62.98%` eligible scheduler cycles.

The final reports are
`/tmp/sw-clustered-ballot-20260729.ncu-rep`,
`/tmp/sw-clustered-lane-force-20260729.ncu-rep`,
`/tmp/sw-clustered-bounds512-20260729.ncu-rep`, and
`/tmp/sw-clustered-cached-type-20260729.ncu-rep`. The final force kernel is
`33.4%` faster than the `144.99 us` legacy full-list baseline. All retained
steps pass the three SW/LAMMPS perturbation cases; the final run reports
`3 passed`.

The old directed full-list SW kernel, its public dispatch, and dead
direct/derived experimental helpers have been removed. Periodic single-rank
SW no longer contributes a full-list build reason and has no silent legacy
fallback. CUDA derives the private relation from generation-matched gmxpacked
endpoint incidence; CPU derives the same symmetric center relation directly
from the native SCI/CJ/exclusion payload and keys it by the native payload
generation. CPU validation initially caught and removed an incorrect
unconditional gmxpacked-capability requirement. The final independent builds
and LAMMPS comparisons pass `3/3` on both CUDA 13 and CPU.

The remaining Phase 3 work is:

1. add the independent canonical SW triplet oracle required by the general
   removal gate;
2. define or reject multi-PP-rank SW until center/J/K ghost-force reversal is
   covered;
3. migrate EDIP next: direct coordination and redistribution passes plus the
   center J/K path, retaining only mathematical `z/dE_dz` state unless its own
   NCU evidence separately satisfies the cache exception;
4. migrate Tersoff after EDIP, initially replaying grouped K for each directed
   edge and retaining zeta/edge scratch only when measured;
5. keep exactly force-only and full variants for every consumer.

#### EDIP clustered execution checkpoint - 2026-07-29

EDIP now uses the shared clustered spatial service on both CUDA and CPU and no
longer requests or consumes the directed legacy full list. Non-periodic input
is rejected explicitly. Multi-PP-rank input is also rejected until typed
`z/dE_dz` halo exchange and reverse ghost-force ownership exist; neither case
falls back to the removed full-list implementation.

The retained implementation derives one EDIP-owned compact center relation
from the authoritative half-pair payload. CUDA decodes the generation-matched
gmxpacked SCI/CJ/exclusion/current-image records and inserts both endpoint
orientations. CPU performs the same symmetric derivation from native SCI/CJ
records. The cache key is
`{provider incarnation, representation payload generation}`. The relation is
private to EDIP and is reused by its three mathematical stages:

1. coordination `z`;
2. pair/J-K force, energy, virial and `dE/dz`;
3. redistribution of the `dE/dz` force contribution.

Each row is filtered by the pair-type-specific EDIP cutoff plus the provider
rebuild skin. Current displacements are recomputed from coordinates, cell and
reciprocal cell in every force stage, so the relation does not retain stale
geometry. Only force-only and full kernels are instantiated; energy-only or
virial-only requests use full with runtime store masks.

The unchanged legacy baseline on the 10,648-atom two-type diamond fixture was:

| legacy stage | duration | registers/thread | achieved occupancy | executed instructions | local spill requests |
|---|---:|---:|---:|---:|---:|
| coordination `Get_Z` | `6.46 us` | 28 | 61.18% | 1.597 M | 0 |
| pair/J-K force full | `43.49 us` | 64 | 57.51% | 18.817 M | 775,940 |
| redistribution full | `10.59 us` | 32 | 61.55% | 3.389 M | 0 |
| total | `60.54 us` | - | - | - | - |

The first clustered full capture measured `6.21 + 48.77 + 11.26 =
66.24 us`. The apparent 9.4% duration regression was a profiler clock
difference rather than extra force work: the clustered and legacy main
kernels both consumed about 107k elapsed SM cycles
(`107,111` versus `107,655`), while the sampled SM clocks were `2.19` and
`2.47 GHz`. The clustered main kernel executed 18.877 M instructions
(`+0.32%`), kept 64 registers and the same 775,940 local-spill requests, and
slightly improved achieved occupancy to 58.73%. This qualifies the retained
force path as an unchanged code-shape result; wall-time comparisons must be
clock matched.

The initial private-relation builder used eight packed-range partitions and
took `16.03 us` for count plus `19.07 us` for fill (`35.10 us` total).
NCU showed a one-wave launch, 29-32% achieved occupancy versus 75%
theoretical occupancy, cross-block workload imbalance and L1TEX
long-scoreboard stalls. Two retained NCU-driven changes followed:

1. sixteen packed-range partitions reduced count/fill to
   `12.48 + 16.48 = 28.96 us`, a 17.5% reduction;
2. squared-distance cutoff comparison reused the already loaded endpoint
   types and removed `sqrtf` plus redundant type loads, reducing the same
   total to `12.22 + 16.03 = 28.25 us`.

The builder is therefore 19.5% faster than its first implementation and runs
only when the provider incarnation or representation payload generation
changes. The final count kernel uses 55 registers, the fill kernel 56, and
neither spills. Their remaining limiting dependency is the atomic slot
allocation/irregular endpoint load path; replacing it would require a larger
scan/layout redesign and is not part of this cleanup slice.

Validation for the current source passes all three LAMMPS perturbations on
both backends:

```text
CUDA: 3 passed
CPU:  3 passed
```

Reports:

```text
/tmp/edip-legacy-baseline-20260729.ncu-rep
/tmp/edip-clustered-force-targeted-20260729.ncu-rep
/tmp/edip-clustered-builder-targeted-20260729.ncu-rep
/tmp/edip-clustered-builder-p16-20260729-sections.ncu-rep
/tmp/edip-clustered-builder-squaredcut-20260729.ncu-rep
```

The independent removal oracle is now implemented by the dedicated
`MANYBODY_CLUSTERED_ORACLE_TEST` CTest target. It links the production EDIP
stages directly without adding a production dump, probe, runtime gate or
diagnostic buffer. A five-atom, two-type periodic fixture compares the actual
generation-keyed relation against an independent O(N²) directed-center
reference, compares the canonical unordered `(i,j,k)` set including both
minimum-image identities, compares production `z` directly and compares
production `dE/dz` against a central finite difference of an independent EDIP
energy implementation. The fixture also covers a clustered exclusion,
asymmetric pair cutoffs and strict rejection at the exact EDIP cutoff.

The oracle passes on both CPU and CUDA:

```text
CPU:  manybody clustered oracles passed: EDIP relation/triplets/z/dE_dz;
      Tersoff relation/directed tuples
CUDA: manybody clustered oracles passed: EDIP relation/triplets/z/dE_dz;
      Tersoff relation/directed tuples
```

This closes the EDIP single-rank removal oracle without retaining a second
production neighbor representation. Multi-rank support remains explicitly
unavailable until typed `z/dE_dz` exchange and reverse ghost-force ownership
are defined.

#### Tersoff clustered execution checkpoint - 2026-07-29

Tersoff now uses the shared clustered spatial service on CUDA and CPU and no
longer requests or consumes the directed legacy full list. Periodic
single-rank execution is the supported scope. Non-periodic and multi-PP-rank
inputs are rejected explicitly because directed-edge/K ghost ownership and
reverse force exchange have not been defined; there is no silent fallback.

The operator owns a compact, generation-keyed center relation derived from the
authoritative half-pair payload. CUDA decodes gmxpacked
SCI/CJ/exclusion/current-image records and inserts both endpoint orientations;
CPU derives the same symmetric relation from native SCI/CJ records. Its cache
identity is `{provider incarnation, representation payload generation}`.
Coordinates and periodic displacements are recomputed on every force call.

Tersoff parameters are directional in `(i,j,k)`. For each center type `i` and
candidate K type `k`, relation construction therefore uses the conservative
cutoff

```text
max over j of R(i,j,k) + D(i,j,k)
```

The force kernel still applies the exact `(i,j,j)` cutoff to every directed
edge and the exact `(i,j,k)` cutoff during both K scans. This preserves the
legacy directed-edge algebra, half-energy ownership, zeta construction and
derivative redistribution without deriving a generic full neighbor-list API.
Only force-only and full kernels remain; energy-only or virial-only requests
use full with runtime output masks.

On the 10,648-atom B/N diamond fixture, the removed full-list
`Tersoff_Force_CUDA<1,1>` baseline measured:

```text
781.57 us, 56 registers/thread, 75% theoretical occupancy,
7.90% achieved occupancy, 107.363 M executed instructions,
79.05% scheduler cycles with no eligible warp, no spills
```

The first clustered force kernel retained one thread per center but scanned
the strict private relation. It measured `201.98 us`, reduced executed
instructions to `29.119 M`, kept 56 registers and no spills, and remained
underfilled at `7.81%` achieved occupancy with only 84 blocks. The relation
builder measured `13.50 + 19.58 = 33.08 us` for count and fill and runs only
when its generation key changes.

NCU then justified assigning one warp to each center and its lanes to directed
J edges. The retained 32-by-32 launch produces 333 blocks and measures:

```text
152.74 us, 56 registers/thread, 64.50% achieved occupancy,
40.431 M executed instructions, 76.35% no-eligible cycles,
5.78 active threads/warp, no spills
```

This scheduling change is 24.4% faster than the initial clustered force and
80.5% faster than the legacy full-list kernel. Full NCU showed that the
remaining delay is not DRAM bandwidth: DRAM throughput is `0.87%`, while
`lg_throttle` and `long_scoreboard` each cost about 11 cycles per issued
instruction. The kernel issued 447,216 global reduction instructions and
1.136 M reduction sectors. Combining the repulsive and direct-attractive
endpoint force for each directed pair removes one three-component atomic
update and gives a smaller retained result of `150.75 us` without changing
register count, occupancy or spills.

A warp-broadcast experiment attempted to share K geometry between directed J
lanes. It was rejected and removed before retention: the existing outer J
loop is divergent, so dynamically captured shuffle masks first produced NaNs
and then an illegal access under a fixed stale mask. `compute-sanitizer`
reported no ordinary out-of-bounds access in the NaN version, confirming that
the fault was collective participation rather than relation storage. Any
future K-geometry reuse must first rewrite the outer loop as a fully converged,
predicated J-tile traversal; no probe, gate or experimental dispatch remains.

The retained source passes all three LAMMPS perturbation cases on both
backends:

```text
CUDA: 3 passed; maximum force differences
      2.0959e-03, 3.4211e-03, 5.3058e-03
CPU:  3 passed; maximum force differences
      1.8822e-03, 3.3963e-03, 5.3516e-03
```

Reports:

```text
/tmp/tersoff-legacy-baseline-20260729.ncu-rep
/tmp/tersoff-clustered-force-initial-20260729.ncu-rep
/tmp/tersoff-clustered-builder-initial-20260729.ncu-rep
/tmp/tersoff-clustered-warpcenter-20260729.ncu-rep
/tmp/tersoff-clustered-pair-atomic-20260729.ncu-rep
```

The same dedicated `MANYBODY_CLUSTERED_ORACLE_TEST` target now closes the
single-rank Tersoff removal oracle. Its independent O(N²) reference verifies
the conservative directed center relation, while an independent O(N³)
enumeration verifies the exact directional `(i,j,k)` tuple set and both
minimum-image identities after the `(i,j,j)` and `(i,j,k)` cutoffs are
applied. The fixture distinguishes a conservative relation candidate from an
accepted J edge and covers clustered exclusions and exact-cutoff behavior.
It passes on both CPU and CUDA.

The exact-boundary case exposed one production mismatch: the legacy force
accepted `r == R + D`, but relation construction used strict `<` and could
discard that edge before the exact force predicate ran. The conservative
relation predicate now uses `<=`, matching the force algebra; no runtime probe
or alternate dispatch was introduced.

Tersoff still needs an explicit multi-rank ownership design or rejection
policy at the public feature level. The old Tersoff full-list kernel and
dispatch are gone, but ReaxFF remains a separate full-list consumer and
therefore still keeps the global legacy builder alive.

#### Phase 3 structural checkpoint - 2026-07-27

The first center-complete structural slice is now production-owned rather than
only a host oracle:

- `LJ_CLUSTER_LAYOUT` owns the gmxpacked endpoint-incidence offsets,
  references, sort keys, error flag, capacities and the exact
  `{provider incarnation, gmxpacked payload generation, SCI count, CJ count,
  supercluster count}` identity;
- compact-payload publication/withdrawal and provider retirement invalidate
  the capability before a view can expose it. `Clear` frees all storage and
  resets all capacities;
- the GPU builder is requested only through
  `need_aux_clustered_metadata=true`. Regular LJ and soft-LJ continue to pass
  `false`, so their peak build/launch path does not allocate, count, sort or
  publish endpoint incidence;
- each packed CJ has eight deterministic structural slots
  (`four JM * native/transposed`). The device builder emits references into
  those slots, rejects overlapping/invalid payload ownership, counts by center
  supercluster, scans offsets and sorts by
  `(center supercluster, source slot)`. Invalid slots sort after the published
  prefix. No SCI/CJ payload is copied to the host;
- view validation rejects stale counts and any reference count above
  `8 * gmxpacked_cjpacked_numbers`. Range access additionally rejects negative,
  reversed or out-of-storage offsets;
- the center cursor filters a supercluster incidence range for one concrete
  center cluster. Native replay yields one J cluster; transposed replay yields
  the original fixed-width I-cluster mask. Pair-shift lookup retains the
  original lane-specific image and inverts it for transposed orientation.

The contract test now covers native replay across two SCI shifts, transposed
replay, partial source masks, cursor reset/termination, malformed ranges,
stale count keys, the structural size bound and all 27-image inversion
semantics. `ClusteredSpatialViewContract` passes 1/1. Full `build-dev-cpu`
and `build-dev-cuda13-lineinfo` `SPONGE` targets link successfully; the latter
compiles the device count/scan/sort implementation.

This does not yet close Phase 3. The remaining structural work is:

1. add the native/CPU production incidence builder rather than relying only on
   the host contract oracle;
2. add the center-lane atom decoder for valid/local masks, exclusions and
   sorted atom IDs;
3. exercise the GPU builder in the first opt-in center-consumer fixture and
   compare its published offsets/references with the host oracle;
4. migrate SW only after that fixture provides an independent canonical
   triplet oracle.

### Phase 4: ReaxFF

1. Replace only spatial candidate discovery for raw bond order, VDW/bond and
   EEQ entries with pair-tile traversal.
2. Compact accepted candidates into stable canonical reaction-edge IDs.
3. Keep corrected bond orders, reaction graph, EEQ sparse matrix, angles,
   torsions and hydrogen bonds as intrinsic ReaxFF structures.
4. Define and test halo exchange for charge, bond-order/Delta state and
   derivative propagation before multi-rank enablement.
5. Remove half/full `ATOM_GROUP` inputs stage by stage; do not use clustered
   records directly as an unstable reaction-edge ID.

#### ReaxFF VDW clustered checkpoint - 2026-07-29

The first ReaxFF slice deliberately separates pairwise spatial evaluation
from reaction-graph construction. VDW has no persistent edge identity and now
consumes the authoritative clustered half-pair payload directly. It does not
derive an `ATOM_GROUP` or a private center-neighbor table. Raw bond order and
EEQ are not folded into this kernel: they require stable ReaxFF-owned
candidate/CSR identities and remain the next migration slice.

For periodic single-rank execution, `REAXFF_VDW_Force_Clustered` validates the
same provider incarnation, lease, cutoff and backend-specific payload contract
as the other clustered consumers. CPU traverses native SCI/CJ/exclusion
records; CUDA traverses generation-matched gmxpacked records and current
pair-shift metadata. Both gather current sorted coordinates and recompute
pair distances every force call. The legacy VDW loop is used only when this
single-rank clustered view is not supplied; no environment gate, probe or
experimental dispatch was added. Other ReaxFF stages still require the legacy
half/full lists, so the global `reaxff_legacy` reason is not removed by this
slice.

The pre-change validation baseline was:

```text
CPU:  H2/dimer/EEQ 11 passed
CUDA: H2/dimer/EEQ 11 passed
PETN 16,240 atoms:
  relative potential-energy difference 3.177584e-04
  maximum charge difference            4.89e-04
  maximum force difference             7.21587e-01
```

The first direct CUDA implementation was already much faster for force-only,
but full output performed one force, energy and six-component virial atomic
update per accepted pair. NCU on the PETN fixture measured:

| path | duration | SM | memory | DRAM | registers | achieved occupancy | spills |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy force-only | `4.43 ms` | `0.82%` | `4.88%` | `0.72%` | 47 | `8.14%` | 0 |
| direct clustered force-only | `436.51 us` | `13.68%` | `73.58%` | `1.17%` | 64 | `49.91%` | 0 |
| legacy full | `4.46 ms` | `0.86%` | `4.88%` | `0.81%` | 47 | `8.14%` | 0 |
| direct clustered full | `4.499 ms` | `1.47%` | `16.15%` | `0.23%` | 64 | `50.25%` | 0 |

The full kernel spent 98.4% of scheduler cycles with no eligible warp and was
dominated by LG-throttle and long-scoreboard stalls. The retained, single
NCU-driven change accumulates the fixed-J endpoint force, energy and virial in
each eight-lane subgroup and performs one final J writeback. Re-profiling the
exact source gives:

| retained path | duration | change | SM | memory | DRAM | achieved occupancy | spills |
|---|---:|---:|---:|---:|---:|---:|---:|
| clustered force-only | `302.11 us` | `-30.8%` vs first direct | `23.30%` | `68.15%` | `2.88%` | `48.90%` | 0 |
| clustered full | `740.64 us` | `-83.5%` vs first direct | `11.96%` | `35.59%` | `0.43%` | `51.45%` | 0 |

Thus force-only is 14.7 times faster than the legacy kernel and full is 6.0
times faster. A 100-step PETN run measures `61.582 step/s`, `0.532070 ns/day`
versus `61.018 step/s`, `0.527193 ns/day` for the retained pre-change
line-info binary. The small end-to-end gain is expected because EEQ, bond
order, angles and torsions still dominate and still build/consume legacy
lists.

The independent `MANYBODY_CLUSTERED_ORACLE_TEST` now also compares ReaxFF VDW
force, total and atom-reduced energy, and six-component virial against an
O(N²) periodic reference with clustered exclusions. It passes on CPU and
CUDA together with the EDIP and Tersoff oracles. Post-change H2/dimer/EEQ
comparisons pass `11/11` on both backends, and the PETN single-frame
comparison retains the baseline error envelope.

Reports:

```text
.tmp/reaxff-vdw-clustered-full-20260729/reaxff_vdw_clustered_full.ncu-rep
.tmp/reaxff-vdw-clustered-force-20260729/reaxff_vdw_clustered_force_false.ncu-rep
.tmp/reaxff-vdw-legacy-force-20260729/reaxff_vdw_legacy_force.ncu-rep
.tmp/reaxff-vdw-subgroup-full-20260729/reaxff_vdw_subgroup_full_retry.ncu-rep
.tmp/reaxff-vdw-subgroup-force-20260729/reaxff_vdw_subgroup_force_retry.ncu-rep
```

Multi-rank ReaxFF is intentionally deferred. The next single-rank target is
raw bond-order discovery: decode clustered half pairs once, apply the strict
bond-order cutoff and `bo_cut`, compact accepted `(min(i,j), max(i,j))` pairs
into ReaxFF-owned edge IDs, then build the existing bond CSR from those IDs.
Clustered SCI/CJ positions must not become reaction-edge IDs.

### Phase 5: external ABI, NO_PBC and legacy removal

The PRIPS plugin API currently exposes host `h_nl` capacity/count/index. This
is an explicit unresolved Phase 5 decision gate and must be closed before the
provider-removal commit. Choose and document one explicit route:

- introduce a versioned C POD `ClusteredSpatialViewV1` capability containing
  sizes, immutable pointers, generation, cutoff superset and backend/device
  kind, then retire the old list functions at an announced ABI boundary; or
- declare the old plugin neighbor ABI unsupported in the backport and fail
  plugin initialization with a precise error.

Do not keep the grid builder solely to emulate the old plugin ABI, and do not
derive `ATOM_GROUP` silently. NO_PBC is a second explicit unresolved Phase 5
decision gate: either support a central-shift-only clustered view with an
explicit periodicity mask or retain a separate documented NO_PBC algorithm.
Record its pair-set and end-to-end cost audit and close the decision before
deleting the provider.

Remove the grid spatial neighbor builder and `ATOM_GROUP` neighbor traversal
only when:

- every active spatial consumer uses clustered traversal;
- NO_PBC policy is explicitly covered;
- the plugin ABI has a versioned replacement or explicit retirement;
- all validation/performance gates pass;
- legacy fallback is not entered by any supported configuration;
- removal does not touch SPONGE Manager-specific code.

At every phase, a performance or correctness failure stops further migration.
Fallback may remain temporarily for comparison, but it must not silently mask
a failed clustered-native implementation.
