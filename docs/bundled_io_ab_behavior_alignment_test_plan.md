# Bundled I/O A/B Behavior Alignment Test Plan

## 1. Goal and Boundary

The production A/B gate must prove that every declared supported bundled I/O
contract has all of the following:

1. A legacy reference surface.
2. A bundled input or output surface.
3. Runtime evidence that the contract was exercised.
4. A deterministic or statistical equivalence assertion.
5. A traceable result in the gate evidence report.

A passing gate proves equivalence only for contracts explicitly registered as
supported. Cross-process VDS reopen-and-append remains unsupported until the
writer implements an append/resume open mode. Unsupported and deferred
contracts must be recorded explicitly and must not contribute to the supported
coverage percentage.

## 2. Mandatory PR and Commit Rule

Implementation is divided into the independent PR scopes in this document.
The following rule is mandatory for all work under this plan:

> Complete exactly one independent PR scope, verify its acceptance criteria,
> and then create exactly one corresponding commit before starting the next PR.

Additional constraints:

- Do not mix changes belonging to different PR scopes in one commit.
- Do not mark a PR complete or commit it before all acceptance checks for that
  PR pass.
- The commit must contain only files owned by that PR and must preserve
  unrelated worktree changes.
- If a PR must be split because its scope is no longer independently
  reviewable, update this plan first and give each new scope its own commit.
- Follow-up fixes discovered after a completed PR belong to a new, explicitly
  named PR scope and a new commit; do not amend an already verified boundary
  unless the user explicitly requests it.
- Record `This commit` and the verification commands in the PR completion log
  within that PR's commit. Report the resulting hash after the commit is
  created; a Git commit cannot contain its own hash.

Recommended commit subjects:

```text
test(bundled-io): add executable A/B contract registry
test(bundled-io): complete contract manifests and fixtures
test(bundled-io): harden statistical equivalence checks
test(bundled-io): close bundled output behavior coverage
test(bundled-io): close bundled input behavior coverage
test(bundled-io): cover rerun and failure semantics
ci(bundled-io): add executable shadow promotion guards
test(bundled-io): complete and promote the A/B gate matrix
```

## 3. Evidence Model

Create `benchmarks/bundled_io/contracts/ab_contracts.json` as the single source
of truth. Every contract entry must contain:

```json
{
  "contract_id": "trajectory.output.velocity",
  "direction": "output",
  "component": "trajectory",
  "status": "supported",
  "minimum_evidence": "E3",
  "legacy_surface": "vel",
  "bundled_surface": "/particles/all/velocity/value",
  "case_ids": ["normal_particle_payload_vds_off"],
  "assertion_ids": ["trajectory_velocity_statistical_equivalence"]
}
```

Evidence levels are normative:

| Level | Required evidence |
|---|---|
| E0 | Schema, key, or dataset path exists. |
| E1 | Converted payload is semantically equal to its legacy source. |
| E2 | Runtime proves that the owning module consumed or produced the payload. |
| E3 | Legacy and bundled runtime behavior is deterministically or statistically equivalent. |
| E4 | Output is reloaded and continued execution remains equivalent. |
| F1 | Invalid input, conflicts, exit status, and stable diagnostics are equivalent. |

Every runtime assertion must emit an `EvidenceRecord`. Case declarations do
not count as evidence. The gate must write `ab_evidence.json` containing case,
contract, assertion, profile, seed, backend, thread/rank configuration,
comparison policy, result, and artifact paths.

## 4. PR 1: Executable Contract Registry

### Scope

- Add the canonical JSON registry and a typed Python loader.
- Replace free-form `AbCase.covers` labels with registered `contract_ids` and
  `assertion_ids`.
- Make assertions return executed evidence records.
- Replace source-string searches in the production-gate manifest with registry
  and case-model validation.
- Generate a supported/deferred/unsupported coverage report.

### Required guards

- Contract IDs, case IDs, and assertion IDs are unique.
- Every supported contract has at least one case and assertion.
- Every referenced contract exists.
- Every executed assertion reports evidence for a declared contract.
- Evidence below `minimum_evidence` fails the gate.
- E0/E1 path checks cannot satisfy an E3 behavior contract.

### Acceptance

- Removing a case, assertion, or evidence report makes the manifest test fail.
- Adding an unmapped supported contract makes the manifest test fail.
- Existing A/B cases still run and produce the new evidence report.

## 5. PR 2: Contract Enumeration and Fixtures

### Enumeration to add

- Topology: improper, nb14, and the `virtual_atoms_in_file` alias.
- H5 output controls: trajectory chunk size and repair policy.
- Legacy output: mdinfo, force trajectory, restart, and QC SCF output.
- Trajectory input particle-stream selection.
- Rerun start, strip, frame limit, and box-update behavior.
- Restart load policies: structural, dynamic, protocol, and full.
- Output families: trajectory-only, observable-only, restart-only, and all.
- Sidecar default suppression, explicit coexistence, provenance, same-path
  idempotence, and different-path conflict.

### Fixture design

- Split the broad synthetic fixture into targeted fixtures whose owning module
  can be proven active.
- Give every targeted module a non-zero, distinguishable sentinel payload.
- Require an owning mdout column, module stream, state transition, or explicit
  runtime activation record for E2.
- Retain the broad fixture for schema and conversion coverage only; do not use
  path presence from that fixture as E3 evidence.

### Acceptance

- Every registered E1 input has a legacy-to-H5 semantic comparison.
- Every supported module fixture reaches E2 and has a non-empty, non-trivial
  runtime result.
- Pure bundled fixtures contain no undeclared legacy dependencies.

## 6. PR 3: Statistical and Deterministic Comparators

### Deterministic policy

- Compare all rerun and short deterministic NVE rows and datasets.
- Require exact step, time, frame count, schema, and non-finite pattern.
- Use contract-specific numeric tolerances; do not use one global tolerance.

### Statistical policy

- Preserve independent replica and block-mean analysis for stochastic MD.
- Use equivalence confidence intervals for means and declared bounds for
  variance ratios.
- Apply Holm correction across observable families.
- Define margins per physical quantity and unit.
- Reject mismatched NaN, positive infinity, and negative infinity patterns.

### Structured trajectory statistics

- Position: center of mass, per-atom MSD, bond/distance distributions, and RDF
  where applicable.
- Velocity: component means/variances, kinetic temperature, and norm
  distribution.
- Force: net force, RMS, quantiles, and per-atom norm distribution.
- Box: matrix components, volume, lengths, and angles.

### Comparator mutation tests

Inject an energy offset, variance increase, atom permutation, local force
offset, missing non-finite value, and frame-schedule change. Every injected
defect must be rejected reliably.

## 7. PR 4: Output Behavior Closure

Add these independent case groups:

| Case group | Required behavior |
|---|---|
| `normal_particle_payload_vds_off/on` | Position, velocity, force, box, step, and time. |
| `normal_output_routing` | All eight legacy output routes, suppression, explicit coexistence, and provenance. |
| `normal_restart_roundtrip_structural` | Legacy and H5 restart reload and E4 continuation. |
| `normal_restart_roundtrip_dynamic` | RNG, thermostat, barostat, and NHC state continuation. |
| `module_output_nhc/sits/meta/qc/reaxff` | Generic observable and dedicated module streams agree. |
| `output_family_modes` | Trajectory-only, observable-only, restart-only, and all enabled. |
| `vds_chunk_boundaries` | Frame counts at chunk-1, chunk, chunk+1, and 2*chunk+1. |
| `vds_repair` | Strict, complete-prefix, failure status, and repair metadata. |

Compare mdinfo as structured key/value content. Normalize line endings before
exactly comparing QC SCF and metadynamics diagnostic text.

## 8. PR 5: Input Behavior Closure

Create focused A/B coverage for:

- Basic topology: mass, charge, residue, exclusions, LJ, and nb14.
- Bonded force fields: bond, angle, dihedral, improper, Urey-Bradley, and CMAP.
- Specialized topology: GB, softcore, subsystem division, and all virtual-atom
  types including periodic-boundary cases.
- Many-body force fields: EAM, SW, EDIP, Tersoff, and ReaxFF.
- Custom pairwise and listed forces.
- Protocol behavior: constraints, CV, restraint, steering, and soft wall.
- Bias/state behavior: SITS, metadynamics, and NHC.
- QC type, energy, spin-square, and SCF text.

Each case must prove that the legacy and bundled branches activate the same
module and compare a module-owned result. Initialization messages or H5 path
existence alone are insufficient.

## 9. PR 6: Rerun and Failure Semantics

### Rerun matrix

- Start/strip: 0/0, 1/0, 0/1, and 1/1.
- Frame limit: 1, exact EOF, beyond EOF, and default unlimited.
- Box update: false and true.
- Input trajectory: velocity present and absent.
- Particle stream: default `all` and one non-default stream.
- Restart: absent, structural, dynamic, protocol, and full.
- Output: VDS off and on.

Use a risk-driven pairwise matrix plus explicit boundary cases instead of the
complete Cartesian product.

### F1 failure matrix

- Atom count, shape, dtype, and schema-version mismatch.
- Missing topology, protocol, restart, or trajectory binding.
- Mixed legacy/H5 restart or trajectory inputs.
- Unsupported sidecar key, key/path length mismatch, and path conflict.
- Invalid chunk size, VDS value, repair policy, and restart policy.
- Dynamic/protocol/full state requested without its owning runtime module.

Compare exit code, error category, and stable diagnostic tokens. Full stderr
byte equality is not required.

## 10. PR 7: Execution Matrix and CI Promotion

PR 7 is split into three independently reviewable commits because the promotion
mechanism can be verified while the runtime matrix is still red. PR 7a must not
claim behavior coverage or enable a release gate; PR 7b owns portable runtime
cases and CPU/MPI evidence; PR 7c owns GPU evidence and eventual promotion.

### PR 7a: Executable Shadow Promotion Guard

- Add a machine-readable matrix containing every required axis and risk-driven
  combination below.
- Validate the matrix structurally and reject missing axes, combinations,
  scenario evidence, or non-executable case references.
- Add a promotion evaluator for supported-contract evidence, scenario status,
  consecutive retry-free runs, and performance budgets.
- Run contract, medium, and production tiers in an explicitly non-release,
  shadow CI workflow. Known runtime failures must remain visible.

Acceptance:

- Mutation tests reject a removed axis, removed risk combination, unresolved
  scenario, missing evidence, fewer than three retry-free runs, and each
  performance-budget violation.
- The current matrix evaluates as not ready while deferred scenarios or
  behavior gaps remain.
- Release workflows do not consume the shadow result.

### PR 7b: Portable Runtime Matrix and CPU/MPI Evidence

- Add pinned same-semantic TIP3P legacy/bundled fixtures for orthogonal and
  nonorthogonal boxes without requiring an online converter during the gate.
- Define executable cases for every declared CPU/GPU, OMP, and MPI scenario.
- Run and compare all CPU rank-1 and rank-2 scenarios, including a single
  rank-0 H5 finalize report and output ownership under MPI.
- Make deterministic position comparisons and statistical position features
  periodic-box aware. Bound pair-distribution work with deterministic sampling.
- Keep GPU scenarios executable but unproven when no GPU runner is available.

Acceptance:

- CPU rank-1 covers deterministic NVE; middle, Bussi, and NHC NVT; orthogonal
  and nonorthogonal NPT; all three constraint modes; and OMP 1/4.
- CPU rank-2 covers deterministic NVE and nonorthogonal statistical NPT, with
  rank-0 output ownership asserted.
- Periodic-image-only position changes pass, while material force/position
  mutations remain rejected.
- Contract, comparator, matrix manifest, and workflow validation tests pass.

### PR 7c: GPU Evidence and Final Promotion

Add a risk-driven execution matrix:

- Deterministic NVE.
- Statistical middle-Langevin, Bussi, and NHC NVT.
- NPT/barostat with orthogonal and non-orthogonal boxes.
- SHAKE, SETTLE, and unconstrained systems.
- CPU and GPU, comparing legacy versus bundled within the same backend.
- `OMP_NUM_THREADS=1` and `4`.
- MPI rank counts 1 and 2, including rank-0 output ownership.

Gate tiers:

| Tier | Contents | Target |
|---|---|---|
| contract | Registry, schema, conversion, and negative contract tests. | Under 5 minutes. |
| medium | Deterministic matrix and shortened statistical A/B. | 30-45 minutes. |
| production | Full replicas, continuation, module, and environment matrix. | Nightly and release gate. |
| soak | Repeated production runs for flake and resource analysis. | Scheduled. |

Production promotion requires:

- 100% of supported contracts at their minimum evidence level.
- All runtime input/output contracts at E3.
- Restart and dynamic state contracts at E4.
- All declared invalid combinations at F1.
- Three consecutive production runs without retry.
- All comparator mutation tests reject their injected defects.
- Every PR 7a matrix scenario is backed by executable case IDs whose evidence
  metadata proves the declared backend, thread count, MPI rank count, and
  rank-0 output ownership; a declaration-only scenario cannot be promoted.

## 11. PR 8: VDS Chunk-Boundary Behavior Closure

This follow-up scope closes the deferred runtime behavior contract for
`output_h5_trajectory_chunk_size` without weakening the unsupported
cross-process append boundary.

- Run same-semantic legacy-input and bundled-input deterministic NVE branches
  with a fixed chunk size.
- Cover frame counts at chunk-1, chunk, chunk+1, and 2*chunk+1.
- Compare mdout plus every trajectory/observable H5 dataset across branches.
- Assert the VDS wrapper frame count and physical shard count at each boundary.
- Emit E3 evidence for the chunk-size contract from the executed cases.

Acceptance:

- The four boundary cases produce exactly 3, 4, 5, and 9 frames for chunk size
  4 and exactly 1, 1, 2, and 3 shards respectively.
- A wrong frame count, shard count, or cross-branch numeric payload fails the
  gate.
- Registry, manifest, comparator, and full contract tests pass.

## 12. PR 9: Focused Non-Zero EDIP Input Behavior Gate

This follow-up replaces the broad fixture's all-zero EDIP sentinel with a
minimal deterministic case whose owning module has distinguishable behavior.

- Generate a two-atom legacy input with `EDIP_in_file`, finite separation, and
  no competing force-field module.
- Convert the same input to native `/manybody/edip` H5 datasets and remove the
  converter's unused legacy-sidecar directory before execution.
- Require non-zero, cross-branch equivalent EDIP energy and force values.
- Compare complete deterministic mdout rows and the non-zero legacy force route
  across the legacy-input and bundled-input branches.
- Keep H5 force/output schedule behavior outside this input-only evidence scope;
  that independent contract must be repaired and gated separately.

Acceptance:

- The bundled mdin has no `EDIP_in_file` and the runtime directory has no
  `legacy_sidecars`; all required EDIP pair/triple/type datasets are present.
- Both branches emit finite `EDIP` values above the declared sentinel floor and
  non-zero force payloads.
- Zero energy, zero force, force mismatch, or mdout mismatch fails the gate.
- Registry, manifest, comparator, real focused A/B, Ruff, and diff checks pass.

## 13. PR 10: H5 Sidecar-Table Failure Semantics

Close the deferred F1 contract for malformed bundled sidecar tables with real
process execution rather than reader-only checks.

- Mutate the topology H5 sidecar table with an unsupported command key.
- Create unequal key/path dataset lengths.
- Duplicate an allowed key with a different path to trigger command conflict.
- Require non-zero exit, `spongeErrorValueErrorCommand`, and stable diagnostic
  tokens for every mutation.
- Keep these cases bundled-only because legacy text inputs have no equivalent
  H5 key/path table representation.

Acceptance:

- All three mutations reach the production sidecar materialization path and are
  rejected before MD execution.
- Unsupported-key evidence names the topology binding and invalid key;
  length-mismatch evidence names the key/path cardinality error; conflict
  evidence names the key and both existing/H5 paths.
- Registry, manifest, all three real CPU failure cases, Ruff, and diff checks
  pass.

## 14. PR 11: Restart Owner-State Failure Semantics

Close the deferred F1 owner-state contract by exercising each non-structural
restart load policy through the production process.

- Request `dynamic` state containing Nose-Hoover-chain payload while the
  current run has no Nose-Hoover-chain thermostat owner.
- Request `protocol` state containing metadynamics payload while the current
  run has no initialized metadynamics owner.
- Request `full` state with the same missing dynamic owner and require it to
  reject before partially applying later protocol state.
- Require non-zero exit, `spongeErrorConflictingCommand`, and owner-specific
  stable diagnostic tokens for all three cases.

Acceptance:

- The three cases use the same bundled restart payload and differ only in the
  declared `input_h5_restart_load` policy.
- Dynamic/full diagnostics name the Nose-Hoover-chain state and missing
  thermostat; protocol diagnostics name metadynamics state and its missing
  module.
- Registry, manifest, all three real CPU failure cases, Ruff, and diff checks
  pass.

## 15. PR 12: Focused Non-Zero Custom-Pair Input Gate

Close the custom-pair input contract with a minimal deterministic fixture whose
module-owned energy and force cannot pass trivially.

- Generate a two-atom legacy input with a named custom pair potential and a
  non-zero parameter payload.
- Convert it to `/forcefield/custom_force/pairwise`, then remove both legacy
  custom-force mdin keys, the H5 sidecar table, and the sidecar directory from
  the bundled branch.
- Require non-zero, cross-branch equivalent `custom_pair` energy and force.
- Require both native materialized inputs to be present and non-empty.

Acceptance:

- The bundled branch has no `pairwise_force_in_file` or
  `custom_pair_in_file`, yet materializes both inputs from native H5 data.
- Complete deterministic mdout rows match, `custom_pair` is non-zero in both
  branches, and all legacy force-route values match within the force tolerance.
- Registry, manifest, real focused CPU A/B, Ruff, and diff checks pass.

## 16. PR 13: Focused Native Exclusion Semantics Gate

Close the exclusions contract with a three-atom, non-periodic Coulomb fixture
whose excluded-pair behavior has an independent analytic oracle.

- Exclude the close `(0, 1)` pair while retaining two non-zero interactions.
- Convert to `/topology/exclusions/{offset,list}` and remove all bundled
  sidecar routes.
- Require cross-branch deterministic mdout and force equivalence.
- Independently require energy `1/4 - 1/3 = -1/12` and the corresponding nine
  Cartesian force values; the unexcluded energy would instead be `-13/12`.

Acceptance:

- Bundled offsets/list are exactly `[0, 1, 1, 1]` and `[1]`, with no
  `exclude_in_file` or legacy sidecar table.
- Both branches emit non-zero Coulomb behavior, match each other, and satisfy
  the analytic energy/force oracle.
- Energy and force mutations representing ignored or corrupted exclusions are
  rejected by unit tests.
- Registry, manifest, real focused CPU A/B, Ruff, and diff checks pass.

## 17. PR 14: Focused Native LJ Soft-Core Behavior Gate

Close the LJ soft-core input contract with a two-atom periodic fixture whose
A-state interaction is absent and B-state interaction produces distinguishable
soft-core energy and force at `lambda_lj = 0.5`.

- Convert the legacy `LJ_soft_core_in_file` payload to native
  `/forcefield/lj_soft_core`, then remove the legacy key, sidecar table, and
  sidecar directory from the bundled branch.
- Preserve source-semantic pair values in H5 and apply the legacy loader's
  `12/6` A/B normalization exactly once when pure native H5 owns the payload.
- Require non-zero, cross-branch equivalent `LJ_soft` energy and force values,
  plus complete deterministic mdout equivalence.
- Keep `/forcefield/subsys_division` absent so this case proves only the
  soft-core parameter contract. Subsystem division remains deferred because
  the current force kernel copies but does not consume its mask.

Acceptance:

- The bundled branch has no `LJ_soft_core_in_file`, subsystem input, or legacy
  sidecars, and contains the exact native atom-type and pair datasets.
- Both branches emit the same non-zero `LJ_soft` result and six-value force
  payload; zero or mismatched energy/force mutations are rejected.
- Legacy text, H5 fixture-equivalence, and native-reader tests retain their
  source-semantic parameter values while runtime behavior matches.
- Registry, manifest, real focused CPU A/B, Ruff, and diff checks pass.

## 18. PR 15: Focused Native Virtual-Atom Behavior Gates

Close the virtual-atom reconstruction contracts with one all-types fixture and
one boundary-crossing fixture. Both bundled branches must be pure native H5,
with no legacy key or sidecar fallback.

- Enumerate types 0, 1, 2, and 3 in one ragged native payload and require exact
  offsets, parent indices, parameters, reconstructed coordinates, non-zero PM
  behavior, real-atom force signal, and full legacy/bundled force equivalence.
- Make type 3 depend on a type 2 site so the runtime case also exercises ordered
  multi-layer reconstruction.
- Put the type 1 parents on opposite sides of a periodic box and require the
  minimum-image coordinate `9.75`; a non-periodic interpolation would produce
  `7.25` and must fail the oracle.
- Refresh virtual coordinates from global records before domain decomposition,
  then switch to local records after `Get_Local`; copy type 3 device records
  from their populated host source.

Acceptance:

- The bundled topology contains the exact `/forcefield/virtual_atom` ragged
  datasets and retains neither virtual-atom legacy key nor sidecar metadata.
- Both branches match the per-type and PBC coordinate oracles, emit finite
  non-zero PM/force behavior, and match complete force and mdout payloads.
- Coordinate, PBC, and trivial-real-force mutations are rejected by unit tests.
- Registry, manifest, real focused CPU A/B, related H5 CTest, Ruff, and diff
  checks pass.

## 19. PR 16: Virtual-Atom Plural Alias Behavior Gate

Close the `virtual_atoms_in_file` compatibility contract independently from the
canonical singular key.

- Reuse the boundary-crossing type 1 fixture, but expose the source only through
  `virtual_atoms_in_file` in the legacy branch.
- Let the legacy native loader resolve the plural alias when the canonical key
  is absent, without changing custom virtual-atom module key resolution.
- Require conversion to the exact `/forcefield/virtual_atom` ragged payload and
  remove both spellings, the sidecar table, and the sidecar directory from the
  bundled branch.
- Apply the same `9.75` PBC coordinate, non-zero PM, real-atom force, complete
  force, and mdout equivalence assertions as the canonical PBC contract.

Acceptance:

- The legacy branch contains only `virtual_atoms_in_file`; the bundled branch
  contains neither spelling and has no fallback sidecar route.
- Both branches satisfy the same runtime coordinate/energy/force semantics.
- Registry, required-semantic inventory, manifest, real focused CPU A/B,
  related H5 CTest, Ruff, clang-format, and diff checks pass.

## 20. Restart-Absent Same-Bootstrap Rerun Gate

Goal: prove that omitting restart input has the same behavior in legacy and
bundled rerun paths, rather than merely proving that both processes start.

Fixture and route requirements:

- Use byte-identical legacy coordinate and velocity files as bootstrap state
  for both branches.
- Keep legacy text trajectory input and bundled H5 trajectory input as the
  only branch-specific state route.
- Remove `input_h5_restart_path`, `input_h5_restart_load`, and the bundled
  `restart.spgr.h5` file so an accidental structural fallback cannot satisfy
  the test.
- Run two selected frames with box update disabled to isolate restart absence
  from the known legacy box-update invalid-free defect.

Behavior requirements:

- Compare every mdout row and column deterministically.
- Check the selected rerun frame indices against an independent oracle.
- Compare bundled trajectory and observable output against the legacy branch,
  including position, box, velocity, force, and scalar observables.
- When no restart exists, derive the expected fixed box from the shared
  orthogonal bootstrap coordinate file; existing structural cases continue to
  use the restart H5 box.

Acceptance:

- The route assertions prove identical bootstrap bytes and complete absence of
  restart bindings and payloads.
- All three E3 assertions pass in a real CPU run.
- Registry, manifest, restart/input CTest, Ruff, smoke, and diff checks pass.

## 21. PR 18: NHC Dynamic Restart Continuation Gate

Close the dynamic restart input, NHC-owned state, and dynamic restart output
contracts with one deterministic producer checkpoint rather than two
independently evolved branches.

- Run one NVT Nose-Hoover-chain producer with chain length three and write the
  particle structural state plus NHC state through both legacy text and H5
  restart routes at the same restart event.
- Require the producer NHC state to be finite and non-zero, and compare the
  legacy and H5 structural/NHC payloads before either continuation starts.
- Fork that single checkpoint into a legacy continuation that binds coordinate,
  velocity, and NHC text restart files and a bundled continuation that binds
  H5 topology/protocol plus `input_h5_restart_load = "dynamic"`.
- Continue both deterministic NHC runs for two steps with identical integrator,
  thermostat, topology, and output settings.
- Compare every mdout column, particle position/velocity/force/box streams, NHC
  coordinate/velocity trajectories, observable-only output, final structural
  restart, final NHC restart, and integrator mode/step/time state.
- Compare legacy NHC trajectory/restart text against the H5 module datasets in
  each branch; account only for the legacy `%f` representation boundary and do
  not relax the existing position or velocity tolerances.

Acceptance:

- The producer state and continued NHC state are non-trivial.
- The bundled mdin contains no legacy coordinate, velocity, or NHC restart
  binding; the legacy mdin contains no H5 binding.
- Any NHC component swap, particle-state mismatch, mdout mismatch, schedule
  mismatch, missing output family, or output-to-input continuation mismatch
  fails the gate.
- The three contracts emit E4 evidence from the same executed case, and
  registry, manifest, real CPU A/B, related CTest, Ruff, and diff checks pass.

## 22. PR 19: Protocol/Full Metadynamics Continuation Gate

Close the protocol and full restart-load policies together with the
metadynamics input contract, using one deterministic checkpoint and three
same-semantic continuations.

- Generate a two-atom periodic NVT producer with a distance collective
  variable, one-dimensional well-tempered metadynamics, and a three-element
  Nose-Hoover chain. Require multiple finite, non-zero hills and a finite
  potential snapshot in the H5 restart.
- Treat the H5 restart event as the checkpoint boundary. Project its stored
  hill and potential text into the legacy branch instead of copying the
  producer's terminal sidecar, because the terminal sidecar contains the next
  hill written after that restart event.
- Fork the checkpoint into: a pure legacy structural/NHC/metadynamics text
  continuation; a `protocol` continuation that loads H5 structural and
  metadynamics state while retaining only the legacy NHC dynamic state; and a
  `full` continuation that loads structural, NHC dynamic, and metadynamics
  state entirely from H5.
- Compare every mdout column; particle position, velocity, force, and box;
  NHC coordinate/velocity trajectories and final state; metadynamics
  `meta/rbias/rct` streams; final hill histories; observable-only output; and
  final restart/integrator state.
- Apply existing deterministic numeric tolerances only at legacy text
  precision boundaries. Reject missing/non-finite/trivial bias, a different
  hill count, a route that mixes owners, or any schedule/shape mismatch.

Acceptance:

- All three continuations use the same producer checkpoint and yield finite,
  non-zero metadynamics work with equivalent particle, thermostat, bias, and
  restart behavior.
- The `protocol` branch proves that only protocol state is added beyond the H5
  structural baseline; the `full` branch contains no legacy restart binding.
- `input.restart_load.protocol`, `input.restart_load.full`, and
  `input.bias.metadynamics` emit E4 evidence from the executed case.
- Registry, manifest, real CPU A/B, related CTest, Ruff, and diff checks pass,
  followed by exactly one PR-scoped commit.

## 23. PR 20: Unrestricted QC Observable and SCF Text Gate

Close the non-trivial QC spin-square and non-empty SCF text contracts with a
deterministic unrestricted-QC rerun under both VDS modes.

- Start from one two-atom `H/N` full-contract fixture and change both branches
  from singlet/restricted to triplet/unrestricted QC. Keep atom indices,
  symbols, charge, coordinates, and every non-QC input unchanged.
- Keep the bundled sidecar QC type payload and its typed `/qc/type` metadata
  synchronized at multiplicity three. This case proves output behavior only;
  it does not promote `input.qc.type`, because the pure bundled runtime still
  does not materialize `/qc/type` into the QC input consumer.
- Require a present, finite, non-zero `QC_S_sq` value in every branch and exact
  deterministic legacy/bundled mdout equivalence. Bridge the same values to
  `/observables/all/qc/spin_square/{step,time,value}`.
- Enable `qc_scf_print_iter`, require non-empty legacy and bundled
  `qc_scf_output`, normalize line endings, and compare the complete text.
  Require both bundled trajectory and observable files to archive exactly the
  same text at `/parameters/sponge/qc/scf_output`.
- Run the focused case with VDS disabled and enabled so both trajectory writer
  implementations carry the QC parameter text while observable behavior stays
  invariant.
- Remove `qc_scf_exact_equivalence` evidence from the old full-contract cases,
  whose SCF files are empty, so successful process startup can no longer
  satisfy this contract.

Acceptance:

- Both VDS modes produce the same two finite, non-zero spin-square frames in
  legacy mdout and bundled observable output.
- SCF output is non-empty and exactly equal across legacy text, bundled text,
  trajectory metadata, and observable metadata.
- Zero spin-square, missing observable mapping, empty SCF text, text mismatch,
  or missing H5 archive fails the gate.
- `input.qc.spin_square`, `input.qc.scf_text`, and
  `output.legacy.qc_scf_output` emit E3 evidence only from the focused cases;
  `input.qc.type` remains explicitly deferred.
- Registry, manifest, both real CPU A/B cases, Ruff, and diff checks pass,
  followed by exactly one PR-scoped commit.

## 24. PR 21: Constraint Sidecar Projection Gate

Close the executable constraint sidecar contract with a deterministic,
non-trivial two-atom projection while keeping native typed `/constraint`
consumption explicitly separate.

- Generate two equal-mass atoms at the target distance `1.5` with opposing
  radial velocities `-1.0` and `1.0`, so merely parsing or initializing the
  constraint module cannot satisfy the gate.
- Convert the same legacy input into a bundle. Require the converter's typed
  `/constraint/default/pairs/{atoms,r0}` payload and the exact
  `constrain_in_file` sidecar binding to coexist, while the bundled mdin does
  not retain a direct legacy key.
- Run four deterministic NVE steps with SHAKE enabled. For every emitted
  frame, require the pair-distance residual to stay below `1e-5` and the
  relative radial-velocity residual to stay below `1e-4` in both branches.
- Compare the complete legacy/bundled position and velocity trajectories with
  the existing deterministic tolerances, in addition to full mdout equality.
- Record this evidence for `input.protocol.constraint.sidecar` only. Keep
  `input.protocol.constraint` deferred until the runtime consumes typed
  `/constraint` without the sidecar table.
- Mutation tests must reject both a wrong constrained distance and an
  unprojected radial velocity, preventing module-startup evidence from passing.

Acceptance:

- Both branches reduce the initial relative radial speed from `2.0` to at most
  `1e-4` and preserve the target distance in every one of four frames.
- The sidecar payload is byte-identical to the legacy constraint file and is
  bound only through the protocol H5 sidecar table in the bundled branch.
- Removing constraint execution, changing the target geometry, retaining
  radial motion, changing the schedule, or diverging either trajectory fails
  the gate.
- `input.protocol.constraint.sidecar` emits E3 evidence from the focused case;
  native typed `input.protocol.constraint` remains explicitly deferred.
- Registry, manifest, real CPU A/B, Ruff, and diff checks pass, followed by
  exactly one PR-scoped commit.

## 25. PR 22: Native Improper Runtime Gate

Separate SPONGE's executable native improper behavior from the still-broken
legacy-to-bundle integration, and close only the former with a non-trivial
four-atom case.

- Generate four non-coplanar equal-mass atoms and one harmonic improper with
  `pk=10.0` and `phi0=0.2`. Require the legacy canonical
  `improper_dihedral_in_file` route to produce finite, non-zero
  `improper_dihedral` energy and force.
- Convert the input, then explicitly normalize the converter's legacy-schema
  `/forcefield/improper/k` dataset to the runtime reader's
  `/forcefield/improper/pk`. Remove the complete H5 sidecar table and sidecar
  directory so the bundled result can only come from typed native state.
- Require the bundled mdin to contain neither `improper_dihedral_in_file` nor
  the `improper_in_file` alias. Validate the exact one-row atom, `pk`, and
  `phi0` payload before execution.
- Compare the module-owned energy row, complete mdout row, and complete binary
  force payload deterministically. Reject trivial energy, trivial force, any
  shape change, or any cross-branch value mismatch.
- Track the passing behavior as `input.topology.improper.native_runtime`. Keep
  `input.topology.improper` deferred for end-to-end conversion: the converter
  currently writes `/k`, canonical sidecars are not allowlisted, and the
  allowlisted alias is not consumed by `Native_Load_Impropers`.

Acceptance:

- Legacy canonical text and pure native typed H5 produce the same finite,
  non-zero improper energy and force without bundled sidecars.
- Removing or ignoring the typed payload makes the semantic gate fail even if
  the process and improper module initialization complete.
- The registry and documentation do not use the native runtime evidence to
  claim that legacy-to-bundle improper conversion is closed.
- Registry, manifest, real CPU A/B, Ruff, and diff checks pass, followed by
  exactly one PR-scoped commit.

## 26. PR 23: GB Native-State/Sidecar-Activation Gate

Close the executable GB hybrid route without using its evidence to claim that
pure native `/forcefield/gb` initialization is supported.

- Generate two non-periodic atoms with charges `+1/-1`, separation `2.0`, and
  identical GB radius/scale parameters `1.5/0.8`. Require finite, non-zero
  module-owned `gb` energy.
- Convert the same input and retain typed `/forcefield/gb/params`. Reduce the
  H5 sidecar table to the single `gb_in_file` binding so mass and charge are
  necessarily consumed from typed native state.
- Treat the remaining GB sidecar as an activation binding, not GB payload
  evidence: native topology state is loaded before sidecar injection, and the
  later binding causes `main` to call `gb.Initial` against that native state.
- Require the bundled mdin to contain no direct `gb_in_file`, while the bound
  sidecar remains byte-identical to the legacy input and the typed parameter
  matrix is validated independently.
- Compare complete mdout and force payloads. In addition to non-zero GB energy,
  require the total force oracle `+/-0.10313021`; reject the Coulomb-only
  `+/-0.25` result so process startup or an ignored GB force cannot pass.
- Track this route as `input.topology.gb.hybrid_activation`. Keep pure native
  `input.topology.gb` deferred until initialization no longer depends on the
  legacy-key sidecar.

Acceptance:

- Legacy text and the typed-state/sidecar-activation bundle produce equivalent
  non-zero GB energy and GB-influenced force.
- Missing or trivial GB energy, Coulomb-only force, sidecar-table contamination,
  typed parameter drift, or any cross-branch mdout/force mismatch fails.
- The registry explicitly distinguishes the executable hybrid route from the
  deferred pure native contract.
- Registry, manifest, real CPU A/B, Ruff, and diff checks pass, followed by
  exactly one PR-scoped commit.

## 27. PR 29: VDS Complete-Prefix Repair Behavior Gate

Close the VDS repair-policy contract with separate evidence for successful
production finalization and terminal-shard failure. A normal no-op is required
behavior, but is not by itself evidence that repair works.

- Run a deterministic five-frame, chunk-size-four SPONGE case from both the
  legacy-input and bundled-input routes with
  `output_h5_trajectory_repair_policy=complete_prefix`.
- Compare every mdout column and every numeric/string H5 dataset across the two
  routes. Require two complete shards, five materialized frames,
  `repair_status=not_applied`, and `repaired_shard_count=0` in both wrappers.
- In the same production gate, execute the real HighFive terminal/resume smoke
  helper with a backend that fails finalization of the terminal shard. Require
  `Finalize_With_Repair` to retain only the valid prefix, rewrite completion
  metadata to one frame, report `repair_status=applied`, and count one repaired
  shard.
- Mutation-test the production no-op oracle against strict policy metadata,
  false applied status, non-zero repaired count, wrong completion frame count,
  and an open manifest entry.
- Keep `output.vds.cross_process_append_resume` unsupported. Complete-prefix
  repair finalizes the current process output; it does not reopen an existing
  wrapper for append.

Acceptance:

- Legacy-input and bundled-input production runs retain identical complete
  five-frame semantics and do not report a repair when every shard finalizes.
- The real backend failure helper proves that a failed terminal shard is
  removed and the wrapper exposes only the valid complete prefix.
- Either half of the evidence being absent, any metadata mutation, an
  incomplete retained manifest, or any cross-branch H5 difference fails the
  gate.
- Registry, manifest, real CPU A/B, real-backend VDS smoke, Ruff, and diff
  checks pass, followed by exactly one PR-scoped commit.

## 28. PR 31: Residue Membership/COM-Virial Follow-up Gate

Strengthen PR 30 from residue-count and whole-molecule evidence to a consumer
whose result changes with residue membership while keeping the declared residue
count fixed.

- Retain PR 30's `normal_residue_sidecar_pbc_mapping` case and its residue-count,
  runtime-partition, force, and whole-molecule PBC mapping mutations. Add the
  COM-virial case as a second consumer of the same sidecar contract instead of
  replacing the existing gate.
- Use two disconnected two-atom molecules with the same `[2,2]` residue
  partition in the legacy and bundled routes. Isolate `residue_in_file` in the
  topology H5 sidecar table and retain only the constraint and restraint
  support bindings in protocol H5.
- Enable `restrain_refcoord_scaling="com_res"`, restraint virial calculation,
  and pressure output. Compare complete mdout and force output, and require
  finite non-trivial `bond`, `restrain`, `pressure`, and `Pxx` behavior.
- Execute a bundled control with `[1,3]`: the declared residue count remains
  two, but the disconnected second residue is split by the runtime. Require
  unchanged bond/restraint energy and force together with a distinct,
  deterministic pressure/Pxx fingerprint. This prevents sidecar parsing or
  module initialization alone from satisfying the gate.
- Fix CPU ownership of atom-to-group maps used by `com_ug`, `com_res`, and
  `com_mol`: allocate independent device storage before copying and then free
  the temporary host map. This avoids a dangling CPU alias that is overwritten
  by later COM scratch allocation.
- Keep typed `/atoms/residue_index` deferred; this PR closes only the isolated
  topology-sidecar behavior route.

Acceptance:

- Legacy and bundled `[2,2]` routes produce equivalent finite mdout,
  trajectory, and non-trivial force behavior at the COM-residue consumer.
- The `[1,3]` control keeps two declared residues and the same energy/force but
  changes both pressure and `Pxx`, proving membership-sensitive virial use.
- Mutation tests reject wrong residue count, pressure, `Pxx`, and trivial force
  without searching production-test source text for implementation tokens.
- Registry, manifest, real CPU A/B, Ruff, clang-format, and diff checks pass,
  followed by exactly one PR-scoped commit.

## 29. PR 32: Pure Native GB Behavior Gate

Close the remaining pure typed generalized-Born input contract without using a
legacy activation sidecar.

- Retain PR 23's native-state plus `gb_in_file` sidecar-activation case as a
  separate compatibility route.
- Add a second deterministic two-atom case whose bundled branch contains
  `/forcefield/gb/params` but no `gb_in_file` command, H5 sidecar table, or
  `legacy_sidecars` directory. The legacy branch continues to consume the same
  radii and scale factors through `gb_in_file`.
- Initialize GB in non-periodic mode when either the legacy command exists or
  `Xponge::system.generalized_born` already contains native parameters.
- Require complete mdout equality, `Coulomb=-0.50`, `gb=-0.25`,
  `potential=-0.75`, and the complete six-value force oracle with maximum
  magnitude `0.10313020646572113`. Reject zero GB output and the Coulomb-only
  force fingerprint.

Acceptance:

- Both the pure typed and hybrid GB cases pass real CPU legacy/bundled A/B.
- The pure typed route cannot satisfy the gate through a retained text binding
  or sidecar payload.
- Registry, manifest, related native-reader CTest, full smoke, Ruff,
  clang-format, and diff checks pass, followed by exactly one PR-scoped commit.

## 30. PR 33: Core Topology Payload-Sensitivity Gate

Replace the existing shared statistical evidence for mass, charge, and
Lennard-Jones input with an additional deterministic, payload-specific behavior
gate.

- Use one two-atom NVE fixture with `dt=0`, no PBC, explicit non-zero velocity,
  charge, and LJ interactions. The legacy branch reads `mass_in_file`,
  `charge_in_file`, and `LJ_in_file`; the bundled branch contains only typed
  `/atoms/mass`, `/atoms/charge`, and `/forcefield/lj` datasets, with no topology
  sidecar table or `legacy_sidecars` directory.
- Require complete one-frame mdout and force equivalence between the two routes.
- Run three independent bundled controls which mutate only one typed owner:
  doubling both masses changes temperature from `26.21` to `52.42` while force
  remains unchanged; zeroing charge changes Coulomb energy from `-0.50` to zero
  and changes force; zeroing both LJ pair arrays changes LJ energy from `-0.02`
  to zero and changes force.
- Make non-periodic LJ initialization consume the same materialized
  `Xponge::system.classical_force_field.lj` state as periodic LJ, while retaining
  named-module legacy loading.

Acceptance:

- The real CPU A/B and all three payload controls pass with finite, non-trivial
  module-owned responses.
- Manifest mutation tests reject trivial correct values, unchanged observables,
  missing required force changes, unexpected mass-only force changes, and
  non-finite force payloads.
- The original statistical TIP3P evidence remains active as a broader regression.
- Registry, full manifest and smoke suites, SPONGE rebuild, related native-reader
  CTests, Ruff, clang-format, and diff checks pass, followed by exactly one
  PR-scoped commit.

## 31. PR 34: Complete H5 Metadata Failure Semantics

Close the complete topology metadata F1 contract by adding the missing schema
version rejection to the existing atom-count, shape, and dtype process gates.

- Preserve the three schema versions emitted by current supported producers:
  legacy unit fixtures (`0`), native H5 fixtures (`1`), and XPONGE conversion
  (`xponge.legacy_to_bundle.v1`).
- Add a bundled-only mutation that replaces `/schema/version` with
  `unsupported.topology.v999` and require a non-zero process exit,
  `spongeErrorValueErrorCommand`, and route-specific stable diagnostics.
- Run validation from the common `Xponge::System::Load_Inputs` entry before any
  native text input loader. Legacy-only inputs remain unaffected.
- Merge the partial three-mutation `failure.h5_metadata.runtime_rejections`
  registry record into the complete `failure.h5_metadata` contract so one owner
  enumerates all four failure semantics.

Acceptance:

- All four real bundled failure cases reject their mutation with the exact
  expected error category and stable diagnostic tokens.
- Known supported schema versions continue to pass existing real A/B and H5
  input-validation tests.
- Registry, full manifest and smoke suites, SPONGE rebuild, related CTests,
  Ruff, clang-format, and diff checks pass, followed by exactly one PR-scoped
  commit.

## 32. PR 35: Subsystem-Division Energy-Partition Behavior Gate

Close the typed subsystem-division contract with a partition-sensitive runtime
observable while preserving the established soft-core dynamics.

- Treat `subsys_division` as an ownership label for short-range soft-core LJ
  energy accounting. A pair whose atom masks are equal contributes to
  `LJ_soft_intra`; a pair whose masks differ contributes to `LJ_soft_inter`.
- Keep the existing pair energy, total energy, force, virial, and lambda
  derivative formulas unchanged. Require the explicit conservation invariant
  `LJ_soft_inter + LJ_soft_intra = LJ_soft_short`.
- Use a deterministic two-atom soft-core fixture. The legacy branch consumes
  `subsys_division_in_file`; the bundled branch consumes only typed
  `/forcefield/subsys_division`, with no retained subsystem command, topology
  sidecar table, or `legacy_sidecars` directory.
- Run the same-semantic `[0,1]` A/B pair and require non-zero inter-system
  energy. Run a second same-semantic `[0,0]` control A/B pair and require the
  same energy to move entirely to the intra-system observable.
- Require complete mdout equivalence for both semantic inputs, unchanged total
  soft-core energy and force across the mask mutation, and exact legacy/bundled
  partition values. This proves runtime consumption without assigning an
  unsupported force-filtering meaning to the historical mask.

Acceptance:

- The `[0,1]` routes match at `LJ_soft_inter=-0.06`,
  `LJ_soft_intra=0.00`, and `LJ_soft_short=-0.06`; the `[0,0]` routes match at
  `LJ_soft_inter=0.00`, `LJ_soft_intra=-0.06`, and the same total.
- Both controls retain the complete non-zero six-value force fingerprint, and
  the bundled branch has only the native subsystem dataset.
- Mutation tests reject non-finite, trivial, mixed, non-conserving, and
  total-energy-changing partition evidence.
- Registry, manifest, real CPU A/B, full smoke, related native-reader CTests,
  SPONGE rebuild, Ruff, clang-format, and diff checks pass, followed by exactly
  one PR-scoped commit.

## 33. PR 36: Typed Residue Runtime-Behavior Gate

Close the native typed residue contract by materializing
`/atoms/residue_index` into `Xponge::system.residues` before MD-core residue
consumers initialize.

- Apply the typed route only to native H5 input. Reject simultaneous
  `residue_in_file` ownership instead of silently selecting either source.
- Require a one-dimensional atom-sized residue-index payload whose labels start
  at zero and remain contiguous and nondecreasing. Convert those labels to the
  per-residue atom counts consumed by the native runtime. When
  `/residues/atom_offset` is also present, require it to start at zero, end at
  atom count, remain nondecreasing, and describe exactly the same mapping.
- Run two deterministic same-semantic A/B pairs. The legacy branches consume
  `residue_in_file`; the bundled branches contain typed
  `/atoms/residue_index=[0,0,1,1]` and
  `/residues/atom_offset=[0,2,4]`, with no topology sidecar table or residue
  sidecar payload.
- Retain only non-target protocol sidecars required to drive the behavior
  oracles. The PBC case proves whole-molecule coordinate mapping; the COM case
  proves residue-membership-sensitive restraint virial and pressure.
- Reuse the `[1,3]` wrong-partition control by mutating the typed payload to
  `[0,1,1,1]`. Require unchanged bond/restraint energy and complete force, but
  a large pressure/Pxx change, proving that the typed membership reaches its
  runtime consumer.

Acceptance:

- Typed and legacy PBC routes both load two residues, produce non-zero
  `bond=2.0`, retain the complete 12-value force, and emit the mapped position
  fingerprint `(19,21,25,28)`.
- Typed and legacy COM routes both produce `bond=2.0`, `restrain=2.0`,
  `pressure=0.04`, and `Pxx=0.11`. The `[1,3]` control preserves energy and
  force while changing pressure/Pxx to `-11.53/-34.60`.
- Both existing residue-sidecar behavior gates continue to pass, demonstrating
  that typed materialization does not override the sidecar route.
- A residue-index/offset disagreement exits with `spongeErrorBadFileFormat`;
  simultaneous typed and `residue_in_file` ownership exits with
  `spongeErrorConflictingCommand`, retaining route-specific diagnostics.
- Registry, manifest, real CPU A/B, full smoke, related native-reader CTests,
  SPONGE rebuild, Ruff, clang-format, and diff checks pass, followed by exactly
  one PR-scoped commit.

## 34. PR 37: Typed SW Pair/Three-Body Behavior Gate

Close `input.manybody.sw` with a same-semantic typed/legacy A/B gate rather
than treating successful SW initialization as behavioral evidence.

- The legacy branch consumes the canonical `SW_in_file`; the bundled branch
  contains only `/manybody/sw` atom-type, pair, and triple datasets, with no SW
  command, H5 sidecar table, or sidecar directory.
- Materialize the typed payload into the existing SW parser so both routes
  share the established validation and force kernel. Validate dataset ranks,
  type ranges, duplicate/missing parameter rows, and complete pair/triple
  parameter coverage before runtime initialization.
- Compare the complete one-step mdout and nine-value force payload. Require
  both branches to produce `SW=194.50` with maximum absolute force
  `404.27862548828125`.
- Mutate only the typed triple lambda from `32.5` to `0`. Require the control
  to produce `SW=158.79`, maximum absolute force `343.52105712890625`, and a
  force delta greater than `1`, proving that the typed three-body parameters
  reach the runtime consumer.

Acceptance:

- The typed bundled route materializes
  `.sponge_h5_native_manybody/sw.txt`; no legacy SW route survives in its
  command file or H5 sidecar metadata.
- The typed and legacy branches match the complete deterministic mdout and
  force behavior, while the lambda-zero counterfactual changes both energy and
  force as expected.
- The existing SW sidecar pair/three-body gate still passes, demonstrating
  that native typed materialization does not override the sidecar route.
- Registry, manifest, real CPU A/B, full smoke, related native-reader CTests,
  SPONGE rebuild, Ruff, clang-format, and diff checks pass, followed by exactly
  one PR-scoped commit.

## 35. PR 38: Typed Constraint Projection Behavior Gate

Close `input.protocol.constraint` by materializing typed protocol pairs into
`Xponge::system.classical_force_field.constraints` before the SHAKE/SETTLE
runtime builds its constraint list.

- Preserve migration compatibility: when `constrain_in_file` is bound by the
  protocol sidecar table, the sidecar remains the owner even if the converted
  H5 file also retains typed constraint datasets.
- For a pure typed route, require both
  `/constraint/default/pairs/atoms` with shape `[n,2]` and
  `/constraint/default/pairs/r0` with shape `[n]`. Reject empty payloads,
  mismatched lengths, out-of-range or self pairs, and non-positive/non-finite
  distances.
- Reuse the deterministic two-atom projection fixture. The legacy branch
  consumes `constrain_in_file`; the bundled branch contains only typed pair
  `[0,1]` with `r0=1.5`, with no constraint command, sidecar table, or sidecar
  directory.
- Compare all four position and velocity frames. Require zero distance error
  and reduction of the initial relative radial speed from `2.0` to at most
  `1e-4` through both routes.
- Change only the typed target and matching bootstrap distance to `2.0` while
  retaining non-zero relative motion. Require all four frames to converge to
  the new target; an implementation that ignores the typed payload must fail.

Acceptance:

- Legacy and pure typed routes have identical complete mdout, position, and
  velocity behavior at `r0=1.5`; both report four projected frames and maximum
  distance residual `0`.
- The `r0=2.0` control reports four projected frames, zero distance residual,
  and radial velocity residual below `1e-4`.
- An out-of-range typed pair `[0,2]` exits with
  `spongeErrorBadFileFormat` and stable dataset/row diagnostics.
- The existing constraint sidecar projection gate remains green, proving that
  typed materialization does not override the migration route.
- Registry, manifest, real CPU A/B, full smoke, related input CTests, SPONGE
  rebuild, Ruff, clang-format, and diff checks pass, followed by exactly one
  PR-scoped commit.

## 36. PR 39: Typed Tersoff Angular Behavior Gate

Close `input.manybody.tersoff` with a pure typed same-semantic behavior gate.

- The legacy branch consumes canonical `TERSOFF_in_file`; the bundled branch
  contains `/manybody/tersoff` atom types, names, map, entry types, raw
  14-parameter rows, and runtime 18-parameter rows, with no Tersoff command,
  sidecar table, or sidecar directory.
- Materialize the typed payload through the established Tersoff text parser.
  Validate atom/type counts, names, entry shapes and indices, duplicate rows,
  map consistency, finite raw values, and agreement between raw and converted
  runtime parameters before force initialization.
- Compare the complete deterministic mdout and nine-value force payload.
  Require both routes to produce `potential=-173.23`,
  `eff_pot=-173.23468`, and maximum absolute force
  `135.94906616210938`.
- Change gamma from `1` to `0` in both typed parameter representations and
  rerun the bundled branch. Require `potential=-196.06`,
  `eff_pot=-196.05984`, maximum absolute force `144.7831268310547`, and
  maximum force delta greater than `25`.

Acceptance:

- Pure typed and legacy routes match the complete energy and force behavior,
  while the gamma-zero counterfactual proves the angular bond-order parameter
  reaches the runtime kernel.
- The bundled route materializes
  `.sponge_h5_native_manybody/tersoff.txt`; no legacy ownership remains.
- A map/entry conflict and a raw/runtime-parameter conflict both exit through
  `spongeErrorBadFileFormat` with dataset-specific diagnostics.
- The existing Tersoff sidecar angular gate remains green, proving typed
  materialization does not override migration-sidecar precedence.
- Registry, manifest, real CPU A/B, full smoke, native input CTests, SPONGE
  rebuild, Ruff, clang-format, and diff checks pass, followed by exactly one
  PR-scoped commit.

## 37. Artifacts and Temporary-Space Policy

- Use `SPONGE_BUNDLED_IO_AB_RUN_ROOT` for all heavy runs.
- Put production artifacts on the workspace filesystem rather than `/tmp`.
- Delete successful per-replica payloads after evidence extraction.
- Retain failed case inputs, stdout/stderr, mdout, H5 metadata, evidence, and a
  minimal reproduction command.
- Record artifact byte counts and enforce a configurable per-case quota.
- Cleanup must only remove directories created by the current gate run.

## 38. PR Completion Log

Append one row immediately after completing and committing each PR.

| PR | Status | Commit | Verification | Notes |
|---|---|---|---|---|
| PR 1: Executable contract registry | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; sidecar rerun VDS-off A/B | 21 contract tests and one real A/B case pass; pure-native rerun exposes a strict column-set gap retained for input behavior closure. |
| PR 2: Contract enumeration and fixtures | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; H5 manifest/native-reader/fixture-equivalence CTest; sidecar rerun VDS-off A/B | Runtime key inventory has exact contract ownership; full-contract mdin/manifest now enumerate output controls, all legacy routes, particle stream, and rerun controls; missing topology runtime cases are explicitly deferred to PR 5. |
| PR 3: Comparators | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real sidecar rerun VDS-off A/B (expected QC failure) | 41 contract/comparator tests pass; TOST plus Holm correction, quantity-specific margins, structured position/velocity/force/box statistics, exact non-finite patterns, and all six comparator mutations are enforced. The real gate still rejects the known QC first-frame behavior gap (`-34183.44` vs `-34139.92`) without weakening tolerances. |
| PR 4: Output behavior closure | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; five output/VDS CTest targets; two-step NVE restart continuation; real TIP3P A/B (expected force-schedule failure) | 45 contract tests and all output writer/backend targets pass. Normal VDS-off/on cases now require position, velocity, force, box, structured mdinfo, explicit route provenance/content, and E4 structural restart continuation. The real gate exposes legacy `frc` 51-frame vs H5 50-frame schedule skew; QC SCF non-empty/exact comparison and dynamic restart continuation remain explicit behavior gaps rather than relaxed assertions. |
| PR 5: Input behavior closure | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; native-topology reader and fixture-equivalence CTest; saved TIP3P and sidecar-rerun production artifacts | 52 contract tests require every supported input contract to produce a present, non-trivial module-owned result and then pass deterministic or statistical A/B comparison. TIP3P mass/charge/LJ pass; the rerun gate rejects the known QC first-frame mismatch (`-34183.44` vs `-34139.92`). Reader/path checks remain E1/E2 only. Residue/exclusions, improper, GB/softcore/subsystem/virtual atoms, SW/EDIP/Tersoff, custom pairwise, constraint/steering, SITS/meta/NHC, and QC type/spin/SCF stay explicitly deferred until dedicated non-trivial fixtures exist. |
| PR 6: Rerun and failure semantics | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; nine process-level F1 cases; five real rerun boundary cases | 60 contract/manifest tests and all nine textual/binding F1 cases pass. The harness now keeps `crd/box/vel` as legacy rerun inputs and inserts overrides before TOML module tables; this removes the earlier QC first-frame skew (`-33917.02` now matches). The 0/0 limit-1 boundary passes, while the gate exposes optional-velocity potential (`1951.16` vs `1947.41`), strip frame counter (`2` vs `1`), and box-update legacy invalid-free failures. H5 atom/shape/dtype/schema and sidecar-table process mutations, restart absent/dynamic/protocol/full continuation, and owner-state F1 cases remain explicitly deferred. |
| PR 7a: Executable shadow promotion guard | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; execution-matrix CLI; workflow YAML parse; `git diff --check` | 70 contract/manifest tests pass. The typed matrix and mutation-tested promotion evaluator reject missing axes/combinations/evidence, unproven environment metadata, retries, comparator-mutation gaps, and runtime/finalize/output-size budget overruns. Medium and production jobs are explicit `continue-on-error` shadow jobs and are absent from the release workflow; runtime evidence remained intentionally deferred at this boundary. |
| PR 7b: Portable runtime matrix and CPU/MPI evidence | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; six CPU rank-1 fast A/B cases; two CPU rank-2 fast A/B cases; Ruff; workflow YAML parse; `git diff --check` | 74 contract/manifest tests and all eight CPU/MPI runtime cases pass. The matrix pins same-semantic fixtures, records backend/OMP/MPI/rank-0 ownership evidence, and compares complete H5 trajectory/observable/restart behavior with periodic-aware deterministic and statistical position checks. Four GPU cases are executable but lack real-device evidence, so promotion remains shadow-only for PR 7c. |
| PR 7c: GPU evidence and final promotion | Pending | | | |
| PR 8: VDS chunk-boundary behavior closure | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; four real CPU VDS chunk-boundary A/B cases; Ruff; `git diff --check` | 76 contract/manifest/comparator tests pass. With chunk size 4, same-semantic deterministic NVE branches produce and compare 3/4/5/9 frames with 1/1/2/3 shards. Frame-count and shard-count mutations are rejected. The cases use `dt=0.0001` to isolate chunk/finalize behavior without relaxing deterministic numeric tolerances. |
| PR 9: Focused non-zero EDIP input behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_edip_nonzero` CPU A/B; Ruff; `git diff --check` | 79 contract/manifest/comparator tests and the focused real A/B pass. The pure bundled branch removes the sidecar table and directory, materializes `/manybody/edip`, and matches legacy at non-zero `EDIP=52.98159` and maximum force `400.30548`. The independently exposed H5 zeroth-frame/force-payload schedule gap remains outside this input-only contract. |
| PR 10: H5 sidecar-table failure semantics | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; three real bundled CPU failure cases; Ruff; `git diff --check` | 79 contract/manifest/comparator tests and all three F1 process cases pass. Unsupported key, key/path length mismatch, and same-key different-path conflict each exit with code 238 and `spongeErrorValueErrorCommand`, while retaining route-specific stable diagnostic tokens. |
| PR 11: Restart owner-state failure semantics | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; three real bundled CPU failure cases; Ruff; `git diff --check` | 79 contract/manifest/comparator tests and all three F1 process cases pass. Dynamic, protocol, and full policies each exit with code 235 and `spongeErrorConflictingCommand`; diagnostics distinguish missing Nose-Hoover-chain and metadynamics owners. |
| PR 12: Focused non-zero custom-pair input gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_custom_pair_nonzero` CPU A/B; Ruff; `git diff --check` | 80 contract/manifest/comparator tests and the focused real A/B pass. The pure bundled branch has no legacy custom-force keys or sidecars, materializes both native inputs, and matches legacy at `custom_pair=31.57` with maximum force `252.554443359375`. |
| PR 13: Focused native exclusion semantics gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_exclusions_coulomb_oracle` CPU A/B; Ruff; `git diff --check` | 83 contract/manifest/comparator tests and the focused real A/B pass. The pure bundled branch uses offsets/list `[0,1,1,1]`/`[1]`, and both branches match the `-1/12` energy and analytic nine-value force oracle with maximum force `0.1111111119389534`. |
| PR 14: Focused native LJ soft-core behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_topology_native_h5_reader|test_h5_input_fixture_equivalence'`; real `normal_lj_soft_core_nonzero` CPU A/B; Ruff; `git diff --check` | 85 contract/manifest/comparator tests, both H5 input contract tests, and the focused real A/B pass. The gate first exposed legacy `potential=-0.06` versus bundled `-0.01`; pure native runtime normalization now matches legacy at `LJ_soft=-0.06`, `eff_pot=-0.056708537`, maximum force `0.1920633763074875`, and zero full-mdout error. Subsystem division remains deferred because its mask is not consumed by the force kernel. |
| PR 15: Focused native virtual-atom behavior gates | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_topology_native_h5_reader|test_h5_input_fixture_equivalence'`; real `normal_virtual_atoms_all_types` and `normal_virtual_atoms_pbc_boundary` CPU A/B; Ruff; clang-format; `git diff --check` | 88 contract/manifest/comparator tests, both H5 input contract tests, and both focused real A/B cases pass. The all-types gate covers exact type 0/1/2/3 ragged payloads, layered type 2-to-3 reconstruction, 24-value coordinate oracles, non-zero PM, and complete force parity; the PBC gate distinguishes `9.75` from the incorrect non-periodic `7.25`. It exposed and fixed the type 3 host/device copy source and the missing global first-frame coordinate refresh. Registry coverage advances to 52 supported, 22 deferred, and 1 unsupported contract. |
| PR 16: Virtual-atom plural alias behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_topology_native_h5_reader|test_h5_input_fixture_equivalence'`; real `normal_virtual_atoms_plural_alias` CPU A/B; Ruff; clang-format; `git diff --check` | 89 contract/manifest/comparator tests, both H5 input contract tests, and the focused real A/B pass. The legacy branch resolves only `virtual_atoms_in_file`; the bundled branch contains the exact native payload with neither key nor sidecars. Both paths match the boundary-crossing coordinate `9.75`, non-zero PM, complete force, and mdout behavior. Registry coverage advances to 53 supported, 21 deferred, and 1 unsupported contract. |
| PR 17: Restart-absent same-bootstrap rerun gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_(restart_h5_reader\|h5_input_validation\|h5_restart_load_runtime_closure)$'`; real `rerun_restart_absent_same_bootstrap_vds_off` CPU A/B; Ruff; `git diff --check` | 90 contract/manifest/comparator tests, two related CTest passes (runtime closure skipped by its build guard), and the real two-frame A/B pass. Both branches use byte-identical coordinate/velocity bootstrap files, the bundled branch has no restart binding or payload, and all mdout columns, frame selection, trajectory, and observable semantics pass E3 comparison. Registry coverage advances to 54 supported, 20 deferred, and 1 unsupported contract. |
| PR 18: NHC dynamic restart continuation gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_(restart_h5_reader\|h5_restart_load_runtime_closure\|module_h5_mappings_with_mock_backend)$'`; real `normal_nhc_dynamic_restart_continuation` CPU A/B; Ruff; `git diff --check` | 91 contract/manifest/comparator tests, two related CTest passes (runtime closure skipped by its build guard), and the real E4 A/B pass. One producer emits non-zero three-chain state through legacy and H5 restart routes, then a text-restart continuation and `dynamic` H5 continuation compare all mdout columns, particle position/velocity/force/box, NHC trajectories, observable output, and final restart state. Registry coverage advances to 57 supported, 17 deferred, and 1 unsupported contract. |
| PR 19: Protocol/full metadynamics continuation gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_(restart_h5_reader\|h5_restart_load_runtime_closure\|module_h5_mappings_with_mock_backend)$'`; real `normal_meta_protocol_full_restart_continuation` CPU A/B; Ruff; `git diff --check` | 92 contract/manifest/comparator tests, two related CTest passes (runtime closure skipped by its build guard), and the real E4 A/B pass. One producer checkpoint with three non-zero hills forks into pure legacy, H5 `protocol` plus legacy NHC, and H5 `full` routes. All mdout columns, particle/NHC/metadynamics streams, final hill/potential state, observable output, and restart state agree within existing text-precision tolerances. Registry coverage advances to 60 supported, 14 deferred, and 1 unsupported contract. |
| PR 20: Unrestricted QC observable and SCF text gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `rerun_qc_unrestricted_sidecar_vds_off/on` CPU A/B; Ruff; `git diff --check` | Triplet unrestricted QC produces two finite non-zero spin-square frames in both branches and archives a non-empty exact SCF trace in trajectory and observable output under both VDS modes. Empty-output evidence was removed from the old full-contract cases. `input.qc.type` remains deferred because pure `/qc/type` is not consumed. Registry coverage advances to 62 supported, 12 deferred, and 1 unsupported contract. |
| PR 21: Constraint sidecar projection gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_constraint_sidecar_projection` CPU A/B; Ruff; `git diff --check` | 97 contract/manifest/comparator tests and the real four-frame A/B pass. Both branches preserve distance `1.5` with zero measured residual and reduce relative radial speed from `2.0` to `3.411315e-6`; complete position and velocity trajectories are identical. The supported sidecar contract is separated from deferred native typed `/constraint` consumption. Registry coverage advances to 63 supported, 12 deferred, and 1 unsupported contract. |
| PR 22: Native improper runtime gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; `ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R 'test_topology_native_h5_reader\|test_h5_input_fixture_equivalence'`; real `normal_improper_native_nonzero` CPU A/B; Ruff; `git diff --check` | 99 contract/manifest/comparator tests, both related H5 input CTests, and the focused real A/B pass. Pure typed `/forcefield/improper/{atoms,pk,phi0}` with no sidecars matches canonical legacy input at `improper_dihedral=31.36`, maximum force `35.415924072265625`, and zero full-mdout error. End-to-end conversion remains deferred for the documented `/k` versus `/pk` and sidecar-key gaps. Registry coverage advances to 64 supported, 12 deferred, and 1 unsupported contract. |
| PR 23: GB native-state/sidecar-activation gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_gb_hybrid_nonzero` CPU A/B; Ruff; `git diff --check` | 101 contract/manifest/comparator tests and the focused real A/B pass. Typed GB state plus the isolated H5 activation binding matches legacy at `gb=-0.25`, total potential `-0.75`, and maximum force `0.10313020646572113`; the gate rejects the Coulomb-only `0.25` force. Pure native GB without the activation sidecar remains deferred. Registry coverage advances to 65 supported, 12 deferred, and 1 unsupported contract. |
| PR 24: SW sidecar pair/three-body behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_sw_sidecar_pair_three_body` CPU A/B; Ruff; `git diff --check` | The isolated H5 `SW_in_file` sidecar route matches legacy at non-zero `SW=194.50` and the complete nine-value force payload. A `lambda=0` control changes the energy to `158.79` and the force fingerprint, so the gate proves both pair and three-body behavior rather than initialization alone. Typed `/manybody/sw` consumption remains deferred. Registry coverage advances to 66 supported, 12 deferred, and 1 unsupported contract. |
| PR 25: Tersoff sidecar angular behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_tersoff_sidecar_angular` CPU A/B; Ruff; `git diff --check` | The isolated H5 `TERSOFF_in_file` sidecar route matches legacy at `potential=-173.23`, `eff_pot=-173.23468`, and the complete nine-value force payload. A `gamma=0` control changes the potential to `-196.06` and the force fingerprint, proving angular bond-order behavior rather than initialization alone. Typed `/manybody/tersoff` consumption remains deferred. Registry coverage advances to 67 supported, 12 deferred, and 1 unsupported contract. |
| PR 26: H5 topology metadata runtime-rejection gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; three real bundled CPU F1 cases; Ruff; `git diff --check` | Production process gates reject topology atom-count mismatch with exit 238/`spongeErrorValueErrorCommand`, and reject non-vector or string `/atoms/mass` payloads with exit 234/`spongeErrorBadFileFormat`, retaining route-specific stable diagnostic tokens. An incompatible `/schema/version` value is currently accepted, so the complete four-mutation metadata contract remains deferred. Registry coverage advances to 68 supported, 12 deferred, and 1 unsupported contract. |
| PR 27: Steering CV protocol-sidecar behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_steering_cv_sidecar_nonzero` CPU A/B; Ruff; `git diff --check` | An isolated distance CV at `1.5` with weight `2.0` produces `steer_cv=3.0` and the analytic six-value force `(2,0,0,-2,0,0)` through both legacy `cv_in_file` and the H5 protocol sidecar route. The gate removes typed `/cv`, rejects zero-weight energy and force mutations, and keeps the unconsumed `steer_cv_in_file`/typed `/steering` contract deferred. Registry coverage advances to 69 supported, 12 deferred, and 1 unsupported contract. |
| PR 28: SITS typed Nk restart behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_sits_nk_typed_restart_nonzero` CPU A/B; Ruff; `git diff --check` | A two-temperature selective LJ/Coulomb SITS fixture consumes legacy `SITS_nk_in_file` with `Nk=(1,4)` versus typed `/parameters/restart/bias/sits/SITS/nk` under protocol restart load. Both routes produce `SITS_AA_kAB=-1.22`, `SITS_bias=-0.5317`, `SITS_fb=0.7049`, `eff_pot=-1.2829471`, and the same non-zero six-value force with maximum magnitude `0.2489061951637268`; the gate removes embedded/external bundled Nk text and rejects initialization-only output, unscaled force, and the symmetric `Nk=(1,1)` control. Typed `/sits` configuration materialization remains deferred. Registry coverage advances to 70 supported, 12 deferred, and 1 unsupported contract. |
| PR 29: VDS complete-prefix repair behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_vds_complete_prefix_noop` CPU A/B; `SPONGE_H5_ENABLE_RUNTIME_SMOKE=1 ctest --test-dir build-dev-cpu-h5-2 --output-on-failure -R '^test_h5_vds_terminal_resume_smoke$'`; Ruff; `git diff --check` | Legacy-input and bundled-input production routes each retain five deterministic frames in two complete shards with `repair_status=not_applied` and zero repaired shards. The same gate runs a real HighFive helper that injects terminal-shard finalize failure and verifies one-shard complete-prefix repair with rewritten completion metadata. Five metadata mutations are rejected. Cross-process reopen/append remains unsupported. Registry coverage advances to 71 supported, 11 deferred, and 1 unsupported contract. |
| PR 30: Residue topology-sidecar runtime behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_residue_sidecar_pbc_mapping` CPU A/B; Ruff; `git diff --check` | An isolated H5 `residue_in_file` topology-sidecar route matches the legacy `[2,2]` residue partition through runtime domain state, non-zero `bond=2.0`, the complete 12-value force payload with maximum magnitude `4.0`, and the cross-PBC whole-molecule position fingerprint `(19,21,25,28)`. Partition and mapping mutations are rejected. Typed `/atoms/residue_index` remains deferred because it is not materialized into `Xponge::system.residues`. Registry coverage advances to 72 supported, 11 deferred, and 1 unsupported contract. |
| PR 31: Residue membership/COM-virial follow-up gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (119 tests); 79-test production manifest; real `normal_residue_sidecar_pbc_mapping` and `normal_residue_sidecar_com_res_virial` CPU A/B; SPONGE rebuild; Ruff; clang-format; `git diff --check` | PR 30's residue-count, runtime-partition, force, and whole-molecule PBC mapping gate remains active. The second isolated `[2,2]` residue-sidecar case drives `com_res` restraint virial behavior with equivalent `bond=2.0`, `restrain=2.0`, `pressure=0.04`, `Pxx=0.11`, and 12-value force output. A same-count `[1,3]` control preserves energy and force but changes pressure/Pxx to `-11.53/-34.60`, proving membership-sensitive consumption. CPU atom-to-group maps now own their storage instead of aliasing freed host buffers. Typed residue input remains deferred. Registry coverage stays at 72 supported, 11 deferred, and 1 unsupported contract. |
| PR 32: Pure native GB behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (120 tests); 80-test production manifest; real `normal_gb_native_nonzero` and `normal_gb_hybrid_nonzero` CPU A/B; two native topology reader CTests; SPONGE rebuild; Ruff; changed-line clang-format; `git diff --check` | The pure bundled branch consumes `/forcefield/gb/params` with no `gb_in_file`, sidecar table, or sidecar files. Both routes match `Coulomb=-0.50`, `gb=-0.25`, `potential=-0.75`, the complete six-value force oracle, and all mdout columns. Non-periodic GB initialization now recognizes preloaded native state while preserving the hybrid activation route. Registry coverage advances to 73 supported, 10 deferred, and 1 unsupported contract. |
| PR 33: Core topology payload-sensitivity gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (128 tests); 88-test production manifest; real `normal_core_topology_payload_sensitivity` CPU A/B plus three typed-payload controls; two native topology reader CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Pure typed mass, charge, and LJ routes contain no topology sidecars and match legacy across the complete one-frame mdout and force payload. Independent controls produce temperature `26.21 -> 52.42` with unchanged force, Coulomb `-0.50 -> 0` with maximum force delta `0.25`, and LJ `-0.02 -> 0` with maximum force delta `0.04541015625`, proving each payload reaches its consumer. The broader `normal_core_h5_output` case remains active and still reports its pre-existing legacy/H5 force schedule mismatch (`51` versus `50` frames), which is not weakened by this input gate. Registry coverage remains 73 supported, 10 deferred, and 1 unsupported contract. |
| PR 34: Complete H5 metadata failure semantics | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (130 tests); 90-test production manifest; four real `failure_h5_topology_*` CPU process gates; three supported-schema real controls; three metadata/input CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Atom-count, mass-shape, mass-dtype, and unknown-schema mutations now share one complete F1 contract. Schema versions `0`, `1`, and `xponge.legacy_to_bundle.v1` each start successfully, while `unsupported.topology.v999` exits 238 with `spongeErrorValueErrorCommand` and route-specific diagnostics. The partial metadata registry record is removed; coverage remains 73 supported, advances to 9 deferred, and retains 1 unsupported contract. |
| PR 35: Subsystem-division energy-partition behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (130 tests); 90-test production manifest; real `normal_lj_soft_core_nonzero` and `normal_subsystem_division_partition` CPU A/B plus all-intra legacy/bundled controls; two native topology reader CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | The legacy `[0,1]` sidecar and pure typed bundled dataset both produce `LJ_soft_inter=-0.06`, `LJ_soft_intra=0.00`, and the same complete force. Same-semantic `[0,0]` controls move the full `-0.06` to the intra observable while preserving total soft-core energy and force, proving mask consumption without changing dynamics. Registry coverage advances to 74 supported, 8 deferred, and 1 unsupported contract. |
| PR 36: Typed residue runtime-behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (131 tests); 91-test production manifest; real typed PBC and COM-virial CPU A/B plus both sidecar residue regressions; two native topology reader CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Pure typed `/atoms/residue_index=[0,0,1,1]` and `/residues/atom_offset=[0,2,4]` materialize into the native `[2,2]` runtime partition without a topology sidecar. They match legacy at `bond=2.0`, the whole-molecule PBC fingerprint `(19,21,25,28)`, `restrain=2.0`, `pressure=0.04`, and `Pxx=0.11`. A typed `[1,3]` control preserves the complete force and energy while changing pressure/Pxx to `-11.53/-34.60`; inconsistent typed datasets and legacy/typed ownership conflicts are rejected. Registry coverage advances to 75 supported, 7 deferred, and 1 unsupported contract. |
| PR 37: Typed SW pair/three-body behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real `normal_sw_typed_pair_three_body` and `normal_sw_sidecar_pair_three_body` CPU A/B cases; related native topology CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Pure typed `/manybody/sw` with no SW command or sidecars matches the legacy pair/three-body route at `SW=194.50` and maximum force `404.27862548828125`. A typed lambda-zero control changes energy to `158.79` and maximum force to `343.52105712890625`, proving that the typed three-body payload reaches the force kernel. Registry coverage advances to 76 supported, 6 deferred, and 1 unsupported contract. |
| PR 38: Typed constraint projection behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (133 tests); 93-test production manifest; real `normal_constraint_typed_projection` and `normal_constraint_sidecar_projection` CPU A/B cases; input-validation and fixture-equivalence CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Pure typed pair `[0,1]` with `r0=1.5` and no constraint sidecar matches the legacy route across four complete position/velocity frames with zero distance residual and radial velocity residual `3.411315e-6`. A typed `r0=2.0` control follows the new target with residual `0`; out-of-range pair `[0,2]` is rejected as `spongeErrorBadFileFormat`. Sidecar precedence remains unchanged. Registry coverage advances to 77 supported, 5 deferred, and 1 unsupported contract. |
| PR 39: Typed Tersoff angular behavior gate | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract` (134 tests); 94-test production manifest; real `normal_tersoff_typed_angular` and `normal_tersoff_sidecar_angular` CPU A/B cases plus two typed failure controls; two native input CTests; SPONGE rebuild; Ruff; clang-format; `git diff --check` | Pure typed `/manybody/tersoff` with no Tersoff command or sidecars matches legacy at `potential=-173.23`, `eff_pot=-173.23468`, and maximum force `135.94906616210938`. A typed gamma-zero control produces `potential=-196.06`, `eff_pot=-196.05984`, and maximum force `144.7831268310547`, proving angular parameter consumption. Conflicting map/entry and raw/runtime parameter payloads are rejected as bad-file-format. Registry coverage advances to 78 supported, 4 deferred, and 1 unsupported contract. |
