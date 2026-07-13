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

## 21. Artifacts and Temporary-Space Policy

- Use `SPONGE_BUNDLED_IO_AB_RUN_ROOT` for all heavy runs.
- Put production artifacts on the workspace filesystem rather than `/tmp`.
- Delete successful per-replica payloads after evidence extraction.
- Retain failed case inputs, stdout/stderr, mdout, H5 metadata, evidence, and a
  minimal reproduction command.
- Record artifact byte counts and enforce a configurable per-case quota.
- Cleanup must only remove directories created by the current gate run.

## 22. PR Completion Log

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
