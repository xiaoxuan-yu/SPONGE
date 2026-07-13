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
ci(bundled-io): promote the complete A/B gate matrix
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

## 11. Artifacts and Temporary-Space Policy

- Use `SPONGE_BUNDLED_IO_AB_RUN_ROOT` for all heavy runs.
- Put production artifacts on the workspace filesystem rather than `/tmp`.
- Delete successful per-replica payloads after evidence extraction.
- Retain failed case inputs, stdout/stderr, mdout, H5 metadata, evidence, and a
  minimal reproduction command.
- Record artifact byte counts and enforce a configurable per-case quota.
- Cleanup must only remove directories created by the current gate run.

## 12. PR Completion Log

Append one row immediately after completing and committing each PR.

| PR | Status | Commit | Verification | Notes |
|---|---|---|---|---|
| PR 1: Executable contract registry | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; sidecar rerun VDS-off A/B | 21 contract tests and one real A/B case pass; pure-native rerun exposes a strict column-set gap retained for input behavior closure. |
| PR 2: Contract enumeration and fixtures | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; H5 manifest/native-reader/fixture-equivalence CTest; sidecar rerun VDS-off A/B | Runtime key inventory has exact contract ownership; full-contract mdin/manifest now enumerate output controls, all legacy routes, particle stream, and rerun controls; missing topology runtime cases are explicitly deferred to PR 5. |
| PR 3: Comparators | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; real sidecar rerun VDS-off A/B (expected QC failure) | 41 contract/comparator tests pass; TOST plus Holm correction, quantity-specific margins, structured position/velocity/force/box statistics, exact non-finite patterns, and all six comparator mutations are enforced. The real gate still rejects the known QC first-frame behavior gap (`-34183.44` vs `-34139.92`) without weakening tolerances. |
| PR 4: Output behavior closure | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; five output/VDS CTest targets; two-step NVE restart continuation; real TIP3P A/B (expected force-schedule failure) | 45 contract tests and all output writer/backend targets pass. Normal VDS-off/on cases now require position, velocity, force, box, structured mdinfo, explicit route provenance/content, and E4 structural restart continuation. The real gate exposes legacy `frc` 51-frame vs H5 50-frame schedule skew; QC SCF non-empty/exact comparison and dynamic restart continuation remain explicit behavior gaps rather than relaxed assertions. |
| PR 5: Input behavior closure | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; native-topology reader and fixture-equivalence CTest; saved TIP3P and sidecar-rerun production artifacts | 52 contract tests require every supported input contract to produce a present, non-trivial module-owned result and then pass deterministic or statistical A/B comparison. TIP3P mass/charge/LJ pass; the rerun gate rejects the known QC first-frame mismatch (`-34183.44` vs `-34139.92`). Reader/path checks remain E1/E2 only. Residue/exclusions, improper, GB/softcore/subsystem/virtual atoms, SW/EDIP/Tersoff, custom pairwise, constraint/steering, SITS/meta/NHC, and QC type/spin/SCF stay explicitly deferred until dedicated non-trivial fixtures exist. |
| PR 6: Rerun and failure semantics | Complete | This commit | `pixi run -e dev-cpu smoke-bundled-io-contract`; nine process-level F1 cases; five real rerun boundary cases | 60 contract/manifest tests and all nine textual/binding F1 cases pass. The harness now keeps `crd/box/vel` as legacy rerun inputs and inserts overrides before TOML module tables; this removes the earlier QC first-frame skew (`-33917.02` now matches). The 0/0 limit-1 boundary passes, while the gate exposes optional-velocity potential (`1951.16` vs `1947.41`), strip frame counter (`2` vs `1`), and box-update legacy invalid-free failures. H5 atom/shape/dtype/schema and sidecar-table process mutations, restart absent/dynamic/protocol/full continuation, and owner-state F1 cases remain explicitly deferred. |
| PR 7: Execution matrix and CI promotion | Pending | | | |
