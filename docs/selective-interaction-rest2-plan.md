# Selective Interaction and REST2 Refactor Plan

## Goal

Refactor the current SITS-specific selective interaction path into a shared
`Selective_Interaction` module, then implement REST2 as another selective
interaction method and expose it as a REMD exchange mode.

The target design should:

- keep existing user-facing `mdin` syntax compatible
- keep existing SITS behavior and output names unchanged
- avoid adding more SITS/REST2-specific branches directly in `main.cpp`
- share atom selection, local atom mapping, and selective pair-interaction
  kernels between SITS and REST2 where possible
- implement REST2 as a static, probe-safe Hamiltonian scaling method
- add a `rest2` replica-exchange mode on top of the existing H-REMD
  cross-evaluation framework

## Design Summary

### User-facing input

Existing SITS input remains valid:

```toml
[SITS]
mode = "production"
atom_in_file = "hot_atoms.txt"
```

REST2 should use its own input namespace:

```toml
[REST2]
mode = "on"
atom_in_file = "hot_atoms.txt"
lambda_m = 0.80
```

The same legacy flat-command style should remain supported if the existing
input parser maps it to the same controller commands:

```text
REST2_mode = on
REST2_atom_in_file = hot_atoms.txt
REST2_lambda_m = 0.80
```

`lambda_m` is the REST2 Hamiltonian scaling factor, usually interpreted as
`T0 / Tm`. The reference replica uses `lambda_m = 1.0`; hotter effective solute
replicas use `lambda_m < 1.0`.

### Source layout

Create a new module folder:

```text
SPONGE/Selective_Interaction/
  Selective_Interaction.h
  Selective_Interaction.cpp
  SITS.h
  SITS.cpp
  REST2.h
  REST2.cpp
```

The current `SPONGE/SITS/SITS.h` and `SPONGE/SITS/SITS.cpp` should move under
this folder with minimal behavioral changes first. Compatibility wrapper headers
may be kept temporarily if needed, but new code should include the
`Selective_Interaction` path.

### Runtime globals

Replace the direct global SITS dependency in `main.cpp`:

```cpp
SITS_INFORMATION sits;
```

with a single facade:

```cpp
SELECTIVE_INTERACTION selective_interaction;
```

The facade owns:

```cpp
SITS_INFORMATION sits;
REST2_INFORMATION rest2;
```

Only one selective interaction method should be active at a time in the first
implementation. If both SITS and REST2 are requested, initialization should
throw a clear configuration error.

## Selective Interaction Interface

`SELECTIVE_INTERACTION` should provide one stable hook surface for `main.cpp`:

```cpp
struct SELECTIVE_INTERACTION
{
    SITS_INFORMATION sits;
    REST2_INFORMATION rest2;
    int is_initialized = 0;

    void Initial(CONTROLLER* controller, int atom_numbers);
    void Get_Local(int* atom_local, int local_atom_numbers, int ghost_numbers);

    void Reset_Force_Energy(int* md_need_potential);

    void LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(...);
    void LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(...);

    void Update_And_Enhance(...);
    void Step_Print(CONTROLLER* controller, float beta0);

    bool Is_Probe_Safe() const;
};
```

The facade should dispatch internally:

```cpp
if (rest2.is_initialized)
{
    rest2.Step_Print(controller);
}
else if (sits.is_initialized)
{
    sits.Step_Print(controller, beta0);
}
```

`main.cpp` should call only `selective_interaction.*`, not `sits.*` or
`rest2.*` directly.

## SITS Migration

### Scope

The first SITS step is a structural migration, not an algorithm rewrite.

Move or wrap:

- `SITS_INFORMATION`
- `CLASSIC_SITS_INFORMATION`
- `SELECT`
- SITS selective LJ/direct Coulomb kernels
- SITS output registration and printing
- SITS local atom mapping

Preserve:

- all `SITS_*` / `[SITS]` user input names
- all SITS mdout column names
- SITS modes: observation, iteration, production, empirical, AMD, GaMD
- existing SITS bias/history behavior

### Probe safety

SITS should remain marked probe-unsafe until its complete history state is
serializable in `RuntimeState`.

`SELECTIVE_INTERACTION::Is_Probe_Safe()` should return:

- `false` for SITS
- `true` for REST2
- `true` when no selective interaction module is active

This keeps existing foreign-state H-REMD probes from silently evaluating SITS
states incorrectly.

## REST2 Runtime Design

### REST2 module state

`REST2_INFORMATION` should contain:

- `is_initialized`
- `atom_numbers`
- `atom_sys_mark`
- `atom_sys_mark_local`
- `local_atom_numbers`
- `ghost_numbers`
- `lambda_m`
- `sqrt_lambda_m`
- device buffers for unscaled selected energy
- device buffers for effective selected energy
- device buffers for bias energy
- optional selected force/virial buffers if useful for debugging and
  verification

Use the same atom-selection conventions as SITS:

- `atom_in_file`: explicit hot-region atom ids
- `atom_numbers`: leading hot-region atom count, or `"ALL"`

For REST2, `"ALL"` is mathematically allowed but not very useful; it should be
accepted only if the resulting scaling semantics are documented clearly.

### Pair scaling

For atom pair classes:

```text
hot-hot:   scale = lambda_m
hot-cold:  scale = sqrt(lambda_m)
cold-cold: scale = 1.0
```

Apply the scale to:

- LJ direct force
- LJ direct energy
- direct short-range Coulomb force
- direct short-range Coulomb energy
- corresponding virial contribution when pressure is requested

The first implementation is explicitly a short-range direct REST2
implementation. Full reciprocal PME Coulomb scaling can be added later.

### Energies printed by REST2

REST2 should register and print its own mdout columns:

```text
REST2_lambda_m
REST2_unscaled
REST2_effective
REST2_bias
```

Definitions:

- `REST2_unscaled`: hot-related pair energy before REST2 scaling
- `REST2_effective`: hot-related pair energy after REST2 scaling
- `REST2_bias = REST2_effective - REST2_unscaled`
- `REST2_lambda_m`: current replica Hamiltonian scaling factor

For `lambda_m = 1.0`, `REST2_bias` should be approximately zero.

SITS output names should not change.

## Force Path Integration

The current SITS branch in `Main_Calculate_Force()` should become a single
selective interaction hook.

Current shape:

```cpp
if (sits.is_initialized && sits.selectively_applied)
{
    sits.SITS_LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(...);
    sits.SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(...);
}
else
{
    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(...);
    lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(...);
}
```

Target shape:

```cpp
if (selective_interaction.is_initialized)
{
    selective_interaction.LJ_Direct_CF_Force_With_Atom_Energy_And_Virial(...);
    selective_interaction.LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(...);
}
else
{
    lj.LJ_PME_Direct_Force_With_Atom_Energy_And_Virial(...);
    lj_soft.LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(...);
}
```

`Reset_Force_Energy`, `Get_Local`, `Update_And_Enhance`, and `Step_Print` should
follow the same facade pattern.

## REST2 and REMD Scheduling

### REMD mode

Add a `rest2` exchange mode beside existing modes:

```text
tremd
hremd
htremd
rest2
```

REST2 exchange should be implemented as a Hamiltonian exchange policy with a
REST2-specific parameter key:

```text
REST2_lambda_m
```

The scheduler/worker should still read physics parameters from each worker's
own `mdin` or `schedules.inputs` overrides. The exchange policy should not own
or invent physical parameters.

### Acceptance rule

Use the H-REMD Metropolis form:

```text
log_acc = -beta * [H_i(x_j) + H_j(x_i) - H_i(x_i) - H_j(x_j)]
```

For REST2, all replicas usually run at the same physical thermostat
temperature, so `beta` is normally the physical simulation beta.

Implementation should reuse the existing foreign-state probe mechanism:

- current state under its own REST2 Hamiltonian: `H_i(x_i)`
- right state under left REST2 Hamiltonian: `H_i(x_j)`
- left state under right REST2 Hamiltonian: `H_j(x_i)`
- right state under its own REST2 Hamiltonian: `H_j(x_j)`

REST2 is probe-safe because it has no history-dependent bias state.

### Runtime-state exchange

Accepted REST2 exchanges should keep the current schedule-local trajectory
semantics:

- schedules are Hamiltonian slots
- accepted exchanges swap complete `RuntimeState` objects
- coordinates, velocities, box, thermostat/barostat state, and RNG state move
  together

Velocity scaling is not required for pure REST2 if physical thermostat
temperature is unchanged.

## Testing and Validation Plan

### Phase 1: SITS facade regression

Goal: prove the `Selective_Interaction` facade preserves current SITS behavior.

Use existing SITS benchmark/validation cases:

```text
benchmarks/performance/sits
```

Checks:

- build succeeds
- existing SITS mdin still runs
- mdout columns are unchanged
- energies and trajectories match current behavior within normal floating-point
  tolerance

### Phase 2: REST2 single-replica correctness

Create a small REST2 validation case, preferably from an existing small
protein/water or peptide/water benchmark fixture.

If no suitable fixture exists, add a minimal repo-local fixture with:

- a small solvated system
- one hot atom selection file
- short NVT mdin
- `lambda_m = 1.0`, `0.8`, and `0.6`

Checks:

- `lambda_m = 1.0` gives `REST2_bias ~= 0`
- decreasing `lambda_m` changes `REST2_effective` monotonically for the same
  saved coordinate frame
- forces are finite
- short runs do not crash
- mdout prints `REST2_lambda_m`, `REST2_unscaled`, `REST2_effective`,
  `REST2_bias`

### Phase 3: REST2 probe correctness

Use one saved runtime state and evaluate it under several `lambda_m` values.

Checks:

- own-Hamiltonian energy equals direct single-worker evaluation
- foreign-state probe does not mutate persistent worker state
- `REST2_bias(lambda=1.0) ~= 0`
- cross-energy values are reproducible

### Phase 4: REST2-REMD smoke test

Use a simple ladder:

```text
lambda_m = 1.00, 0.90, 0.80, 0.70
```

Run:

- short blocks first, for functional correctness
- then a longer run for acceptance and performance

Checks:

- exchange log includes `rest2`
- random values advance across epochs
- acceptance probability is in `[0, 1]`
- accepted exchanges swap complete runtime states
- no SITS probe-safety error is triggered for REST2
- output directories remain schedule-local and do not overwrite one another

### Phase 5: Benchmark and overhead

Compare:

- baseline no selective interaction
- REST2 `lambda_m = 1.0`
- REST2 ladder
- existing SITS case after facade migration

Metrics:

- ns/day or aggregate steps/s
- manager/scheduler overhead for REST2-REMD
- REST2 kernel overhead relative to baseline LJ/direct Coulomb
- exchange acceptance and walker diffusion

## Implementation Order

### Step 1: Create `Selective_Interaction` facade

- Add `SPONGE/Selective_Interaction/Selective_Interaction.h`
- Add `SPONGE/Selective_Interaction/Selective_Interaction.cpp`
- Move or wrap current SITS files under `SPONGE/Selective_Interaction/`
- Keep compatibility includes if required
- Update CMake source lists

Exit criteria:

- SPONGE builds
- existing non-SITS jobs behave unchanged

### Step 2: Route SITS through facade

- Replace global `sits` usage in `main.cpp` with `selective_interaction`
- Preserve SITS internals
- Preserve SITS output names
- Preserve SITS input names

Exit criteria:

- existing SITS benchmark passes
- mdout columns are unchanged

### Step 3: Add REST2 runtime module

- Add `REST2_INFORMATION`
- Add REST2 atom selection
- Add REST2 local atom mapping
- Add REST2 selective LJ/direct Coulomb kernel
- Add REST2 energy printing
- Add REST2 probe-safe reporting

Exit criteria:

- `REST2_lambda_m = 1.0` behaves like baseline for selected direct terms
- `REST2_bias` prints and is near zero at `lambda_m = 1.0`

### Step 4: Add worker input override support

- Ensure `schedules.inputs.REST2_lambda_m` can override the worker's mdin value
- Keep mdin as the primary external interface
- Avoid adding REST2-specific physics ownership to scheduler core

Exit criteria:

- one shared mdin can create multiple REST2 workers through schedule input
  overrides
- separate mdin files also work

### Step 5: Add `rest2` exchange policy

- Add `REST2ReplicaExchangePolicy` or a REST2 specialization of H-REMD
- Reuse foreign-state probe infrastructure
- Require `selective_interaction.Is_Probe_Safe()`
- Use physical beta from worker observables or schedule input

Exit criteria:

- short REST2-REMD smoke test runs and writes exchange logs

### Step 6: Add validation and benchmark tests

- Add single-replica REST2 validation
- Add REST2 probe validation
- Add REST2-REMD benchmark
- Document command-line usage

Exit criteria:

- CI/local benchmark command can validate REST2 behavior
- acceptance, energy, and overhead metrics are reported

## Open Questions

- Should the first REST2 version scale only direct short-range Coulomb, or
  should full PME reciprocal scaling be implemented immediately?
- Should REST2 include bonded terms involving hot atoms in the first version?
- Should REST2 scale 1-4 LJ/Coulomb terms in the first version?
- What should be the canonical REST2 test system: existing SITS benchmark,
  existing FEP fixture, or a new small solvated peptide fixture?
- Should `REST2_lambda_m` be allowed to change dynamically at runtime, or only
  at worker initialization / schedule override time?

## Recommended First Milestone

Do not implement REST2 first. First land the facade-only migration:

```text
main.cpp -> selective_interaction -> SITS
```

with no behavior change. This reduces risk and gives REST2 a clean insertion
point. After that, REST2 can be added without further expanding direct
algorithm-specific branches in `main.cpp`.
