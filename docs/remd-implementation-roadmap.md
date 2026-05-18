# REMD Implementation Roadmap

## Goal

Build a generic external scheduling framework for SPONGE first, then implement
replica-exchange methods on top of it without coupling the exchange logic into
the current SPONGE MPI/controller layer.

The final target is:

- one reusable single-replica worker runtime
- one generic multi-worker manager
- pluggable exchange policies for:
  - T-REMD
  - H-REMD
  - HT-REMD

## Design summary

### Worker side

The SPONGE executable remains a normal single-replica worker.

Worker-facing responsibilities:

- initialize one SPONGE runtime
- run one step or one block
- expose lightweight runtime snapshots
- export/import runtime state
- apply post-swap maintenance
- evaluate exchange observables

The current `sponge::SpongeScheduler` is the worker lifecycle wrapper.

### Manager side

`SPONGE_MANAGER` is the high-level orchestrator.

Manager responsibilities:

- define logical schedules
- launch and supervise workers
- assign block execution plans
- collect exchange observables
- compute acceptance according to policy
- swap complete runtime states between schedules
- write exchange logs and recovery metadata

### Exchange semantics

Schedules are fixed thermodynamic slots.

- `schedule i` always corresponds to one target state
- accepted exchanges swap runtime states between schedules
- schedule-local trajectories remain naturally associated with thermodynamic
  states

This is the intended GROMACS-like semantic model.

## Layering

### 1. `scheduler`

Single-worker runtime control API.

Current:

- initialize
- run one step
- run N steps
- run to end
- snapshot
- finalize

To add:

- export runtime state
- import runtime state
- scale velocities
- invalidate neighbor list / force rebuild
- collect exchange observables
- evaluate foreign-state energy

### 2. `manager core`

Generic orchestration model independent of REMD formulas.

Core data model:

- `ScheduleInputs`
- `RuntimeStateRef`
- `ExchangeObservable`
- `WorkerConfig`
- `ScheduleConfig`
- `ScheduleRecord`
- `Manager`

To add:

- `WorkerHandle`
- `BlockExecutionPlan`
- `BlockExecutionResult`
- `ExchangePair`
- `ExchangeAttempt`
- `ExchangeResult`

### 3. `worker protocol`

Boundary between manager and worker.

Recommended evolution:

- phase A: in-process manager prototype
- phase B: local child-process workers + files
- phase C: long-lived workers + pipe/socket protocol

### 4. `exchange policies`

Algorithms built on manager core:

- `TemperatureReplicaExchangePolicy`
- `HamiltonianReplicaExchangePolicy`
- `TemperatureHamiltonianReplicaExchangePolicy`

## FEP test relevance

The test directory:

- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/step2_mdin.txt`
- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/0`
- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/1`
- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/2`
- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/3`

looks like a natural Hamiltonian/FEP ladder:

- one shared production-style `mdin`
- multiple state directories
- state-dependent charge/LJ-soft-core data

This makes it a strong validation target for:

- worker launch planning
- schedule layout
- runtime-state swap semantics
- later H-REMD and HT-REMD acceptance logic

## Recommended implementation order

### Phase 0: stabilize current foundations

Scope:

- keep `SPONGE` directly runnable
- keep `SPONGE_MANAGER` independently buildable
- preserve current scheduler wrapper behavior

Exit criteria:

- `SPONGE` still runs ordinary jobs unchanged
- `SPONGE_MANAGER` can print a valid manager plan

### Phase 1: finish scheduler as a worker API

Scope:

- formalize scheduler-facing worker API
- define the minimum runtime-state schema

Add to scheduler:

- `ExportRuntimeState()`
- `ImportRuntimeState(...)`
- `CollectExchangeObservables()`
- `ScaleVelocities(factor)`
- `InvalidateNeighborList()`

Recommended runtime-state fields:

- coordinates
- velocities
- box
- step
- time
- thermostat hidden state
- barostat hidden state
- restart metadata

Exit criteria:

- one worker can run a block, export state, import the same state, and continue
- resumed trajectory matches uninterrupted execution for the tested window

Validation:

- use one FEP state directory as a single-worker regression case

### Phase 2: complete manager core

Scope:

- move from plan-printing skeleton to executable orchestration core

Add:

- `WorkerHandle`
- `BlockExecutionPlan`
- `BlockExecutionResult`
- schedule state bookkeeping
- block loop skeleton

Manager should support:

- building schedule records from config
- assigning workers to schedules
- driving one block on one or more workers
- storing runtime-state references

Exit criteria:

- `SPONGE_MANAGER` can drive one schedule for one block and collect a result
- `SPONGE_MANAGER` can drive multiple schedules
- child-process schedules can be dispatched in parallel and gathered by the
  manager

Validation:

- use FEP directories `0/1/2/3` as four schedules without any exchanges
- verify each schedule can produce its own block summary

### Phase 3: add local worker protocol

Scope:

- make manager capable of launching real worker processes

Recommended first transport:

- child-process launch
- file-based command/result exchange

Why:

- easiest to debug
- robust enough for early REMD validation
- decouples logical replicas from hardware process count

Exit criteria:

- manager can launch a worker against one FEP directory
- manager can run a block and recover `Snapshot + RuntimeStateRef +
  ExchangeObservable`

### Phase 4: implement T-REMD on top of manager core

Scope:

- first full exchange algorithm

Add:

- odd/even adjacent pairing
- temperature ladder definition
- acceptance calculation using potential energy
- runtime-state swap
- post-swap velocity scaling
- neighbor-list invalidation

Exit criteria:

- temperature exchange attempts are logged
- accepted swaps move runtime state between schedules correctly

Validation:

- use a simpler temperature-only test first
- do not force the FEP sample into this phase unless needed

### Phase 5: implement H-REMD worker support

Scope:

- expose foreign-state energy evaluation required by Hamiltonian exchange

Recommended first implementation path:

- start from existing FEP/soft-core machinery
- support neighboring-state foreign evaluation first
- avoid full all-to-all state matrix in the first iteration

Likely additions:

- `EvaluateForeignStateEnergy(target_state)`
- state-local Hamiltonian descriptors derived from FEP input files

Exit criteria:

- one worker running state `i` can report energy for state `i`
- the same worker can evaluate energy for adjacent state `j`

Validation:

- use the FEP sample as the primary phase-5 regression target

### Phase 6: implement H-REMD policy

Scope:

- add Hamiltonian exchange policy on top of manager core

Add:

- neighboring schedule pairing
- Hamiltonian acceptance calculation
- runtime-state swap without velocity scaling

Exit criteria:

- manager can run H-REMD across the FEP ladder
- exchange log includes local/foreign energy terms and acceptance decisions

Validation:

- primary validation on `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD`

### Phase 7: implement HT-REMD

Scope:

- combine temperature and Hamiltonian exchange

Add:

- combined `beta + foreign Hamiltonian` reduced-potential calculation
- velocity scaling after accepted swaps

Exit criteria:

- same manager framework can switch among T/H/HT policies

## Immediate coding backlog

1. Define the concrete `RuntimeState` structure and serialization format.
2. Extend `SpongeScheduler` with explicit export/import hooks.
3. Add `CollectExchangeObservables()` for one worker.
4. Turn `Manager` from a descriptive skeleton into a block executor.
5. Design a `manager.toml`-style configuration schema for schedules and workers.
6. Add a file-based worker protocol for `SPONGE_MANAGER`.
7. Use the FEP sample as the first multi-schedule no-exchange manager test.
8. After manager execution is stable, add H-REMD support before HT-REMD.

## Practical recommendation

Use two validation tracks in parallel:

- a minimal temperature-only system for fast T-REMD debugging
- the provided FEP sample for H-REMD and later HT-REMD

This keeps the generic framework honest while avoiding overfitting the manager
design to one exchange mode.

## Current implementation status

The repository now contains a working first-pass external scheduling prototype:

- `SpongeScheduler` exposes runtime-state export/import, observable collection,
  velocity scaling, and neighbor-list invalidation.
- the exported `RuntimeState` now carries the main thermostat/barostat hidden
  state used by current REMD smoke paths, including Nose-Hoover chain data,
  pressure-based barostat RNG state, Bussi/Andersen/Langevin RNG state, and
  MC-barostat adaptive proposal counters plus its serialized RNG engine.
- manager execution now goes through a `worker_protocol` abstraction with an
  in-process backend, so future child-process or long-lived worker backends can
  replace it without changing the REMD policy layer.
- the current `worker_protocol` already includes a file-based child-process
  path that reuses the existing `SPONGE` executable in a hidden worker mode,
  while preserving normal direct-run behavior for ordinary users.
- `persistent = true` child-process workers now use the TCP loopback protocol
  instead of temporary request/response files, reusing the same serialized
  runtime-state payloads as the file protocol.
- `SPONGE_MANAGER` can drive multiple schedules block-by-block against the FEP
  sample in `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD`.
- `SPONGE_MANAGER` now supports both the current FEP shortcut CLI and an
  explicit `--config manager.toml` path for declaring schedules/workers
  directly.
- manager-side runtime-state swapping preserves `walker_id`, so trajectory
  identity follows the swapped runtime state instead of being reset to the
  schedule slot.
- T-REMD, H-REMD, and HT-REMD each have an initial manager-side policy with:
  - odd/even pairing
  - Metropolis acceptance
  - complete runtime-state swap between schedules
  - temperature-dependent velocity scaling where required
- H-REMD and HT-REMD currently obtain cross-Hamiltonian energies by launching a
  temporary probe worker under the target schedule configuration and importing
  the source runtime state.

Validated smoke paths:

- multi-schedule no-exchange execution on the FEP sample
- T-REMD epoch execution with runtime-state swap
- H-REMD epoch execution with FEP cross-state probes
- HT-REMD epoch execution with combined temperature/hamiltonian acceptance
- child-process worker execution for T-REMD, H-REMD, and HT-REMD
- bundled smoke regression script for the current FEP-based manager/REMD paths
- bundled fail-closed probe guard smoke for sink metadynamics and SITS cases
- bundled scheduler runtime-state roundtrip smoke against the FEP state-0
  worker path

Known current limitations:

- `RuntimeState` coverage is much better than the initial prototype, but it is
  still not exhaustive; remaining control-module edge cases, especially outside
  the currently validated thermostat/barostat set, still need auditing.
- the scheduler runtime-state roundtrip path is now validated by a tolerance-
  based worker smoke on the FEP sample, which is strong enough to catch major
  import/export regressions but not yet a proof of bitwise restart identity.
- exchange observable probing reuses the current force-evaluation path, which
  is appropriate for the present FEP-style validation but still needs hardening
  for more stateful bias/enhanced-sampling modules. The current prototype now
  fails closed for foreign-state probes when sink metadynamics or SITS is
  enabled, because their history-dependent state is not serialized yet.
- H-REMD/HT-REMD cross-state evaluation currently prioritizes correctness and
  low coupling over performance.
- the TCP worker path currently removes file transport but does not yet keep
  worker sessions alive across multiple manager requests; long-lived session
  reuse is the next protocol performance step.
