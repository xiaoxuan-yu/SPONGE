# Manager Architecture Draft

## Goal

Build a general-purpose external scheduler for SPONGE that can orchestrate
multiple single-replica workers. Replica-exchange methods such as T-REMD,
H-REMD, and HT-REMD should be implemented as algorithms on top of this common
manager framework instead of being coupled directly into the SPONGE MPI layer.

## Design principles

- Keep `SPONGE` usable as a normal single-replica executable.
- Reuse one worker lifecycle API for direct CLI runs and manager-driven runs.
- Let the manager own orchestration, exchange policy, logging, and recovery.
- Exchange complete runtime state between schedules, not only thermodynamic
  parameters, so that trajectories remain schedule-local and downstream rerun
  workflows stay natural.
- Separate generic manager abstractions from REMD-specific acceptance rules.

## Roles

### Worker

A worker is one SPONGE runtime instance. It is responsible for:

- initialization from input arguments
- block-wise MD execution
- exporting and importing runtime state
- reporting exchange observables
- applying post-exchange maintenance such as velocity scaling and neighbor-list
  invalidation

The current `sponge::SpongeScheduler` is the first process-local worker wrapper.

### Schedule

A schedule is a fixed logical slot managed by the manager. Its schedule-local
configuration lives in one `inputs` table, which is used both for worker input
overrides and for exchange-policy field discovery.

Examples:

- temperature schedule `T_i`
- Hamiltonian schedule `H_i`
- combined schedule `(T_i, H_i)`

A schedule should stay associated with one logical trajectory/output lane. When
an exchange is accepted, the schedule receives another worker runtime state.

### Runtime state

Runtime state is the complete worker state required to continue integration
correctly after a swap.

The minimal long-term target should include:

- coordinates
- velocities
- box/cell
- current step and time
- thermostat hidden variables
- barostat hidden variables
- any required history terms for continued integration

### Manager

The manager coordinates all schedules. It is responsible for:

- starting workers
- assigning work blocks
- collecting exchange observables
- selecting exchange pairs
- evaluating acceptance criteria
- swapping runtime states between schedules
- emitting exchange logs and statistics

## Layered architecture

### 1. `sponge_runtime`

Library for single-replica execution and runtime-state manipulation.

Key responsibilities:

- worker lifecycle
- runtime snapshot
- runtime-state export/import
- maintenance hooks after swaps

### 2. `sponge_worker_protocol`

Communication layer between manager and worker.

Possible evolution path:

- phase 1: process launch + files
- phase 2: child process + TCP loopback request/response
- phase 3: long-lived child process + session reuse for `RUN_BLOCK`
- phase 4: distributed protocol if needed

Current protocol foundation:

- `worker_protocol/message_protocol.h` defines the persistent-worker message
  header shared by `SPONGE` and `SPONGE_MANAGER`.
- `worker_protocol/tcp_socket.h` provides the first cross-platform loopback TCP
  transport wrapper.
- `worker_protocol/tcp_protocol.h` sends the existing worker request/response
  payloads through length-prefixed TCP messages.
- The header is length-prefixed and contains `magic`, `version`,
  `message_type`, `request_id`, and `payload_size`.
- Message integers are serialized explicitly in little-endian order so the
  eventual TCP backend is not tied to the host byte order.
- The initial message type set covers `HELLO`, `RUN_BLOCK`, `RUN_RESULT`,
  `IMPORT_STATE`, `PROBE_OBSERVABLE`, `PROBE_RESULT`, `GET_STATUS`,
  `SHUTDOWN`, and `ERROR`.
- `SPONGE --worker-tcp host:port` is the hidden worker entrypoint used by
  manager-launched persistent child-process workers.
- `manager.transport = "tcp"` keeps child-process run workers alive across
  blocks. Foreign-state probe requests are still isolated in one-shot workers so
  they do not overwrite a schedule's own runtime state.

### 3. `sponge_manager_core`

Generic orchestration layer without REMD-specific formulas.

Suggested core types:

- `WorkerHandle`
- `ScheduleRecord`
- `ScheduleInputs`
- `RuntimeState`
- `ExchangeObservable`
- `BlockExecutionPlan`
- `BlockExecutionResult`
- `Manager`

### 4. `sponge_manager_algorithms`

Algorithm plugins that use manager core.

Planned first algorithms:

- temperature replica exchange
- Hamiltonian replica exchange
- temperature + Hamiltonian replica exchange

Future candidates:

- REST2
- generalized Gibbs-style exchange
- expanded ensemble schedulers

## Recommended executable layout

- `SPONGE`
  Single-replica worker executable
- `SPONGE_MANAGER`
  High-level orchestration executable

Internally these should share the same runtime library.

## Exchange semantics

The intended schedule-manager REMD semantics are:

- each schedule owns one fixed thermodynamic state
- accepted exchanges swap worker runtime state between schedules
- schedule-local outputs therefore remain tied to one thermodynamic slot

This matches the desired trajectory semantics more closely than parameter-only
swaps.

## REMD algorithm mapping

### T-REMD

Needs:

- potential energy per schedule
- odd/even neighbor exchange scheduling
- velocity scaling after accepted swaps
- neighbor-list rebuild before the next block

### H-REMD

Needs:

- local energy under the local Hamiltonian
- foreign-state energy evaluation for candidate schedules

### HT-REMD

Needs:

- all H-REMD quantities
- state temperatures and corresponding beta values

## Phased implementation plan

### Phase 1: runtime library split

- keep CLI behavior unchanged
- split runtime code from the executable entry point
- add a dedicated `SPONGE_MANAGER` target skeleton

### Phase 2: runtime-state API

- define `RuntimeState`
- add export/import helpers
- add post-swap maintenance hooks

### Phase 3: manager core

- define core manager data types
- implement block execution orchestration
- support multiple schedules and workers

### Phase 4: T-REMD

- implement odd/even exchange policy
- compute temperature REMD acceptance
- swap runtime states between schedules

### Phase 5: H-REMD

- add foreign-state energy evaluation in worker/runtime layer
- implement Hamiltonian exchange acceptance

### Phase 6: HT-REMD

- combine temperature and Hamiltonian reduced-potential criteria

## Immediate next steps

1. Finalize the runtime/library split.
2. Introduce a minimal manager core target and API surface.
3. Design the first `RuntimeState` schema.
4. Implement block-level execution and state persistence for one worker.

## Current implementation status

- `Phase 1` is in progress:
  - `SPONGE` now uses a scheduler-backed CLI entry point
  - `SPONGE_MANAGER` target skeleton has been introduced
  - runtime code is being split into a reusable `sponge_runtime` library
- `Phase 3` has a first core data-model scaffold:
  - `ScheduleInputs`
  - `RuntimeStateRef`
  - `ExchangeObservable`
  - `ScheduleConfig` and `ScheduleRecord`
  - `Manager` plan/introspection skeleton
- `Phase 2` remains the next functional milestone:
  - define the concrete serialized runtime-state schema
  - implement export/import hooks in the runtime layer
