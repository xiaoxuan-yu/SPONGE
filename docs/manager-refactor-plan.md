# SPONGE Manager Refactor Plan

## Goal

Refactor the current scheduler/manager prototype into a general-purpose task
orchestration framework for SPONGE.

The target architecture should:

- keep `SPONGE` usable as a normal single-replica executable
- treat one SPONGE runtime as one worker
- let `SPONGE_MANAGER` orchestrate multiple workers
- exchange complete `RuntimeState` objects between schedules
- avoid coupling manager core to FEP- or lambda-specific concepts
- avoid file-based request/response transport for multi-process execution
- support Linux, macOS, and Windows with one primary communication backend

## Current issues

The current prototype works, but it still has several design debts:

- manager core still exposes `lambda`-centric fields
- CLI usage still centers on the FEP sample shortcut
- file transport remains available as a compatibility/debug fallback, now also
  using persistent workers to avoid repeated SPONGE initialization
- child-process block execution can now be dispatched in parallel through
  persistent file, TCP, or shared-memory transports
- manager currently knows more about scenario-specific parameters than it
  should
- some prototype files do not yet follow SPONGE source style, including BOM,
  naming, and CMake formatting
- temporary toolchain compatibility shims such as `isoc23_compat.cpp` should
  not be part of the long-term implementation

## Design direction

### 1. Keep SPONGE as the single-replica runtime

`SPONGE` should continue to own:

- parsing `mdin`
- initializing one simulation runtime
- running one block of MD
- importing and exporting one `RuntimeState`
- evaluating local or foreign-state observables when requested

### 2. Treat the manager as an orchestrator, not a physics owner

`SPONGE_MANAGER` should own:

- worker lifecycle
- schedule lifecycle
- block dispatch
- exchange pairing
- acceptance calculation
- state swap
- logging
- recovery

It should **not** become the source of truth for physical parameters such as:

- temperature
- pressure
- lambda
- soft-core parameters
- other force-field or enhanced-sampling settings

Those should come from each worker's own `mdin` or worker args.

### 3. Use complete runtime-state exchange

The manager should continue to exchange complete `RuntimeState` objects instead
of parameter-only remapping.

This keeps:

- trajectories schedule-local
- rerun workflows natural
- semantics close to full-state replica exchange

### 4. Use persistent workers for every transport

For real multi-process execution, the primary direction should be:

- persistent worker processes
- long-lived communication channels
- no repeated SPONGE runtime initialization between manager blocks

### 5. Prefer one cross-platform transport

Because SPONGE explicitly targets Linux, macOS, and Windows, the fastest
cross-platform path is:

- TCP loopback
- length-prefixed binary messages
- long-lived manager/worker connections

This is preferred over:

- Unix-domain-socket-only designs
- OS-specific named pipe first
- file-based protocols

### 6. Follow SPONGE source style from the start

The refactored implementation should follow the existing SPONGE code style,
not a separate manager-specific style.

Required conventions:

- C++ source/header files use UTF-8 with BOM
- 4-space indentation
- Allman braces
- 80-column formatting
- struct/class names follow the existing SPONGE style, such as
  `RUNTIME_STATE` or `SPONGE_SCHEDULER`
- methods use the existing PascalCase / underscore style seen in nearby code
- CMake files pass the repository `cmake-format` check

Before treating the refactor as merge-ready, run:

```bash
pixi run -e dev-cuda13 format
pixi run -e dev-cuda13 format-check
```

### 7. Do not hide toolchain issues in source shims

The current prototype used a temporary `isoc23_compat.cpp` shim to bridge
missing libc symbols in one build environment. That is acceptable as a local
unblocker, but it should not be kept in the formal manager implementation.

The long-term solution should be one of:

- make compile-time and link-time libc/sysroot come from the same toolchain
- adjust the CMake/pixi environment so generated binaries link against the
  libc version expected by the headers
- avoid compiler/library combinations that emit unavailable `__isoc23_*`
  symbols

The source tree should not carry project-level compatibility definitions for
glibc private `__isoc23_*` symbols.

## Core concepts

### Worker

A worker is one SPONGE runtime instance.

Responsibilities:

- initialize from args
- run a block
- import/export `RuntimeState`
- report observables
- service probe requests

### Schedule

A schedule is one logical output lane managed by the manager.

Each schedule is associated with one thermodynamic or Hamiltonian slot.
Accepted exchanges swap runtime state between schedules.

### RuntimeState

`RuntimeState` is the complete worker state needed to continue integration.

It should contain:

- coordinates
- velocities
- box
- step/time
- thermostat hidden state
- barostat hidden state
- required history terms
- relevant RNG state

### Schedule inputs

Each schedule should expose one unified config block:

- `schedules.inputs`

This block should be the only schedule-local source for:

- worker input overrides
- manager-side exchange field discovery

There should be no separate `state` section in the long-term config model.

Typical examples inside `schedules.inputs`:

- `target_temperature`
- `target_pressure`
- `hamiltonian_id`
- `lambda_lj`
- `default_out_file_prefix`

The worker uses these values as input overrides on top of the shared base
`mdin`, and the manager extracts the subset relevant to the selected exchange
mode.

## Recommended manager/worker boundary

### Worker owns

- actual `mdin`
- actual force-field parameters
- actual thermostat/barostat settings
- actual lambda or alchemical settings

### Manager owns

- worker process/session lifecycle
- schedule ordering
- runtime-state routing
- exchange decisions
- schedule identity and validation

## Communication protocol

### Recommended primary backend

- persistent TCP loopback connections for the default transport
- binary request/response messages

### Message shape

Suggested fixed header fields:

- `magic`
- `version`
- `message_type`
- `request_id`
- `payload_size`

Suggested first message types:

- `HELLO`
- `RUN_BLOCK`
- `RUN_RESULT`
- `IMPORT_STATE`
- `PROBE_OBSERVABLE`
- `PROBE_RESULT`
- `GET_STATUS`
- `SHUTDOWN`
- `ERROR`

### Serialization

The current hand-written binary serialization style can be retained, but the
transport should switch from:

- file read/write

to:

- socket read/write

## Execution model

### Current prototype

- multiple schedules exist
- workers run as child processes
- child-process mode can use file, TCP loopback, or shared-memory payload
  transport
- all transports use persistent workers and are dispatched in parallel and
  gathered by the manager

### Target execution model

1. manager launches `N` persistent workers
2. each worker connects back to the manager
3. manager dispatches `RUN_BLOCK` to all active workers in parallel
4. manager gathers all results
5. manager computes exchange attempts
6. manager swaps `RuntimeState` objects in memory
7. manager sends updated states back to affected workers
8. next epoch begins

## Decoupling from FEP and lambda

### What should move out of manager core

- hardcoded `lambda` field in the core schedule descriptor
- separate `state` config section
- FEP-specific shortcut assumptions in the main manager interface
- scenario-specific ladder construction in core logic

### What can remain as scenario adapters

- `--fep-root`
- FEP sample auto-discovery
- lambda ladder convenience builders

These should be treated as:

- sample adapters
- scenario builders
- convenience frontends

not as the core manager API.

## Suggested phased refactor

## Execution discipline

Before each implementation round, reread this document and use it as the active
source of truth for the refactor. This avoids drifting back to prototype
assumptions such as file-based transport, FEP/lambda coupling, temporary libc
compatibility shims, or non-SPONGE naming style.

Also follow the repository-local `AGENTS.md` instructions for the active
workspace. In this checkout, shell commands should be run through the RTK
wrapper, for example:

```bash
rtk proxy bash -lc '...'
```

This command discipline is part of the implementation plan, not just a local
preference, because the manager refactor must remain reproducible in the same
environment used by the rest of SPONGE development.

Before merge-ready cleanup, audit the source tree to ensure the formal manager
implementation does not contain temporary toolchain workaround targets or files
such as `isoc23_compat.cpp` or `iso23-compact`. Toolchain consistency should be
handled in CMake/pixi configuration rather than by project-level libc symbol
shims.

### Phase 0: style and prototype cleanup

- remove temporary compatibility shims from the planned implementation path
- ensure new files are UTF-8 with BOM
- rename new core types to match SPONGE conventions
- format C++ and CMake files with repository tooling
- run commands through the repository-required RTK wrapper
- remove any temporary `isoc23`/`iso23` compatibility files, targets, or naming
- keep the current prototype behavior available only as a migration reference

### Phase 1: decouple manager core from FEP/lambda

- remove the hardcoded `lambda` field from manager core
- remove the separate schedule `state` section
- adopt unified `schedules.inputs`
- keep scenario-specific inputs only in adapters/frontends
- make `manager.toml` the primary interface

### Phase 2: define persistent worker protocol

- introduce a persistent manager/worker protocol abstraction
- define message headers and binary payload shapes
- keep the session abstraction independent from the selected transport

### Phase 3: implement persistent child-process worker backends

- add persistent worker modes to `SPONGE`
- add manager-side session handling
- keep worker sessions alive across multiple blocks rather than launching a
  fresh worker per request
- support file, TCP, and shared-memory transport under the same session model

### Phase 4: parallel block execution

- dispatch block runs to all active workers concurrently
- gather block results concurrently
- keep exchange and routing centralized in manager

### Phase 5: reattach T/H/HT-REMD policies

- adapt T-REMD to `inputs`-derived field usage
- adapt H-REMD to foreign-state probe usage
- adapt HT-REMD to combined criteria

### Phase 6: scenario adapters and validation polish

- rebuild FEP shortcuts as scenario adapters on top of the general manager
- add validation between schedule inputs and worker-reported parameters
- improve docs and examples

## What users do today

Current users can run:

- `SPONGE` directly for single-replica simulations
- `SPONGE_MANAGER` through:
  - shortcut CLI flags such as `--fep-root`
  - or `--config manager.toml`

Current manager usage is still prototype-oriented and still exposes FEP-heavy
examples in the docs.

## Target user experience

The long-term manager UX should look like:

1. prepare one `manager.toml`
2. define one shared worker/base `mdin` template
3. define multiple schedules
4. let each schedule provide its own `inputs` overrides
5. launch `SPONGE_MANAGER --config manager.toml`
6. let manager handle block execution and exchange internally

## Recommended `manager.toml` model

The config should stay small and orchestration-oriented.

Suggested top-level structure:

- `[manager]`
- `[exchange]`
- `[worker_defaults]`
- `[[schedules]]`

### `[manager]`

Holds scheduler-level behavior only:

- `name`
- `block_steps`
- `epochs`
- `transport`
- `emit_output`
- `log_path`

### `[exchange]`

Holds exchange control:

- `enabled`
- `mode`
- `pairing`
- `start_round`

### `[worker_defaults]`

Holds the shared base worker launch template:

- `executable`
- `args`
- optional shared base `mdin`
- optional working-directory root

This is where a shared `mdin` should usually be defined.

### `[[schedules]]`

Each schedule should carry:

- `schedule_id`
- `label`
- `working_directory`
- `inputs`

Example shape:

```toml
[manager]
name = "my-remd"
block_steps = 1000
epochs = 100
transport = "tcp"
emit_output = false

[exchange]
enabled = true
mode = "tremd"
pairing = "odd_even"

[worker_defaults]
executable = "/path/to/SPONGE"
args = ["-mdin", "/data/remd/base.mdin.toml"]
working_directory_root = "/data/remd"

[[schedules]]
schedule_id = 0
label = "replica0"
working_directory = "replica0"

[schedules.inputs]
target_temperature = 300.0
default_out_file_prefix = "mdout"

[[schedules]]
schedule_id = 1
label = "replica1"
working_directory = "replica1"

[schedules.inputs]
target_temperature = 320.0
default_out_file_prefix = "mdout"
```

## Output collision policy

Different schedules must never silently overwrite each other's outputs.

The recommended rule is:

- users may override the schedule-local output prefix through
  `schedules.inputs.default_out_file_prefix`
- if the user provides no explicit prefix, manager supplies a default one
- manager appends the schedule id to the effective prefix so outputs stay
  distinct

Example:

- user sets `default_out_file_prefix = "mdout"`
- schedule `0` becomes `mdout_0`
- schedule `1` becomes `mdout_1`

This keeps the syntax simple while making collisions unlikely by default.

## Exchange-mode field discovery

Because there is no separate `state` section, manager should derive the fields
it needs from `schedules.inputs`.

Examples:

- `tremd` requires `target_temperature`
- `hremd` requires a Hamiltonian identity field such as `hamiltonian_id`
- `htremd` requires both

The manager should validate required fields at startup and fail fast if they
are missing.

## Open questions for follow-up

- how much schedule metadata should be discovered from worker inputs versus
  required in config?
- should worker `HELLO` include the effective temperature/pressure/hamiltonian
  summary parsed from input?
- should the first transport backend be pure blocking TCP or an async event-loop
  implementation?
- should scenario adapters live in the main CLI or in separate helper tools?
