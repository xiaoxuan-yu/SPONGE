# Scheduler Interface

SPONGE now exposes a lightweight process-local scheduling wrapper through
[scheduler.h](/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/SPONGE/scheduler/scheduler.h).

## Design goals

- Keep the existing `SPONGE` command-line behavior unchanged.
- Provide a stable lifecycle interface for higher-level orchestration.
- Avoid coupling replica scheduling to the current MPI communication layer.
- Support the common deployment model where an external scheduler launches
  multiple independent SPONGE processes.

## Current interface

`sponge::SpongeScheduler` currently provides:

- `InitializeFromArgv(...)`
- `InitializeFromArgs(...)`
- `RunSingleStep(...)`
- `RunSteps(...)`
- `RunToEnd(...)`
- `Snapshot()`
- `ExportRuntimeState()`
- `ImportRuntimeState(...)`
- `CollectExchangeObservables()`
- `ScaleVelocities(factor)`
- `InvalidateNeighborList(...)`
- `Finalize()`

`Snapshot()` returns lightweight runtime metadata such as:

- next step index
- last completed step
- step limit
- simulation time
- instantaneous and target temperature
- instantaneous and target pressure
- total/effective potential
- box lengths

`ExportRuntimeState()` currently returns an in-process minimal runtime-state
buffer that includes:

- coordinates
- velocities
- box lengths and angles
- step / step limit
- start time / current time

This is a first scheduler-facing state representation and is not yet a fully
serialized cross-process worker protocol payload.

## Important limitation

This is a **single-runtime-per-process** wrapper around the current global
SPONGE runtime. It is intended to be used by:

- one scheduler process launching many SPONGE worker processes, or
- one embedding process driving one SPONGE runtime at a time

It is **not** yet a multi-instance in-process API.

## Near-term extension points

The next scheduler-facing features should be:

- explicit state export/import helpers
- block-level checkpoint/restart control
- forced neighbor-list rebuild hooks after parameter changes
- foreign-state energy evaluation for H/HT-REMD and REST2
