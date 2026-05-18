# SPONGE Manager Usage

## Overview

`SPONGE_MANAGER` is the high-level orchestration entrypoint for the current
scheduler/manager prototype.

It can be driven in two ways:

- shortcut CLI flags such as `--fep-root`, which are convenient for the current
  FEP validation sample
- an explicit `--config manager.toml`, which declares schedules, workers, and
  manager-level execution settings directly

It can run schedules in two worker modes:

- `in_process`: the manager reuses the SPONGE runtime in the same process
- `child_process`: the manager launches external `SPONGE` workers

For `child_process`, the transport is selected by `[manager].transport`:

- `transport = "file"`: prototype file request/response protocol
- `transport = "tcp"`: TCP loopback request/response protocol using
  `SPONGE --worker-tcp host:port`
- `transport = "shm"`: TCP loopback control messages plus shared-memory
  request/response payloads

Advanced users may still override the transport per worker with
`worker_defaults.persistent` or `schedules.worker.persistent`. The higher-level
`manager.transport` field is the preferred user-facing knob.

When every schedule uses `child_process`, block execution is dispatched in
parallel and gathered by the manager. Mixed or `in_process` execution remains
serial for now because the in-process SPONGE runtime still owns process-global
state such as the current working directory and CUDA runtime initialization.

The current CLI keeps `SPONGE` itself directly runnable for ordinary
single-replica usage. The worker mode is only used internally by the manager.

## FEP sample root

The current Hamiltonian-style validation sample is:

- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD`

Shared input:

- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/step2_mdin.txt`

Example state directories:

- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/0`
- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/1`

## Basic manager smoke test

Run two schedules with no exchange:

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --fep-root /media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD \
  --state-ids 0,1 \
  --lambda-lj-list 0.0,0.333333 \
  --block-steps 1 \
  --emit-output 0
```

## Manager config file

`SPONGE_MANAGER` also supports:

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --config /path/to/manager.toml
```

Minimal shape:

```toml
[manager]
block_steps = 1
epochs = 1
transport = "tcp"
emit_output = false
log_path = "/tmp/manager_exchange.log"

[exchange]
enabled = true
mode = "hremd"
start_round = 0

[worker_defaults]
launch = "child_process"
args = [
  "-mdin", "/abs/path/to/step2_mdin.txt",
  "-workspace", ".",
  "-default_in_file_prefix", "TMP",
]
working_directory_root = "/abs/path/to/remd_root"

[[schedules]]
schedule_id = 0
label = "state_0"
working_directory = "0"

[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = 0
lambda_lj = 0.0
default_out_file_prefix = "manager_smoke"

[schedules.worker]
name = "worker_0"
```

Notes:

- `schedules.inputs` is the unified schedule-local input block. REMD policies
  discover fields such as `target_temperature` and `hamiltonian_id` from here.
- scalar `schedules.inputs` entries are appended to the worker command line as
  SPONGE input overrides, except manager-only metadata such as
  `hamiltonian_id`.
- relative `working_directory`, `executable_path`, and `log_path` are resolved
  relative to the config file location, or to `working_directory_root` when it
  is set.
- the effective `default_out_file_prefix` is made schedule-local by appending
  the schedule id, for example `manager_smoke_0` and `manager_smoke_1`.
- if `worker_defaults.launch = "child_process"` and a schedule omits
  `worker.executable_path`, the manager defaults to a sibling `SPONGE`
  executable next to `SPONGE_MANAGER`.
- `manager.transport = "tcp"` enables the TCP loopback worker protocol and
  avoids temporary request/response files for child-process dispatch.
- `manager.transport = "shm"` keeps TCP as the control channel, but moves the
  serialized worker request/response payload through shared memory. On
  Linux/macOS this uses POSIX shared memory; on Windows it uses named file
  mapping objects. This is still host-memory IPC, not CUDA IPC or direct GPU
  pointer exchange.

## T-REMD examples

In-process worker mode:

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --fep-root /media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD \
  --state-ids 0,1 \
  --lambda-lj-list 0.0,0.333333 \
  --thermo-temperatures 300,600 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch in_process \
  --remd-mode tremd \
  --exchange-round 0
```

Child-process worker mode:

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --fep-root /media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD \
  --state-ids 0,1 \
  --lambda-lj-list 0.0,0.333333 \
  --thermo-temperatures 300,600 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode tremd \
  --exchange-round 0
```

## H-REMD example

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --fep-root /media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD \
  --state-ids 0,1 \
  --lambda-lj-list 0.0,0.333333 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode hremd \
  --exchange-round 0
```

## HT-REMD example

```bash
/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/build-dev-cuda13/SPONGE_MANAGER \
  --fep-root /media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD \
  --state-ids 0,1 \
  --lambda-lj-list 0.0,0.333333 \
  --thermo-temperatures 300,600 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode htremd \
  --exchange-round 0
```

## Logs

The manager writes exchange history to:

- `/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD/manager_exchange.log`

Current record types:

- `exchange_attempt`
- `schedule_state`

## Smoke regression script

A bundled smoke script exercises the currently validated FEP-based paths:

- `/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/scripts/remd_smoke_fep.sh`

It currently runs:

- multi-schedule no-exchange manager execution
- config-driven H-REMD manager execution
- child-process T-REMD
- child-process H-REMD
- child-process HT-REMD

There is also a fail-closed probe guard smoke for history-dependent bias
modules:

- `/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/scripts/remd_probe_guard_smoke.sh`

It currently verifies that a hidden worker `probe_only` request aborts on a
`sinkmeta` or `SITS` case instead of silently returning a foreign-state
observable.

And there is a scheduler/runtime-state roundtrip smoke:

- `/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE/scripts/scheduler_state_roundtrip_smoke.sh`

It verifies, on the FEP state-0 sample, that a `2`-step direct worker run stays
close to a `1 + import + 1` resumed run, while keeping the exported observable
summary aligned.

## Current caveats

- `manager.transport = "tcp"` child-process workers now use TCP loopback
  instead of temporary request/response files. `RUN_BLOCK` workers are kept
  alive across blocks and epochs; foreign-state probe workers remain one-shot
  sessions so probes do not overwrite a schedule's own runtime state.
- `manager.transport = "shm"` uses the same worker/session protocol as TCP, but
  wraps the TCP control channel with shared-memory bulk payload transfer. It has
  passed the current two-worker T-REMD and H-REMD smoke paths, including
  foreign-state probes.
- `manager.transport = "file"` child-process workers keep the older file
  protocol as a migration fallback.
- H-REMD and HT-REMD foreign-state energies are still obtained by target-state
  probe workers rather than a specialized lightweight rerun kernel.
- Foreign-state probe workers currently reject configurations with sink
  metadynamics or SITS enabled, because their history-dependent bias state is
  not serialized into `RuntimeState` yet.
- `RuntimeState` already includes several hidden thermostat/barostat states, but
  it is not yet an exhaustive image of every stochastic or enhanced-sampling
  module in SPONGE.
