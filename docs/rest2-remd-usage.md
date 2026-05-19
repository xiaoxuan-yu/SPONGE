# REST2 and REST2-REMD Usage

This document describes the first REST2 implementation under the shared
`Selective_Interaction` runtime and the `rest2` manager exchange mode.

## Scope

The current REST2 implementation scales selected short-range direct
Lennard-Jones and direct Coulomb interactions:

- hot-hot pairs: `lambda_m`
- hot-cold pairs: `sqrt(lambda_m)`
- cold-cold pairs: `1.0`

The first implementation does not yet scale reciprocal PME, bonded terms, or
1-4 terms. It is designed as a static, probe-safe Hamiltonian for manager-driven
REST2 replica exchange.

## Single-Replica REST2

REST2 can be enabled from a normal SPONGE `mdin` without changing the external
SPONGE invocation.

```text
REST2_mode = on
REST2_atom_numbers = 22
REST2_lambda_m = 0.80
```

or with an explicit hot-atom file:

```text
REST2_mode = on
REST2_atom_in_file = hot_atoms.txt
REST2_lambda_m = 0.80
```

`REST2_lambda_m = 1.0` is the reference Hamiltonian. Hotter effective solute
replicas normally use `REST2_lambda_m < 1.0`.

REST2 adds these mdout columns:

```text
REST2_lambda_m
REST2_unscaled
REST2_effective
REST2_bias
```

For `REST2_lambda_m = 1.0`, `REST2_bias` should be approximately zero.

## REST2-REMD With SPONGE_MANAGER

Use `exchange.mode = "rest2"` and provide each schedule's `REST2_lambda_m` in
`schedules.inputs`. The manager passes those inputs to each worker as command
line overrides, so one shared `mdin` can define the base system while the
schedule controls the Hamiltonian slot.

```toml
[manager]
block_steps = 1000
epochs = 100
transport = "tcp"
log_path = "manager_exchange.log"

[exchange]
enabled = true
mode = "rest2"

[worker_defaults]
mdin = "mdin.spg.toml"
emit_output = false
args = ["-dont_check_input", "1"]
working_directory_root = "replicas"

[worker_defaults.inputs]
target_temperature = 300.0
default_out_file_prefix = "rest2"

[schedules]
ids = [0, 1]

[schedules.inputs]
REST2_lambda_m = [1.0, 0.9]
```

The `rest2` exchange policy reuses the Hamiltonian REMD acceptance form:

```text
log_acc = -beta * [H_i(x_j) + H_j(x_i) - H_i(x_i) - H_j(x_j)]
```

Accepted exchanges swap complete `RuntimeState` objects between Hamiltonian
slots. Coordinates, velocities, box, thermostat/barostat state, and RNG state
move together. No velocity scaling is applied for pure REST2 because all slots
normally use the same physical thermostat temperature.

## Validation Commands

Run the REST2 validation suite:

```bash
pixi run -e dev-cuda13 python -m pytest \
  benchmarks/validation/rest2/tests \
  --sponge-cmd /path/to/SPONGE \
  --manager-cmd /path/to/SPONGE_MANAGER
```

Run the lightweight REST2 performance micro-benchmark:

```bash
pixi run -e dev-cuda13 python -m pytest \
  benchmarks/performance/rest2/tests \
  --rest2-perf-sponge-cmd /path/to/SPONGE \
  --rest2-perf-steps 100
```

The benchmark writes a JSON summary under
`benchmarks/performance/rest2/outputs/`.
