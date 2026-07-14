# SPONGE Input Inventory and Layered Container Classification

This document reorganizes the current SPONGE input surface under the proposed
four-file input model:

```text
topology.spgt.h5      # physical topology and Hamiltonian container
protocol.spgp.h5      # simulation protocol container
restart.spgr.h5       # restart/checkpoint state container
run.mdin              # human-readable binding/control deck
```

Rerun and trajectory-analysis launches may additionally bind a trajectory H5MD
container:

```text
trajectory.spg.h5md   # trajectory/history container for rerun-style input
```

Core ownership rule:

```text
container owns canonical data/state
mdin owns binding + small editable run policy
resolver owns compatibility check and runtime assembly
```

The inventory is based on a code scan of `Command_Exist`, `Command`,
`Command_Choice`, `Check_Int`, `Check_Float`, `Get_Bool`, explicit `*_in_file`
readers, TOML schedule decoders, and known dynamic prefix-style inputs.

## 1. Layer Boundaries

### `topology.spgt.h5`

Owns the physical topology and Hamiltonian foundation.

Examples:

- Atom identity and atom ordering
- Mass, charge, atom type, residue, molecule topology
- Bond, angle, dihedral, improper, CMAP, LJ, 1-4, GB parameters
- Exclusions and static connectivity
- Water/ion model and force-field definition
- Intrinsic constraint topology when it is part of the prepared topology or force-field model
- topology-intrinsic atom groups or selections
- Intrinsic reference coordinates or initial box if they are part of the prepared topology identity

It answers: what is the physical topology and force field?

### `protocol.spgp.h5`

Owns the algorithmic layer attached to a topology.

Examples:

- CV definitions
- Restraint definitions and target definitions
- Enhanced sampling method definitions
- Metadynamics, umbrella, ABF, SITS, OPES-like method configuration
- Bias update rules and adaptive sampling policy
- Collective-variable output definitions
- Protocol-level analysis observables
- Resolved CV/restraint atom maps when tied to a topology hash

It answers: what sampling/protocol algorithm is attached to the topology?

### `restart.spgr.h5`

Owns concrete numerical state.

Examples:

- Restart coordinates, velocities, box
- Current step and time
- Checkpoint state
- Integrator, thermostat, barostat internal state
- RNG state when reproducibility requires it
- Bias history, metadynamics hills/grid, accumulated bias
- SITS/adaptive sampling state
- Parent run reference, lineage, and state hash
- Optional resolved launch snapshot for provenance/debugging

It answers: what concrete state is this run starting from or continuing from?
Every launch must bind a restart container, and that container stores one
concrete restart state. Even an initial launch starts from an initial
`restart.spgr.h5` state.

### `run.mdin`

Owns binding and small editable run policy.

Examples:

- Bind `topology.spgt.h5`, `protocol.spgp.h5`, and `restart.spgr.h5`
- Continue/restart/append mode
- Stateless component load policy
- Ensemble
- Timestep
- Step count or target duration
- Temperature and pressure target policy
- Thermostat/barostat method selection
- Output/log/checkpoint cadence
- Seed policy or explicit seed
- Execution/debug/resource options
- Temporary compatibility overrides

It answers: which containers should be bound, and how should this launch run?

`run.mdin` must not expose HDF5 internal paths or symbolic references back into
the restart file. It declares H5 container bindings under `[input.h5.*]` and a
restart component load policy under `[input.h5.restart]`; the resolver loads the
single restart state declared by `restart.spgr.h5`.

For `mode = rerun`, `run.mdin` may bind an H5MD trajectory input. This is not a
restart binding: it selects a history container whose frames are streamed by the
rerun reader.

### Trajectory H5MD Input

Owns trajectory/history data used as input to rerun or trajectory-analysis
launches.

Examples:

- Coordinate frames
- Optional velocity frames
- Optional force frames for analysis modes that consume forces
- Box frames
- Step and time arrays for the trajectory frame axis
- Observable streams used only as analysis input, not as launch state

It answers: which trajectory frames should rerun consume?

Trajectory H5MD input must not be treated as canonical topology, protocol, or
launchable restart state. A rerun launch still needs topology/protocol context
and a launch state for any non-trajectory state required by the engine. A
trajectory frame may be converted into `restart.spgr.h5`, but that conversion is
an explicit state-materialization step, not normal rerun input.

## 2. Current Input Inventory by Owner

### 2.1 `run.mdin`: Binding, Launch Policy, Paths, and Overrides

These inputs should stay in the human-editable deck because they are run-specific
choices, resource/path selections, output policy, or debug overrides.
The deck path itself is selected by CLI or launcher state, not by a field inside
`run.mdin`. `command_only` is also external launch control and must not be active
for a normal `run.mdin`-based launch.

#### Binding and Global Control

| Current/proposed item | Type | Classification |
|---|---|---|
| `input_h5_topology_path` from `[input.h5.topology] path` | path | Resolver binding to `topology.spgt.h5`. |
| `input_h5_protocol_path` from `[input.h5.protocol] path` | path | Resolver binding to `protocol.spgp.h5`. |
| `input_h5_restart_path` from `[input.h5.restart] path` | path | Resolver binding to `restart.spgr.h5`. |
| `input_h5_restart_load` from `[input.h5.restart] load` | enum | Restart component load policy: `structural`, `dynamic`, `protocol`, `full`, or `custom`. |
| `input_h5_trajectory_path` from `[input.h5.trajectory] path` | path | H5MD trajectory input path for rerun/analysis; recommended suffix `*.spg.h5md`. |
| `input_h5_trajectory_particle_stream` from `[input.h5.trajectory] particle_stream` | string | Optional H5MD particle stream name; default `all`. |
| `workspace` | path | Path resolution context. |
| `buffer_frame` | int | Runtime/file buffering policy. |
| `device` | int/list/string | Resource binding. |
| `device_optimized_block` | int | Runtime/GPU tuning. |
| `dont_check_input` | bool/int | Input warning behavior. |
| `end_pause` | bool/int | Terminal behavior. |
| `plugin` | string/list | Runtime plugin selection. |
| `default_in_file_prefix` | path prefix | Legacy fallback/override mechanism only. |
| `default_out_file_prefix` | path prefix | Legacy output naming policy only. |
| `output_h5_trajectory_path` from `[output.h5.trajectory] path` | path | Canonical trajectory H5MD output path; recommended suffix `*.spg.h5md`. |
| `output_h5_trajectory_vds` from `[output.h5.trajectory] vds` | bool | Use chunked H5MD shards plus HDF5 VDS wrapper for trajectory output. |
| `output_h5_trajectory_chunk_size` from `[output.h5.trajectory] chunk_size` | int | VDS file-level shard size in trajectory frames; default `20`. |
| `output_h5_trajectory_repair_policy` from `[output.h5.trajectory] repair_policy` | string | VDS finalize policy: `strict` by default, or explicit `complete_prefix`. |
| `output_h5_restart_path` from `[output.h5.restart] path` | path | Canonical restart H5 output path; recommended suffix `*.spgr.h5`. |
| `output_h5_observable_path` from `[output.h5.observable] path` | path | Optional observable-only H5MD output path; recommended suffix `*.obs.spg.h5md`. |
| `mdout` | path | Legacy compatibility output path. |
| `mdinfo` | path | Legacy compatibility output path. |

#### MD Launch Policy

| Current item | Type | Classification |
|---|---|---|
| `mode` | enum | Launch mode: NVE/NVT/NPT/min/rerun. |
| `dt` | float | Timestep policy. |
| `step_limit` | int | Run length. |
| `frame_limit` | int | Rerun length. |
| `rerun_frame_limit` | int | Rerun length alias. |
| `target_temperature` | float | Temperature policy. |
| `target_pressure` | float | Pressure policy. |
| `target_temperature_schedule_mode` | enum | Run policy. |
| `target_temperature_schedule_steps` | TOML array | Inline run policy. |
| `target_pressure_schedule_mode` | enum | Run policy. |
| `target_pressure_schedule_steps` | TOML array | Inline run policy. |
| `pbc` | bool/int | Runtime/launch policy; must be compatible with topology/run box state. |
| `skin` | float | Neighbor/runtime setting. |
| `cutoff` | float | Runtime nonbonded policy. |
| `velocity_max` | float | Runtime clamp. |
| `make_output_whole` | enum/string | Output behavior. |
| `force_whole_output` | bool/int | Output behavior. |

Schedule note: `target_temperature_schedule_file` and
`target_pressure_schedule_file` are not structurally required when `run.mdin` is
TOML. The same information can be represented inline with
`target_temperature_schedule_steps` or `target_pressure_schedule_steps`, each as
an array of `{ step, value }` objects. Inline steps and a schedule file are
mutually exclusive in the current implementation. Schedule files remain useful
as a legacy/external form or when a long policy is shared across many decks.

#### Output and Checkpoint Policy

| Current item | Type | Classification |
|---|---|---|
| `write_information_interval` | int | Log/output cadence. |
| `write_trajectory_interval` | int | Trajectory output cadence. |
| `write_mdout_interval` | int | Energy/log output cadence. |
| `write_restart_file_interval` | int | Checkpoint request cadence. |
| `[write.interval] information` | int | TOML alias normalized to `write_information_interval`. |
| `[write.interval] trajectory` | int | TOML alias normalized to `write_trajectory_interval`. |
| `[write.interval] mdout` | int | TOML alias normalized to `write_mdout_interval`. |
| `[write.interval] restart` or `restart_file` | int | TOML alias normalized to `write_restart_file_interval`. |
| `print_pressure` | bool/int | Output selection. |
| `print_zeroth_frame` | bool/int | Output selection. |
| `max_restart_export_count` | int | Checkpoint retention policy. |
| `rst` | path | Legacy restart output base; canonical checkpoint writes belong to `output_h5_restart_path`. |
| `crd` | path | Legacy trajectory output or rerun trajectory input. |
| `box` | path | Legacy box output or rerun box input. |
| `vel` | path | Legacy velocity output or rerun velocity input. |
| `frc` | path | Legacy force trajectory output. |
| `qc_scf_output` | path | Legacy quantum-chemistry SCF diagnostic output path; H5 output stores the log under `/parameters/sponge/qc/scf_output`. |
| `rerun_start` | int | Rerun policy. |
| `rerun_strip` | int | Rerun policy. |
| `rerun_need_box_update` | bool/int | Rerun policy. |

Rerun trajectory inputs are trajectory data. They should not be folded into
`topology.spgt.h5`, `protocol.spgp.h5`, or `restart.spgr.h5`; they can remain
legacy external trajectory paths or be represented by a trajectory H5MD
container.

When `input_h5_trajectory_path` is set, the rerun reader consumes trajectory
frames from that H5MD file and `crd`, `box`, and `vel` are not required. A deck
must not request both `input_h5_trajectory_path` and legacy rerun trajectory
inputs (`crd`, `box`, or `vel`) for the same rerun launch unless a future
resolver defines an explicit override policy. `rerun_start`, `rerun_strip`, and
`rerun_frame_limit` apply to the H5MD trajectory frame axis exactly as they apply
to legacy rerun frames.

If any canonical H5 output is enabled, legacy output files are disabled by
default. Legacy files are written only when their legacy path keys are
explicitly set. Legacy sidecars are compatibility artifacts, not canonical data
owners.

`output_h5_trajectory_chunk_size` is only meaningful when
`output_h5_trajectory_vds = true`. It counts trajectory frames, not MD steps and
not HDF5 internal dataset chunks. It does not affect striping, compression,
`buffer_frame`, or `write_trajectory_interval`.

#### Thermostat, Barostat, and Runtime Resources

| Current item | Type | Classification |
|---|---|---|
| `thermostat`, `thermostat_mode` | enum | Launch policy. |
| `thermostat_tau` | float | Prefix form of `thermostat.tau`; launch policy. |
| `thermostat_seed` | int | Seed policy unless persisted in `restart.spgr.h5` RNG state. |
| `barostat`, `barostat_mode` | enum | Launch policy. |
| `nose_hoover_chain_length` | int | Thermostat method policy. |
| `nose_hoover_chain_restart_output` | path | Legacy output path. |
| `nose_hoover_chain_crd`, `nose_hoover_chain_vel` | path | Legacy thermostat trajectory output. |
| `monte_carlo_barostat_update_interval` | int | Barostat policy. |
| `monte_carlo_barostat_check_interval` | int | Barostat policy. |
| `monte_carlo_barostat_initial_ratio` | float | Barostat policy. |
| `monte_carlo_barostat_accept_rate_low` | float | Barostat policy. |
| `monte_carlo_barostat_accept_rate_high` | float | Barostat policy. |
| `monte_carlo_barostat_couple_dimension` | enum/int | Barostat policy. |
| `monte_carlo_barostat_only_direction` | enum/int | Barostat policy. |
| `monte_carlo_barostat_surface_number` | int | Barostat policy. |
| `monte_carlo_barostat_surface_tension` | float | Barostat policy. |
| `[barostat.monte_carlo] *` | mixed | TOML alias group normalized to `monte_carlo_barostat_*`. |
| `DOM_DEC_update_interval` | int | Domain decomposition runtime policy. |
| `DOM_DEC_split_nx`, `DOM_DEC_split_ny`, `DOM_DEC_split_nz` | int | Resource/runtime partition policy. |
| `neighbor_list_refresh_interval` | int | Runtime tuning. |
| `neighbor_list_skin_permit` | float | Runtime tuning. |
| `neighbor_list_throw_error_when_overflow` | bool/int | Runtime checking. |
| `neighbor_list_max_neighbor_numbers` | int | Runtime capacity. |
| `neighbor_list_check_overflow_interval` | int | Runtime checking. |
| `neighbor_list_max_atom_in_grid_numbers` | int | Runtime capacity. |
| `neighbor_list_max_ghost_in_grid_numbers` | int | Runtime capacity. |

#### Minimization and Constraint Method Policy

| Current item | Type | Classification |
|---|---|---|
| `minimization_max_move` | float | Launch policy. |
| `minimization_momentum_keep` | float | Launch policy. |
| `minimization_dynamic_dt` | bool/int | Launch policy. |
| `minimization_beta1` | float | Method policy. |
| `minimization_beta2` | float | Intended method policy; code scan suggests current guard may need review. |
| `minimization_epsilon` | float | Method policy. |
| `constrain_mode` | enum | Method selection. |
| `constrain_angle` | bool/int | Method policy. |
| `constrain_mass` | float | Method policy. |
| `SHAKE_iteration_numbers` | int | Solver policy. |
| `SHAKE_step_length` | float | Solver policy. |
| `settle_disable` | bool/int | Method switch. |

Constraint ownership depends on semantics. Intrinsic force-field/topology
constraints belong to `topology.spgt.h5`; protocol-imposed extra constraints,
including the current `constrain_in_file` path, belong to `protocol.spgp.h5`.
Method selection and solver policy belong to `run.mdin`.

#### Force and Runtime Method Scalars

| Current item | Type | Classification |
|---|---|---|
| `lambda_lj` | float | FEP/soft-core launch policy. |
| `soft_core_alpha` | float | Soft-core launch policy. |
| `soft_core_powfer` | float | Existing spelling in code; soft-core launch policy. |
| `soft_core_sigma` | float | Soft-core launch policy. |
| `soft_core_sigma_min` | float | Soft-core launch policy. |
| `PM_fftx`, `PM_ffty`, `PM_fftz` | int | PME/PM runtime grid policy. |
| `PM_grid_spacing` | float | PME/PM runtime grid policy. |
| `PM_Direct_Tolerance` | float | PME/PM accuracy/runtime policy. |
| `PM_MPI_size` | int | PME/PM runtime policy. |
| `PM_print_detail` | bool/int | Runtime output policy. |
| `gb_epsilon`, `gb_radii_cutoff`, `gb_radii_offset` | float | NO_PBC GB runtime/model policy. If fixed as part of a force-field definition, resolver may require it in `topology.spgt.h5`. |
| `hard_wall_x_low`, `hard_wall_y_low`, `hard_wall_z_low` | float | Wall/confinement protocol geometry; canonical value belongs to `protocol.spgp.h5`. |
| `hard_wall_x_high`, `hard_wall_y_high`, `hard_wall_z_high` | float | Wall/confinement protocol geometry; canonical value belongs to `protocol.spgp.h5`. |

#### Legacy Compatibility Paths and Debug Overrides

AMBER and GROMACS input paths remain legacy compatibility paths and are out of
scope for this container specification version. This document does not define
how `amber_parm7`, `amber_rst7`, `gromacs_top`, `gromacs_gro`,
`gromacs_include_dir`, or `gromacs_define` are converted or represented inside
`topology.spgt.h5`, `protocol.spgp.h5`, or `restart.spgr.h5`.

| Current item | Type | Classification |
|---|---|---|
| Any explicit `*_in_file` | path | Debug/override path. Canonical owner depends on semantic data type. |

### 2.2 `topology.spgt.h5`: Physical topology and Hamiltonian

These inputs define the topology and Hamiltonian. They should become canonical
datasets or metadata in `topology.spgt.h5`.

#### Native HDF5 Contract Rule

`topology.spgt.h5` is a native HDF5 topology/Hamiltonian container. Runtime
materialization must read typed HDF5 datasets from the container, not inline
legacy text files and not HDF5 string datasets that simply embed the old
`*_in_file` payload. Legacy filenames and raw imported source snippets may be
kept only as provenance under `/provenance` or as non-launchable compatibility
records under `/compatibility/legacy_import`; they are not canonical runtime
inputs.

Every canonical dataset below uses zero-based atom indices, row-major array
layout, and explicit unit metadata when values are not dimensionless. Optional
module groups must include a local `schema_version` dataset or attribute so a
module reader can reject unknown layouts before mutating runtime state.

#### Root Metadata

| HDF5 path/attr | Type | Required | Constraint |
|---|---|---:|---|
| `/schema/name` | string | yes | e.g. `"sponge.topology"` |
| `/schema/version` | int/string | yes | Supported by resolver. |
| `/identity/uuid` | string | yes | Stable container identity. |
| `/identity/content_hash` | string | yes | Hash of canonical topology content. |
| `/provenance/generator` | string | no | Producer. |
| `/provenance/generator_version` | string | no | Producer version. |
| `/provenance/source_files` | string/JSON | no | Original source summary. |
| `/topology/atom_count` | int64 | yes | `> 0` |
| `/topology/atom_order_hash` | string | yes | Used by resolver compatibility check. |
| `/topology/topology_hash` | string | yes | Used by resolver compatibility check. |
| `/topology/forcefield_hash` | string | yes | Used by resolver compatibility check. |
| `/topology/box_type` | enum string | conditional | Required if topology has intrinsic box/PBC definition. |

#### Atom Identity and Static Topology

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/atoms/name` | string table, length `N` | external/native topology |
| `/atoms/element` | string table or `int32[N]` atomic number | external/native topology |
| `/atoms/mass` | `float32[N]`, `>0` | `mass_in_file` |
| `/atoms/charge` | `float32[N]` | `charge_in_file` |
| `/atoms/type` | `int32[N]` or string table refs | `LJ_in_file`, external topology |
| `/atoms/residue_index` | `int32[N]` | `residue_in_file` |
| `/residues/name` | string table | `residue_in_file` |
| `/residues/atom_offset` | `int64[n_residue+1]` CSR | `residue_in_file` |
| `/topology/exclusions/offset` | `int64[N+1]` CSR | `exclude_in_file` |
| `/topology/exclusions/list` | `int32[*]` atom index | `exclude_in_file` |
| `/topology/molecules/atom_offset` | `int64[n_molecule+1]` CSR | external/native topology |
| `/topology/molecules/atom_index` | `int32[N]` atom index | external/native topology |
| `/topology/named_groups/<name>` | `int32[*]` | topology-intrinsic selections only. |

The native reader must validate `N == /topology/atom_count`, all CSR offsets are
monotonic, and all atom indices are in `[0, N)`. `atom_order_hash` is computed
from the canonical atom ordering datasets, not from an imported filename.

#### Bonded and Nonbonded Parameters

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/forcefield/bond/atoms` | `int32[nbond,2]` | `bond_in_file` |
| `/forcefield/bond/k` | `float32[nbond]` | `bond_in_file` |
| `/forcefield/bond/r0` | `float32[nbond]` | `bond_in_file` |
| `/forcefield/angle/atoms` | `int32[nangle,3]` | `angle_in_file` |
| `/forcefield/angle/k` | `float32[nangle]` | `angle_in_file` |
| `/forcefield/angle/theta0` | `float32[nangle]` with unit metadata | `angle_in_file` |
| `/forcefield/dihedral/atoms` | `int32[ndihedral,4]` | `dihedral_in_file` |
| `/forcefield/dihedral/pk` | `float32[ndihedral]` | `dihedral_in_file` |
| `/forcefield/dihedral/pn` | `float32[ndihedral]` | `dihedral_in_file` |
| `/forcefield/dihedral/ipn` | `int32[ndihedral]` | `dihedral_in_file` |
| `/forcefield/dihedral/gamc` | `float32[ndihedral]` | `dihedral_in_file` |
| `/forcefield/dihedral/gams` | `float32[ndihedral]` | `dihedral_in_file` |
| `/forcefield/improper/atoms` | `int32[n,4]` | `improper_dihedral_in_file` |
| `/forcefield/improper/pk` | `float32[n]` | `improper_dihedral_in_file` |
| `/forcefield/improper/phi0` | `float32[n]` | `improper_dihedral_in_file` |
| `/forcefield/lj/type` | `int32[N]` | `LJ_in_file` |
| `/forcefield/lj/atom_type_count` | `int32` | `LJ_in_file` |
| `/forcefield/lj/params` | `float32[n_lj,2]`, columns `[pair_A_12, pair_B_6]` | `LJ_in_file` |
| `/forcefield/nb14/atoms` | `int32[n14,2]` | `nb14_in_file` |
| `/forcefield/nb14/params` | `float32[n14,3]`, columns `[A_12, B_6, cf_scale_factor]` | `nb14_in_file` |
| `/forcefield/nb14_extra/atoms` | `int32[n,2]` | `nb14_extra_in_file` |
| `/forcefield/nb14_extra/params` | `float32[n,ncol]` | `nb14_extra_in_file` |
| `/forcefield/urey_bradley/atoms` | `int32[n,3]` | `urey_bradley_in_file` |
| `/forcefield/urey_bradley/angle_k` | `float32[n]` | `urey_bradley_in_file` |
| `/forcefield/urey_bradley/angle_theta0` | `float32[n]` | `urey_bradley_in_file` |
| `/forcefield/urey_bradley/bond_k` | `float32[n]` | `urey_bradley_in_file` |
| `/forcefield/urey_bradley/bond_r0` | `float32[n]` | `urey_bradley_in_file` |
| `/forcefield/cmap/atoms` | `int32[n,5]` | `cmap_in_file` |
| `/forcefield/cmap/type` | `int32[n]` | `cmap_in_file` |
| `/forcefield/cmap/resolution` | `int32[n_type]` | `cmap_in_file` |
| `/forcefield/cmap/grid_value` | `float32[sum(resolution^2)]` | `cmap_in_file` |
| `/forcefield/gb/params` | `float32[N,2]`, columns `[radius, scale_factor]` | `gb_in_file` |
| `/forcefield/virtual_atom/type` | `int32[n_vatom]` | `virtual_atom_in_file` |
| `/forcefield/virtual_atom/atom` | `int32[n_vatom]` | `virtual_atom_in_file` |
| `/forcefield/virtual_atom/from_offset` | `int64[n_vatom+1]` CSR | `virtual_atom_in_file` |
| `/forcefield/virtual_atom/from` | `int32[*]` | `virtual_atom_in_file` |
| `/forcefield/virtual_atom/parameter_offset` | `int64[n_vatom+1]` CSR | `virtual_atom_in_file` |
| `/forcefield/virtual_atom/parameter` | `float32[*]` | `virtual_atom_in_file` |
| `/forcefield/lj_soft_core/atom_type_A` | `int32[N]` | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/atom_type_B` | `int32[N]` | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/atom_type_count_A` | `int32` | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/atom_type_count_B` | `int32` | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/pair_AA` | `float32[nA_pair]`, runtime-ready `A_12` values | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/pair_AB` | `float32[nA_pair]`, runtime-ready `B_6` values | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/pair_BA` | `float32[nB_pair]`, runtime-ready `A_12` values | `LJ_soft_core_in_file` |
| `/forcefield/lj_soft_core/pair_BB` | `float32[nB_pair]`, runtime-ready `B_6` values | `LJ_soft_core_in_file` |
| `/forcefield/subsys_division` | `int32[N]` | `subsys_division_in_file` |

Each force-field subgroup must define a minimal local schema:

| HDF5 path/attr | Type | Meaning |
|---|---|---|
| `<group>/schema_version` | string/int | Module-local layout version. |
| `<group>/count` | int64 | Number of interaction records when applicable. |
| `<group>/unit/*` | string attrs/datasets | Unit names for numeric columns. |
| `<group>/source_hash` | string | Optional hash of the canonical subgroup payload. |

Native readers must materialize directly into the existing runtime structures
from these arrays. If an interaction type still lacks a native reader, the
container may advertise that fact in `/capabilities/missing_native_reader`, and
input validation must fail for launches requiring that interaction type instead
of falling back to legacy text.

#### Special Force-Field Resources

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/manybody/eam/tables/<name>/x` | `float32[n]` | `EAM_in_file` |
| `/manybody/eam/tables/<name>/y` | `float32[n]` | `EAM_in_file` |
| `/manybody/eam/atom_type` | `int32[N]` or string table refs | `EAM_atom_type_in_file` |
| `/manybody/sw/records` | compound or columnar typed datasets | `SW_in_file` |
| `/manybody/edip/records` | compound or columnar typed datasets | `EDIP_in_file` |
| `/manybody/tersoff/records` | compound or columnar typed datasets | `TERSOFF_in_file` |
| `/manybody/reaxff/parameters/<section>/<field>` | typed numeric/string tables | `REAXFF_in_file` |
| `/manybody/reaxff/type` | `int32[N]` or string table refs | `REAXFF_type_in_file` |
| `/qc/type` | `int32[N]` or enum table refs | `qc_type_in_file`, if QM region/type definition is considered topology-intrinsic. |
| `/forcefield/custom_force/pairwise/<force_name>/atom_type_pairs` | `int32[n,2]` or string refs | `pairwise_force_in_file`. |
| `/forcefield/custom_force/pairwise/<force_name>/parameters` | `float32[n,p]` plus field names | `<force_name>_in_file` from pairwise config. |
| `/forcefield/custom_force/listed/<name>/atoms` | `int32[n,k]` | listed force definition config. |
| `/forcefield/custom_force/listed/<name>/parameters` | `float32[n,p]` plus field names | `<listed_force>_in_file`. |

For many-body and custom-force modules that still use a textual DSL internally,
the native H5 contract is still not the legacy file text. The canonical payload
is either typed HDF5 tables or a versioned module DSL object:

```text
/<module>/<object>/dsl/name
/<module>/<object>/dsl/schema_version
/<module>/<object>/dsl/ast_nodes
/<module>/<object>/dsl/numeric_tables
```

Raw legacy text may be stored only in `/compatibility/legacy_import/<key>` for
round-trip diagnostics and must not be used by the native bundle reader.

Custom forces are force-field extensions and belong to `topology.spgt.h5`. In
the current implementation, `pairwise_force` extends the nonbonded Hamiltonian
with typed pair parameters and optional direct electrostatic code, while
`listed_forces` can also define listed interactions and modify connectivity or
constraint-distance data through `connected_atoms` and `constrain_distance`.
Walls are not classified here; wall and confinement definitions belong to
`protocol.spgp.h5`.

### 2.3 `protocol.spgp.h5`: CV, Restraint, Bias, and Sampling Protocol

These inputs describe the algorithmic layer attached to the topology. They should
not contain restart coordinates, velocities, checkpoint state, or force-field
canonical data.

`protocol.spgp.h5` is also native HDF5-first. It must not use
`/parameters/restart/protocol_sidecars`, embedded legacy `cv_in_file` text, or
other `*_in_file` text blobs as canonical protocol content. Protocol modules
may keep imported source text under `/compatibility/legacy_import`, but native
launches read typed protocol objects from the groups below.

#### Root Metadata

| HDF5 path/attr | Type | Required | Constraint |
|---|---|---:|---|
| `/schema/name` | string | yes | e.g. `"sponge.protocol"` |
| `/schema/version` | int/string | yes | Supported by resolver. |
| `/identity/uuid` | string | yes | Protocol identity. |
| `/identity/content_hash` | string | yes | Hash of canonical protocol content. |
| `/protocol/topology_compatibility/topology_hash` | string | conditional | Required for resolved, topology-specific protocols. |
| `/protocol/cv_count` | int64 | no | Resolver-checkable metadata. |
| `/protocol/restraint_count` | int64 | no | Resolver-checkable metadata. |
| `/protocol/enhanced_sampling/method` | string | no | If protocol uses enhanced sampling. |
| `/protocol/enhanced_sampling/state_schema_version` | string/int | no | Bias-state compatibility. |

#### CV Definitions

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/cv/<name>/type` | string attr/dataset | `<cv>_CV_type` |
| `/cv/<name>/schema_version` | string/int | CV module layout version. |
| `/cv/<name>/dimension` | int64 | Number of scalar components produced. |
| `/cv/<name>/selection_expression` | string | CV atom selection expression if unresolved. |
| `/cv/<name>/atom_indices` | `int32[*]` | `<cv>_atom_in_file`, resolved map tied to topology hash. |
| `/cv/<name>/parameter/<field>` | typed numeric/string datasets | `<cv>_parameter_in_file`, tabulated/other static CV parameter. |
| `/cv/<name>/period` | `float32[dimension]` | `CV_period` style CV parameter. |
| `/cv/<name>/sigma` | `float32[dimension]` | `CV_sigma` style CV parameter. |
| `/cv/<name>/rotate` | bool attr | `RMSD_rotate` |
| `/cv/<name>/function` | string attr/text | `combine_function` |
| `/cv/<name>/min_padding` | float attr | `tabulated_min_padding` |
| `/cv/<name>/max_padding` | float attr | `tabulated_max_padding` |

Dynamic CV parameter files (`<cv>_<parameter>_in_file`) must be classified by
semantic meaning. Static parameters belong to `protocol.spgp.h5`; reference
coordinates belong to `restart.spgr.h5`.

#### Restraints

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/restraint/<name>/enabled_default` | bool attr | Optional protocol default. |
| `/restraint/<name>/type` | enum string | harmonic positional, RMSD, CV restraint, etc. |
| `/restraint/<name>/schema_version` | string/int | Restraint module layout version. |
| `/restraint/<name>/atom_indices` | `int32[n]` | `restrain_atom_id` |
| `/restraint/<name>/selection_expression` | string | Future unresolved selection expression. |
| `/restraint/<name>/weight` | `float32[n,3]` | `restrain_weight_in_file` |
| `/restraint/<name>/single_weight_default` | float attr | `restrain_single_weight`, only if part of reusable protocol. |
| `/restraint/<name>/refcoord_scaling_default` | enum attr | `restrain_refcoord_scaling`, only if part of reusable protocol. |
| `/restraint/<name>/calc_virial_default` | bool attr | `restrain_calc_virial`, only if part of reusable protocol. |

Reference coordinates for restraints belong to `restart.spgr.h5`, not `protocol.spgp.h5`,
unless they are truly intrinsic to the prepared physical topology, in which case
they may be stored in `topology.spgt.h5` as topology reference data and copied/linked by
the resolver as derived state.

#### Protocol Constraints

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/constraint/<name>/schema_version` | string/int | Constraint protocol layout version. |
| `/constraint/<name>/pairs/atoms` | `int32[n,2]` | `constrain_in_file` |
| `/constraint/<name>/pairs/r0` | `float32[n]` | `constrain_in_file` |

The current `constrain_in_file` adds extra distance constraint pairs for the
constraint protocol. It is not the SHAKE/SETTLE method selector and should not be
treated as intrinsic molecular topology unless the constraints originate from a
force-field/topology conversion.

#### Enhanced Sampling and Bias Definitions

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/sits/method/mode` | enum/string | `SITS_mode` if protocol-stable. |
| `/sits/method/k_numbers` | int64 | `SITS_k_numbers` if protocol-stable. |
| `/sits/method/temperature_ladder` | `float32[*]` or expression | `SITS_T`, `SITS_T_low`, `SITS_T_high` if protocol-stable. |
| `/sits/method/pe_a`, `/sits/method/pe_b` | float | `SITS_pe_a`, `SITS_pe_b` if protocol-stable. |
| `/sits/method/fb_interval`, `/sits/method/fb_bias` | int/float | SITS policy if protocol-stable. |
| `/sits/atom_indices` | `int32[*]` | `SITS_atom_in_file` |
| `/sits/atom_numbers_policy` | int/enum | `SITS_atom_numbers` if reused as protocol definition. |
| `/meta/<name>/cv_refs` | string/list | `meta_CV` |
| `/meta/<name>/schema_version` | string/int | Metadynamics protocol layout version. |
| `/meta/<name>/ndim` | int64 | `meta_Ndim` |
| `/meta/<name>/sigma` | `float32[ndim]` | `meta_CV_sigma` |
| `/meta/<name>/period` | `float32[ndim]` | `meta_CV_period` |
| `/meta/<name>/grid/min` | `float32[ndim]` | `meta_CV_minimal` |
| `/meta/<name>/grid/max` | `float32[ndim]` | `meta_CV_maximum` |
| `/meta/<name>/grid/count` | `int64[ndim]` | `meta_CV_grid` |
| `/meta/<name>/hill_height_default` | float | `meta_height` if protocol-stable. |
| `/meta/<name>/cutoff` | `float32[ndim]` | `meta_cutoff` if protocol-stable. |
| `/meta/<name>/method_flags` | attrs | `mask`, `sink`, `subhill`, `kde`, `dip`, `convmeta`, `grw` if protocol-stable. |
| `/meta/<name>/sumhill_freq_default` | int64 | `meta_sumhill_freq` if protocol-stable. |
| `/steer/cv_refs` | string/list | `steer_CV` from `cv_in_file`. |
| `/steer/weight` | `float32[n_cv]` | `steer_weight` from `cv_in_file`. |

The canonical representation of a protocol is a graph of typed objects:

```text
/cv/<name>/*
/restraint/<name>/*
/constraint/<name>/*
/sits/*
/meta/<name>/*
/steer/*
```

References between protocol objects are by stable object name plus optional
content hash, not by command-file order. A native protocol reader must validate
that `cv_refs` point to existing `/cv/<name>` groups and that dimensions match
before runtime initialization.

Accumulated bias potential, scatter state, edge state, and sumhill history
belong to `restart.spgr.h5`.

Current `steer` input is easy to miss with a top-level mdin scan because it has
no independent `steer_in_file` and is not read from the global `CONTROLLER`
namespace. `STEER_CV::Initial` asks the `COLLECTIVE_VARIABLE_CONTROLLER` for
`steer_CV` and `steer_weight` after `cv_in_file` has been loaded. The current
implementation is a static linear CV bias, `E = sum_i weight_i * CV_i`; it does
not own restartable history state. If future steering modes add time-dependent
targets, protocols, or accumulated work histories, static steering definitions
belong here, while path-dependent accumulated state belongs to
`restart.spgr.h5`.

#### Wall and Confinement Protocol Definitions

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/wall/hard/bounds_low` | `float32[3]` with nullable/infinite entries | `hard_wall_x_low`, `hard_wall_y_low`, `hard_wall_z_low`. |
| `/wall/hard/bounds_high` | `float32[3]` with nullable/infinite entries | `hard_wall_x_high`, `hard_wall_y_high`, `hard_wall_z_high`. |
| `/wall/hard/allow_npt` | bool attr, normally `false` | Current implementation rejects hard wall in NPT mode. |
| `/wall/soft/config` | UTF-8 text or structured sections | `soft_walls_in_file`. |
| `/wall/soft/<name>/potential` | UTF-8 text or expression | `potential` section from `soft_walls_in_file`. |

Walls are treated as boundary/confinement protocol, not as topology-owned
force-field data. `hard_wall` is an integration-time reflection boundary rather
than a conventional potential term. `soft_walls` defines external coordinate
potentials over all atoms and is best grouped with reproducible confinement
protocols.

### 2.4 `restart.spgr.h5`: Concrete Runtime State

These inputs are state-bearing and should become canonical state in `restart.spgr.h5`.

#### Root Metadata

| HDF5 path/attr | Type | Required | Constraint |
|---|---|---:|---|
| `/schema/name` | string | yes | e.g. `"sponge.restart"` |
| `/schema/version` | int/string | yes | Supported by resolver. |
| `/identity/uuid` | string | yes | Run identity. |
| `/identity/content_hash` | string | no | Optional whole-container hash. |
| `/run/topology_hash` | string | yes | Must match bound topology. |
| `/run/producer_protocol_hash` | string | no | Provenance only; not a global compatibility lock. |
| `/run/state_type` | enum string | yes | `initial`, `restart`, `checkpoint`, etc. |
| `/run/current_step` | int64 | yes | `>=0` |
| `/run/current_time` | float64 | yes | `>=0` |
| `/run/parent_run_uuid` | string | no | Lineage. |
| `/run/state_hash` | string | yes | Hash of the single restart state. |
| `/run/checkpoint_schema_version` | string/int | no | Continuation compatibility. |

#### State Data

`restart.spgr.h5` stores exactly one launchable restart state. It is a lightweight
H5MD-compatible HDF5 container, not a trajectory container. It should not contain
append-style frame history. Dense coordinate/velocity histories belong to
trajectory output and should not be transferred with cloud restart handoff paths.

Standard particle state uses the H5MD `particles` layout with exactly one frame.
Only SPONGE-specific or otherwise non-standard restart fields are stored under
`/parameters/restart`, following the H5MD/MindSPONGE extension convention.

```text
/h5md/
/particles/all/
/parameters/restart/
```

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/particles/all/step` | `int64[1]` | shared restart step. |
| `/particles/all/time` | `float64[1]` | shared restart time. |
| `/particles/all/position/step` | hard link to `/particles/all/step` | coordinate/restart metadata |
| `/particles/all/position/time` | hard link to `/particles/all/time` | coordinate/restart metadata |
| `/particles/all/position/value` | `float32[1,N,3]` | `coordinate_in_file` |
| `/particles/all/velocity/step` | hard link to `/particles/all/step` | coordinate/restart metadata |
| `/particles/all/velocity/time` | hard link to `/particles/all/time` | coordinate/restart metadata |
| `/particles/all/velocity/value` | `float32[1,N,3]` | `velocity_in_file` |
| `/particles/all/box` attrs | `dimension=3`, `boundary[3]` | PBC metadata. |
| `/particles/all/box/edges/step` | hard link to `/particles/all/step` | box/restart metadata |
| `/particles/all/box/edges/time` | hard link to `/particles/all/time` | box/restart metadata |
| `/particles/all/box/edges/value` | `float32[1,3,3]` | native coordinate tail or restart metadata |
| `/particles/all/species` | `int32[N]` | optional H5MD-compatible species/type index cache; canonical atom identity remains in `topology.spgt.h5`. |
| `/parameters/restart/rng_state/<module>` | native byte/string engine state | Reproducibility state, module-versioned. |
| `/parameters/restart/integrator_state/<field>` | scalar/string datasets | Integrator continuation metadata. |
| `/parameters/restart/thermostat/nose_hoover_chain` | `float32[nchain,2]` | `nose_hoover_chain_restart_input` |
| `/parameters/restart/thermostat/<module>/<field>` | typed datasets | Thermostat continuation state. |
| `/parameters/restart/barostat/<module>/<field>` | typed datasets | Barostat continuation state. |

For PIMD or replica-style restart states, use additional H5MD particle groups
such as `/particles/replica0`, `/particles/replica1`, or a documented
SPONGE/MindSPONGE module convention. Do not put standard particle coordinates or
velocities in `/parameters/restart` unless representing a non-H5MD extension
that cannot be expressed as an H5MD particle element.

`step` and `time` should not be physically duplicated for every element. Store
one shared dataset per particle group, then expose element-local `step` and
`time` entries as HDF5 hard links. This follows the H5MD convention used by
MindSPONGE trajectory files while keeping the restart file lightweight.

#### Restart Load Policy

The restart container stores one launchable state, but that state is
componentized. `[input.h5.restart] load` selects which components are restored.
The parser-visible flattened key is `input_h5_restart_load`.

| Value | Meaning |
|---|---|
| `structural` | Load coordinates, box, step/time, and velocity when required by launch policy. This is the default and cannot be disabled. |
| `dynamic` | Load `structural` plus compatible engine dynamic state such as integrator, thermostat, barostat, and RNG state. |
| `protocol` | Load `structural` plus compatible protocol-owned continuation state such as CV/restraint references, SITS state, and metadynamics state. |
| `full` | Load `structural`, `dynamic`, and `protocol` components. |
| `custom` | Load `structural` plus an explicit future component list. Until component-list keys are implemented, `custom` is reserved. |

`fresh` is intentionally invalid. A launch must always start from one concrete
state in the bound `restart.spgr.h5`; a newly built initial configuration should
first be materialized as an initial restart container.

#### Reference and Adaptive State

The native restart contract stores protocol-owned continuation state as typed
HDF5 groups. It must not inline legacy `SITS_nk_in_file`, `meta_*_in_file`,
`restrain_coordinate_in_file`, or CV reference files as raw text. A compatibility
importer may preserve original text under `/compatibility/legacy_import`, but
`load = protocol` and `load = full` read the canonical groups below.

| HDF5 path | Type/shape | Existing input |
|---|---|---|
| `/parameters/restart/references/restraint/<name>/coordinate` | `float32[N,3]` | `restrain_coordinate_in_file`, `restrain_amber_rst7` |
| `/parameters/restart/references/restraint/<name>/atom_indices` | `int32[n]` optional copy/cache | `restrain_atom_id` compatibility |
| `/parameters/restart/references/restraint/<name>/state_hash` | string | Reference-state compatibility. |
| `/parameters/restart/references/cv/<name>/coordinate` | `float32[n,3]` | `<cv>_coordinate_in_file`, RMSD reference |
| `/parameters/restart/references/cv/<name>/value` | `float32[*]` | Materialized CV reference value. |
| `/parameters/restart/references/cv/<name>/state_hash` | string | Reference-state compatibility. |
| `/parameters/restart/bias/sits/<module>/nk` | `float32[k]` | `SITS_nk_in_file` |
| `/parameters/restart/bias/sits/<module>/log_norm` | `float32[k]` | SITS continuation state. |
| `/parameters/restart/bias/sits/<module>/log_nk` | `float32[k]` | SITS continuation state. |
| `/parameters/restart/bias/sits/<module>/schema_version` | string/int | SITS state layout version. |
| `/parameters/restart/bias/meta/<name>/grid/min` | `float32[ndim]` | `meta_edge_in_file`, protocol-compatible copy. |
| `/parameters/restart/bias/meta/<name>/grid/max` | `float32[ndim]` | `meta_edge_in_file`, protocol-compatible copy. |
| `/parameters/restart/bias/meta/<name>/grid/count` | `int64[ndim]` | `meta_edge_in_file`, protocol-compatible copy. |
| `/parameters/restart/bias/meta/<name>/potential/value` | `float32[grid_count...]` | `meta_potential_in_file` |
| `/parameters/restart/bias/meta/<name>/scatter/position` | `float32[n,ndim]` | `meta_scatter_in_file` |
| `/parameters/restart/bias/meta/<name>/scatter/weight` | `float32[n]` | `meta_scatter_in_file` |
| `/parameters/restart/bias/meta/<name>/hills/step` | `int64[n_hill]` | `myhill.log` |
| `/parameters/restart/bias/meta/<name>/hills/time` | `float64[n_hill]` | `myhill.log` |
| `/parameters/restart/bias/meta/<name>/hills/center` | `float32[n_hill,ndim]` | `myhill.log` |
| `/parameters/restart/bias/meta/<name>/hills/height` | `float32[n_hill]` | `myhill.log` |
| `/parameters/restart/bias/meta/<name>/hills/sigma` | `float32[n_hill,ndim]` | `myhill.log` |
| `/parameters/restart/bias/meta/<name>/history/<field>` | typed datasets | `history.log`, sinkmeta history. |
| `/parameters/restart/protocol_state/cv/<name>/<field>` | typed datasets | CV module continuation state. |
| `/parameters/restart/protocol_state/restraint/<name>/<field>` | typed datasets | Restraint module continuation state. |

`/parameters/restart/protocol_sidecars/<command_key>` is deprecated as a
canonical state path. It is allowed only in the compatibility bridge used to
launch older modules before their native H5 readers exist. New writers must
prefer the typed paths above, and validation should fail native-mode launches
when the required state exists only as a protocol sidecar text payload.

#### Resolved Launch Snapshot

For provenance and debugging, a run launch may append:

```text
/resolved_inputs/<launch_id>/
```

Suggested fields:

| HDF5 path | Type |
|---|---|
| `/resolved_inputs/<launch_id>/mdin_hash` | string |
| `/resolved_inputs/<launch_id>/topology_hash` | string |
| `/resolved_inputs/<launch_id>/protocol_hash` | string |
| `/resolved_inputs/<launch_id>/state_hash` | string |
| `/resolved_inputs/<launch_id>/resolved_policy` | JSON/text |
| `/resolved_inputs/<launch_id>/engine_version` | string |
| `/resolved_inputs/<launch_id>/command_line` | string |

This snapshot is derived/cache data, not a human-editable input source.

### 2.5 `trajectory.spg.h5md`: Rerun and Trajectory-Analysis Input

Trajectory H5MD input is a history container, not a launchable restart
container. It is used when `mode = rerun` or a future analysis mode needs to
stream frames through the force/evaluation pipeline.

The canonical logical layout follows the output H5MD trajectory contract:

```text
/h5md/
/particles/<stream>/
/observables/
/parameters/sponge/
```

For ordinary rerun, the default stream is `/particles/all`. A different stream
may be selected by `input_h5_trajectory_particle_stream`.

| HDF5 path | Type/shape | Existing input it replaces |
|---|---|---|
| `/particles/<stream>/step` | `int64[n_frame]` | implicit legacy frame index |
| `/particles/<stream>/time` | `float64[n_frame]` | optional legacy timing |
| `/particles/<stream>/position/value` | `float32[n_frame,N,3]` | `crd` binary trajectory |
| `/particles/<stream>/box/edges/value` | `float32[n_frame,3,3]` | `box` text trajectory |
| `/particles/<stream>/velocity/value` | `float32[n_frame,N,3]`, optional | `vel` binary trajectory |
| `/particles/<stream>/force/value` | `float32[n_frame,N,3]`, optional | future force rerun input or analysis input |

`position/value` and `box/edges/value` are required for rerun. Velocity is
optional and should be loaded only when present and requested by the consuming
mode. Force frames are not required for ordinary rerun force recomputation.

The H5MD trajectory input may be a single H5MD file or a VDS wrapper produced by
the output bundle writer. In VDS mode, the reader consumes the wrapper path from
`input_h5_trajectory_path`; shard paths remain writer-derived internals and must
not appear as separate `run.mdin` fields.

The rerun reader maps `rerun_start`, `rerun_strip`, and `rerun_frame_limit` to
the selected H5MD frame axis. End-of-trajectory handling must match legacy rerun:
when a required frame cannot be read completely, the run stops at the last
complete consumed frame.

## 3. Compatibility Checks

The resolver must hard-error on incompatible bindings unless `run.mdin`
explicitly requests a supported repair/conversion.

### `topology.spgt.h5`

- Supported schema version
- Atom count exists and is positive
- Atom ordering hash exists
- Topology hash exists
- Force-field hash exists
- Parameter tables are complete

### `protocol.spgp.h5` vs `topology.spgt.h5`

- Protocol topology hash matches, or unresolved selections can be resolved against the topology
- CV atom selections resolve to valid atoms
- Restraint selections resolve to valid atoms
- Enhanced sampling method is compatible with selected CVs
- CV dimensions match any continuation bias state dimensions

### `restart.spgr.h5` vs `topology.spgt.h5`

- Restart atom count equals topology atom count
- Atom ordering hash matches topology atom ordering hash
- Box/PBC state is compatible with topology and selected ensemble
- Velocity existence matches requested launch mode
- Checkpoint/integrator state is compatible with selected integrator policy
- Default snapshot exists and contains required structural state

### `restart.spgr.h5` vs `protocol.spgp.h5`

`restart.spgr.h5` is not globally locked to one protocol. The producer protocol
hash is provenance by default. Protocol compatibility is checked only for
protocol-owned state components requested by the component load policy.

- Bias state method matches protocol enhanced sampling method, if bias state is loaded
- Bias CV dimension matches protocol CV definition, if bias state is loaded
- Bias grid metadata matches protocol CV domain, if bias state is loaded
- Adaptive state schema version matches protocol adaptive policy version, if adaptive state is loaded
- Restraint/CV reference state matches the selected protocol definition, if used

### `trajectory.spg.h5md` vs `topology.spgt.h5` for rerun

- Bound trajectory file exists and has a supported H5MD/output schema
- Selected particle stream exists; default stream is `all`
- Position and box datasets exist for the selected stream
- Position atom count equals topology atom count
- Atom ordering hash matches when recorded; otherwise the resolver must require
  an explicit compatibility override or keep legacy behavior
- Frame count is sufficient for `rerun_start`, `rerun_strip`, and
  `rerun_frame_limit`, or end-of-trajectory behavior is explicitly accepted
- Velocity dataset shape matches position frames and atom count when velocity is
  loaded
- Box/PBC frame data is compatible with the selected rerun mode and
  `rerun_need_box_update`
- VDS wrappers must resolve to readable, complete shard mappings; incomplete
  shard manifests require an explicit repair/complete-prefix policy

### `run.mdin` vs Containers

- Bound files exist
- Bound files have supported schemas
- Bound UUID/hash metadata is compatible
- Requested MD `mode` is valid for the current SPONGE parser: nve/nvt/npt/minimization/rerun
- Requested ensemble is valid for available state
- Requested state load policy is valid and never omits required coordinates/box
- Temperature/pressure policies are valid
- Seed policy is deterministic or explicitly random

## 4. Runtime Assembly

Startup flow:

```text
1. Read run.mdin
2. Resolve [input.h5.topology] path and load topology.spgt.h5
3. Resolve [input.h5.protocol] path and load protocol.spgp.h5
4. Resolve [input.h5.restart] path and open restart.spgr.h5
5. Validate compatibility
6. Load restart components selected by [input.h5.restart] load
7. Assemble RuntimeInput in memory
8. Execute MD
9. Write a new restart.spgr.h5 state or replace/update the bound restart state,
   depending on launch policy
```

Additional rerun trajectory flow:

```text
1. Read run.mdin
2. Load topology/protocol/restart context as required by the run
3. If mode = rerun and input_h5_trajectory_path is set, open the H5MD trajectory
   input; otherwise use legacy crd/box/vel inputs
4. Validate trajectory/topology compatibility
5. For each rerun iteration, read the selected trajectory frame into RuntimeInput
6. Recompute forces/observables and write requested outputs
```

Resolution priority for legacy compatibility:

```text
[input.h5.*] bindings and explicit run.mdin launch overrides
> selected restart components from restart.spgr.h5
> legacy default_in_file_prefix
> old missing/default behavior
```

TOML examples must remain compatible with the existing SPONGE parser. Current
SPONGE TOML support flattens tables into underscore-separated keys. Grouped
TOML is therefore allowed and preferred when the flattened key is an existing
SPONGE command. For example, `[write] trajectory_interval` becomes
`write_trajectory_interval`, and `[thermostat] mode` becomes
`thermostat_mode`. The SPONGE-distributed schema under `schemas/` is the
authoritative schema source for editor integration and downstream tools such as
Mokda.

The grouping template should follow the existing Mokda schema style: use module
or command-prefix tables such as `[write]`, `[thermostat]`, `[barostat]`,
`[constrain]`, `[SHAKE]`, `[restrain]`, `[PM]`, `[gb]`, `[neighbor_list]`,
`[hard_wall]`, `[SITS]`, `[REAXFF]`, `[DOM_DEC]`, `[minimization]`, and
`[rerun]`. H5 input containers are grouped under `[input.h5.*]`, and H5 output
containers are grouped under `[output.h5.*]`. The write cadence exception is
`[write.interval]`, a parser-supported TOML alias group normalized to the
existing `write_*_interval` commands. Monte Carlo barostat parameters may use
`[barostat.monte_carlo]`, a parser-supported alias group normalized to existing
`monte_carlo_barostat_*` commands. Do not invent semantic aliases whose
flattened names are not parser commands unless the parser normalizes them, such
as `[run] steps` or `[output] trajectory_interval`.

Use existing mdin keys for run policy. TOML top-level run keys such as `mode`,
`step_limit`, `dt`, `target_temperature`, and `target_pressure` must appear
before the first table header; otherwise TOML assigns them to the preceding
table and the flattened keys become wrong. H5 input container bindings must be
registered and consumed by the resolver before the normal unused-input check:

```toml
mode = "npt"
step_limit = 2500000
dt = 0.002
target_temperature = 300.0
target_pressure = 1.0

[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/metadyn.protocol.spgp.h5"

[input.h5.restart]
path = "runs/prod_0007.restart.spgr.h5"
load = "structural"

[thermostat]
mode = "middle_langevin"
seed = 123456
tau = 1.0

[barostat]
mode = "monte_carlo_barostat"

[barostat.monte_carlo]
update_interval = 100

[output.h5.trajectory]
path = "prod.spg.h5md"
vds = true
chunk_size = 20
repair_policy = "strict"

[output.h5.restart]
path = "prod.spgr.h5"

[output.h5.observable]
path = "prod.obs.spg.h5md"

[write.interval]
trajectory = 5000
information = 500
mdout = 500
restart = 500000
```

Rerun with H5MD trajectory input:

```toml
mode = "rerun"

[input.h5.topology]
path = "topologies/protein.topology.spgt.h5"

[input.h5.protocol]
path = "protocols/analysis.protocol.spgp.h5"

[input.h5.restart]
path = "runs/prod_0007.restart.spgr.h5"
load = "protocol"

[input.h5.trajectory]
path = "runs/prod_0007.spg.h5md"
particle_stream = "all"

[rerun]
frame_limit = 1000
start = 0
strip = 0
need_box_update = true

[write.interval]
mdout = 1
```

This example intentionally does not set `crd`, `box`, or `vel`; those keys are
legacy rerun inputs and are mutually exclusive with `input_h5_trajectory_path`
unless a future resolver defines an explicit override policy.

`run.mdin` must not contain direct HDF5 paths such as
`restart.spgr.h5:/particles/all/position/value` or
`trajectory.spg.h5md:/particles/all/position/value`, and it must not contain
symbolic HDF5 handles such as `$RUN.latest_state`.

## 5. Migration Notes for Current SPONGE

- Native topology and force-field files map mainly to `topology.spgt.h5`.
- `coordinate_in_file` and `velocity_in_file` map to `restart.spgr.h5`.
- AMBER and GROMACS compatibility inputs are out of scope for this container
  specification version.
- `cv_in_file` and most CV/restrain/enhanced-sampling definitions map to
  `protocol.spgp.h5`.
- Restraint atom IDs and weights are protocol definitions; restraint reference
  coordinates are restart state unless explicitly intrinsic to the prepared topology.
- SITS atom selection is protocol definition; SITS Nk/log-normalization is run
  adaptive state.
- Metadynamics grid/domain/method definition is protocol data; accumulated
  potential/scatter/edge/sumhill is restart state.
- Rerun `crd`, `box`, and `vel` inputs map to a trajectory H5MD input such as
  `trajectory.spg.h5md`, not to `restart.spgr.h5`.
- `restart.spgr.h5` remains a single-state launch container; it must not absorb
  dense trajectory history just to support rerun.
- Canonical H5 outputs are selected with `[output.h5.trajectory]`,
  `[output.h5.restart]`, and `[output.h5.observable]`; legacy output paths are
  compatibility sidecars and are disabled by default when canonical H5 output is
  enabled.
- `run.mdin` remains the human edit point for launch policy, but not for
  canonical topology, protocol definition, or state arrays.
