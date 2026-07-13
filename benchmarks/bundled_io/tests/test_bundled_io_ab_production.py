from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pytest

from benchmarks.bundled_io.ab_contracts import (
    AssertionEvidence,
    build_case_evidence,
    load_contract_registry,
    update_evidence_report,
    validate_complete_evidence_report,
    validate_contract_registry,
)
from benchmarks.bundled_io.ab_statistics import (
    StatisticalEquivalencePolicy,
    compare_replicas,
)
from benchmarks.utils import Extractor
from benchmarks.validation.thermostat.tests.utils import (
    read_mass_values,
    write_velocity_file_for_temperature,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "h5_bundle" / "fixtures" / "input_matrix"
XPONGE_DEV_ROOT = REPO_ROOT.parent / "XPONGE"
SITS_FF19SB_CMAP_FIXTURE = "sits_ff19sb_cmap_peptide"
SITS_FF19SB_CMAP_SOURCE = (
    REPO_ROOT / "benchmarks" / "performance" / "sits" / "statics" / "ala2_sits"
)
PROFILE = os.environ.get("SPONGE_BUNDLED_IO_AB_PROFILE", "medium").strip()
EVIDENCE_RUN_ID = os.environ.get(
    "SPONGE_BUNDLED_IO_AB_RUN_ID", f"{os.getpid()}-{time.time_ns()}"
)

PROFILE_LIMITS = {
    "medium": {
        "normal_step_limit": 1000,
        "normal_interval": 20,
        "normal_replicas": 10,
        "normal_burn_in_frames": 10,
        "normal_block_size": 8,
        "normal_relative_margin": 5.0e-2,
        "normal_absolute_margin": 1.5e-1,
        "rerun_frame_limit": 2,
    },
    "production": {
        "normal_step_limit": 10000,
        "normal_interval": 100,
        "normal_replicas": 5,
        "normal_burn_in_frames": 20,
        "normal_block_size": 10,
        "normal_relative_margin": 1.0e-2,
        "normal_absolute_margin": 5.0e-2,
        "rerun_frame_limit": 2,
    },
}

STATISTICAL_CONFIDENCE_Z = 3.0
STATISTICAL_MAXIMUM_STD_RATIO = 1.5
STATISTICAL_MINIMUM_BLOCKS_PER_REPLICA = 4

FULL_CONTRACT_INPUT_REQUIRED_PATHS = {
    "topology.spgt.h5": {
        "/atoms/mass",
        "/atoms/charge",
        "/atoms/residue_index",
        "/forcefield/bond/atoms",
        "/forcefield/angle/atoms",
        "/forcefield/dihedral/atoms",
        "/forcefield/cmap/grid_value",
        "/forcefield/custom_force/pairwise",
        "/forcefield/custom_force/pairwise/data/custom_pair",
        "/forcefield/custom_force/listed",
        "/forcefield/custom_force/listed/data/custom_bond",
        "/manybody/eam/atom_type",
        "/manybody/sw/atom_type",
        "/manybody/edip/atom_type",
        "/manybody/tersoff/atom_type",
        "/manybody/reaxff/parameters",
        "/manybody/reaxff/type",
        "/qc/type",
    },
    "protocol.spgp.h5": {
        "/cv/config/section/name",
        "/constraint/default/pairs/atoms",
        "/sits/atom_indices",
        "/restraint/config/section/name",
        "/restraint/cv/config/section/name",
        "/restraint/default/atom_indices",
        "/restraint/default/weight",
        "/meta/default/grid",
        "/wall/soft/potential",
        "/steer/config/section/name",
    },
    "restart.spgr.h5": {
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
        "/parameters/restart/thermostat/nose_hoover_chain",
        "/parameters/restart/bias/sits/SITS/nk",
        "/parameters/restart/bias/meta/default/potential_export",
        "/parameters/restart/bias/meta/default/scatter",
        "/parameters/restart/bias/meta/default/hills",
        "/parameters/restart/references/restraint/default/coordinate",
        "/parameters/restart/protocol_sidecars/cv_in_file",
        "/parameters/restart/protocol_sidecars/SITS_in_file",
        "/parameters/restart/protocol_sidecars/meta_potential_in_file",
    },
    "trajectory.spg.h5md": {
        "/particles/all/position/value",
        "/particles/all/velocity/value",
        "/particles/all/box/edges/value",
        "/particles/all/step",
        "/particles/all/time",
    },
}

H5_COMPARE_DATASETS = (
    "/parameters/sponge/output/frame_count",
    "/parameters/sponge/output/last_complete_step",
    "/parameters/sponge/output/last_complete_time",
    "/particles/all/step",
    "/particles/all/time",
    "/particles/all/position/value",
    "/particles/all/box/edges/value",
)

RESTART_COMPARE_DATASETS = (
    "/particles/all/position/value",
    "/particles/all/velocity/value",
    "/particles/all/box/edges/value",
)

TRAJECTORY_REL = Path("output") / "ab.spg.h5md"
OBSERVABLE_REL = Path("output") / "ab.obs.spg.h5md"
RESTART_REL = Path("output") / "ab.spgr.h5"


@dataclass(frozen=True)
class AbCase:
    name: str
    fixture_case: str
    legacy_subdir: str
    bundled_subdir: str
    mode: str
    vds: bool
    statistical_md: bool
    restart_load_policy: str
    contract_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]


@dataclass
class AbRun:
    replica_index: int
    replica_seed: int
    legacy_dir: Path
    bundled_dir: Path
    legacy_metrics: dict[str, object]
    bundled_metrics: dict[str, object]
    legacy_output_contract: dict[str, object]
    bundled_output_contract: dict[str, object]


@dataclass(frozen=True)
class H5VirtualAtomRecord:
    atom: int
    kind: int
    source_atoms: tuple[int, ...]
    parameters: tuple[float, ...]


def _cases_for_profile() -> list[AbCase]:
    cases = [
        AbCase(
            name="normal_core_h5_output",
            fixture_case="tip3p_validation_generated",
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=True,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.normal_md",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.restart",
                "output.trajectory.vds_off",
            ),
            assertion_ids=(
                "mdout_statistical_equivalence",
                "h5_statistical_equivalence",
            ),
        ),
        AbCase(
            name="normal_sits_ff19sb_cmap_peptide",
            fixture_case=SITS_FF19SB_CMAP_FIXTURE,
            legacy_subdir="generated_legacy",
            bundled_subdir="generated_bundled",
            mode="normal",
            vds=False,
            statistical_md=True,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.normal_md",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.restart",
                "output.trajectory.vds_off",
                "input.topology.cmap",
                "input.protocol.sits",
                "system.ff19sb_ace_ala_nme",
            ),
            assertion_ids=(
                "mdout_statistical_equivalence",
                "h5_statistical_equivalence",
                "cmap_runtime_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_pure_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
                "input.full_contract.inventory",
                "input.full_contract.pure_native",
                "input.restart_load.structural",
                "input.topology.cmap",
                "input.manybody.reaxff",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "cmap_runtime_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_pure_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
                "input.full_contract.inventory",
                "input.full_contract.pure_native",
                "input.restart_load.structural",
                "input.topology.cmap",
                "input.manybody.reaxff",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "cmap_runtime_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_sidecar_vds_off",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=False,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_off",
                "input.full_contract.inventory",
                "input.full_contract.sidecar",
                "input.restart_load.structural",
                "input.topology.cmap",
                "input.manybody.reaxff",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "cmap_runtime_equivalence",
            ),
        ),
        AbCase(
            name="rerun_full_contract_sidecar_vds_on",
            fixture_case="full_contract_rerun",
            legacy_subdir="legacy_input",
            bundled_subdir="bundled_input_with_legacy_sidecar/bundle",
            mode="rerun",
            vds=True,
            statistical_md=False,
            restart_load_policy="structural",
            contract_ids=(
                "runtime.rerun",
                "output.legacy.mdout",
                "output.trajectory",
                "output.observable",
                "output.trajectory.vds_on",
                "input.full_contract.inventory",
                "input.full_contract.sidecar",
                "input.restart_load.structural",
                "input.topology.cmap",
                "input.manybody.reaxff",
            ),
            assertion_ids=(
                "full_contract_input_inventory",
                "mdout_deterministic_equivalence",
                "h5_rerun_semantic_equivalence",
                "cmap_runtime_equivalence",
            ),
        ),
    ]
    return cases


@pytest.fixture(scope="module", autouse=True)
def _require_complete_evidence_for_full_matrix(request):
    yield
    if request.session.testsfailed:
        return
    selected_case_ids = {
        item.callspec.params["case"].name
        for item in request.session.items
        if hasattr(item, "callspec")
        and isinstance(item.callspec.params.get("case"), AbCase)
    }
    expected_case_ids = {case.name for case in _cases_for_profile()}
    if selected_case_ids != expected_case_ids:
        return
    validate_complete_evidence_report(
        _output_root() / "ab_evidence.json",
        load_contract_registry(),
        EVIDENCE_RUN_ID,
    )


@pytest.mark.parametrize(
    "case", _cases_for_profile(), ids=lambda case: case.name
)
def test_legacy_and_bundled_ab_behavior(case: AbCase):
    if PROFILE not in PROFILE_LIMITS:
        raise AssertionError(
            "SPONGE_BUNDLED_IO_AB_PROFILE must be one of "
            f"{sorted(PROFILE_LIMITS)}"
        )
    contracts = load_contract_registry()
    validate_contract_registry(contracts, _cases_for_profile())
    root = _output_root()
    case_root = root / case.name
    runs = []
    for replica_index in range(_replica_count(case)):
        replica_seed = _replica_seed(replica_index)
        replica_root = case_root / f"replica_{replica_index:02d}"
        legacy_dir, bundled_dir = _prepare_case_pair(
            case, replica_root, replica_seed
        )
        _prepare_mdin(
            legacy_dir,
            "mdin.spg.toml",
            case,
            branch="legacy",
            replica_seed=replica_seed,
        )
        _prepare_mdin(
            bundled_dir,
            "mdin.bundled.spg.toml",
            case,
            branch="bundled",
            replica_seed=replica_seed,
        )

        legacy_metrics = _run_sponge(legacy_dir, _mdin_name(legacy_dir))
        bundled_metrics = _run_sponge(bundled_dir, _mdin_name(bundled_dir))
        runs.append(
            AbRun(
                replica_index=replica_index,
                replica_seed=replica_seed,
                legacy_dir=legacy_dir,
                bundled_dir=bundled_dir,
                legacy_metrics=legacy_metrics,
                bundled_metrics=bundled_metrics,
                legacy_output_contract=_validate_branch_output_contract(
                    case, legacy_dir, branch="legacy"
                ),
                bundled_output_contract=_validate_branch_output_contract(
                    case, bundled_dir, branch="bundled"
                ),
            )
        )

    comparison, assertion_evidence = _compare_outputs(case, runs)
    evidence = build_case_evidence(contracts, case, assertion_evidence)
    metrics = {
        "profile": PROFILE,
        "case": case.name,
        "contract_ids": list(case.contract_ids),
        "assertion_ids": list(case.assertion_ids),
        "evidence": [record.as_dict() for record in evidence],
        "vds": case.vds,
        "statistical_md": case.statistical_md,
        "restart_load_policy": case.restart_load_policy,
        "replica_count": len(runs),
        "replicas": [
            {
                "index": run.replica_index,
                "seed": run.replica_seed,
                "legacy": run.legacy_metrics,
                "bundled": run.bundled_metrics,
                "legacy_output_contract": run.legacy_output_contract,
                "bundled_output_contract": run.bundled_output_contract,
            }
            for run in runs
        ],
        "comparison": comparison,
        "contract_boundary": {
            "cross_process_vds_reopen_append_resume": "unsupported",
            "vds_resume_covered_semantics": (
                "complete-prefix terminal tail repair and no-op policy path"
            ),
        },
    }
    metrics_path = case_root / "ab_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_evidence_report(
        root / "ab_evidence.json",
        contracts,
        case,
        evidence,
        {
            "profile": PROFILE,
            "replica_seeds": [run.replica_seed for run in runs],
            "vds": case.vds,
            "statistical_md": case.statistical_md,
            "restart_load_policy": case.restart_load_policy,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            "mpi_rank_count": os.environ.get("OMPI_COMM_WORLD_SIZE", "1"),
            "sponge_executable": str(_sponge_executable()),
            "metrics_path": str(metrics_path),
        },
        EVIDENCE_RUN_ID,
    )
    print(f"\nBundled I/O A/B metrics: {metrics_path}")


def test_sidecar_rerun_cmap_potential_is_branch_and_vds_invariant():
    cases = {
        case.vds: case
        for case in _cases_for_profile()
        if case.name.startswith("rerun_full_contract_sidecar_vds_")
    }
    if set(cases) != {False, True}:
        raise AssertionError(
            "sidecar rerun A/B cases must cover VDS off and on"
        )

    root = _output_root() / "sidecar_rerun_cmap_potential_regression"
    potential_by_branch: dict[str, dict[bool, list[float]]] = {
        "legacy": {},
        "bundled": {},
    }
    for vds, case in cases.items():
        case_root = root / ("vds_on" if vds else "vds_off")
        legacy_dir, bundled_dir = _prepare_case_pair(case, case_root, 20260709)
        _prepare_mdin(
            legacy_dir,
            "mdin.spg.toml",
            case,
            branch="legacy",
            replica_seed=20260709,
        )
        _prepare_mdin(
            bundled_dir,
            "mdin.bundled.spg.toml",
            case,
            branch="bundled",
            replica_seed=20260709,
        )
        _run_sponge(legacy_dir, _mdin_name(legacy_dir))
        _run_sponge(bundled_dir, _mdin_name(bundled_dir))

        legacy_mdout = _read_mdout(legacy_dir / "mdout.txt")
        bundled_mdout = _read_mdout(bundled_dir / "mdout.txt")
        _require_matching_mdout_columns(
            legacy_mdout, bundled_mdout, f"{case.name} potential regression"
        )
        if "potential" not in legacy_mdout["columns"]:
            raise AssertionError(f"{case.name} mdout is missing potential")
        legacy_potential = [row["potential"] for row in legacy_mdout["rows"]]
        bundled_potential = [row["potential"] for row in bundled_mdout["rows"]]
        if not any(math.isfinite(value) for value in legacy_potential):
            raise AssertionError(
                f"{case.name} legacy potential has no finite frame"
            )
        if not any(math.isfinite(value) for value in bundled_potential):
            raise AssertionError(
                f"{case.name} bundled potential has no finite frame"
            )
        _assert_numeric_sequences_close(
            f"{case.name} potential",
            legacy_potential,
            bundled_potential,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )
        potential_by_branch["legacy"][vds] = legacy_potential
        potential_by_branch["bundled"][vds] = bundled_potential

    for branch, potentials in potential_by_branch.items():
        _assert_numeric_sequences_close(
            f"sidecar rerun {branch} potential VDS off/on",
            potentials[False],
            potentials[True],
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )


def _output_root() -> Path:
    configured = os.environ.get("SPONGE_BUNDLED_IO_AB_OUTPUT_DIR")
    if configured:
        root = Path(configured)
        root.mkdir(parents=True, exist_ok=True)
        return root
    cached = os.environ.get("SPONGE_BUNDLED_IO_AB_RUN_ROOT")
    if cached:
        return Path(cached)
    root = Path(tempfile.mkdtemp(prefix=f"sponge_bundled_io_ab_{PROFILE}_"))
    os.environ["SPONGE_BUNDLED_IO_AB_RUN_ROOT"] = str(root)
    return root


def _copy_case(case: AbCase, branch: str, subdir: str, case_root: Path) -> Path:
    source = FIXTURE_ROOT / case.fixture_case / subdir
    destination = case_root / branch
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def _replica_count(case: AbCase) -> int:
    if not case.statistical_md:
        return 1
    return int(PROFILE_LIMITS[PROFILE]["normal_replicas"])


def _replica_seed(replica_index: int) -> int:
    return 20260709 + replica_index * 104729


def _prepare_case_pair(
    case: AbCase, case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    if case.mode == "normal":
        if case.fixture_case == SITS_FF19SB_CMAP_FIXTURE:
            return _prepare_sits_ff19sb_cmap_pair(case_root, replica_seed)
        return _prepare_normal_tip3p_pair(case_root, replica_seed)

    legacy_dir = _copy_case(case, "legacy", case.legacy_subdir, case_root)
    bundled_dir = _copy_case(case, "bundled", case.bundled_subdir, case_root)
    _validate_full_contract_input(case, bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_normal_tip3p_pair(
    case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    source = (
        REPO_ROOT
        / "benchmarks"
        / "validation"
        / "thermostat"
        / "statics"
        / "tip3p_water"
    )
    legacy_source = case_root / "generated_legacy_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(source, legacy_source)
    _reset_xponge_coordinate_start_time(legacy_source / "tip3p_coordinate.txt")
    masses = read_mass_values(legacy_source / "tip3p_mass.txt")
    constrain_pair_count = Extractor.read_first_field_int(
        legacy_source / "tip3p_bond.txt"
    )
    write_velocity_file_for_temperature(
        legacy_source / "initial_velocity.txt",
        masses,
        temperature=300.0,
        seed=replica_seed,
        degrees_of_freedom=3 * len(masses) - constrain_pair_count,
    )
    _write_tip3p_mdin(legacy_source)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    return legacy_dir, bundled_dir


def _prepare_sits_ff19sb_cmap_pair(
    case_root: Path, replica_seed: int
) -> tuple[Path, Path]:
    legacy_source = case_root / "sits_ff19sb_cmap_source"
    legacy_dir = case_root / "legacy"
    converted_dir = case_root / "converted_sits_ff19sb_cmap_bundle"
    bundled_dir = case_root / "bundled"
    for path in (legacy_source, legacy_dir, converted_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(SITS_FF19SB_CMAP_SOURCE, legacy_source)
    _write_sits_ff19sb_cmap_mdin(legacy_source, replica_seed)
    shutil.copytree(legacy_source, legacy_dir)
    _convert_legacy_case(legacy_source, converted_dir)
    shutil.copytree(converted_dir / "bundle", bundled_dir)
    return legacy_dir, bundled_dir


def _write_sits_ff19sb_cmap_mdin(case_dir: Path, replica_seed: int):
    _assert_sits_ff19sb_cmap_fixture(case_dir / "ALA_cmap.txt")
    masses = read_mass_values(case_dir / "ALA_mass.txt")
    constrain_pair_count = Extractor.read_first_field_int(
        case_dir / "ALA_bond.txt"
    )
    write_velocity_file_for_temperature(
        case_dir / "initial_velocity.txt",
        masses,
        temperature=300.0,
        seed=replica_seed,
        degrees_of_freedom=3 * len(masses) - constrain_pair_count,
    )
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(
            [
                'md_name = "bundled io ab sits ff19sb cmap"',
                'mode = "nve"',
                "step_limit = 10",
                "dt = 0.002",
                "cutoff = 8.0",
                "dont_check_input = 1",
                'default_in_file_prefix = "ALA"',
                'velocity_in_file = "initial_velocity.txt"',
                "print_zeroth_frame = 1",
                "write_mdout_interval = 1",
                "write_information_interval = 1",
                'constrain_mode = "SHAKE"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _assert_sits_ff19sb_cmap_fixture(path: Path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 4:
        raise AssertionError(f"SITS ff19SB CMAP fixture is too short: {path}")
    if lines[0] != "1 1":
        raise AssertionError(
            f"SITS ff19SB ACE-ALA-NME must contain one CMAP entry: {path}"
        )
    if lines[1] != "24":
        raise AssertionError(
            f"SITS ff19SB ACE-ALA-NME CMAP resolution must be 24: {path}"
        )
    if lines[-1] != "4 6 8 14 16 0":
        raise AssertionError(
            "SITS ff19SB ACE-ALA-NME CMAP atom/type entry changed: "
            f"{lines[-1]!r}"
        )


def _reset_xponge_coordinate_start_time(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"empty coordinate file: {path}")
    fields = lines[0].split()
    if len(fields) >= 2:
        fields[1] = "0.0000000000"
        lines[0] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tip3p_mdin(case_dir: Path):
    (case_dir / "mdin.spg.toml").write_text(
        "\n".join(
            [
                'md_name = "bundled io ab tip3p normal"',
                'mode = "nve"',
                "step_limit = 10",
                "dt = 0.002",
                "cutoff = 8.0",
                'default_in_file_prefix = "tip3p"',
                "print_zeroth_frame = 1",
                "write_mdout_interval = 1",
                "write_information_interval = 1",
                'constrain_mode = "SETTLE"',
                'velocity_in_file = "initial_velocity.txt"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _xponge_python() -> Path:
    configured = os.environ.get("SPONGE_XPONGE_PYTHON")
    if configured:
        return Path(configured)
    dev_python = XPONGE_DEV_ROOT / ".venv" / "bin" / "python"
    if dev_python.exists():
        return dev_python
    return Path(sys.executable)


def _xponge_env() -> dict[str, str]:
    env = dict(os.environ)
    configured_root = os.environ.get("SPONGE_XPONGE_ROOT")
    xponge_root = Path(configured_root) if configured_root else XPONGE_DEV_ROOT
    if xponge_root.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(xponge_root)
            if not existing
            else str(xponge_root) + os.pathsep + existing
        )
    return env


def _convert_legacy_case(case_root: Path, output_dir: Path):
    _run(
        [
            _xponge_python(),
            "-m",
            "Xponge",
            "legacy-to-bundle",
            case_root,
            "-o",
            output_dir,
            "-m",
            "mdin.spg.toml",
        ],
        env=_xponge_env(),
    )
    assert (output_dir / "bundle" / "mdin.bundled.spg.toml").exists()


def _mdin_name(case_dir: Path) -> str:
    bundled = case_dir / "mdin.bundled.spg.toml"
    if bundled.exists():
        return bundled.name
    return "mdin.spg.toml"


def _prepare_mdin(
    case_dir: Path,
    mdin_name: str,
    case: AbCase,
    *,
    branch: str,
    replica_seed: int,
):
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    mdin_path = case_dir / mdin_name
    text = mdin_path.read_text(encoding="utf-8")
    remove_keys = {
        "step_limit",
        "write_mdout_interval",
        "write_trajectory_interval",
        "write_restart_file_interval",
        "output_h5_trajectory_path",
        "output_h5_trajectory_vds",
        "output_h5_trajectory_chunk_size",
        "output_h5_restart_path",
        "output_h5_observable_path",
        "rerun_start",
        "rerun_strip",
        "rerun_frame_limit",
        "rerun_need_box_update",
        "input_h5_restart_load",
        "mdout",
        "mdinfo",
    }
    if case.statistical_md:
        remove_keys.update(
            {
                "mode",
                "thermostat",
                "thermostat_seed",
                "thermostat_tau",
                "target_temperature",
            }
        )
    text = _remove_key_lines(
        text,
        remove_keys,
    )
    limits = PROFILE_LIMITS[PROFILE]
    if case.mode == "normal":
        step_limit = limits["normal_step_limit"]
        interval = limits["normal_interval"]
        additions = [
            'mode = "nvt"',
            'thermostat = "middle_langevin"',
            f"thermostat_seed = {replica_seed}",
            "thermostat_tau = 0.1",
            "target_temperature = 300.0",
            f"step_limit = {step_limit}",
            f"write_mdout_interval = {interval}",
            f"write_trajectory_interval = {interval}",
            f"write_restart_file_interval = {step_limit}",
        ]
    else:
        frame_limit = limits["rerun_frame_limit"]
        additions = [
            "rerun_start = 0",
            "rerun_strip = 0",
            f"rerun_frame_limit = {frame_limit}",
            "rerun_need_box_update = 0",
            "write_mdout_interval = 1",
            "write_trajectory_interval = 1",
            "write_restart_file_interval = 0",
        ]
        if _has_key_line(text, "input_h5_restart_path"):
            additions.append(
                f'input_h5_restart_load = "{case.restart_load_policy}"'
            )
    additions.extend(
        [
            'mdout = "mdout.txt"',
            'mdinfo = "mdinfo.txt"',
            f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
            f"output_h5_trajectory_vds = {'true' if case.vds else 'false'}",
            "output_h5_trajectory_chunk_size = 1",
            f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
        ]
    )
    if case.mode == "normal":
        additions.append(f'output_h5_restart_path = "{RESTART_REL.as_posix()}"')
    mdin_path.write_text(
        text.rstrip() + "\n" + "\n".join(additions) + "\n",
        encoding="utf-8",
    )
    assert branch in {"legacy", "bundled"}


def _remove_key_lines(text: str, keys: set[str]) -> str:
    output = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in keys:
            output.append(line)
    return "\n".join(output)


def _has_key_line(text: str, key: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split("=", 1)[0].strip() == key:
            return True
    return False


def _run_sponge(case_dir: Path, mdin_name: str) -> dict[str, object]:
    start = time.perf_counter()
    result = subprocess.run(
        [_sponge_executable(), "-mdin", mdin_name],
        cwd=case_dir,
        text=True,
        capture_output=True,
        check=False,
        timeout=int(os.environ.get("SPONGE_BUNDLED_IO_AB_TIMEOUT", "7200")),
    )
    elapsed_s = time.perf_counter() - start
    (case_dir / "run.stdout").write_text(result.stdout, encoding="utf-8")
    (case_dir / "run.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(
            f"SPONGE failed in {case_dir} with code {result.returncode}\n"
            f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
        )
    return _collect_metrics(case_dir, elapsed_s)


def _sponge_executable() -> str:
    configured = os.environ.get("SPONGE_BUNDLED_IO_AB_SPONGE")
    if configured:
        return configured
    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if not env_name:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            env_name = Path(conda_prefix).name
    candidates = []
    if env_name:
        candidates.append(REPO_ROOT / f"build-{env_name}" / "SPONGE")
    candidates.extend(
        [
            REPO_ROOT / "build-dev-cuda13" / "SPONGE",
            REPO_ROOT / "build-dev-cpu-h5-2" / "SPONGE",
            REPO_ROOT / "build" / "SPONGE",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    path_executable = shutil.which("SPONGE")
    if path_executable:
        return path_executable
    raise AssertionError(
        "SPONGE executable was not found. Set "
        "SPONGE_BUNDLED_IO_AB_SPONGE=/path/to/SPONGE or build SPONGE first."
    )


def _collect_metrics(case_dir: Path, elapsed_s: float) -> dict[str, object]:
    trajectory = case_dir / TRAJECTORY_REL
    observable = case_dir / OBSERVABLE_REL
    restart = case_dir / RESTART_REL
    h5_finalize = _parse_h5_finalize_timing(case_dir / "run.stdout")
    return {
        "elapsed_s": elapsed_s,
        "mdout_rows": len(_read_mdout(case_dir / "mdout.txt")["rows"]),
        "trajectory_h5_bytes": _file_size(trajectory),
        "observable_h5_bytes": _file_size(observable),
        "restart_h5_bytes": _file_size(restart),
        "vds_shard_count": _vds_shard_count(trajectory),
        "h5_files_total_bytes": sum(
            _file_size(path) for path in (trajectory, observable, restart)
        ),
        "flush_finalize_elapsed_s": h5_finalize["total_s"],
        "h5_finalize_timing": h5_finalize,
    }


def _parse_h5_finalize_timing(stdout_path: Path) -> dict[str, float]:
    pattern = re.compile(
        r"H5 I/O finalize timing: trajectory=(?P<trajectory>[0-9.eE+-]+) s, "
        r"observable=(?P<observable>[0-9.eE+-]+) s, "
        r"restart=(?P<restart>[0-9.eE+-]+) s, total=(?P<total>[0-9.eE+-]+) s"
    )
    text = stdout_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise AssertionError(
            f"H5 finalize timing was not reported in {stdout_path}"
        )
    return {
        "trajectory_s": float(match.group("trajectory")),
        "observable_s": float(match.group("observable")),
        "restart_s": float(match.group("restart")),
        "total_s": float(match.group("total")),
    }


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _vds_shard_count(wrapper_path: Path) -> int:
    shard_root = wrapper_path.with_suffix("").with_suffix(".shards")
    if not shard_root.exists():
        # ab.spg.h5md -> ab.spg.shards
        shard_root = wrapper_path.parent / "ab.spg.shards"
    if not shard_root.exists():
        return 0
    return len(list(shard_root.glob("segment_*.spg.h5md")))


def _compare_outputs(
    case: AbCase, runs: Sequence[AbRun]
) -> tuple[dict[str, object], tuple[AssertionEvidence, ...]]:
    evidence = []
    if case.statistical_md:
        mdout_comparison = _compare_mdout_statistically(case, runs)
        h5_comparison = _compare_h5_outputs_statistically(case, runs)
        evidence.extend(
            (
                AssertionEvidence(
                    assertion_id="mdout_statistical_equivalence",
                    evidence_level="E3",
                    details={
                        "method": mdout_comparison["method"],
                        "column_count": len(mdout_comparison["columns"]),
                    },
                ),
                AssertionEvidence(
                    assertion_id="h5_statistical_equivalence",
                    evidence_level="E3",
                    details={
                        "families": sorted(h5_comparison),
                        "method": "all_dataset_schema_and_statistical_values",
                    },
                ),
            )
        )
    else:
        mdout_comparison = _compare_mdout_deterministically(case, runs[0])
        h5_comparison = _compare_h5_outputs_deterministically(case, runs[0])
        evidence.extend(
            (
                AssertionEvidence(
                    assertion_id="mdout_deterministic_equivalence",
                    evidence_level="E3",
                    details={
                        "method": mdout_comparison["method"],
                        "row_count": mdout_comparison["rows"],
                        "columns": mdout_comparison["columns"],
                    },
                ),
                AssertionEvidence(
                    assertion_id="h5_rerun_semantic_equivalence",
                    evidence_level="E3",
                    details={
                        "method": h5_comparison["method"],
                        "trajectory_frame_count": h5_comparison[
                            "trajectory_frame_count"
                        ],
                    },
                ),
            )
        )

    comparison: dict[str, object] = {
        "mdout": mdout_comparison,
        "h5": h5_comparison,
    }
    if "input.full_contract.inventory" in case.contract_ids:
        inventory = _validate_full_contract_input(case, runs[0].bundled_dir)
        comparison["full_contract_input_inventory"] = inventory
        evidence.append(
            AssertionEvidence(
                assertion_id="full_contract_input_inventory",
                evidence_level="E0",
                details=inventory,
            )
        )
    if "input.topology.cmap" in case.contract_ids:
        comparison["cmap"] = [
            _compare_cmap_materialization(run.legacy_dir, run.bundled_dir)
            for run in runs
        ]
        evidence.append(
            AssertionEvidence(
                assertion_id="cmap_runtime_equivalence",
                evidence_level="E3",
                details={
                    "replica_count": len(runs),
                    "mdout_equivalence_assertion": (
                        "mdout_statistical_equivalence"
                        if case.statistical_md
                        else "mdout_deterministic_equivalence"
                    ),
                    "both_branches_initialized": True,
                },
            )
        )
    return comparison, tuple(evidence)


def _compare_cmap_materialization(
    legacy_dir: Path, bundled_dir: Path
) -> dict[str, object]:
    legacy_stdout = (legacy_dir / "run.stdout").read_text(encoding="utf-8")
    bundled_stdout = (bundled_dir / "run.stdout").read_text(encoding="utf-8")
    if "START INITIALIZING CMAP" not in legacy_stdout:
        raise AssertionError("legacy run did not initialize CMAP")
    if "START INITIALIZING CMAP" not in bundled_stdout:
        raise AssertionError("bundled run did not initialize CMAP")

    topology = bundled_dir / "topology.spgt.h5"
    paths = _h5_paths(topology)
    required = {
        "/forcefield/cmap/atoms",
        "/forcefield/cmap/type",
        "/forcefield/cmap/resolution",
        "/forcefield/cmap/grid_value",
    }
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(
            f"bundled CMAP native H5 datasets missing: {missing}"
        )
    return {
        "legacy_initialized": True,
        "bundled_initialized": True,
        "bundled_required_datasets": sorted(required),
    }


def _read_mdout(path: Path) -> dict[str, object]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise AssertionError(f"mdout has no data rows: {path}")
    columns = lines[0].split()
    rows = []
    for raw in lines[1:]:
        fields = raw.split()
        if len(fields) != len(columns):
            continue
        row = {}
        for column, field in zip(columns, fields):
            row[column] = _parse_float(field)
        rows.append(row)
    if not rows:
        raise AssertionError(f"mdout has no parseable rows: {path}")
    return {"columns": columns, "rows": rows}


def _parse_float(value: str) -> float:
    normalized = value.lower().replace("-nan(ind)", "nan")
    normalized = normalized.replace("nan(ind)", "nan")
    return float(normalized)


def _statistical_policy() -> StatisticalEquivalencePolicy:
    limits = PROFILE_LIMITS[PROFILE]
    return StatisticalEquivalencePolicy(
        burn_in_frames=int(limits["normal_burn_in_frames"]),
        block_size=int(limits["normal_block_size"]),
        minimum_blocks_per_replica=STATISTICAL_MINIMUM_BLOCKS_PER_REPLICA,
        confidence_z=STATISTICAL_CONFIDENCE_Z,
        relative_margin=float(limits["normal_relative_margin"]),
        absolute_margin=float(limits["normal_absolute_margin"]),
        maximum_std_ratio=STATISTICAL_MAXIMUM_STD_RATIO,
    )


def _require_matching_mdout_columns(
    left: dict[str, object], right: dict[str, object], label: str
) -> list[str]:
    left_columns = list(left["columns"])
    right_columns = list(right["columns"])
    if left_columns != right_columns:
        raise AssertionError(
            f"{label} mdout columns differ: legacy={left_columns}, "
            f"bundled={right_columns}"
        )
    if not left_columns:
        raise AssertionError(f"{label} has no mdout columns")
    return left_columns


def _same_nonfinite_value(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isinf(left) and math.isinf(right) and left == right


def _assert_numeric_sequences_close(
    label: str,
    left: Sequence[float],
    right: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if len(left) != len(right):
        raise AssertionError(
            f"{label} length mismatch: legacy={len(left)}, bundled={len(right)}"
        )
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if not _same_nonfinite_value(left_value, right_value):
                raise AssertionError(
                    f"{label} non-finite mismatch at index {index}: "
                    f"legacy={left_value}, bundled={right_value}"
                )
            continue
        if not math.isclose(
            left_value,
            right_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(
                f"{label} mismatch at index {index}: legacy={left_value}, "
                f"bundled={right_value}"
            )


def _compare_mdout_statistically(
    case: AbCase, runs: Sequence[AbRun]
) -> dict[str, object]:
    parsed = [
        (
            _read_mdout(run.legacy_dir / "mdout.txt"),
            _read_mdout(run.bundled_dir / "mdout.txt"),
        )
        for run in runs
    ]
    columns = _require_matching_mdout_columns(
        parsed[0][0], parsed[0][1], f"{case.name} replica 0"
    )
    for replica_index, (left, right) in enumerate(parsed):
        replica_columns = _require_matching_mdout_columns(
            left, right, f"{case.name} replica {replica_index}"
        )
        if replica_columns != columns:
            raise AssertionError(
                f"{case.name} mdout columns changed between replicas: "
                f"first={columns}, replica_{replica_index}={replica_columns}"
            )
        if len(left["rows"]) != len(right["rows"]):
            raise AssertionError(
                f"{case.name} replica {replica_index} mdout row count mismatch: "
                f"legacy={len(left['rows'])}, bundled={len(right['rows'])}"
            )

    policy = _statistical_policy()
    comparison: dict[str, object] = {
        "method": "independent_replicas_block_mean_equivalence",
        "replicas": len(runs),
        "policy": {
            "burn_in_frames": policy.burn_in_frames,
            "block_size": policy.block_size,
            "minimum_blocks_per_replica": policy.minimum_blocks_per_replica,
            "confidence_z": policy.confidence_z,
            "relative_margin": policy.relative_margin,
            "absolute_margin": policy.absolute_margin,
            "maximum_std_ratio": policy.maximum_std_ratio,
        },
        "columns": {},
    }
    for column in columns:
        legacy_replicas = [
            [row[column] for row in left["rows"]] for left, _ in parsed
        ]
        bundled_replicas = [
            [row[column] for row in right["rows"]] for _, right in parsed
        ]
        if column in {"step", "frame", "time"}:
            for replica_index, (legacy, bundled) in enumerate(
                zip(legacy_replicas, bundled_replicas)
            ):
                _assert_numeric_sequences_close(
                    f"{case.name} mdout {column} replica {replica_index}",
                    legacy,
                    bundled,
                    relative_tolerance=0.0,
                    absolute_tolerance=1.0e-12,
                )
            comparison["columns"][column] = {"method": "exact_schedule"}
            continue

        all_values = [
            value
            for replica in (*legacy_replicas, *bundled_replicas)
            for value in replica
        ]
        if any(not math.isfinite(value) for value in all_values):
            for replica_index, (legacy, bundled) in enumerate(
                zip(legacy_replicas, bundled_replicas)
            ):
                _assert_numeric_sequences_close(
                    f"{case.name} mdout {column} replica {replica_index}",
                    legacy,
                    bundled,
                    relative_tolerance=0.0,
                    absolute_tolerance=0.0,
                )
            comparison["columns"][column] = {
                "method": "exact_nonfinite_pattern",
                "nonfinite_count": sum(
                    not math.isfinite(value) for value in all_values
                ),
            }
            continue

        comparison["columns"][column] = compare_replicas(
            f"{case.name} mdout {column}",
            legacy_replicas,
            bundled_replicas,
            policy,
        )
    return comparison


def _compare_mdout_deterministically(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    left = _read_mdout(run.legacy_dir / "mdout.txt")
    right = _read_mdout(run.bundled_dir / "mdout.txt")
    columns = _require_matching_mdout_columns(left, right, case.name)
    left_rows = left["rows"]
    right_rows = right["rows"]
    if len(left_rows) != len(right_rows):
        raise AssertionError(
            f"{case.name} mdout row count mismatch: legacy={len(left_rows)}, "
            f"bundled={len(right_rows)}"
        )
    maximum_abs_error = 0.0
    for column in columns:
        legacy_values = [row[column] for row in left_rows]
        bundled_values = [row[column] for row in right_rows]
        _assert_numeric_sequences_close(
            f"{case.name} deterministic mdout {column}",
            legacy_values,
            bundled_values,
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )
        for legacy_value, bundled_value in zip(legacy_values, bundled_values):
            if math.isfinite(legacy_value) and math.isfinite(bundled_value):
                maximum_abs_error = max(
                    maximum_abs_error, abs(legacy_value - bundled_value)
                )
    return {
        "method": "deterministic_all_rows_all_columns",
        "rows": len(left_rows),
        "columns": columns,
        "maximum_abs_error": maximum_abs_error,
    }


H5_OBSERVABLE_REQUIRED_PATHS = {
    "/observables/all/step",
    "/observables/all/time",
    "/parameters/sponge/mdout/columns/original_name",
    "/parameters/sponge/mdout/columns/hdf5_name",
}


def _validate_full_contract_input(
    case: AbCase, bundled_dir: Path
) -> dict[str, object]:
    if case.fixture_case != "full_contract_rerun":
        return {"file_count": 0, "path_count": 0, "has_sidecars": False}
    all_paths: set[str] = set()
    for file_name, required_paths in FULL_CONTRACT_INPUT_REQUIRED_PATHS.items():
        file_path = bundled_dir / file_name
        paths = _h5_paths(file_path)
        missing = sorted(required_paths - paths)
        if missing:
            raise AssertionError(
                f"{case.name} full-contract input {file_name} is missing "
                f"required paths: {missing}"
            )
        all_paths.update(paths)
    has_sidecars = any("legacy_sidecars" in path for path in all_paths)
    if "input.full_contract.pure_native" in case.contract_ids and has_sidecars:
        raise AssertionError(
            f"{case.name} pure bundle retains legacy sidecar tables"
        )
    if "input.full_contract.sidecar" in case.contract_ids and not has_sidecars:
        raise AssertionError(
            f"{case.name} sidecar bundle has no legacy sidecar tables"
        )
    return {
        "file_count": len(FULL_CONTRACT_INPUT_REQUIRED_PATHS),
        "path_count": len(all_paths),
        "has_sidecars": has_sidecars,
        "validation": "required_paths_present",
    }


def _output_h5_files(case: AbCase, root: Path) -> dict[str, Path]:
    files = {
        "trajectory": root / TRAJECTORY_REL,
        "observable": root / OBSERVABLE_REL,
    }
    if case.mode == "normal":
        files["restart"] = root / RESTART_REL
    return files


def _validate_branch_output_contract(
    case: AbCase, case_dir: Path, *, branch: str
) -> dict[str, object]:
    if branch not in {"legacy", "bundled"}:
        raise AssertionError(f"unknown A/B branch: {branch}")
    if case.mode == "rerun" and branch == "legacy":
        if not (case_dir / "mdout.txt").exists():
            raise AssertionError(f"{case.name} legacy rerun did not emit mdout")
        # Rerun reaches its frame loop before the H5 output initializers on the
        # legacy-text path.  Bundled output is therefore checked against the
        # legacy mdout/trajectory semantics in _compare_rerun_h5_output.
        return {
            "h5_emission": "legacy_rerun_h5_output_not_available",
            "mdout_rows": len(_read_mdout(case_dir / "mdout.txt")["rows"]),
        }
    files = _output_h5_files(case, case_dir)
    summary: dict[str, object] = {}
    for name, path in files.items():
        if not path.exists():
            raise AssertionError(
                f"{case.name} did not emit {name} H5 output: {path}"
            )
        summary[name] = {
            "dataset_count": len(_h5_dataset_paths(path)),
            "path_count": len(_h5_paths(path)),
            "bytes": _file_size(path),
        }

    _validate_trajectory_output(case.name, files["trajectory"])
    observable_summary = _validate_observable_output(
        case.name, files["observable"], case_dir / "mdout.txt"
    )
    summary["observable"].update(observable_summary)
    if "restart" in files:
        _validate_restart_output(case.name, files["restart"])
    if case.vds and _vds_shard_count(files["trajectory"]) <= 0:
        raise AssertionError(
            f"{case.name} VDS run did not create trajectory shards"
        )
    return summary


def _validate_trajectory_output(case_name: str, path: Path) -> None:
    paths = _h5_paths(path)
    missing = sorted(set(H5_COMPARE_DATASETS) - paths)
    if missing:
        raise AssertionError(
            f"{case_name} trajectory output is missing required datasets: {missing}"
        )
    steps = _h5_numeric_values(path, "/particles/all/step")
    times = _h5_numeric_values(path, "/particles/all/time")
    frame_counts = _h5_numeric_values(
        path, "/parameters/sponge/output/frame_count"
    )
    if not steps or len(steps) != len(times):
        raise AssertionError(
            f"{case_name} trajectory timeline is empty or inconsistent: "
            f"steps={len(steps)}, times={len(times)}"
        )
    if not frame_counts or int(frame_counts[-1]) != len(steps):
        raise AssertionError(
            f"{case_name} trajectory completion frame count does not match "
            f"timeline: completion={frame_counts}, frames={len(steps)}"
        )
    for dataset in (
        "/particles/all/position/value",
        "/particles/all/box/edges/value",
    ):
        shape = _h5_dataset_shape(path, dataset)
        if not shape or shape[0] != len(steps):
            raise AssertionError(
                f"{case_name} trajectory {dataset} frame shape mismatch: "
                f"shape={shape}, frames={len(steps)}"
            )


def _validate_restart_output(case_name: str, path: Path) -> None:
    paths = _h5_paths(path)
    missing = sorted(set(RESTART_COMPARE_DATASETS) - paths)
    if missing:
        raise AssertionError(
            f"{case_name} restart output is missing required datasets: {missing}"
        )
    for dataset in RESTART_COMPARE_DATASETS:
        values = _h5_numeric_values(path, dataset)
        if not values:
            raise AssertionError(
                f"{case_name} restart output is empty: {dataset}"
            )


def _validate_observable_output(
    case_name: str, path: Path, mdout_path: Path
) -> dict[str, object]:
    paths = _h5_paths(path)
    missing = sorted(H5_OBSERVABLE_REQUIRED_PATHS - paths)
    if missing:
        raise AssertionError(
            f"{case_name} observable output is missing required paths: {missing}"
        )
    original_names = _h5_string_values(
        path, "/parameters/sponge/mdout/columns/original_name"
    )
    hdf5_names = _h5_string_values(
        path, "/parameters/sponge/mdout/columns/hdf5_name"
    )
    if len(original_names) != len(hdf5_names) or not original_names:
        raise AssertionError(
            f"{case_name} observable column mapping is invalid: "
            f"original={original_names}, hdf5={hdf5_names}"
        )
    if len(set(hdf5_names)) != len(hdf5_names):
        raise AssertionError(
            f"{case_name} observable HDF5 names are not unique"
        )

    mdout = _read_mdout(mdout_path)
    mdout_columns = list(mdout["columns"])
    expected_original_names = mdout_columns
    if original_names != expected_original_names:
        raise AssertionError(
            f"{case_name} observable mapping does not exactly match mdout "
            f"semantic order: mapping={original_names}, "
            f"expected={expected_original_names}"
        )
    rows = mdout["rows"]
    steps = _h5_numeric_values(path, "/observables/all/step")
    times = _h5_numeric_values(path, "/observables/all/time")
    if "step" in mdout_columns:
        _assert_numeric_sequences_close(
            f"{case_name} observable step stream",
            [row["step"] for row in rows],
            steps,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )
    elif len(steps) != len(rows):
        raise AssertionError(
            f"{case_name} observable step stream length mismatch: "
            f"h5={len(steps)}, mdout={len(rows)}"
        )
    if "time" in mdout_columns:
        _assert_numeric_sequences_close(
            f"{case_name} observable time stream",
            [row["time"] for row in rows],
            times,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-8,
        )
    elif len(times) != len(rows):
        raise AssertionError(
            f"{case_name} observable time stream length mismatch: "
            f"h5={len(times)}, mdout={len(rows)}"
        )
    for original_name, hdf5_name in zip(original_names, hdf5_names):
        dataset = f"/observables/all/{hdf5_name}/value"
        if dataset not in paths:
            raise AssertionError(
                f"{case_name} observable mapping has no value dataset: {dataset}"
            )
        _assert_numeric_sequences_close(
            f"{case_name} observable {original_name}",
            [row[original_name] for row in rows],
            _h5_numeric_values(path, dataset),
            relative_tolerance=1.0e-4,
            absolute_tolerance=1.0e-8,
        )
    return {"mapped_mdout_columns": original_names}


def _compare_h5_outputs_deterministically(
    case: AbCase, run: AbRun
) -> dict[str, object]:
    if case.mode == "rerun":
        return _compare_rerun_h5_output(case, run)

    compared: dict[str, object] = {}
    legacy_files = _output_h5_files(case, run.legacy_dir)
    bundled_files = _output_h5_files(case, run.bundled_dir)
    if set(legacy_files) != set(bundled_files):
        raise AssertionError(
            f"{case.name} H5 output families differ: legacy={sorted(legacy_files)}, "
            f"bundled={sorted(bundled_files)}"
        )
    for name in legacy_files:
        datasets = _assert_h5_schema_equivalent(
            legacy_files[name], bundled_files[name], f"{case.name} {name}"
        )
        for dataset in datasets:
            left = _normalize_h5dump(
                _h5dump_dataset(legacy_files[name], dataset)
            )
            right = _normalize_h5dump(
                _h5dump_dataset(bundled_files[name], dataset)
            )
            if left != right:
                raise AssertionError(
                    f"{case.name} deterministic H5 dataset differs: "
                    f"{name}:{dataset}"
                )
        compared[name] = {
            "method": "deterministic_all_datasets",
            "dataset_count": len(datasets),
            "legacy_shard_count": _vds_shard_count(legacy_files[name]),
            "bundled_shard_count": _vds_shard_count(bundled_files[name]),
        }
    if case.vds:
        for branch, path in (
            ("legacy", legacy_files["trajectory"]),
            ("bundled", bundled_files["trajectory"]),
        ):
            if _vds_shard_count(path) <= 0:
                raise AssertionError(
                    f"{case.name} {branch} VDS output has no shards"
                )
    return compared


def _compare_rerun_h5_output(case: AbCase, run: AbRun) -> dict[str, object]:
    """Bridge the bundled H5 writer to legacy rerun observables and frames."""

    output_files = _output_h5_files(case, run.bundled_dir)
    trajectory_output = output_files["trajectory"]
    observable_output = output_files["observable"]
    input_trajectory = run.bundled_dir / "trajectory.spg.h5md"
    if not input_trajectory.exists():
        raise AssertionError(
            f"{case.name} bundled rerun input trajectory is missing: {input_trajectory}"
        )

    output_steps = _h5_numeric_values(trajectory_output, "/particles/all/step")
    input_steps = _h5_numeric_values(input_trajectory, "/particles/all/step")
    if not output_steps:
        raise AssertionError(
            f"{case.name} bundled rerun output has no trajectory frames"
        )
    matching_indices = []
    for output_step in output_steps:
        try:
            matching_indices.append(input_steps.index(output_step))
        except ValueError as error:
            raise AssertionError(
                f"{case.name} output step {output_step} is absent from rerun input"
            ) from error

    input_position_dataset = "/particles/all/position/value"
    input_box_dataset = "/particles/all/box/edges/value"
    input_position_values = _h5_numeric_values(
        input_trajectory, input_position_dataset
    )
    input_position_shape = _h5_dataset_shape(
        input_trajectory, input_position_dataset
    )
    input_box_values = _h5_numeric_values(input_trajectory, input_box_dataset)
    input_box_shape = _h5_dataset_shape(input_trajectory, input_box_dataset)
    expected_positions = _rerun_expected_position_values(
        run,
        matching_indices,
        input_position_values,
        input_position_shape,
        input_box_values,
        input_box_shape,
    )
    expected_boxes = _rerun_expected_box_values(
        run, matching_indices, input_box_values, input_box_shape
    )

    for dataset in (
        input_position_dataset,
        input_box_dataset,
    ):
        output_values = _h5_numeric_values(trajectory_output, dataset)
        input_shape = (
            input_position_shape
            if dataset == input_position_dataset
            else input_box_shape
        )
        output_shape = _h5_dataset_shape(trajectory_output, dataset)
        if len(input_shape) < 2 or len(output_shape) < 2:
            raise AssertionError(
                f"{case.name} rerun trajectory payload is not frame-shaped: {dataset}"
            )
        input_frame_width = math.prod(input_shape[1:])
        output_frame_width = math.prod(output_shape[1:])
        if input_frame_width != output_frame_width:
            raise AssertionError(
                f"{case.name} rerun {dataset} frame width mismatch: "
                f"input={input_frame_width}, output={output_frame_width}"
            )
        expected = (
            expected_positions
            if dataset == input_position_dataset
            else expected_boxes
        )
        _assert_numeric_sequences_close(
            f"{case.name} rerun output matches bundled input {dataset}",
            expected,
            output_values,
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-6,
        )

    # _compare_mdout_deterministically already proves legacy and bundled mdout
    # equivalence. The bundled observable output was checked against bundled
    # mdout in _validate_observable_output, which gives a transitive semantic
    # comparison for every legacy observable column.
    observable_columns = _validate_observable_output(
        case.name,
        observable_output,
        run.bundled_dir / "mdout.txt",
    )
    if case.vds and _vds_shard_count(trajectory_output) <= 0:
        raise AssertionError(f"{case.name} bundled VDS output has no shards")
    return {
        "method": "legacy_mdout_to_bundled_h5_semantic_bridge",
        "legacy_rerun_h5_output": "not_available",
        "trajectory_frame_count": len(output_steps),
        "observable": observable_columns,
        "bundled_shard_count": _vds_shard_count(trajectory_output),
    }


def _h5_integer_values(path: Path, dataset: str) -> list[int]:
    values = _h5_numeric_values(path, dataset)
    integers = []
    for value in values:
        if not value.is_integer():
            raise AssertionError(
                f"H5 integer dataset has non-integer value: {path}:{dataset}"
            )
        integers.append(int(value))
    return integers


def _read_h5_virtual_atoms(topology: Path) -> list[H5VirtualAtomRecord]:
    root = "/forcefield/virtual_atom"
    required = {
        f"{root}/atom",
        f"{root}/type",
        f"{root}/from",
        f"{root}/from_offset",
        f"{root}/parameter",
        f"{root}/parameter_offset",
    }
    paths = _h5_paths(topology)
    if not required & paths:
        return []
    missing = sorted(required - paths)
    if missing:
        raise AssertionError(f"virtual atom topology is incomplete: {missing}")

    atoms = _h5_integer_values(topology, f"{root}/atom")
    kinds = _h5_integer_values(topology, f"{root}/type")
    source_atoms = _h5_integer_values(topology, f"{root}/from")
    source_offsets = _h5_integer_values(topology, f"{root}/from_offset")
    parameters = _h5_numeric_values(topology, f"{root}/parameter")
    parameter_offsets = _h5_integer_values(topology, f"{root}/parameter_offset")
    record_count = len(atoms)
    if len(kinds) != record_count:
        raise AssertionError(
            "virtual atom type count does not match atom count"
        )
    if len(source_offsets) != record_count + 1:
        raise AssertionError(
            "virtual atom source offsets have an invalid length"
        )
    if len(parameter_offsets) != record_count + 1:
        raise AssertionError(
            "virtual atom parameter offsets have an invalid length"
        )
    if source_offsets[0] != 0 or source_offsets[-1] != len(source_atoms):
        raise AssertionError(
            "virtual atom source offsets do not cover all values"
        )
    if parameter_offsets[0] != 0 or parameter_offsets[-1] != len(parameters):
        raise AssertionError(
            "virtual atom parameter offsets do not cover all values"
        )

    records = []
    for index, (atom, kind) in enumerate(zip(atoms, kinds)):
        source_begin, source_end = source_offsets[index : index + 2]
        parameter_begin, parameter_end = parameter_offsets[index : index + 2]
        if source_begin > source_end or parameter_begin > parameter_end:
            raise AssertionError("virtual atom offsets must be monotonic")
        expected_counts = {0: (1, 1), 1: (2, 1), 2: (3, 2), 3: (3, 2)}
        if kind not in expected_counts:
            raise AssertionError(
                f"unsupported virtual atom type in H5 topology: {kind}"
            )
        expected_sources, expected_parameters = expected_counts[kind]
        if source_end - source_begin != expected_sources:
            raise AssertionError(
                f"virtual atom type {kind} has an invalid source count"
            )
        if parameter_end - parameter_begin != expected_parameters:
            raise AssertionError(
                f"virtual atom type {kind} has an invalid parameter count"
            )
        records.append(
            H5VirtualAtomRecord(
                atom=atom,
                kind=kind,
                source_atoms=tuple(source_atoms[source_begin:source_end]),
                parameters=tuple(parameters[parameter_begin:parameter_end]),
            )
        )
    return records


def _rerun_updates_box(case_dir: Path) -> bool:
    mdin_path = case_dir / _mdin_name(case_dir)
    for raw_line in mdin_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key != "rerun_need_box_update":
            continue
        normalized = value.strip("\"'").lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise AssertionError(f"invalid rerun_need_box_update value: {value!r}")
    return False


def _lower_triangular_periodic_displacement(
    left: Sequence[float], right: Sequence[float], box_edges: Sequence[float]
) -> tuple[float, float, float]:
    if len(box_edges) != 9:
        raise AssertionError("rerun box frame must contain nine edge values")
    a11, a21, a22, a31, a32, a33 = (
        box_edges[0],
        box_edges[3],
        box_edges[4],
        box_edges[6],
        box_edges[7],
        box_edges[8],
    )
    if min(abs(a11), abs(a22), abs(a33)) == 0.0:
        raise AssertionError(
            "rerun virtual atom calculation requires an invertible box"
        )
    dr_x = left[0] - right[0]
    dr_y = left[1] - right[1]
    dr_z = left[2] - right[2]
    r11 = 1.0 / a11
    r21 = -a21 / (a11 * a22)
    r22 = 1.0 / a22
    r31 = (a21 * a32 - a22 * a31) / (a11 * a22 * a33)
    r32 = -a32 / (a22 * a33)
    r33 = 1.0 / a33
    shift_x = math.floor(dr_x * r11 + dr_y * r21 + dr_z * r31 + 0.5)
    shift_y = math.floor(dr_y * r22 + dr_z * r32 + 0.5)
    shift_z = math.floor(dr_z * r33 + 0.5)
    return (
        dr_x - shift_x * a11 - shift_y * a21 - shift_z * a31,
        dr_y - shift_y * a22 - shift_z * a32,
        dr_z - shift_z * a33,
    )


def _canonicalize_virtual_atom_positions(
    positions: Sequence[float],
    atom_count: int,
    box_edges: Sequence[float],
    records: Sequence[H5VirtualAtomRecord],
) -> list[float]:
    if len(positions) != atom_count * 3:
        raise AssertionError(
            "rerun position frame does not match topology atom count"
        )
    coordinates = [
        list(positions[index : index + 3])
        for index in range(0, len(positions), 3)
    ]
    levels = [0] * atom_count
    records_by_level: dict[int, list[H5VirtualAtomRecord]] = {}
    for record in records:
        if record.atom < 0 or record.atom >= atom_count:
            raise AssertionError(
                f"virtual atom index is out of range: {record.atom}"
            )
        if any(
            source < 0 or source >= atom_count for source in record.source_atoms
        ):
            raise AssertionError("virtual atom source index is out of range")
        level = max(levels[source] for source in record.source_atoms) + 1
        levels[record.atom] = level
        records_by_level.setdefault(level, []).append(record)

    for level in sorted(records_by_level):
        for record in records_by_level[level]:
            source = [coordinates[index] for index in record.source_atoms]
            if record.kind == 0:
                h_double = 2.0 * record.parameters[0]
                coordinates[record.atom] = [
                    source[0][0],
                    source[0][1],
                    2.0 * h_double - source[0][2],
                ]
            elif record.kind == 1:
                delta = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                coordinates[record.atom] = [
                    source[0][axis] + record.parameters[0] * delta[axis]
                    for axis in range(3)
                ]
            elif record.kind == 2:
                delta_2 = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                delta_3 = _lower_triangular_periodic_displacement(
                    source[2], source[0], box_edges
                )
                coordinates[record.atom] = [
                    source[0][axis]
                    + record.parameters[0] * delta_2[axis]
                    + record.parameters[1] * delta_3[axis]
                    for axis in range(3)
                ]
            else:
                delta_21 = _lower_triangular_periodic_displacement(
                    source[1], source[0], box_edges
                )
                delta_32 = _lower_triangular_periodic_displacement(
                    source[2], source[1], box_edges
                )
                direction = [
                    delta_21[axis] + record.parameters[1] * delta_32[axis]
                    for axis in range(3)
                ]
                direction_norm = math.sqrt(
                    sum(value * value for value in direction)
                )
                if direction_norm == 0.0:
                    raise AssertionError(
                        "virtual atom type 3 has a zero-length direction"
                    )
                coordinates[record.atom] = [
                    source[0][axis]
                    + record.parameters[0] * direction[axis] / direction_norm
                    for axis in range(3)
                ]
    return [value for coordinate in coordinates for value in coordinate]


def _rerun_expected_position_values(
    run: AbRun,
    matching_indices: Sequence[int],
    input_positions: Sequence[float],
    input_position_shape: Sequence[int],
    input_boxes: Sequence[float],
    input_box_shape: Sequence[int],
) -> list[float]:
    if len(input_position_shape) != 3 or input_position_shape[2] != 3:
        raise AssertionError(
            "rerun position input must have shape [frame,atom,3]"
        )
    if len(input_box_shape) != 3 or input_box_shape[1:] != (3, 3):
        raise AssertionError("rerun box input must have shape [frame,3,3]")
    atom_count = input_position_shape[1]
    position_width = math.prod(input_position_shape[1:])
    box_width = math.prod(input_box_shape[1:])
    records = _read_h5_virtual_atoms(run.bundled_dir / "topology.spgt.h5")
    use_trajectory_box = _rerun_updates_box(run.bundled_dir)
    if use_trajectory_box:
        restart_box = []
    else:
        restart_box = _h5_numeric_values(
            run.bundled_dir / "restart.spgr.h5",
            "/particles/all/box/edges/value",
        )
        if len(restart_box) != box_width:
            raise AssertionError(
                "structural rerun restart box has an invalid shape"
            )

    expected = []
    for frame_index in matching_indices:
        position_begin = frame_index * position_width
        box_begin = frame_index * box_width
        frame_box = (
            input_boxes[box_begin : box_begin + box_width]
            if use_trajectory_box
            else restart_box
        )
        expected.extend(
            _canonicalize_virtual_atom_positions(
                input_positions[
                    position_begin : position_begin + position_width
                ],
                atom_count,
                frame_box,
                records,
            )
        )
    return expected


def _rerun_expected_box_values(
    run: AbRun,
    matching_indices: Sequence[int],
    input_boxes: Sequence[float],
    input_box_shape: Sequence[int],
) -> list[float]:
    if len(input_box_shape) < 2:
        raise AssertionError("rerun box input is not frame-shaped")
    box_width = math.prod(input_box_shape[1:])
    if _rerun_updates_box(run.bundled_dir):
        return [
            value
            for frame_index in matching_indices
            for value in input_boxes[
                frame_index * box_width : (frame_index + 1) * box_width
            ]
        ]
    restart_boxes = _h5_numeric_values(
        run.bundled_dir / "restart.spgr.h5", "/particles/all/box/edges/value"
    )
    if len(restart_boxes) != box_width:
        raise AssertionError(
            "structural rerun restart box has an invalid shape"
        )
    return restart_boxes * len(matching_indices)


def _compare_h5_outputs_statistically(
    case: AbCase, runs: Sequence[AbRun]
) -> dict[str, object]:
    expected_families = set(_output_h5_files(case, runs[0].legacy_dir))
    summaries: dict[str, object] = {}
    for name in sorted(expected_families):
        baseline_datasets: set[str] | None = None
        for run in runs:
            legacy_path = _output_h5_files(case, run.legacy_dir)[name]
            bundled_path = _output_h5_files(case, run.bundled_dir)[name]
            datasets = _assert_h5_schema_equivalent(
                legacy_path,
                bundled_path,
                f"{case.name} {name} replica {run.replica_index}",
            )
            if baseline_datasets is None:
                baseline_datasets = datasets
            elif baseline_datasets != datasets:
                raise AssertionError(
                    f"{case.name} {name} schema changed across replicas"
                )

        assert baseline_datasets is not None
        dataset_summaries: dict[str, object] = {}
        for dataset in sorted(baseline_datasets):
            legacy_path = _output_h5_files(case, runs[0].legacy_dir)[name]
            kind = _h5_dataset_kind(legacy_path, dataset)
            if kind == "numeric":
                dataset_summaries[dataset] = (
                    _compare_h5_numeric_dataset_statistics(
                        case, runs, name, dataset
                    )
                )
            else:
                for run in runs:
                    legacy_file = _output_h5_files(case, run.legacy_dir)[name]
                    bundled_file = _output_h5_files(case, run.bundled_dir)[name]
                    left = _normalize_h5dump(
                        _h5dump_dataset(legacy_file, dataset)
                    )
                    right = _normalize_h5dump(
                        _h5dump_dataset(bundled_file, dataset)
                    )
                    if left != right:
                        raise AssertionError(
                            f"{case.name} {name} metadata dataset differs: {dataset}"
                        )
                dataset_summaries[dataset] = {"method": "exact_metadata"}
        summaries[name] = {
            "method": "all_dataset_schema_and_statistical_values",
            "dataset_count": len(baseline_datasets),
            "datasets": dataset_summaries,
        }
    return summaries


def _compare_h5_numeric_dataset_statistics(
    case: AbCase, runs: Sequence[AbRun], name: str, dataset: str
) -> dict[str, object]:
    legacy_replicas = []
    bundled_replicas = []
    shape: tuple[int, ...] | None = None
    for run in runs:
        legacy_path = _output_h5_files(case, run.legacy_dir)[name]
        bundled_path = _output_h5_files(case, run.bundled_dir)[name]
        legacy_values = _h5_numeric_values(legacy_path, dataset)
        bundled_values = _h5_numeric_values(bundled_path, dataset)
        _assert_matching_numeric_shape(
            f"{case.name} {name}:{dataset} replica {run.replica_index}",
            legacy_path,
            bundled_path,
            dataset,
            legacy_values,
            bundled_values,
        )
        if shape is None:
            shape = _h5_dataset_shape(legacy_path, dataset)
        legacy_replicas.append(legacy_values)
        bundled_replicas.append(bundled_values)

    if _is_deterministic_timeline_dataset(dataset):
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_numeric_sequences_close(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
                relative_tolerance=0.0,
                absolute_tolerance=1.0e-12,
            )
        return {"method": "exact_timeline_or_completion_metadata"}

    if any(
        not math.isfinite(value)
        for replica in (*legacy_replicas, *bundled_replicas)
        for value in replica
    ):
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_numeric_sequences_close(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
                relative_tolerance=0.0,
                absolute_tolerance=0.0,
            )
        return {"method": "exact_nonfinite_pattern"}

    policy = _statistical_policy()
    flat_policy = replace(policy, burn_in_frames=0)
    result: dict[str, object] = {}
    if _can_use_statistics(legacy_replicas, flat_policy):
        result["flat_values"] = compare_replicas(
            f"{case.name} {name}:{dataset} flat values",
            legacy_replicas,
            bundled_replicas,
            flat_policy,
        )
    else:
        for replica_index, (legacy, bundled) in enumerate(
            zip(legacy_replicas, bundled_replicas)
        ):
            _assert_numeric_sequences_close(
                f"{case.name} {name}:{dataset} replica {replica_index}",
                legacy,
                bundled,
                relative_tolerance=1.0e-4,
                absolute_tolerance=1.0e-8,
            )
        result["flat_values"] = {"method": "exact_short_series"}

    if shape and len(shape) >= 1 and shape[0] > 1:
        legacy_frame_means = [
            _frame_summary_series(values, shape, rms=False)
            for values in legacy_replicas
        ]
        bundled_frame_means = [
            _frame_summary_series(values, shape, rms=False)
            for values in bundled_replicas
        ]
        if _can_use_statistics(legacy_frame_means, policy):
            result["frame_means"] = compare_replicas(
                f"{case.name} {name}:{dataset} frame means",
                legacy_frame_means,
                bundled_frame_means,
                policy,
            )
            legacy_frame_rms = [
                _frame_summary_series(values, shape, rms=True)
                for values in legacy_replicas
            ]
            bundled_frame_rms = [
                _frame_summary_series(values, shape, rms=True)
                for values in bundled_replicas
            ]
            result["frame_rms"] = compare_replicas(
                f"{case.name} {name}:{dataset} frame RMS",
                legacy_frame_rms,
                bundled_frame_rms,
                policy,
            )
    return result


def _can_use_statistics(
    replicas: Sequence[Sequence[float]], policy: StatisticalEquivalencePolicy
) -> bool:
    required = policy.burn_in_frames + (
        policy.minimum_blocks_per_replica * policy.block_size
    )
    return all(len(replica) >= required for replica in replicas)


def _frame_summary_series(
    values: Sequence[float], shape: tuple[int, ...], *, rms: bool
) -> list[float]:
    frame_count = shape[0]
    if frame_count <= 0 or len(values) % frame_count != 0:
        raise AssertionError(
            f"cannot build frame statistics for shape={shape}, values={len(values)}"
        )
    width = len(values) // frame_count
    series = []
    for frame_index in range(frame_count):
        frame = values[frame_index * width : (frame_index + 1) * width]
        if rms:
            series.append(
                math.sqrt(statistics.fmean(value * value for value in frame))
            )
        else:
            series.append(statistics.fmean(frame))
    return series


def _assert_matching_numeric_shape(
    label: str,
    legacy_path: Path,
    bundled_path: Path,
    dataset: str,
    legacy_values: Sequence[float],
    bundled_values: Sequence[float],
) -> None:
    legacy_shape = _h5_dataset_shape(legacy_path, dataset)
    bundled_shape = _h5_dataset_shape(bundled_path, dataset)
    if legacy_shape != bundled_shape or len(legacy_values) != len(
        bundled_values
    ):
        raise AssertionError(
            f"{label} numeric shape mismatch: legacy_shape={legacy_shape}, "
            f"bundled_shape={bundled_shape}, legacy_values={len(legacy_values)}, "
            f"bundled_values={len(bundled_values)}"
        )


def _is_deterministic_timeline_dataset(dataset: str) -> bool:
    return (
        dataset.endswith("/step")
        or dataset.endswith("/time")
        or dataset.endswith("/frame_count")
        or dataset.endswith("/last_complete_step")
        or dataset.endswith("/last_complete_time")
    )


def _assert_h5_schema_equivalent(
    legacy_path: Path, bundled_path: Path, label: str
) -> set[str]:
    legacy_paths = _h5_paths(legacy_path)
    bundled_paths = _h5_paths(bundled_path)
    if legacy_paths != bundled_paths:
        raise AssertionError(
            f"{label} H5 path sets differ: "
            f"legacy_only={sorted(legacy_paths - bundled_paths)}, "
            f"bundled_only={sorted(bundled_paths - legacy_paths)}"
        )
    legacy_datasets = _h5_dataset_paths(legacy_path)
    bundled_datasets = _h5_dataset_paths(bundled_path)
    if legacy_datasets != bundled_datasets:
        raise AssertionError(
            f"{label} H5 dataset sets differ: "
            f"legacy_only={sorted(legacy_datasets - bundled_datasets)}, "
            f"bundled_only={sorted(bundled_datasets - legacy_datasets)}"
        )
    for dataset in legacy_datasets:
        legacy_header = _h5_dataset_header(legacy_path, dataset)
        bundled_header = _h5_dataset_header(bundled_path, dataset)
        if legacy_header != bundled_header:
            raise AssertionError(f"{label} H5 schema differs at {dataset}")
    return legacy_datasets


def _h5_paths(path: Path) -> set[str]:
    output = _run(["h5dump", "-n", path]).stdout
    paths = set()
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] in {"group", "dataset"}:
            paths.add(fields[1])
    return paths


def _h5_dataset_paths(path: Path) -> set[str]:
    output = _run(["h5dump", "-n", path]).stdout
    datasets = set()
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] == "dataset":
            datasets.add(fields[1])
    return datasets


def _h5dump_dataset(path: Path, dataset: str) -> str:
    return _run(["h5dump", "-d", dataset, path]).stdout


def _h5_dataset_header(path: Path, dataset: str) -> str:
    return _normalize_h5dump(_run(["h5dump", "-H", "-d", dataset, path]).stdout)


def _h5_dataset_kind(path: Path, dataset: str) -> str:
    header = _h5_dataset_header(path, dataset)
    match = re.search(
        r"DATASET\s+.*?\{\s*DATATYPE\s+([^\n]+)", header, re.DOTALL
    )
    if match is None:
        raise AssertionError(
            f"cannot determine H5 dataset type: {path}:{dataset}"
        )
    datatype = match.group(1)
    if "H5T_STRING" in datatype:
        return "string"
    if any(
        marker in datatype
        for marker in ("H5T_IEEE", "H5T_STD_I", "H5T_STD_U", "H5T_NATIVE_")
    ):
        return "numeric"
    return "other"


def _h5_dataset_shape(path: Path, dataset: str) -> tuple[int, ...]:
    header = _h5_dataset_header(path, dataset)
    match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s+\(\s*([^)]*?)\s*\)", header)
    if match is None:
        return ()
    dimensions = [item.strip() for item in match.group(1).split(",")]
    if not dimensions or dimensions == [""]:
        return ()
    return tuple(int(item) for item in dimensions)


def _h5_data_text(path: Path, dataset: str) -> str:
    dump = _run(["h5dump", "-d", dataset, "-y", "-w", "0", path]).stdout
    marker = re.search(r"\bDATA\s*\{", dump)
    if marker is None:
        raise AssertionError(
            f"H5 dataset has no data section: {path}:{dataset}"
        )
    start = marker.end()
    depth = 1
    in_string = False
    escaped = False
    for index in range(start, len(dump)):
        character = dump[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return dump[start:index]
    raise AssertionError(
        f"H5 dataset has unterminated data section: {path}:{dataset}"
    )


def _h5_numeric_values(path: Path, dataset: str) -> list[float]:
    if _h5_dataset_kind(path, dataset) != "numeric":
        raise AssertionError(f"H5 dataset is not numeric: {path}:{dataset}")
    data = re.sub(r'"(?:\\.|[^"\\])*"', "", _h5_data_text(path, dataset))
    tokens = re.findall(
        r"(?<![A-Za-z0-9_])[-+]?(?:nan|inf(?:inity)?|"
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?![A-Za-z0-9_])",
        data,
        flags=re.IGNORECASE,
    )
    return [float(token) for token in tokens]


def _h5_string_values(path: Path, dataset: str) -> list[str]:
    if _h5_dataset_kind(path, dataset) != "string":
        raise AssertionError(
            f"H5 dataset is not a string array: {path}:{dataset}"
        )
    return re.findall(r'"((?:\\.|[^"\\])*)"', _h5_data_text(path, dataset))


def _normalize_h5dump(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("HDF5 "):
            continue
        normalized = re.sub(r'FILE\s+"[^"]+"', 'FILE "<vds-source>"', line)
        lines.append(normalized.strip())
    return "\n".join(lines)


def _run(
    cmd: list[object], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with code {result.returncode}: "
            f"{' '.join(str(part) for part in cmd)}\n"
            f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
        )
    return result
