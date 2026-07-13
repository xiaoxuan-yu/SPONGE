from __future__ import annotations

import json
import os
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    OBSERVABLE_REL,
    PROFILE,
    PROFILE_LIMITS,
    REPO_ROOT,
    RESTART_REL,
    TRAJECTORY_REL,
    AbCase,
    AbRun,
    _collect_metrics,
    _compare_h5_outputs_deterministically,
    _compare_h5_outputs_statistically,
    _compare_mdout_deterministically,
    _compare_mdout_statistically,
    _insert_root_toml_keys,
    _mdin_name,
    _output_root,
    _remove_key_lines,
    _reset_xponge_coordinate_start_time,
    _run_sponge,
    _sponge_executable,
)

TIP3P_SOURCE = (
    REPO_ROOT
    / "benchmarks"
    / "validation"
    / "thermostat"
    / "statics"
    / "tip3p_water"
)
MATRIX_FIXTURE_ROOT = (
    REPO_ROOT / "benchmarks" / "bundled_io" / "fixtures" / "tip3p_matrix"
)
MATRIX_SELECTOR_ENV = "SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS"
MATRIX_BACKEND_ENV = "SPONGE_BUNDLED_IO_AB_BACKEND"
MATRIX_EVIDENCE_NAME = "ab_matrix_evidence.json"


@dataclass(frozen=True)
class MatrixRuntimeCase:
    scenario_id: str
    ensemble: str
    thermostat: str
    barostat: str
    box_geometry: str
    constraint: str
    backend: str
    omp_threads: int
    mpi_ranks: int
    comparison: str
    tier: str

    @property
    def statistical(self) -> bool:
        return self.comparison == "statistical"

    def axis_values(self) -> dict[str, object]:
        return {
            "ensemble": self.ensemble,
            "thermostat": self.thermostat,
            "box_geometry": self.box_geometry,
            "constraint": self.constraint,
            "backend": self.backend,
            "omp_threads": self.omp_threads,
            "mpi_ranks": self.mpi_ranks,
            "comparison": self.comparison,
        }


MATRIX_RUNTIME_CASES = (
    MatrixRuntimeCase(
        scenario_id="nve_unconstrained_cpu_omp1_rank1",
        ensemble="nve",
        thermostat="none",
        barostat="none",
        box_geometry="orthogonal",
        constraint="unconstrained",
        backend="cpu",
        omp_threads=1,
        mpi_ranks=1,
        comparison="deterministic",
        tier="medium",
    ),
    MatrixRuntimeCase(
        scenario_id="middle_nvt_shake_cpu_omp1_rank1",
        ensemble="nvt",
        thermostat="middle_langevin",
        barostat="none",
        box_geometry="orthogonal",
        constraint="SHAKE",
        backend="cpu",
        omp_threads=1,
        mpi_ranks=1,
        comparison="statistical",
        tier="medium",
    ),
    MatrixRuntimeCase(
        scenario_id="bussi_nvt_settle_cpu_omp4_rank1",
        ensemble="nvt",
        thermostat="bussi_thermostat",
        barostat="none",
        box_geometry="orthogonal",
        constraint="SETTLE",
        backend="cpu",
        omp_threads=4,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="nhc_nvt_unconstrained_cpu_omp1_rank1",
        ensemble="nvt",
        thermostat="nose_hoover_chain",
        barostat="none",
        box_geometry="orthogonal",
        constraint="unconstrained",
        backend="cpu",
        omp_threads=1,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="npt_orthogonal_settle_cpu_omp1_rank1",
        ensemble="npt",
        thermostat="middle_langevin",
        barostat="andersen_barostat",
        box_geometry="orthogonal",
        constraint="SETTLE",
        backend="cpu",
        omp_threads=1,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="npt_nonorthogonal_shake_cpu_omp4_rank1",
        ensemble="npt",
        thermostat="middle_langevin",
        barostat="andersen_barostat",
        box_geometry="nonorthogonal",
        constraint="SHAKE",
        backend="cpu",
        omp_threads=4,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="nve_settle_cpu_omp4_rank2",
        ensemble="nve",
        thermostat="none",
        barostat="none",
        box_geometry="orthogonal",
        constraint="SETTLE",
        backend="cpu",
        omp_threads=4,
        mpi_ranks=2,
        comparison="deterministic",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="npt_nonorthogonal_shake_cpu_omp4_rank2",
        ensemble="npt",
        thermostat="middle_langevin",
        barostat="andersen_barostat",
        box_geometry="nonorthogonal",
        constraint="SHAKE",
        backend="cpu",
        omp_threads=4,
        mpi_ranks=2,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="middle_nvt_settle_gpu_omp4_rank1",
        ensemble="nvt",
        thermostat="middle_langevin",
        barostat="none",
        box_geometry="orthogonal",
        constraint="SETTLE",
        backend="gpu",
        omp_threads=4,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="nhc_nvt_unconstrained_gpu_omp1_rank1",
        ensemble="nvt",
        thermostat="nose_hoover_chain",
        barostat="none",
        box_geometry="orthogonal",
        constraint="unconstrained",
        backend="gpu",
        omp_threads=1,
        mpi_ranks=1,
        comparison="statistical",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="nve_unconstrained_gpu_omp4_rank1",
        ensemble="nve",
        thermostat="none",
        barostat="none",
        box_geometry="orthogonal",
        constraint="unconstrained",
        backend="gpu",
        omp_threads=4,
        mpi_ranks=1,
        comparison="deterministic",
        tier="production",
    ),
    MatrixRuntimeCase(
        scenario_id="npt_nonorthogonal_settle_gpu_omp1_rank2",
        ensemble="npt",
        thermostat="middle_langevin",
        barostat="andersen_barostat",
        box_geometry="nonorthogonal",
        constraint="SETTLE",
        backend="gpu",
        omp_threads=1,
        mpi_ranks=2,
        comparison="statistical",
        tier="production",
    ),
)


@pytest.mark.parametrize(
    "matrix_case", MATRIX_RUNTIME_CASES, ids=lambda case: case.scenario_id
)
def test_legacy_and_bundled_execution_matrix_behavior(
    matrix_case: MatrixRuntimeCase, monkeypatch
):
    if not _selected(matrix_case):
        pytest.skip(
            f"set {MATRIX_SELECTOR_ENV} to opt into execution-matrix runs"
        )
    if PROFILE not in PROFILE_LIMITS:
        raise AssertionError(
            "SPONGE_BUNDLED_IO_AB_PROFILE must be medium or production"
        )
    configured_backend = os.environ.get(MATRIX_BACKEND_ENV)
    if configured_backend != matrix_case.backend:
        raise AssertionError(
            f"{matrix_case.scenario_id} requires backend "
            f"{matrix_case.backend}, got {configured_backend!r}"
        )
    monkeypatch.setenv("OMP_NUM_THREADS", str(matrix_case.omp_threads))
    case_root = _output_root() / "execution_matrix" / matrix_case.scenario_id
    runs = []
    for replica_index in range(_replica_count(matrix_case)):
        seed = 20260709 + replica_index * 104729
        replica_root = case_root / f"replica_{replica_index:02d}"
        legacy_dir, bundled_dir = _prepare_pair(matrix_case, replica_root)
        _prepare_runtime_mdin(matrix_case, legacy_dir, "mdin.spg.toml", seed)
        _prepare_runtime_mdin(
            matrix_case,
            bundled_dir,
            "mdin.bundled.spg.toml",
            seed,
        )
        legacy_metrics = _run_matrix_sponge(
            matrix_case, legacy_dir, _mdin_name(legacy_dir)
        )
        bundled_metrics = _run_matrix_sponge(
            matrix_case, bundled_dir, _mdin_name(bundled_dir)
        )
        _assert_rank0_output_ownership(matrix_case, legacy_dir)
        _assert_rank0_output_ownership(matrix_case, bundled_dir)
        runs.append(
            AbRun(
                replica_index=replica_index,
                replica_seed=seed,
                legacy_dir=legacy_dir,
                bundled_dir=bundled_dir,
                legacy_metrics=legacy_metrics,
                bundled_metrics=bundled_metrics,
                legacy_output_contract={},
                bundled_output_contract={},
            )
        )

    ab_case = _as_ab_case(matrix_case)
    if matrix_case.statistical:
        mdout = _compare_mdout_statistically(ab_case, runs)
        h5 = _compare_h5_outputs_statistically(ab_case, runs)
    else:
        mdout = _compare_mdout_deterministically(ab_case, runs[0])
        h5 = _compare_h5_outputs_deterministically(ab_case, runs[0])

    artifact_bytes = _directory_bytes(case_root)
    quota = int(
        os.environ.get(
            "SPONGE_BUNDLED_IO_AB_CASE_QUOTA_BYTES", str(2 * 1024**3)
        )
    )
    if artifact_bytes > quota:
        raise AssertionError(
            f"{matrix_case.scenario_id} artifacts exceed quota: "
            f"{artifact_bytes}>{quota}"
        )
    performance = _performance_summary(runs)
    _record_matrix_evidence(
        matrix_case,
        mdout=mdout,
        h5=h5,
        performance=performance,
        artifact_bytes=artifact_bytes,
    )
    if os.environ.get("SPONGE_BUNDLED_IO_AB_RETAIN_SUCCESS") != "1":
        shutil.rmtree(case_root)


def _selected(case: MatrixRuntimeCase) -> bool:
    raw = os.environ.get(MATRIX_SELECTOR_ENV, "").strip()
    if not raw:
        return False
    selectors = {item.strip() for item in raw.split(",") if item.strip()}
    return (
        "all" in selectors
        or case.scenario_id in selectors
        or ("cpu-rank1" in selectors and case.backend == "cpu" and case.mpi_ranks == 1)
        or ("cpu-rank2" in selectors and case.backend == "cpu" and case.mpi_ranks == 2)
        or ("gpu-rank1" in selectors and case.backend == "gpu" and case.mpi_ranks == 1)
        or ("gpu-rank2" in selectors and case.backend == "gpu" and case.mpi_ranks == 2)
        or case.tier in selectors
    )


def _run_matrix_sponge(
    case: MatrixRuntimeCase, case_dir: Path, mdin_name: str
) -> dict[str, object]:
    if case.mpi_ranks == 1:
        return _run_sponge(case_dir, mdin_name)
    launcher = shlex.split(
        os.environ.get("SPONGE_BUNDLED_IO_AB_MPI_LAUNCHER", "mpirun")
    )
    if not launcher or shutil.which(launcher[0]) is None:
        raise AssertionError(
            "MPI execution matrix requires mpirun or "
            "SPONGE_BUNDLED_IO_AB_MPI_LAUNCHER"
        )
    command = [
        *launcher,
        "-np",
        str(case.mpi_ranks),
        _sponge_executable(),
        "-mdin",
        mdin_name,
    ]
    env = dict(os.environ)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=case_dir,
        env=env,
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
            f"MPI SPONGE failed in {case_dir} with code {result.returncode}\n"
            f"command={' '.join(command)}\n"
            f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"
        )
    return _collect_metrics(case_dir, elapsed_s)


def _assert_rank0_output_ownership(
    case: MatrixRuntimeCase, case_dir: Path
) -> None:
    stdout = (case_dir / "run.stdout").read_text(encoding="utf-8")
    finalize_reports = stdout.count("H5 I/O finalize timing:")
    if finalize_reports != 1:
        raise AssertionError(
            f"{case.scenario_id} expected one rank-0 H5 finalize report, "
            f"got {finalize_reports}"
        )
    required = (
        case_dir / "mdout.txt",
        case_dir / TRAJECTORY_REL,
        case_dir / OBSERVABLE_REL,
        case_dir / RESTART_REL,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(
            f"{case.scenario_id} rank-0 outputs are missing: {missing}"
        )


def _replica_count(case: MatrixRuntimeCase) -> int:
    if not case.statistical:
        return 1
    if os.environ.get("SPONGE_BUNDLED_IO_AB_MATRIX_FAST") == "1":
        return 4
    return int(PROFILE_LIMITS[PROFILE]["normal_replicas"])


def _step_and_interval(case: MatrixRuntimeCase) -> tuple[int, int]:
    if not case.statistical:
        return 5, 1
    if os.environ.get("SPONGE_BUNDLED_IO_AB_MATRIX_FAST") == "1":
        return 64, 1
    limits = PROFILE_LIMITS[PROFILE]
    return int(limits["normal_step_limit"]), int(limits["normal_interval"])


def _prepare_pair(
    case: MatrixRuntimeCase, case_root: Path
) -> tuple[Path, Path]:
    legacy_dir = case_root / "legacy"
    bundled_dir = case_root / "bundled"
    for path in (legacy_dir, bundled_dir):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(TIP3P_SOURCE, legacy_dir)
    coordinate = legacy_dir / "tip3p_coordinate.txt"
    _reset_xponge_coordinate_start_time(coordinate)
    if case.box_geometry == "nonorthogonal":
        _make_nonorthogonal(coordinate)
    shutil.copyfile(
        MATRIX_FIXTURE_ROOT / "initial_velocity.txt",
        legacy_dir / "initial_velocity.txt",
    )
    (legacy_dir / "mdin.spg.toml").write_text("", encoding="utf-8")

    shutil.copytree(MATRIX_FIXTURE_ROOT / "common", bundled_dir)
    shutil.copyfile(
        MATRIX_FIXTURE_ROOT / case.box_geometry / "restart.spgr.h5",
        bundled_dir / "restart.spgr.h5",
    )
    return legacy_dir, bundled_dir


def _make_nonorthogonal(coordinate: Path) -> None:
    lines = coordinate.read_text(encoding="utf-8").splitlines()
    box = lines[-1].split()
    if len(box) < 6:
        raise AssertionError(f"coordinate has no six-value box: {coordinate}")
    box[3:6] = ["80.0000000", "100.0000000", "110.0000000"]
    lines[-1] = " ".join(box)
    coordinate.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_runtime_mdin(
    case: MatrixRuntimeCase, case_dir: Path, mdin_name: str, seed: int
) -> None:
    mdin_path = case_dir / mdin_name
    text = _remove_key_lines(mdin_path.read_text(encoding="utf-8"), _runtime_key_names())
    step_limit, interval = _step_and_interval(case)
    additions = _runtime_keys(
        case,
        seed,
        step_limit=step_limit,
        interval=interval,
        legacy_input=mdin_name == "mdin.spg.toml",
    )
    additions.extend(
        [
            'mdout = "mdout.txt"',
            'mdinfo = "mdinfo.txt"',
            'crd = "output/legacy.crd"',
            'box = "output/legacy.box"',
            'vel = "output/legacy.vel"',
            'frc = "output/legacy.frc"',
            'rst = "output/legacy_restart"',
            f'output_h5_trajectory_path = "{TRAJECTORY_REL.as_posix()}"',
            "output_h5_trajectory_vds = false",
            "output_h5_trajectory_chunk_size = 8",
            f'output_h5_observable_path = "{OBSERVABLE_REL.as_posix()}"',
            f'output_h5_restart_path = "{RESTART_REL.as_posix()}"',
        ]
    )
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    mdin_path.write_text(
        _insert_root_toml_keys(text, additions), encoding="utf-8"
    )


def _runtime_keys(
    case: MatrixRuntimeCase,
    seed: int,
    *,
    step_limit: int,
    interval: int,
    legacy_input: bool = True,
) -> list[str]:
    keys = [
        f'md_name = "bundled io matrix {case.scenario_id}"',
        f'mode = "{case.ensemble}"',
        f"step_limit = {step_limit}",
        f"dt = {0.0001 if case.constraint == 'unconstrained' else 0.002}",
        "cutoff = 8.0",
        "print_zeroth_frame = 1",
        f"write_mdout_interval = {interval}",
        f"write_information_interval = {interval}",
        f"write_trajectory_interval = {interval}",
        f"write_restart_file_interval = {step_limit}",
    ]
    if legacy_input:
        keys.extend(
            [
                'default_in_file_prefix = "tip3p"',
                'velocity_in_file = "initial_velocity.txt"',
            ]
        )
    if case.backend == "gpu" and case.mpi_ranks > 1:
        device_map = os.environ.get(
            "SPONGE_BUNDLED_IO_AB_GPU_DEVICES", "0"
        ).strip()
        if not device_map:
            raise AssertionError(
                "SPONGE_BUNDLED_IO_AB_GPU_DEVICES must not be empty"
            )
        keys.append(f'device = "{device_map}"')
    if case.thermostat != "none":
        keys.extend(
            [
                f'thermostat = "{case.thermostat}"',
                f"thermostat_seed = {seed}",
                "thermostat_tau = 0.1",
                "target_temperature = 300.0",
            ]
        )
        if case.thermostat == "bussi_thermostat":
            keys.append('thermostat_mode = "bussi_thermostat"')
    if case.barostat != "none":
        keys.extend(
            [
                f'barostat = "{case.barostat}"',
                "target_pressure = 1.0",
                "barostat_tau = 1.0",
                "barostat_update_interval = 10",
                'barostat_isotropy = "isotropic"',
            ]
        )
    if case.constraint != "unconstrained":
        keys.append(f'constrain_mode = "{case.constraint}"')
    return keys


def _runtime_key_names() -> set[str]:
    return {
        "md_name",
        "mode",
        "step_limit",
        "dt",
        "cutoff",
        "device",
        "default_in_file_prefix",
        "velocity_in_file",
        "print_zeroth_frame",
        "write_mdout_interval",
        "write_information_interval",
        "write_trajectory_interval",
        "write_restart_file_interval",
        "thermostat",
        "thermostat_mode",
        "thermostat_seed",
        "thermostat_tau",
        "target_temperature",
        "barostat",
        "target_pressure",
        "barostat_tau",
        "barostat_update_interval",
        "barostat_isotropy",
        "constrain_mode",
        "mdout",
        "mdinfo",
        "crd",
        "box",
        "vel",
        "frc",
        "rst",
        "output_h5_trajectory_path",
        "output_h5_trajectory_vds",
        "output_h5_trajectory_chunk_size",
        "output_h5_observable_path",
        "output_h5_restart_path",
    }


def _as_ab_case(case: MatrixRuntimeCase) -> AbCase:
    return AbCase(
        name=case.scenario_id,
        fixture_case="tip3p_validation_generated",
        legacy_subdir="generated_legacy",
        bundled_subdir="generated_bundled",
        mode="normal",
        vds=False,
        statistical_md=case.statistical,
        restart_load_policy="structural",
        contract_ids=(),
        assertion_ids=(),
    )


def _performance_summary(runs: list[AbRun]) -> dict[str, float]:
    runtime_ratios = [
        float(run.bundled_metrics["elapsed_s"])
        / max(float(run.legacy_metrics["elapsed_s"]), 1.0e-12)
        for run in runs
    ]
    finalize_fractions = [
        float(run.bundled_metrics["flush_finalize_elapsed_s"])
        / max(float(run.bundled_metrics["elapsed_s"]), 1.0e-12)
        for run in runs
    ]
    byte_ratios = [
        float(run.bundled_metrics["h5_files_total_bytes"])
        / max(float(run.legacy_metrics["h5_files_total_bytes"]), 1.0)
        for run in runs
    ]
    return {
        "runtime_ratio": statistics.median(runtime_ratios),
        "finalize_fraction": max(finalize_fractions),
        "output_bytes_ratio": statistics.median(byte_ratios),
    }


def _record_matrix_evidence(
    case: MatrixRuntimeCase,
    *,
    mdout: dict[str, object],
    h5: dict[str, object],
    performance: dict[str, float],
    artifact_bytes: int,
) -> None:
    path = _output_root() / MATRIX_EVIDENCE_NAME
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"schema_version": "1", "cases": {}}
    cases = report.setdefault("cases", {})
    cases[case.scenario_id] = {
        "metadata": {
            **case.axis_values(),
            "omp_num_threads": case.omp_threads,
            "mpi_rank_count": case.mpi_ranks,
            "rank0_output_owner": True,
            "profile": PROFILE,
            "performance": performance,
            "artifact_bytes": artifact_bytes,
        },
        "records": [
            {
                "contract_id": "runtime.execution_matrix",
                "evidence_level": "E3",
                "status": "passed",
                "details": {
                    "mdout_method": mdout["method"],
                    "h5_families": sorted(h5),
                },
            }
        ],
    }
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
