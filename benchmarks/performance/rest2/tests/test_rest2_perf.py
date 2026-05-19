import json
import math
import shutil
import time
from pathlib import Path

from benchmarks.utils import Outputer, Runner
from benchmarks.validation.rest2.tests.utils import (
    copy_ala2_case,
    parse_exchange_log,
    repo_root_from_test_file,
    resolve_executable,
    runtime_env,
    write_rest2_manager_config,
)
from benchmarks.validation.utils import parse_mdout_rows

FEP_REST2_LAMBDAS = (0.0, 0.1, 0.2, 0.3)
FEP_REST2_LAMBDA_MS = (1.0, 0.9, 0.8, 0.7)
FEP_REST2_HOT_ATOMS = 55


def _write_mdin(case_dir, *, step_limit, lambda_m=None):
    lines = [
        'md_name = "performance alanine_dipeptide_tip3p_water REST2"',
        'mode = "nvt"',
        f"step_limit = {step_limit}",
        "dt = 0.002",
        "cutoff = 8.0",
        'thermostat = "middle_langevin"',
        "thermostat_tau = 1.0",
        "thermostat_seed = 2026",
        "target_temperature = 300.0",
        'default_in_file_prefix = "ALA"',
        "print_zeroth_frame = 1",
        "write_mdout_interval = 50",
        "write_information_interval = 50",
        "write_trajectory_interval = 0",
        "write_restart_file_interval = 0",
        'constrain_mode = "SHAKE"',
    ]
    if lambda_m is not None:
        lines.extend(
            [
                'REST2_mode = "on"',
                "REST2_atom_numbers = 22",
                f"REST2_lambda_m = {lambda_m}",
            ]
        )
    (case_dir / "mdin.spg.toml").write_text("\n".join(lines) + "\n")


def test_rest2_micro_benchmark(
    outputs_path, sponge_cmd, rest2_perf_steps, rest2_perf_timeout
):
    repo_root = repo_root_from_test_file(__file__)
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)
    env = runtime_env(repo_root)
    cases = [
        ("baseline", None),
        ("rest2_lambda_1", 1.0),
        ("rest2_lambda_08", 0.8),
    ]
    summaries = []

    for label, lambda_m in cases:
        case_dir = copy_ala2_case(repo_root, outputs_path, label)
        _write_mdin(case_dir, step_limit=rest2_perf_steps, lambda_m=lambda_m)
        start = time.perf_counter()
        Runner.run_sponge(
            case_dir,
            timeout=rest2_perf_timeout,
            sponge_cmd=resolved_sponge,
            env=env,
        )
        elapsed_s = time.perf_counter() - start
        rows = parse_mdout_rows(
            case_dir / "mdout.txt", ("step", "potential"), int_columns=("step",)
        )
        summary = {
            "case": label,
            "lambda_m": lambda_m,
            "steps": rest2_perf_steps,
            "elapsed_s": elapsed_s,
            "steps_per_s": rest2_perf_steps / elapsed_s,
            "last_step": rows[-1]["step"],
            "last_potential": rows[-1]["potential"],
        }
        if lambda_m is not None:
            rest2_rows = parse_mdout_rows(
                case_dir / "mdout.txt",
                ("REST2_lambda_m", "REST2_bias"),
                int_columns=(),
            )
            summary["rest2_lambda_m"] = rest2_rows[-1]["REST2_lambda_m"]
            summary["rest2_bias"] = rest2_rows[-1]["REST2_bias"]
        summaries.append(summary)

    (outputs_path / "rest2_micro_benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    Outputer.print_table(
        ["Case", "Steps", "Elapsed(s)", "Steps/s", "Potential"],
        [
            [
                row["case"],
                row["steps"],
                f"{row['elapsed_s']:.3f}",
                f"{row['steps_per_s']:.3f}",
                f"{row['last_potential']:.3f}",
            ]
            for row in summaries
        ],
        title="Performance Benchmark: REST2 ALA2 Micro-Benchmark",
    )
    assert all(row["steps_per_s"] > 0.0 for row in summaries)
    assert math.isclose(summaries[1]["rest2_bias"], 0.0, abs_tol=1.0e-4)
    assert not math.isclose(summaries[2]["rest2_bias"], 0.0, abs_tol=1.0e-4)


def test_rest2_remd_manager_micro_benchmark(
    outputs_path,
    sponge_cmd,
    manager_cmd,
    rest2_perf_steps,
    rest2_perf_timeout,
):
    repo_root = repo_root_from_test_file(__file__)
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)
    resolved_manager = resolve_executable(
        manager_cmd, "SPONGE_MANAGER", repo_root
    )
    run_dir = outputs_path / "rest2_remd_manager"
    lambdas = (1.0, 0.9, 0.8, 0.7)
    block_steps = max(1, min(rest2_perf_steps, 10))
    epochs = 3
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    for schedule_id in range(len(lambdas)):
        case_dir = copy_ala2_case(repo_root, run_dir, str(schedule_id))
        _write_mdin(case_dir, step_limit=1000, lambda_m=1.0)

    config_path, log_path = write_rest2_manager_config(
        run_dir,
        lambdas=lambdas,
        block_steps=block_steps,
        epochs=epochs,
        sponge_cmd=resolved_sponge,
    )
    start = time.perf_counter()
    Runner.run_command(
        [resolved_manager, "--config", config_path],
        cwd=run_dir,
        timeout=rest2_perf_timeout,
        env=runtime_env(repo_root),
    )
    elapsed_s = time.perf_counter() - start
    attempts, states = parse_exchange_log(log_path)
    accepted = [row for row in attempts if row["accepted"] == "1"]
    final_walkers = [int(row["walker_id"]) for row in states[-len(lambdas) :]]
    total_replica_steps = len(lambdas) * block_steps * epochs
    summary = {
        "case": "rest2_remd_manager",
        "replicas": len(lambdas),
        "lambdas": lambdas,
        "block_steps": block_steps,
        "epochs": epochs,
        "total_replica_steps": total_replica_steps,
        "elapsed_s": elapsed_s,
        "aggregate_steps_per_s": total_replica_steps / elapsed_s,
        "exchange_attempts": len(attempts),
        "accepted_exchanges": len(accepted),
        "acceptance_ratio": len(accepted) / len(attempts),
        "final_walker_ids": final_walkers,
    }
    (outputs_path / "rest2_remd_manager_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    assert summary["aggregate_steps_per_s"] > 0.0
    assert summary["exchange_attempts"] == 5
    assert 0.0 <= summary["acceptance_ratio"] <= 1.0
    assert sorted(final_walkers) == list(range(len(lambdas)))
    assert final_walkers != list(range(len(lambdas)))


def _required_fep_files(fep_root):
    files = []
    for schedule_id in range(len(FEP_REST2_LAMBDAS)):
        replica_root = Path(fep_root) / str(schedule_id)
        files.extend(
            [
                replica_root / "TMP_LJ_soft_core.txt",
                replica_root / "TMP_coordinate.txt",
                replica_root / "TMP_velocity.txt",
                replica_root / "TMP_mass.txt",
                replica_root / "TMP_charge.txt",
                replica_root / "TMP_subsys_division.txt",
            ]
        )
    return files


def _write_fep_rest2_manager_config(
    run_dir,
    *,
    fep_root,
    sponge_cmd,
    block_steps,
    epochs,
):
    log_path = Path(run_dir) / "manager_exchange.log"
    ids = list(range(len(FEP_REST2_LAMBDAS)))
    prefixes = [
        Path(fep_root) / str(schedule_id) / "TMP" for schedule_id in ids
    ]
    coordinates = [
        Path(fep_root) / str(schedule_id) / "TMP_coordinate.txt"
        for schedule_id in ids
    ]
    velocities = [
        Path(fep_root) / str(schedule_id) / "TMP_velocity.txt"
        for schedule_id in ids
    ]
    lines = [
        "[manager]",
        f"block_steps = {block_steps}",
        f"epochs = {epochs}",
        'transport = "tcp"',
        f'log_path = "{log_path}"',
        "",
        "[exchange]",
        "enabled = true",
        'mode = "rest2"',
        "",
        "[worker_defaults]",
        f'executable = "{sponge_cmd}"',
        'args = ["-dont_check_input", "1"]',
        f'working_directory_root = "{run_dir}"',
        "",
        "[worker_defaults.inputs]",
        'md_name = "FEP NPT REST2 manager benchmark"',
        'mode = "NPT"',
        "dt = 0.002",
        "cutoff = 8.0",
        'constrain_mode = "SHAKE"',
        'barostat = "andersen_barostat"',
        'thermostat = "middle_langevin"',
        "thermostat_tau = 0.1",
        "thermostat_seed = 2026",
        "target_temperature = 300.0",
        "velocity_max = 20",
        'REST2_mode = "on"',
        f"REST2_atom_numbers = {FEP_REST2_HOT_ATOMS}",
        'default_out_file_prefix = "fep_rest2"',
        "write_information_interval = 1000",
        "write_mdout_interval = 1000",
        "write_trajectory_interval = 1000",
        "write_restart_file_interval = 1000",
        "",
        "[schedules]",
        "ids = [" + ", ".join(str(schedule_id) for schedule_id in ids) + "]",
        "",
        "[schedules.inputs]",
        "lambda_lj = ["
        + ", ".join(f"{lambda_value:g}" for lambda_value in FEP_REST2_LAMBDAS)
        + "]",
        "REST2_lambda_m = ["
        + ", ".join(f"{lambda_m:g}" for lambda_m in FEP_REST2_LAMBDA_MS)
        + "]",
        "default_in_file_prefix = ["
        + ", ".join(f'"{path.as_posix()}"' for path in prefixes)
        + "]",
        "coordinate_in_file = ["
        + ", ".join(f'"{path.as_posix()}"' for path in coordinates)
        + "]",
        "velocity_in_file = ["
        + ", ".join(f'"{path.as_posix()}"' for path in velocities)
        + "]",
        "",
    ]
    for schedule_id in ids:
        (Path(run_dir) / str(schedule_id)).mkdir(parents=True, exist_ok=True)
    config_path = Path(run_dir) / "manager.toml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path, log_path


def test_fep_npt_rest2_manager_smoke(
    outputs_path,
    sponge_cmd,
    manager_cmd,
    rest2_perf_steps,
    rest2_perf_timeout,
    fep_rest2_root,
):
    missing = [
        path
        for path in _required_fep_files(fep_rest2_root)
        if not path.exists()
    ]
    if missing:
        import pytest

        pytest.skip(
            "external FEP+REST2 fixture is not available; first missing file: "
            f"{missing[0]}"
        )

    repo_root = repo_root_from_test_file(__file__)
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)
    resolved_manager = resolve_executable(
        manager_cmd, "SPONGE_MANAGER", repo_root
    )
    run_dir = outputs_path / "fep_npt_rest2_manager"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    block_steps = max(2, min(rest2_perf_steps, 5))
    epochs = 1
    config_path, log_path = _write_fep_rest2_manager_config(
        run_dir,
        fep_root=fep_rest2_root,
        sponge_cmd=resolved_sponge,
        block_steps=block_steps,
        epochs=epochs,
    )
    start = time.perf_counter()
    Runner.run_command(
        [resolved_manager, "--config", config_path],
        cwd=run_dir,
        timeout=rest2_perf_timeout,
        env=runtime_env(repo_root),
    )
    elapsed_s = time.perf_counter() - start

    summaries = []
    attempts, states = parse_exchange_log(log_path)
    assert attempts
    assert states

    for schedule_id, (lambda_lj, lambda_m) in enumerate(
        zip(FEP_REST2_LAMBDAS, FEP_REST2_LAMBDA_MS)
    ):
        schedule_dir = run_dir / str(schedule_id)
        mdout = schedule_dir / f"fep_rest2_{schedule_id}.out"
        mdinfo = schedule_dir / f"fep_rest2_{schedule_id}.info"
        coordinate_out = (
            schedule_dir / f"fep_rest2_{schedule_id}_coordinate.txt"
        )
        velocity_out = schedule_dir / f"fep_rest2_{schedule_id}_velocity.txt"
        assert mdout.exists()
        assert mdinfo.exists()
        mdout_text = mdout.read_text(encoding="utf-8")
        assert "REST2_lambda_m" in mdout_text
        assert "REST2_bias" in mdout_text
        mdinfo_text = mdinfo.read_text()
        assert f"FEP lj lambda: {lambda_lj:.6f}" in mdinfo_text
        assert f"REST2 lambda_m set to {lambda_m:.6f}" in mdinfo_text
        summaries.append(
            {
                "schedule_id": schedule_id,
                "lambda_lj": lambda_lj,
                "rest2_lambda_m": lambda_m,
                "mdout_has_rest2_columns": True,
                "coordinate_output_bytes": (
                    coordinate_out.stat().st_size
                    if coordinate_out.exists()
                    else 0
                ),
                "velocity_output_bytes": (
                    velocity_out.stat().st_size if velocity_out.exists() else 0
                ),
            }
        )

    summary = {
        "case": "fep_npt_rest2_manager",
        "fep_root": str(fep_rest2_root),
        "replicas": len(FEP_REST2_LAMBDAS),
        "block_steps": block_steps,
        "epochs": epochs,
        "elapsed_s": elapsed_s,
        "aggregate_steps_per_s": (
            len(FEP_REST2_LAMBDAS) * block_steps * epochs / elapsed_s
        ),
        "exchange_attempts": len(attempts),
        "accepted_exchanges": len(
            [row for row in attempts if row["accepted"] == "1"]
        ),
        "schedules": summaries,
    }
    (outputs_path / "fep_npt_rest2_manager_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    assert summary["aggregate_steps_per_s"] > 0.0
    assert summary["exchange_attempts"] == len(FEP_REST2_LAMBDAS) // 2
