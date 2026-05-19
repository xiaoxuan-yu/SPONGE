import json
import math
import time

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
        case_dir = copy_ala2_case(
            repo_root, run_dir, f"schedule_{schedule_id}"
        )
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
