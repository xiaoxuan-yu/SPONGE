import csv
import json
import os
import shutil
import time
from pathlib import Path

from benchmarks.utils import Outputer, Runner


def repo_root_from_test_file(test_file):
    return Path(test_file).resolve().parents[4]


def resolve_executable(explicit_path, executable_name, repo_root):
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Executable not found: {explicit_path}")
        return explicit_path

    found = shutil.which(executable_name)
    if found:
        return Path(found).resolve()

    candidates = [
        repo_root / "build" / executable_name,
        repo_root / "build-dev-cuda13" / executable_name,
        repo_root / "build-dev-cuda12" / executable_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find {executable_name}. Set --{executable_name.lower()}-cmd "
        "or add it to PATH."
    )


def build_runtime_env(repo_root):
    env = os.environ.copy()
    pixi_env = repo_root / ".pixi" / "envs" / "dev-cuda13"
    library_paths = [
        pixi_env / "lib",
        pixi_env / "targets" / "x86_64-linux" / "lib",
    ]
    existing = env.get("LD_LIBRARY_PATH")
    path_text = os.pathsep.join(
        str(path) for path in library_paths if path.exists()
    )
    if path_text:
        env["LD_LIBRARY_PATH"] = (
            path_text if not existing else path_text + os.pathsep + existing
        )
    return env


def write_ice_mdin(run_dir, ice_root, block_steps, epochs):
    mdin_path = run_dir / "ice_tremd.spg.toml"
    input_prefix = ice_root / "ice_Ih_cubic_8064w"
    coordinate_file = ice_root / "pre_eq" / "pre_eq_coordinate.txt"
    velocity_file = ice_root / "pre_eq" / "pre_eq_velocity.txt"
    mdin_path.write_text(
        "\n".join(
            [
                'md_name = "ice Ih T-REMD benchmark"',
                'mode = "npt"',
                "pbc = true",
                "dt = 0.002",
                "# Manager must override this when running multiple epochs.",
                "step_limit = 1",
                f'default_in_file_prefix = "{input_prefix}"',
                f'coordinate_in_file = "{coordinate_file}"',
                f'velocity_in_file = "{velocity_file}"',
                "target_pressure = 1",
                "",
                "[write]",
                f"information_interval = {block_steps}",
                "restart_file_interval = 0",
                "trajectory_interval = 0",
                "",
                "[thermostat]",
                'mode = "middle_langevin"',
                "",
                "[barostat]",
                'mode = "berendsen_barostat"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return mdin_path


def write_manager_config(
    run_dir,
    manager_mode,
    temperatures,
    block_steps,
    epochs,
    mdin_path,
    sponge_cmd,
):
    log_path = run_dir / "manager_exchange.log"
    ids = list(range(len(temperatures)))
    lines = [
        "[manager]",
        f"block_steps = {block_steps}",
        f"epochs = {epochs}",
        f'transport = "{manager_mode}"',
        f'log_path = "{log_path}"',
        "",
        "[exchange]",
        "enabled = true",
        'mode = "tremd"',
        "start_round = 0",
        "",
        "[worker_defaults]",
        f'mdin = "{mdin_path}"',
        "emit_output = false",
    ]
    _ = sponge_cmd
    lines.extend(
        [
            "args = [",
            '  "-workspace", ".",',
            '  "-dont_check_input", "1",',
            "]",
            f'working_directory_root = "{run_dir}"',
            "",
            "[worker_defaults.inputs]",
            'default_out_file_prefix = "ice_tremd"',
            "",
            "[schedules]",
            "ids = ["
            + ", ".join(str(schedule_id) for schedule_id in ids)
            + "]",
            "",
            "[schedules.inputs]",
            "target_temperature = ["
            + ", ".join(f"{temperature:g}" for temperature in temperatures)
            + "]",
            "",
        ]
    )

    for schedule_id in ids:
        (run_dir / str(schedule_id)).mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "manager.toml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path, log_path


def parse_exchange_log(log_path):
    with Path(log_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        attempts = []
        states = []
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "exchange_attempt":
                attempts.append(
                    {
                        "accepted": fields[-1],
                        "acceptance_probability": fields[-3],
                    }
                )
            elif fields[0] == "schedule_state":
                states.append(
                    {
                        "schedule_id": fields[4],
                        "walker_id": fields[5],
                    }
                )
    return attempts, states


def summarize_remd_run(
    *,
    mode,
    temperatures,
    block_steps,
    epochs,
    elapsed_s,
    attempts,
    states,
):
    accepted = sum(1 for row in attempts if row.get("accepted") == "1")
    total_steps = len(temperatures) * block_steps * epochs
    simulated_ps = total_steps * 0.002
    elapsed_days = elapsed_s / 86400.0
    aggregate_ns_per_day = (
        (simulated_ps / 1000.0) / elapsed_days if elapsed_days > 0 else 0.0
    )
    final_states = states[-len(temperatures) :] if states else []
    return {
        "mode": mode,
        "replicas": len(temperatures),
        "temperatures": temperatures,
        "block_steps": block_steps,
        "epochs": epochs,
        "total_replica_steps": total_steps,
        "simulated_ps_aggregate": simulated_ps,
        "elapsed_s": elapsed_s,
        "aggregate_steps_per_s": total_steps / elapsed_s,
        "aggregate_ns_per_day": aggregate_ns_per_day,
        "exchange_attempts": len(attempts),
        "accepted_exchanges": accepted,
        "acceptance_ratio": accepted / len(attempts) if attempts else 0.0,
        "final_walker_ids": [
            int(row["walker_id"])
            for row in final_states
            if row.get("walker_id")
        ],
    }


def run_ice_tremd_case(
    *,
    run_dir,
    mode,
    ice_root,
    temperatures,
    block_steps,
    epochs,
    manager_cmd,
    sponge_cmd,
    timeout,
    repo_root,
):
    mdin_path = write_ice_mdin(run_dir, ice_root, block_steps, epochs)
    config_path, log_path = write_manager_config(
        run_dir,
        mode,
        temperatures,
        block_steps,
        epochs,
        mdin_path,
        sponge_cmd,
    )

    env = build_runtime_env(repo_root)
    start = time.perf_counter()
    output = Runner.run_command(
        [manager_cmd, "--config", config_path],
        cwd=run_dir,
        timeout=timeout,
        env=env,
    )
    elapsed_s = time.perf_counter() - start

    attempts, states = parse_exchange_log(log_path)
    summary = summarize_remd_run(
        mode=mode,
        temperatures=temperatures,
        block_steps=block_steps,
        epochs=epochs,
        elapsed_s=elapsed_s,
        attempts=attempts,
        states=states,
    )
    (run_dir / "manager_stdout.log").write_text(output, encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def print_summary_table(summaries):
    Outputer.print_table(
        [
            "Mode",
            "Replicas",
            "Steps",
            "Elapsed(s)",
            "Steps/s",
            "Agg ns/day",
            "Attempts",
            "Accept",
        ],
        [
            [
                row["mode"],
                row["replicas"],
                row["total_replica_steps"],
                f"{row['elapsed_s']:.3f}",
                f"{row['aggregate_steps_per_s']:.3f}",
                f"{row['aggregate_ns_per_day']:.6f}",
                row["exchange_attempts"],
                f"{row['acceptance_ratio']:.3f}",
            ]
            for row in summaries
        ],
        title="Performance Benchmark: Ice Ih 5-Replica T-REMD",
    )
