import csv
import os
import shutil
from pathlib import Path


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

    for build_dir in (
        "build-dev-cuda13-hostcc",
        "build-dev-cuda13",
        "build-dev-cuda12",
        "build",
    ):
        candidate = repo_root / build_dir / executable_name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find {executable_name}; pass --{executable_name.lower()}-cmd"
    )


def runtime_env(repo_root):
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


def copy_ala2_case(repo_root, outputs_path, run_name):
    source = repo_root / "benchmarks/performance/sits/statics/ala2_sits"
    if not source.exists():
        raise FileNotFoundError(f"ALA2 fixture not found: {source}")
    case_dir = Path(outputs_path) / run_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, case_dir)
    return case_dir


def write_rest2_mdin(case_dir, lambda_m, *, step_limit=20):
    mdin = "\n".join(
        [
            'md_name = "validation alanine_dipeptide_tip3p_water REST2"',
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
            "write_mdout_interval = 5",
            "write_information_interval = 10",
            'constrain_mode = "SHAKE"',
            'REST2_mode = "on"',
            "REST2_atom_numbers = 22",
            f"REST2_lambda_m = {lambda_m}",
        ]
    )
    (Path(case_dir) / "mdin.spg.toml").write_text(mdin + "\n")


def write_rest2_manager_config(
    run_dir,
    *,
    lambdas,
    block_steps,
    epochs,
    sponge_cmd,
):
    log_path = Path(run_dir) / "manager_exchange.log"
    ids = list(range(len(lambdas)))
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
        "emit_output = false",
        f'executable = "{sponge_cmd}"',
        'args = ["-mdin", "mdin.spg.toml", "-dont_check_input", "1"]',
        f'working_directory_root = "{run_dir}"',
        "",
        "[worker_defaults.inputs]",
        "target_temperature = 300.0",
        'default_out_file_prefix = "rest2_smoke"',
        "",
        "[schedules]",
        "ids = [" + ", ".join(str(schedule_id) for schedule_id in ids) + "]",
        "",
        "[schedules.inputs]",
        "REST2_lambda_m = ["
        + ", ".join(f"{lambda_m:g}" for lambda_m in lambdas)
        + "]",
        "",
    ]
    config_path = Path(run_dir) / "manager.toml"
    config_path.write_text("\n".join(lines))
    return config_path, log_path


def parse_exchange_log(log_path):
    attempts = []
    states = []
    with Path(log_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["record_type"] == "exchange_attempt":
                attempts.append(row)
            elif row["record_type"] == "schedule_state":
                states.append(row)
    return attempts, states
