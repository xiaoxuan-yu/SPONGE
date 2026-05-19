import math
import shutil

import pytest

from benchmarks.utils import Runner
from benchmarks.validation.rest2.tests.utils import (
    copy_ala2_case,
    parse_exchange_log,
    repo_root_from_test_file,
    resolve_executable,
    runtime_env,
    write_rest2_manager_config,
    write_rest2_mdin,
)
from benchmarks.validation.utils import parse_mdout_rows


REST2_COLUMNS = (
    "REST2_lambda_m",
    "REST2_unscaled",
    "REST2_effective",
    "REST2_bias",
)


def _repo_root():
    return repo_root_from_test_file(__file__)


def _run_sponge(case_dir, *, sponge_cmd, timeout):
    repo_root = _repo_root()
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)
    Runner.run_sponge(
        case_dir,
        timeout=timeout,
        sponge_cmd=resolved_sponge,
        env=runtime_env(repo_root),
    )


def test_rest2_single_replica_energy_columns_and_scaling(
    outputs_path, sponge_cmd, rest2_timeout
):
    repo_root = _repo_root()
    first_rows = {}
    for lambda_m in (1.0, 0.8, 0.6):
        run_name = f"single_lambda_{str(lambda_m).replace('.', 'p')}"
        case_dir = copy_ala2_case(repo_root, outputs_path, run_name)
        write_rest2_mdin(case_dir, lambda_m, step_limit=20)
        _run_sponge(case_dir, sponge_cmd=sponge_cmd, timeout=rest2_timeout)

        rows = parse_mdout_rows(
            case_dir / "mdout.txt", REST2_COLUMNS, int_columns=()
        )
        assert rows
        first_rows[lambda_m] = rows[0]
        for row in rows:
            assert row["REST2_lambda_m"] == pytest.approx(lambda_m)
            for column in REST2_COLUMNS[1:]:
                assert math.isfinite(row[column])

    assert first_rows[1.0]["REST2_bias"] == pytest.approx(0.0, abs=1.0e-4)
    assert first_rows[1.0]["REST2_effective"] == pytest.approx(
        first_rows[1.0]["REST2_unscaled"], abs=1.0e-4
    )
    assert first_rows[0.8]["REST2_bias"] > first_rows[1.0]["REST2_bias"]
    assert first_rows[0.6]["REST2_bias"] > first_rows[0.8]["REST2_bias"]


def test_rest2_manager_smoke_with_schedule_input_overrides(
    outputs_path, sponge_cmd, manager_cmd, rest2_timeout
):
    repo_root = _repo_root()
    resolved_sponge = resolve_executable(sponge_cmd, "SPONGE", repo_root)
    resolved_manager = resolve_executable(
        manager_cmd, "SPONGE_MANAGER", repo_root
    )
    run_dir = outputs_path / "manager_rest2_smoke"
    lambdas = (1.0, 0.9, 0.8, 0.7)
    block_steps = 3
    epochs = 3
    if run_dir.exists():
        shutil.rmtree(run_dir)

    for schedule_id in range(len(lambdas)):
        case_dir = copy_ala2_case(
            repo_root, run_dir, f"schedule_{schedule_id}"
        )
        write_rest2_mdin(case_dir, 1.0, step_limit=1000)

    config_path, log_path = write_rest2_manager_config(
        run_dir,
        lambdas=lambdas,
        block_steps=block_steps,
        epochs=epochs,
        sponge_cmd=resolved_sponge,
    )

    output = Runner.run_command(
        [resolved_manager, "--config", config_path],
        cwd=run_dir,
        timeout=rest2_timeout,
        env=runtime_env(repo_root),
    )
    assert "REST2-REMD epoch" in output
    assert log_path.exists()

    attempts, states = parse_exchange_log(log_path)
    assert len(attempts) == 5
    assert len(states) == len(lambdas) * epochs
    assert {row["mode"] for row in attempts} == {"rest2"}
    assert {row["mode"] for row in states} == {"rest2"}

    probabilities = [float(row["acceptance_probability"]) for row in attempts]
    random_values = [float(row["random_value"]) for row in attempts]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert all(0.0 <= value <= 1.0 for value in random_values)
    assert len(set(random_values)) > 1

    accepted = [row for row in attempts if row["accepted"] == "1"]
    assert accepted
    final_walker_ids = [
        int(row["walker_id"]) for row in states[-len(lambdas) :]
    ]
    assert sorted(final_walker_ids) == list(range(len(lambdas)))
    assert final_walker_ids != list(range(len(lambdas)))

    for schedule_id, lambda_m in enumerate(lambdas):
        mdout = (
            run_dir
            / f"schedule_{schedule_id}"
            / f"rest2_smoke_{schedule_id}.out"
        )
        mdinfo = (
            run_dir
            / f"schedule_{schedule_id}"
            / f"rest2_smoke_{schedule_id}.info"
        )
        assert mdout.exists()
        assert mdinfo.exists()
        mdout_header = mdout.read_text().splitlines()[0]
        for column in REST2_COLUMNS:
            assert column in mdout_header
        assert f"REST2 lambda_m set to {lambda_m:.6f}" in mdinfo.read_text()
