import json
import sys

import pytest

from benchmarks.performance.clustered_lj.run_migration_matrix import (
    ACTIVE_VIEW_ENV,
    DNA_AB_WARNING,
    HISTORICAL_BASELINE_DNA_ONLY_ENV,
    HISTORICAL_BASELINE_DNA_PROBES,
    LIFECYCLE_ENV,
    ROLLING_CACHE_ENV,
    STALE_DIRECT_ENV,
    GpuIdleSample,
    MatrixCase,
    MatrixRunError,
    build_case_plan,
    build_process_environment,
    infer_lj_mode,
    main,
    parse_timings,
    require_finite_mdout,
    resolve_case_environment,
    run_case,
    stage_case,
    validate_baseline_environment,
    validate_environment,
)


def valid_environment():
    return {
        ACTIVE_VIEW_ENV: "1",
        LIFECYCLE_ENV: "outer",
    }


def valid_historical_baseline_environment():
    return valid_environment() | {
        **{key: "1" for key in HISTORICAL_BASELINE_DNA_PROBES},
        "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING": "1",
        STALE_DIRECT_ENV: "1",
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({}, f"{ACTIVE_VIEW_ENV}=1 is required"),
        (
            {ACTIVE_VIEW_ENV: "1"},
            f"{LIFECYCLE_ENV} must be 'outer' or 'outer-source'",
        ),
        (
            {
                ACTIVE_VIEW_ENV: "1",
                LIFECYCLE_ENV: "outer",
                STALE_DIRECT_ENV: "1",
            },
            "has no production reader",
        ),
        (
            {
                ACTIVE_VIEW_ENV: "1",
                LIFECYCLE_ENV: "outer",
                "SPONGE_CLUSTERED_EXPERIMENT_PROBE": "1",
            },
            "may not contain probe keys",
        ),
        (
            {
                ACTIVE_VIEW_ENV: "1",
                LIFECYCLE_ENV: "outer",
                ROLLING_CACHE_ENV: "1",
            },
            "is unsupported",
        ),
    ],
)
def test_environment_contract_rejects_incomplete_or_obsolete_paths(
    update, message
):
    with pytest.raises(MatrixRunError, match=message):
        validate_environment(update)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            {"SPONGE_CLUSTERED_UNKNOWN_PROBE": "1"},
            "contains unsupported probes",
        ),
        (
            {
                next(iter(HISTORICAL_BASELINE_DNA_PROBES)): "1",
                "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING": "1",
            },
            "complete frozen DNA split3/split2 probe set",
        ),
        (
            {
                **{key: "1" for key in HISTORICAL_BASELINE_DNA_PROBES},
                next(iter(HISTORICAL_BASELINE_DNA_PROBES)): "0",
                "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING": "1",
            },
            "complete frozen DNA split3/split2 probe set",
        ),
        (
            {
                **{key: "1" for key in HISTORICAL_BASELINE_DNA_PROBES},
            },
            "require.*FULL_DENSE_PADDING=1",
        ),
        (
            {
                **{key: "1" for key in HISTORICAL_BASELINE_DNA_PROBES},
                "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING": "1",
            },
            f"require.*{STALE_DIRECT_ENV}=1",
        ),
    ],
)
def test_historical_baseline_rejects_ambiguous_probe_selection(update, message):
    environment = valid_environment() | update
    with pytest.raises(MatrixRunError, match=message):
        validate_baseline_environment(environment)


def test_historical_baseline_probes_are_dna_only():
    baseline_environment = valid_historical_baseline_environment()
    current_environment = valid_environment()
    validate_baseline_environment(baseline_environment)

    baseline_water = resolve_case_environment(
        MatrixCase(1, "baseline", "wat160k", "nvt", 10_000),
        baseline_environment=baseline_environment,
        current_environment=current_environment,
    )
    baseline_dna = resolve_case_environment(
        MatrixCase(1, "baseline", "dna_cou", "nvt", 10_000),
        baseline_environment=baseline_environment,
        current_environment=current_environment,
    )
    current_dna = resolve_case_environment(
        MatrixCase(1, "current", "dna_cou", "nvt", 10_000),
        baseline_environment=baseline_environment,
        current_environment=current_environment,
    )

    assert not (HISTORICAL_BASELINE_DNA_ONLY_ENV & baseline_water.keys())
    assert baseline_water[STALE_DIRECT_ENV] == "1"
    assert HISTORICAL_BASELINE_DNA_PROBES <= baseline_dna.keys()
    assert baseline_dna["SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING"] == "1"
    assert not (HISTORICAL_BASELINE_DNA_ONLY_ENV & current_dna.keys())
    assert STALE_DIRECT_ENV not in current_dna


def test_process_environment_clears_unvalidated_sponge_state(monkeypatch):
    monkeypatch.setenv("SPONGE_CLUSTERED_EXPERIMENT_PROBE", "1")
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")

    process_environment = build_process_environment(valid_environment())

    assert "SPONGE_CLUSTERED_EXPERIMENT_PROBE" not in process_environment
    assert process_environment[ACTIVE_VIEW_ENV] == "1"
    assert process_environment["CUDA_DEVICE_MAX_CONNECTIONS"] == "8"


def test_three_cycle_plan_is_the_complete_alternating_matrix():
    cases = build_case_plan(3, 10_000)

    assert len(cases) == 36
    assert len({case.name for case in cases}) == 36
    expected_cells = {
        (cycle, implementation, system, ensemble)
        for cycle in range(1, 4)
        for implementation in ("baseline", "current")
        for system in ("wat160k", "wat600k", "dna_cou")
        for ensemble in ("nvt", "npt")
    }
    assert {
        (case.cycle, case.implementation, case.system, case.ensemble)
        for case in cases
    } == expected_cells
    assert cases[0].implementation == "baseline"
    assert cases[12].implementation == "current"


@pytest.mark.parametrize(
    ("runs", "steps", "message"),
    [
        (2, 10_000, "runs must be at least 3"),
        (3, 9_999, "steps must be at least 10000"),
    ],
)
def test_plan_cannot_weaken_acceptance(runs, steps, message):
    with pytest.raises(MatrixRunError, match=message):
        build_case_plan(runs, steps)


def test_stage_uses_tracked_inputs_and_generates_ensemble_mdin(tmp_path):
    water = MatrixCase(1, "current", "wat160k", "npt", 10_000)
    dna = MatrixCase(1, "current", "dna_cou", "nvt", 10_000)

    water_dir = stage_case(water, tmp_path)
    dna_dir = stage_case(dna, tmp_path)

    assert (water_dir / "water.top").is_symlink()
    assert (water_dir / "water_npt_eq.gro").is_symlink()
    water_mdin = (water_dir / "mdin.spg.toml").read_text()
    assert 'mode = "npt"' in water_mdin
    assert "step_limit = 10000" in water_mdin
    assert "skin = 2.0" in water_mdin
    assert 'barostat = "andersen_barostat"' in water_mdin
    assert 'direct_kernel = "clustered"' in water_mdin

    assert (dna_dir / "2m2c_LJ.txt").is_symlink()
    dna_mdin = (dna_dir / "mdin.spg.toml").read_text()
    assert 'mode = "nvt"' in dna_mdin
    assert "skin = 2.0" in dna_mdin
    assert 'velocity_in_file = "Pmin_velocity.txt"' in dna_mdin
    assert "barostat =" not in dna_mdin


def test_timing_finiteness_and_route_parsers(tmp_path):
    mdinfo = tmp_path / "mdinfo.txt"
    mdinfo.write_text(
        "Rank   0 | Calculate_Force | 4.25 seconds\n"
        "Core Run Wall Time: 5.50 seconds\n"
        "Core Run Speed: 150.25 ns/day\n"
    )
    mdout = tmp_path / "mdout.txt"
    mdout.write_text("step temperature\n10000 300.0\n")

    assert parse_timings(mdinfo) == (4.25, 5.5, 150.25)
    require_finite_mdout(mdout)
    assert infer_lj_mode("wat160k", "") == "comb"
    assert infer_lj_mode("dna_cou", DNA_AB_WARNING) == "packed-ab"

    mdinfo.write_text(
        "Rank   0 | Calculate_Force | 1.25 minutes\n"
        "Core Run Wall Time: 90.0 seconds (1.50 minutes)\n"
        "Core Run Speed: 10.0 ns/day\n"
    )
    assert parse_timings(mdinfo) == (75.0, 90.0, 10.0)

    mdout.write_text("10000 -nan(ind)\n")
    with pytest.raises(MatrixRunError, match="non-finite"):
        require_finite_mdout(mdout)
    with pytest.raises(MatrixRunError, match="required packed-AB"):
        infer_lj_mode("dna_cou", "")

    mdinfo.write_text(
        "Calculate_Force : 4.25 milliseconds\n"
        "Core Run Wall Time: 5.50 seconds\n"
        "Core Run Speed: 150.25 ns/hour\n"
    )
    with pytest.raises(MatrixRunError, match="force_s, speed_ns_day"):
        parse_timings(mdinfo)


def test_idle_contention_records_invalid_without_launch(tmp_path, monkeypatch):
    case = MatrixCase(1, "current", "wat160k", "nvt", 10_000)
    case_dir = stage_case(case, tmp_path)
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_migration_matrix.query_idle_gpu",
        lambda _command: GpuIdleSample(25.0, 225.0, 31.0),
    )

    row = run_case(
        case,
        case_dir,
        binary=tmp_path / "must-not-run",
        environment=valid_environment(),
        timeout=1,
        nvidia_smi="nvidia-smi",
        max_idle_sm=5.0,
    )

    assert row["valid"] == 0
    assert "idle SM 25.0%" in row["reason"]
    assert not (case_dir / "run.stdout").exists()


def test_gpu_probe_failure_records_invalid_without_launch(
    tmp_path, monkeypatch
):
    case = MatrixCase(1, "current", "wat160k", "nvt", 10_000)
    case_dir = stage_case(case, tmp_path)

    def fail_telemetry(_command):
        raise MatrixRunError("no GPU rows")

    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_migration_matrix.query_idle_gpu",
        fail_telemetry,
    )

    row = run_case(
        case,
        case_dir,
        binary=tmp_path / "must-not-run",
        environment=valid_environment(),
        timeout=1,
        nvidia_smi="nvidia-smi",
        max_idle_sm=5.0,
    )

    assert row["valid"] == 0
    assert row["reason"] == "pre-run GPU telemetry failed: no GPU rows"
    assert not (case_dir / "run.stdout").exists()


def test_dry_run_stages_manifest_without_gpu(tmp_path):
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(valid_environment()))
    baseline_environment_path = tmp_path / "baseline-environment.json"
    baseline_environment_path.write_text(
        json.dumps(valid_historical_baseline_environment())
    )
    output_root = tmp_path / "matrix"

    exit_code = main(
        [
            "--baseline-bin",
            sys.executable,
            "--current-bin",
            sys.executable,
            "--environment",
            str(environment_path),
            "--baseline-environment",
            str(baseline_environment_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["mode"] == "dry-run"
    assert len(manifest["cases"]) == 36
    assert set(manifest["sources"]) == {"wat160k", "wat600k", "dna_cou"}
    assert manifest["environment"] == valid_environment()
    assert manifest["schema"] == 2
    environments = manifest["environments"]
    assert environments["process_policy"]["clear_prefixes"] == ["SPONGE_"]
    assert environments["current_configured"] == valid_environment()
    assert (
        environments["baseline_configured"]
        == valid_historical_baseline_environment()
    )
    for system in ("wat160k", "wat600k"):
        assert not (
            HISTORICAL_BASELINE_DNA_ONLY_ENV
            & environments["resolved"]["baseline"][system].keys()
        )
    assert (
        HISTORICAL_BASELINE_DNA_PROBES
        <= environments["resolved"]["baseline"]["dna_cou"].keys()
    )
    assert not (
        HISTORICAL_BASELINE_DNA_ONLY_ENV
        & environments["resolved"]["current"]["dna_cou"].keys()
    )
    assert (
        manifest["mdin_sha256"]["01_wat160k_nvt_baseline"]
        == manifest["mdin_sha256"]["01_wat160k_nvt_current"]
    )
    assert not (output_root / "matrix.tsv").exists()
