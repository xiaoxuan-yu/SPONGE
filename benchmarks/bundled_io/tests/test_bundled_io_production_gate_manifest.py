from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from benchmarks.bundled_io.ab_contracts import (
    load_contract_registry,
    validate_contract_registry,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    PROFILE_LIMITS,
    _cases_for_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PIXI_TOML = REPO_ROOT / "pixi.toml"
H5_BUNDLE_RUNNER = REPO_ROOT / "tests" / "h5_bundle" / "run_h5_bundle_tests.sh"
AUDIT_DOC = REPO_ROOT / "docs" / "sponge_h5_bundle_unit_test_audit_matrix.md"


def _dev_tasks() -> dict[str, object]:
    with PIXI_TOML.open("rb") as handle:
        data = tomllib.load(handle)
    return data["feature"]["dev"]["tasks"]


def _task_command(tasks: dict[str, object], name: str) -> str:
    value = tasks[name]
    if isinstance(value, str):
        return value
    assert isinstance(value, dict), f"{name} must be a command task"
    command = value.get("cmd")
    assert isinstance(command, str), f"{name} must provide cmd"
    return command


def _task_depends(tasks: dict[str, object], name: str) -> list[str]:
    value = tasks[name]
    assert isinstance(value, dict), f"{name} must be a dependency task"
    depends = value.get("depends-on")
    assert isinstance(depends, list), f"{name} must use depends-on"
    assert all(isinstance(item, str) for item in depends)
    return depends


def test_bundled_io_production_gate_keeps_all_required_stages():
    tasks = _dev_tasks()

    assert _task_depends(tasks, "smoke-bundled-io") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
    ]
    assert _task_depends(tasks, "smoke-bundled-io-production") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
        "smoke-bundled-io-staged",
        "smoke-gpu-medium",
    ]

    contract = _task_command(tasks, "smoke-bundled-io-contract")
    assert "test_bundled_io_contract_smoke.py" in contract
    assert "test_bundled_io_ab_statistics.py" in contract
    assert "test_bundled_io_ab_contract_registry.py" in contract
    assert "test_bundled_io_production_gate_manifest.py" in contract
    assert "test_bundled_io_ab_production.py" not in contract
    assert "test-smoke-runtime" in _task_command(
        tasks, "smoke-bundled-io-runtime"
    )
    assert "run_h5_bundle_tests.sh staged" in _task_command(
        tasks, "smoke-bundled-io-staged"
    )


def test_gpu_medium_smoke_uses_real_validation_and_performance_scenarios():
    tasks = _dev_tasks()

    assert _task_depends(tasks, "smoke-gpu-medium") == [
        "smoke-gpu-medium-preflight",
        "smoke-gpu-medium-validation",
        "smoke-gpu-medium-reaxff",
        "smoke-gpu-medium-nonortho",
    ]

    preflight = _task_command(tasks, "smoke-gpu-medium-preflight")
    assert "test_cuda_runtime_preflight.py" in preflight

    validation = _task_command(tasks, "smoke-gpu-medium-validation")
    assert (
        "benchmarks/validation/thermostat/tests/test_thermostat.py"
        in validation
    )
    assert "middle_langevin" in validation
    assert "benchmarks/validation/cv/tests/test_dihedral.py" in validation
    assert "benchmarks/validation/misc/tests/test_softcore.py" in validation

    reaxff = _task_command(tasks, "smoke-gpu-medium-reaxff")
    assert "benchmarks/performance/reaxff/tests/test_petn_nve_perf.py" in reaxff
    assert "--steps=1000" in reaxff

    nonortho = _task_command(tasks, "smoke-gpu-medium-nonortho")
    assert (
        "benchmarks/performance/nonortho/tests/test_long_run_rdf.py" in nonortho
    )
    assert "--mode=NVT" in nonortho
    assert "--steps=2000" in nonortho


def test_ab_bundled_io_gates_keep_medium_and_production_profiles():
    tasks = _dev_tasks()

    medium = _task_command(tasks, "ab-bundled-io-medium-runtime")
    assert "SPONGE_BUNDLED_IO_AB_PROFILE=medium" in medium
    assert "test_bundled_io_ab_production.py" in medium

    production = _task_command(tasks, "ab-bundled-io-production-runtime")
    assert "SPONGE_BUNDLED_IO_AB_PROFILE=production" in production
    assert "test_bundled_io_ab_production.py" in production

    assert _task_depends(tasks, "ab-bundled-io-medium") == [
        "smoke-bundled-io-contract",
        "smoke-bundled-io-runtime",
        "ab-bundled-io-medium-runtime",
    ]
    assert _task_depends(tasks, "ab-bundled-io-production") == [
        "smoke-bundled-io-production",
        "ab-bundled-io-production-runtime",
    ]


def test_release_boundary_keeps_ab_opt_in_shadow_only():
    tasks = _dev_tasks()
    default_smoke = _task_depends(tasks, "smoke-bundled-io")
    production_smoke = _task_depends(tasks, "smoke-bundled-io-production")

    assert "ab-bundled-io-medium" not in default_smoke
    assert "ab-bundled-io-production" not in default_smoke
    assert "ab-bundled-io-medium" not in production_smoke
    assert "ab-bundled-io-production" not in production_smoke

    doc = " ".join(AUDIT_DOC.read_text(encoding="utf-8").split())
    assert (
        "Legacy/bundled behavior A/B is a separate opt-in release gate" in doc
    )
    assert "not part of the default smoke task" in doc
    assert "opt-in/shadow path" in doc
    assert "legacy-behavior replacement candidate" in doc
    assert (
        "proves equivalence for the explicitly enumerated supported input/output "
        "contract and runtime scenarios"
    ) in doc
    assert "cross-process VDS reopen-and-append resume" in doc


def test_ab_production_harness_has_executable_contract_coverage():
    contracts = load_contract_registry()
    cases = _cases_for_profile()
    summary = validate_contract_registry(contracts, cases)

    assert set(PROFILE_LIMITS) == {"medium", "production"}
    assert PROFILE_LIMITS["medium"] == {
        "normal_step_limit": 1000,
        "normal_interval": 20,
        "normal_replicas": 10,
        "normal_burn_in_frames": 10,
        "normal_block_size": 8,
        "normal_relative_margin": 5.0e-2,
        "normal_absolute_margin": 1.5e-1,
        "rerun_frame_limit": 2,
    }
    assert PROFILE_LIMITS["production"] == {
        "normal_step_limit": 10000,
        "normal_interval": 100,
        "normal_replicas": 5,
        "normal_burn_in_frames": 20,
        "normal_block_size": 10,
        "normal_relative_margin": 1.0e-2,
        "normal_absolute_margin": 5.0e-2,
        "rerun_frame_limit": 2,
    }
    assert {case.name for case in cases} == {
        "normal_core_h5_output",
        "normal_sits_ff19sb_cmap_peptide",
        "rerun_full_contract_pure_vds_off",
        "rerun_full_contract_pure_vds_on",
        "rerun_full_contract_sidecar_vds_off",
        "rerun_full_contract_sidecar_vds_on",
    }
    assert summary["status_counts"]["supported"] > 0
    assert contracts["output.vds.cross_process_append_resume"].status == (
        "unsupported"
    )
    assert contracts["output.vds.complete_prefix_repair"].status == "deferred"


def test_h5_bundle_runner_defaults_to_current_pixi_build_dir():
    source = H5_BUNDLE_RUNNER.read_text(encoding="utf-8")

    assert "SPONGE_H5_BUNDLE_TEST_BUILD_DIR" in source
    assert 'elif [[ -n "${PIXI_ENVIRONMENT_NAME:-}" ]]' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-${PIXI_ENVIRONMENT_NAME}"' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-h5-tests"' in source
