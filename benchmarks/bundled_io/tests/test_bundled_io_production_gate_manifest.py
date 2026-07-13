from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from benchmarks.bundled_io.ab_contracts import (
    load_contract_registry,
    load_implementation_inventory,
    validate_contract_registry,
    validate_implementation_inventory,
)
from benchmarks.bundled_io.input_semantics import (
    REQUIRED_INPUT_SEMANTIC_CONTRACTS,
    InputSemanticSpec,
    assert_module_semantics,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    INPUT_SEMANTIC_SPECS_BY_CASE,
    MDINFO_CONTRACT_KEYS,
    PROFILE_LIMITS,
    RERUN_INPUT_SEMANTIC_SPECS,
    _cases_for_profile,
    _expected_rerun_frame_indices,
    _insert_root_toml_keys,
    _parse_mdinfo_key_values,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PIXI_TOML = REPO_ROOT / "pixi.toml"
H5_BUNDLE_RUNNER = REPO_ROOT / "tests" / "h5_bundle" / "run_h5_bundle_tests.sh"
AUDIT_DOC = REPO_ROOT / "docs" / "sponge_h5_bundle_unit_test_audit_matrix.md"
VDS_WRITER_TEST = (
    REPO_ROOT
    / "tests"
    / "h5_bundle"
    / "test_vds_trajectory_writer_with_mock_backend.cpp"
)
OUTPUT_PLAN_TEST = REPO_ROOT / "tests/h5_bundle/test_h5_output_plan.cpp"
WRITER_TEST = (
    REPO_ROOT / "tests/h5_bundle/test_h5md_writers_with_mock_backend.cpp"
)
MODULE_TEST = (
    REPO_ROOT / "tests/h5_bundle/test_module_h5_mappings_with_mock_backend.cpp"
)
HIGHFIVE_TEST = REPO_ROOT / "tests/h5_bundle/test_highfive_backend_io.cpp"


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
    assert "test_bundled_io_contract_inventory.py" in contract
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
    validate_implementation_inventory(
        contracts, load_implementation_inventory()
    )

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
        "rerun_boundary_start0_strip0_limit1_vds_off",
        "rerun_boundary_start1_strip0_unlimited_no_velocity_vds_on",
        "rerun_boundary_start0_strip1_beyond_selected_vds_on",
        "rerun_boundary_start0_strip0_exact_eof_box_vds_off",
        "rerun_boundary_start1_strip1_limit1_selected_no_velocity_vds_off",
        "failure_missing_trajectory_binding",
        "failure_invalid_output_chunk_size",
        "failure_invalid_output_vds_value",
        "failure_invalid_output_repair_policy",
        "failure_invalid_restart_policy",
        "failure_missing_topology_binding",
        "failure_missing_protocol_binding",
        "failure_mixed_legacy_h5_trajectory",
        "failure_mixed_legacy_h5_restart",
    }
    assert summary["status_counts"]["supported"] > 0
    assert contracts["output.vds.cross_process_append_resume"].status == (
        "unsupported"
    )
    assert contracts["output.vds.complete_prefix_repair"].status == "deferred"


def test_rerun_boundary_matrix_covers_semantic_axes_and_eof_boundaries():
    cases = [
        case
        for case in _cases_for_profile()
        if case.name.startswith("rerun_boundary_")
    ]

    assert {(case.rerun_start, case.rerun_strip) for case in cases} == {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    }
    assert {case.rerun_frame_limit for case in cases} == {1, 2, 3, None}
    assert {case.rerun_need_box_update for case in cases} == {False, True}
    assert {case.rerun_velocity_present for case in cases} == {False, True}
    assert {case.trajectory_particle_stream for case in cases} == {
        "all",
        "selected",
    }
    assert {case.vds for case in cases} == {False, True}
    assert all(
        "rerun_selection_equivalence" in case.assertion_ids for case in cases
    )


@pytest.mark.parametrize(
    ("case_name", "expected_indices"),
    [
        ("rerun_boundary_start0_strip0_limit1_vds_off", [0]),
        (
            "rerun_boundary_start1_strip0_unlimited_no_velocity_vds_on",
            [1],
        ),
        ("rerun_boundary_start0_strip1_beyond_selected_vds_on", [0]),
        ("rerun_boundary_start0_strip0_exact_eof_box_vds_off", [0, 1]),
        (
            "rerun_boundary_start1_strip1_limit1_selected_no_velocity_vds_off",
            [1],
        ),
    ],
)
def test_rerun_selection_oracle_is_independent_of_runtime_frame_counter(
    case_name, expected_indices
):
    case = next(case for case in _cases_for_profile() if case.name == case_name)

    assert (
        _expected_rerun_frame_indices(case, frame_count=2) == expected_indices
    )


def test_failure_matrix_requires_exit_category_and_stable_tokens():
    cases = [
        case
        for case in _cases_for_profile()
        if case.name.startswith("failure_")
    ]

    assert {case.failure_mutation for case in cases} == {
        "missing_trajectory",
        "invalid_chunk_size",
        "invalid_vds_value",
        "invalid_repair_policy",
        "invalid_restart_policy",
        "missing_topology",
        "missing_protocol",
        "mixed_trajectory",
        "mixed_restart",
    }
    assert all(case.expected_error_category for case in cases)
    assert all(case.expected_diagnostic_tokens for case in cases)
    assert all(
        case.assertion_ids == ("stable_failure_semantics",) for case in cases
    )
    assert {
        case.failure_mutation
        for case in cases
        if set(case.failure_branches) == {"legacy", "bundled"}
    } == {
        "missing_trajectory",
        "invalid_chunk_size",
        "invalid_vds_value",
        "invalid_repair_policy",
    }


def test_h5_bundle_runner_defaults_to_current_pixi_build_dir():
    source = H5_BUNDLE_RUNNER.read_text(encoding="utf-8")

    assert "SPONGE_H5_BUNDLE_TEST_BUILD_DIR" in source
    assert 'elif [[ -n "${PIXI_ENVIRONMENT_NAME:-}" ]]' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-${PIXI_ENVIRONMENT_NAME}"' in source
    assert 'BUILD_DIR="${ROOT_DIR}/build-h5-tests"' in source


def test_output_behavior_cases_cover_particle_fields_and_both_vds_modes():
    contracts = load_contract_registry()
    cases = {case.name: case for case in _cases_for_profile()}

    assert cases["normal_core_h5_output"].vds is False
    assert cases["normal_sits_ff19sb_cmap_peptide"].vds is True
    for case_name in (
        "normal_core_h5_output",
        "normal_sits_ff19sb_cmap_peptide",
    ):
        case = cases[case_name]
        assert {
            "output.legacy.crd",
            "output.legacy.box",
            "output.legacy.velocity",
            "output.legacy.force",
        } <= set(case.contract_ids)
        assert "particle_legacy_coexistence" in case.assertion_ids

    for contract_id in (
        "output.legacy.crd",
        "output.legacy.box",
        "output.legacy.velocity",
        "output.legacy.force",
    ):
        assert contracts[contract_id].status == "supported"
        assert contracts[contract_id].minimum_evidence == "E3"


def test_mdinfo_parser_uses_declared_structured_keys(tmp_path):
    mdinfo = tmp_path / "mdinfo.txt"
    mdinfo.write_text(
        "Working Directory: /dynamic/path\n"
        + "\n".join(f"{key} is value for {key}" for key in MDINFO_CONTRACT_KEYS)
        + "\nStart Wall Time: dynamic\n",
        encoding="utf-8",
    )

    parsed = _parse_mdinfo_key_values(mdinfo)

    assert set(parsed) == MDINFO_CONTRACT_KEYS
    assert all(len(values) == 1 for values in parsed.values())


def test_vds_chunk_boundary_gate_enumerates_all_required_edges():
    source = VDS_WRITER_TEST.read_text(encoding="utf-8")

    assert "Test_Vds_Chunk_Boundary_Frame_Counts" in source
    for boundary in ("{1, {1}}", "{2, {2}}", "{3, {2, 1}}", "{5, {2, 2, 1}}"):
        assert boundary in source


def test_output_behavior_closure_keeps_family_module_restart_and_repair_gates():
    required_functions = {
        OUTPUT_PLAN_TEST: {
            "Test_H5_Output_Path_Keys_Enable_Only_Their_Bundle",
            "Test_All_H5_Output_Bundles_Can_Be_Enabled_Together",
            "Test_Legacy_Output_Plan_All_Keys",
            "Test_Resolve_Legacy_Output_Plan_Matrix",
        },
        WRITER_TEST: {
            "Test_Trajectory_Optional_Velocity_And_Force_Paths",
            "Test_Observable_Only_Writer",
            "Test_Restart_Module_State_And_Legacy_Provenance",
            "Test_Legacy_Provenance_On_Trajectory_And_Observable",
        },
        MODULE_TEST: {
            "Test_Nhc_And_Sits_Mappings",
            "Test_Metadynamics_And_Diagnostics",
            "Test_Qc_And_Reaxff_Mappings",
        },
        HIGHFIVE_TEST: {
            "Test_Trajectory_Optional_Particle_Fields_With_Real_Backend",
            "Test_Restart_Writer_With_Real_Backend",
            "Test_Module_Metad_And_Reaxff_With_Real_Backend",
            "Test_Vds_Complete_Prefix_Repair_With_Real_Backend",
        },
    }
    for path, functions in required_functions.items():
        source = path.read_text(encoding="utf-8")
        for function in functions:
            assert function in source, (
                f"{path.name} no longer covers {function}"
            )


def test_input_semantic_gate_requires_nontrivial_owned_result():
    result = assert_module_semantics(
        "bond",
        [{"bond": 2.0}, {"bond": 3.0}],
        [{"bond": 2.0}, {"bond": 3.0}],
        InputSemanticSpec("input.topology.bond", ("bond",)),
        deterministic=True,
    )

    assert result["legacy_nontrivial"] is True
    assert result["bundled_nontrivial"] is True


def test_input_semantic_registry_uses_owned_observables_not_initialization_logs():
    specs = [
        *RERUN_INPUT_SEMANTIC_SPECS,
        *(
            spec
            for case_specs in INPUT_SEMANTIC_SPECS_BY_CASE.values()
            for spec in case_specs
        ),
    ]

    assert specs
    assert all(spec.observables for spec in specs)
    assert all(
        "initial" not in observable.lower()
        for spec in specs
        for observable in spec.observables
    )


def test_input_semantic_contract_inventory_is_explicit_and_evidence_gated():
    contracts = load_contract_registry()
    runtime_spec_ids = {
        spec.contract_id
        for specs in INPUT_SEMANTIC_SPECS_BY_CASE.values()
        for spec in specs
    } | {spec.contract_id for spec in RERUN_INPUT_SEMANTIC_SPECS}

    assert REQUIRED_INPUT_SEMANTIC_CONTRACTS <= set(contracts)
    assert runtime_spec_ids == {
        contract_id
        for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS
        if contracts[contract_id].status == "supported"
    }
    for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS:
        contract = contracts[contract_id]
        assert contract.minimum_evidence == "E3"
        if contract.status == "supported":
            assert "input_semantic_equivalence" in contract.assertion_ids
        else:
            assert contract.status == "deferred"
            assert contract.reason


@pytest.mark.parametrize(
    ("legacy", "bundled", "message"),
    [
        ([{"other": 1.0}], [{"bond": 1.0}], "missing module-owned observable"),
        ([{"bond": 0.0}], [{"bond": 0.0}], "all trivial"),
        ([{"bond": 1.0}], [{"bond": 2.0}], "mismatch at row"),
    ],
)
def test_input_semantic_gate_rejects_activation_only_evidence(
    legacy, bundled, message
):
    with pytest.raises(AssertionError, match=message):
        assert_module_semantics(
            "bond",
            legacy,
            bundled,
            InputSemanticSpec("input.topology.bond", ("bond",)),
            deterministic=True,
        )


def test_rerun_overrides_remain_root_keys_before_module_tables():
    updated = _insert_root_toml_keys(
        'mode = "rerun"\n[REAXFF]\nin_file = "reaxff.txt"\n',
        ["rerun_start = 1", 'crd = "traj.dat"'],
    )

    parsed = tomllib.loads(updated)
    assert parsed["rerun_start"] == 1
    assert parsed["crd"] == "traj.dat"
    assert parsed["REAXFF"] == {"in_file": "reaxff.txt"}
