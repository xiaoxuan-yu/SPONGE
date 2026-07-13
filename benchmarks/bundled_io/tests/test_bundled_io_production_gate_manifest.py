from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from benchmarks.bundled_io.ab_contracts import (
    ContractSpec,
    load_contract_registry,
    load_implementation_inventory,
    validate_contract_registry,
    validate_implementation_inventory,
)
from benchmarks.bundled_io.execution_matrix import (
    ProductionRun,
    evaluate_promotion_readiness,
    load_execution_matrix,
    validate_execution_matrix,
)
from benchmarks.bundled_io.input_semantics import (
    REQUIRED_INPUT_SEMANTIC_CONTRACTS,
    InputSemanticSpec,
    assert_module_semantics,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_execution_matrix import (
    MATRIX_RUNTIME_CASES,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    FOCUSED_CONSTRAINT_SIDECAR_FIXTURE,
    FOCUSED_CUSTOM_PAIR_FIXTURE,
    FOCUSED_EDIP_FIXTURE,
    FOCUSED_EXCLUSIONS_FIXTURE,
    FOCUSED_GB_HYBRID_FIXTURE,
    FOCUSED_IMPROPER_NATIVE_FIXTURE,
    FOCUSED_LJ_SOFT_CORE_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
    FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
    INPUT_SEMANTIC_SPECS_BY_CASE,
    MDINFO_CONTRACT_KEYS,
    PROFILE_LIMITS,
    RERUN_INPUT_SEMANTIC_SPECS,
    _assert_constraint_projection_oracle,
    _assert_exclusion_coulomb_oracle,
    _assert_gb_force_oracle,
    _assert_nontrivial_equivalent_forces,
    _assert_virtual_atom_oracle,
    _cases_for_profile,
    _expected_rerun_frame_indices,
    _h5_string_values,
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
AB_SHADOW_WORKFLOW = REPO_ROOT / ".github/workflows/bundled-io-ab-shadow.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release.yml"
MATRIX_FIXTURE_ROOT = (
    REPO_ROOT / "benchmarks/bundled_io/fixtures/tip3p_matrix"
)


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
        "normal_nhc_dynamic_restart_continuation",
        "normal_meta_protocol_full_restart_continuation",
        "normal_sits_ff19sb_cmap_peptide",
        "normal_edip_nonzero",
        "normal_custom_pair_nonzero",
        "normal_exclusions_coulomb_oracle",
        "normal_gb_hybrid_nonzero",
        "normal_improper_native_nonzero",
        "normal_lj_soft_core_nonzero",
        "normal_virtual_atoms_all_types",
        "normal_virtual_atoms_pbc_boundary",
        "normal_virtual_atoms_plural_alias",
        "normal_constraint_sidecar_projection",
        "normal_vds_chunk_minus_one",
        "normal_vds_chunk_exact",
        "normal_vds_chunk_plus_one",
        "normal_vds_chunk_two_plus_one",
        "rerun_full_contract_pure_vds_off",
        "rerun_full_contract_pure_vds_on",
        "rerun_full_contract_sidecar_vds_off",
        "rerun_full_contract_sidecar_vds_on",
        "rerun_qc_unrestricted_sidecar_vds_off",
        "rerun_qc_unrestricted_sidecar_vds_on",
        "rerun_restart_absent_same_bootstrap_vds_off",
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
        "failure_sidecar_unsupported_key",
        "failure_sidecar_key_path_length_mismatch",
        "failure_sidecar_path_conflict",
        "failure_restart_dynamic_without_owner",
        "failure_restart_protocol_without_owner",
        "failure_restart_full_without_owner",
    }
    assert summary["status_counts"]["supported"] > 0
    assert contracts["output.vds.cross_process_append_resume"].status == (
        "unsupported"
    )
    assert contracts["output.vds.complete_prefix_repair"].status == "deferred"


def test_nhc_dynamic_restart_uses_one_checkpoint_and_e4_continuation():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_nhc_dynamic_restart_continuation"
    )
    expected_contracts = {
        "input.restart_load.dynamic",
        "input.bias.nhc",
        "output.restart.dynamic_continuation",
    }

    assert case.mode == "dynamic_continuation"
    assert case.restart_load_policy == "dynamic"
    assert case.statistical_md is False
    assert set(case.contract_ids) == expected_contracts
    assert case.assertion_ids == (
        "restart_dynamic_continuation_equivalence",
    )
    for contract_id in expected_contracts:
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert contract.case_ids == (case.name,)
        assert contract.assertion_ids == case.assertion_ids
    assert contracts["input.restart_load.dynamic"].minimum_evidence == "E4"
    assert (
        contracts["output.restart.dynamic_continuation"].minimum_evidence
        == "E4"
    )


def test_meta_protocol_full_restart_uses_one_checkpoint_and_e4_continuation():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_meta_protocol_full_restart_continuation"
    )
    expected_contracts = {
        "input.restart_load.protocol",
        "input.restart_load.full",
        "input.bias.metadynamics",
    }

    assert case.mode == "protocol_full_continuation"
    assert case.restart_load_policy == "protocol/full"
    assert case.statistical_md is False
    assert set(case.contract_ids) == expected_contracts
    assert case.assertion_ids == (
        "restart_protocol_full_continuation_equivalence",
    )
    for contract_id in expected_contracts:
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert contract.case_ids == (case.name,)
        assert contract.assertion_ids == case.assertion_ids
    assert contracts["input.restart_load.protocol"].minimum_evidence == "E4"
    assert contracts["input.restart_load.full"].minimum_evidence == "E4"


def test_vds_chunk_boundary_cases_cover_required_frame_transitions():
    contract = load_contract_registry()["output.trajectory.chunk_size"]
    cases = {
        case.name: case
        for case in _cases_for_profile()
        if case.mode == "chunk_boundary"
    }

    assert {
        name: (
            case.output_chunk_size,
            case.expected_trajectory_frames,
            case.normal_interval,
            case.normal_dt,
        )
        for name, case in cases.items()
    } == {
        "normal_vds_chunk_minus_one": (4, 3, 1, 0.0001),
        "normal_vds_chunk_exact": (4, 4, 1, 0.0001),
        "normal_vds_chunk_plus_one": (4, 5, 1, 0.0001),
        "normal_vds_chunk_two_plus_one": (4, 9, 1, 0.0001),
    }
    assert all(case.vds for case in cases.values())
    assert all(not case.statistical_md for case in cases.values())
    assert all(
        "h5_chunk_boundary_equivalence" in case.assertion_ids
        for case in cases.values()
    )
    assert contract.status == "supported"
    assert contract.minimum_evidence == "E3"
    assert set(contract.case_ids) == set(cases)
    assert contract.assertion_ids == ("h5_chunk_boundary_equivalence",)


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


def test_restart_absent_case_requires_same_bootstrap_and_behavior_outputs():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "rerun_restart_absent_same_bootstrap_vds_off"
    )
    contract = contracts["input.restart_load.absent"]

    assert case.restart_load_policy == "absent"
    assert case.statistical_md is False
    assert case.rerun_frame_limit == 2
    assert case.rerun_need_box_update is False
    assert "input.restart_load.structural" not in case.contract_ids
    assert {
        "input.restart_load.absent",
        "output.legacy.mdout",
        "output.trajectory",
        "output.observable",
        "output.trajectory.vds_off",
    } <= set(case.contract_ids)
    assert case.assertion_ids == (
        "mdout_deterministic_equivalence",
        "rerun_selection_equivalence",
        "h5_rerun_semantic_equivalence",
    )
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == case.assertion_ids


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
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
        "restart_dynamic_without_owner",
        "restart_protocol_without_owner",
        "restart_full_without_owner",
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
    sidecar_cases = [
        case
        for case in cases
        if case.contract_ids == ("failure.sidecar_table",)
    ]
    assert {case.failure_mutation for case in sidecar_cases} == {
        "unsupported_sidecar_key",
        "sidecar_length_mismatch",
        "sidecar_path_conflict",
    }
    assert all(case.failure_branches == ("bundled",) for case in sidecar_cases)
    restart_owner_cases = [
        case
        for case in cases
        if case.contract_ids == ("failure.restart_owner_state",)
    ]
    assert {
        case.failure_mutation: case.restart_load_policy
        for case in restart_owner_cases
    } == {
        "restart_dynamic_without_owner": "dynamic",
        "restart_protocol_without_owner": "protocol",
        "restart_full_without_owner": "full",
    }
    assert all(
        case.failure_branches == ("bundled",) for case in restart_owner_cases
    )


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


def test_focused_edip_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case for case in _cases_for_profile() if case.name == "normal_edip_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_EDIP_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.manybody.edip",
    )
    assert spec == (
        InputSemanticSpec("input.manybody.edip", ("EDIP",), 1.0e-6),
    )
    assert contracts["input.manybody.edip"].status == "supported"
    assert contracts["input.manybody.edip"].case_ids == (case.name,)
    assert contracts["input.manybody.edip"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_custom_pair_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_custom_pair_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_CUSTOM_PAIR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.custom.pairwise",
    )
    assert spec == (
        InputSemanticSpec(
            "input.custom.pairwise", ("custom_pair",), 1.0e-6
        ),
    )
    assert contracts["input.custom.pairwise"].status == "supported"
    assert contracts["input.custom.pairwise"].case_ids == (case.name,)
    assert contracts["input.custom.pairwise"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_exclusions_case_requires_native_payload_and_coulomb_oracle():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_exclusions_coulomb_oracle"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_EXCLUSIONS_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.exclusions",
    )
    assert spec == (
        InputSemanticSpec("input.topology.exclusions", ("Coulomb",), 1.0e-6),
    )
    assert contracts["input.topology.exclusions"].status == "supported"
    assert contracts["input.topology.exclusions"].case_ids == (case.name,)
    assert contracts["input.topology.exclusions"].assertion_ids == (
        "input_semantic_equivalence",
    )


def test_focused_gb_case_requires_native_state_and_sidecar_activation_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_gb_hybrid_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_GB_HYBRID_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.gb.hybrid_activation",
    )
    assert spec == (
        InputSemanticSpec(
            "input.topology.gb.hybrid_activation", ("gb",), 1.0e-6
        ),
    )
    hybrid = contracts["input.topology.gb.hybrid_activation"]
    assert hybrid.status == "supported"
    assert hybrid.case_ids == (case.name,)
    assert hybrid.assertion_ids == ("input_semantic_equivalence",)
    native = contracts["input.topology.gb"]
    assert native.status == "deferred"
    assert "only calls GB initialization when gb_in_file exists" in native.reason
    assert "tracked separately" in native.reason


def test_focused_gb_gate_rejects_activation_only_or_coulomb_only_behavior():
    spec = InputSemanticSpec(
        "input.topology.gb.hybrid_activation", ("gb",), 1.0e-6
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "GB",
            [{"gb": 0.0}],
            [{"gb": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match=r"GB\+Coulomb force oracle"):
        _assert_gb_force_oracle(
            "GB mutation", (0.25, 0.0, 0.0, -0.25, 0.0, 0.0)
        )


def test_focused_improper_case_requires_pure_native_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_improper_native_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_IMPROPER_NATIVE_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.improper.native_runtime",
    )
    assert spec == (
        InputSemanticSpec(
            "input.topology.improper.native_runtime",
            ("improper_dihedral",),
            1.0e-6,
        ),
    )
    native = contracts["input.topology.improper.native_runtime"]
    assert native.status == "supported"
    assert native.case_ids == (case.name,)
    assert native.assertion_ids == ("input_semantic_equivalence",)
    conversion = contracts["input.topology.improper"]
    assert conversion.status == "deferred"
    assert "requires /pk" in conversion.reason
    assert "not consumed" in conversion.reason


def test_focused_improper_gate_rejects_activation_only_evidence():
    spec = InputSemanticSpec(
        "input.topology.improper.native_runtime",
        ("improper_dihedral",),
        1.0e-6,
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "improper",
            [{"improper_dihedral": 0.0}],
            [{"improper_dihedral": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match="bundled force is all trivial"):
        _assert_nontrivial_equivalent_forces(
            "improper force", [1.0, -1.0], [0.0, 0.0]
        )


def test_focused_lj_soft_core_case_requires_pure_h5_nonzero_energy_and_force():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_lj_soft_core_nonzero"
    )
    spec = INPUT_SEMANTIC_SPECS_BY_CASE[case.name]

    assert case.fixture_case == FOCUSED_LJ_SOFT_CORE_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.topology.lj_soft_core",
    )
    assert spec == (
        InputSemanticSpec("input.topology.lj_soft_core", ("LJ_soft",), 1.0e-6),
    )
    contract = contracts["input.topology.lj_soft_core"]
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("input_semantic_equivalence",)
    subsystem = contracts["input.topology.subsystem_division"]
    assert subsystem.status == "deferred"
    assert "not consumed" in subsystem.reason


def test_focused_lj_soft_core_gate_rejects_trivial_energy_and_force():
    spec = InputSemanticSpec(
        "input.topology.lj_soft_core", ("LJ_soft",), 1.0e-6
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "LJ soft-core",
            [{"LJ_soft": 0.0}],
            [{"LJ_soft": 0.0}],
            spec,
            deterministic=True,
        )
    with pytest.raises(AssertionError, match="bundled force is all trivial"):
        _assert_nontrivial_equivalent_forces(
            "LJ soft-core force", [1.0, -1.0], [0.0, 0.0]
        )


@pytest.mark.parametrize(
    ("case_name", "fixture", "contract_id", "observable"),
    [
        (
            "normal_virtual_atoms_all_types",
            FOCUSED_VIRTUAL_ATOMS_ALL_TYPES_FIXTURE,
            "input.topology.virtual_atoms",
            "PM",
        ),
        (
            "normal_virtual_atoms_pbc_boundary",
            FOCUSED_VIRTUAL_ATOMS_PBC_FIXTURE,
            "input.topology.virtual_atoms_pbc",
            "PM",
        ),
        (
            "normal_virtual_atoms_plural_alias",
            FOCUSED_VIRTUAL_ATOMS_ALIAS_FIXTURE,
            "input.topology.virtual_atoms_alias",
            "PM",
        ),
    ],
)
def test_focused_virtual_atom_cases_require_native_runtime_oracles(
    case_name: str, fixture: str, contract_id: str, observable: str
):
    contracts = load_contract_registry()
    case = next(case for case in _cases_for_profile() if case.name == case_name)

    assert case.fixture_case == fixture
    assert case.statistical_md is False
    assert case.normal_step_limit == 1
    assert case.normal_dt == 0.0
    assert case.input_behavior_only is True
    assert case.contract_ids == ("output.legacy.mdout", contract_id)
    assert INPUT_SEMANTIC_SPECS_BY_CASE[case.name] == (
        InputSemanticSpec(contract_id, (observable,), 1.0e-6),
    )
    contract = contracts[contract_id]
    assert contract.status == "supported"
    assert contract.case_ids == (case.name,)
    assert contract.assertion_ids == ("input_semantic_equivalence",)


def test_focused_virtual_atom_gate_rejects_semantic_mutations():
    expected_coordinates = (
        9.5,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        9.75,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
    )
    redistributed_forces = (
        0.25,
        0.0,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    )
    wrong_pbc_coordinates = list(expected_coordinates)
    wrong_pbc_coordinates[6] = 7.25
    with pytest.raises(AssertionError, match="coordinate oracle"):
        _assert_virtual_atom_oracle(
            "PBC mutation",
            expected_coordinates,
            (2,),
            wrong_pbc_coordinates,
            redistributed_forces,
        )

    with pytest.raises(AssertionError, match="real-atom force is all trivial"):
        _assert_virtual_atom_oracle(
            "force redistribution mutation",
            expected_coordinates,
            (2,),
            expected_coordinates,
            (0.0,) * len(expected_coordinates),
        )


def test_focused_constraint_sidecar_case_requires_projected_runtime_behavior():
    contracts = load_contract_registry()
    case = next(
        case
        for case in _cases_for_profile()
        if case.name == "normal_constraint_sidecar_projection"
    )

    assert case.fixture_case == FOCUSED_CONSTRAINT_SIDECAR_FIXTURE
    assert case.statistical_md is False
    assert case.normal_step_limit == 4
    assert case.normal_interval == 1
    assert case.normal_dt == 0.001
    assert case.input_behavior_only is True
    assert case.contract_ids == (
        "output.legacy.mdout",
        "input.protocol.constraint.sidecar",
    )
    assert case.assertion_ids == (
        "mdout_deterministic_equivalence",
        "constraint_geometry_equivalence",
    )

    sidecar = contracts["input.protocol.constraint.sidecar"]
    assert sidecar.status == "supported"
    assert sidecar.case_ids == (case.name,)
    assert sidecar.assertion_ids == ("constraint_geometry_equivalence",)
    typed = contracts["input.protocol.constraint"]
    assert typed.status == "deferred"
    assert "does not materialize" in typed.reason
    assert "Sidecar behavior is tracked separately" in typed.reason


@pytest.mark.parametrize(
    ("positions", "velocities", "message"),
    [
        (
            (0.0, 0.0, 0.0, 1.6, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "distance residual exceeds tolerance",
        ),
        (
            (0.0, 0.0, 0.0, 1.5, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            "radial velocity residual exceeds tolerance",
        ),
    ],
)
def test_focused_constraint_gate_rejects_ignored_or_wrong_constraint(
    positions, velocities, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_constraint_projection_oracle(
            "constraint mutation", positions, velocities
        )


@pytest.mark.parametrize(
    ("rows", "forces", "message"),
    [
        (
            [{"eff_pot": -13.0 / 12.0}],
            (
                -1.0 / 16.0,
                0.0,
                0.0,
                1.0 / 9.0,
                0.0,
                0.0,
                1.0 / 16.0 - 1.0 / 9.0,
                0.0,
                0.0,
            ),
            "exclusion energy mismatch",
        ),
        (
            [{"eff_pot": -1.0 / 12.0}],
            (0.0,) * 9,
            "exclusion force oracle mismatch",
        ),
    ],
)
def test_focused_exclusions_oracle_rejects_ignored_or_wrong_payload(
    rows, forces, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_exclusion_coulomb_oracle("exclusions", rows, forces)


@pytest.mark.parametrize(
    ("legacy", "bundled", "message"),
    [
        ([0.0, 0.0], [0.0, 0.0], "legacy force is all trivial"),
        ([1.0, -1.0], [1.0, 0.0], "mismatch at index"),
    ],
)
def test_focused_edip_force_gate_rejects_trivial_or_mismatched_force(
    legacy, bundled, message
):
    with pytest.raises(AssertionError, match=message):
        _assert_nontrivial_equivalent_forces("EDIP force", legacy, bundled)


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


def test_unrestricted_qc_gate_requires_nontrivial_spin_and_nonempty_scf_text():
    contracts = load_contract_registry()
    cases = {
        case.name: case
        for case in _cases_for_profile()
        if case.name.startswith("rerun_qc_unrestricted_sidecar_vds_")
    }
    expected_case_ids = {
        "rerun_qc_unrestricted_sidecar_vds_off",
        "rerun_qc_unrestricted_sidecar_vds_on",
    }

    assert set(cases) == expected_case_ids
    assert {case.vds for case in cases.values()} == {False, True}
    for case in cases.values():
        assert case.fixture_case == "full_contract_rerun"
        assert case.bundled_subdir == (
            "bundled_input_with_legacy_sidecar/bundle"
        )
        assert case.restart_load_policy == "structural"
        assert not case.statistical_md
        assert case.contract_ids == (
            "output.legacy.mdout",
            "output.legacy.qc_scf_output",
            "input.qc.spin_square",
            "input.qc.scf_text",
        )
        assert case.assertion_ids == (
            "mdout_deterministic_equivalence",
            "h5_rerun_semantic_equivalence",
            "qc_scf_exact_equivalence",
            "input_semantic_equivalence",
        )

    spin_spec = next(
        spec
        for spec in RERUN_INPUT_SEMANTIC_SPECS
        if spec.contract_id == "input.qc.spin_square"
    )
    assert spin_spec == InputSemanticSpec(
        "input.qc.spin_square", ("QC_S_sq",), 1.0e-4
    )
    with pytest.raises(AssertionError, match="all trivial"):
        assert_module_semantics(
            "QC spin-square",
            [{"QC_S_sq": 0.0}],
            [{"QC_S_sq": 0.0}],
            spin_spec,
            deterministic=True,
        )

    spin_contract = contracts["input.qc.spin_square"]
    assert spin_contract.status == "supported"
    assert set(spin_contract.case_ids) == expected_case_ids
    assert spin_contract.assertion_ids == (
        "input_semantic_equivalence",
        "h5_rerun_semantic_equivalence",
    )
    for contract_id in ("input.qc.scf_text", "output.legacy.qc_scf_output"):
        contract = contracts[contract_id]
        assert contract.status == "supported"
        assert set(contract.case_ids) == expected_case_ids
        assert contract.assertion_ids == ("qc_scf_exact_equivalence",)

    qc_type = contracts["input.qc.type"]
    assert qc_type.status == "deferred"
    assert "does not materialize /qc/type" in qc_type.reason


def test_h5_string_reader_preserves_multiline_scf_text(tmp_path):
    path = tmp_path / "qc.obs.spg.h5md"
    dataset = "/parameters/sponge/qc/scf_output"
    expected = "Step 0\n  indented continuation\nStep 1\n"
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            dataset,
            data=expected,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    assert _h5_string_values(path, dataset) == [expected]


def test_input_semantic_contract_inventory_is_explicit_and_evidence_gated():
    contracts = load_contract_registry()
    specialized_semantic_assertions = {
        "input.bias.nhc": "restart_dynamic_continuation_equivalence",
        "input.bias.metadynamics": (
            "restart_protocol_full_continuation_equivalence"
        ),
        "input.protocol.constraint.sidecar": (
            "constraint_geometry_equivalence"
        ),
        "input.qc.scf_text": "qc_scf_exact_equivalence",
    }
    runtime_spec_ids = {
        spec.contract_id
        for specs in INPUT_SEMANTIC_SPECS_BY_CASE.values()
        for spec in specs
    } | {spec.contract_id for spec in RERUN_INPUT_SEMANTIC_SPECS}

    assert REQUIRED_INPUT_SEMANTIC_CONTRACTS <= set(contracts)
    assert runtime_spec_ids | set(specialized_semantic_assertions) == {
        contract_id
        for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS
        if contracts[contract_id].status == "supported"
    }
    for contract_id in REQUIRED_INPUT_SEMANTIC_CONTRACTS:
        contract = contracts[contract_id]
        assert contract.minimum_evidence == "E3"
        if contract.status == "supported":
            expected_assertion = specialized_semantic_assertions.get(
                contract_id, "input_semantic_equivalence"
            )
            assert expected_assertion in contract.assertion_ids
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


def _ready_promotion_fixture():
    matrix = load_execution_matrix()
    contract_id = "runtime.synthetic_matrix"
    scenarios = []
    report_cases = {}
    for index, scenario in enumerate(matrix.scenarios):
        case_id = f"synthetic_matrix_case_{index}"
        scenarios.append(
            replace(
                scenario,
                status="executable",
                case_ids=(case_id,),
                reason="",
            )
        )
        report_cases[case_id] = {
            "metadata": {
                **scenario.axis_values(),
                "omp_num_threads": scenario.omp_threads,
                "mpi_rank_count": scenario.mpi_ranks,
                "rank0_output_owner": True,
            },
            "records": [
                {
                    "contract_id": contract_id,
                    "evidence_level": "E3",
                    "status": "passed",
                }
            ],
        }
    matrix = replace(
        matrix,
        promotion_state="candidate",
        scenarios=tuple(scenarios),
    )
    contracts = {
        contract_id: ContractSpec(
            contract_id=contract_id,
            direction="runtime",
            component="execution_matrix",
            status="supported",
            minimum_evidence="E3",
            legacy_surface="synthetic legacy",
            bundled_surface="synthetic bundled",
            case_ids=tuple(report_cases),
            assertion_ids=("synthetic_equivalence",),
            inventory_refs=(),
        )
    }
    runs = tuple(
        ProductionRun(
            run_id=f"run-{index}",
            passed=True,
            retry_count=0,
            runtime_ratio=1.0,
            finalize_fraction=0.1,
            output_bytes_ratio=1.0,
            comparator_mutations_rejected=True,
        )
        for index in range(matrix.required_consecutive_production_runs)
    )
    return matrix, contracts, {"cases": report_cases}, runs


def test_execution_matrix_enumerates_every_required_axis_and_risk_pair():
    matrix = load_execution_matrix()
    runtime_cases = {case.scenario_id: case for case in MATRIX_RUNTIME_CASES}
    case_ids = {case.name for case in _cases_for_profile()} | set(runtime_cases)

    validate_execution_matrix(matrix, case_ids)

    assert set(matrix.required_axes) == {
        "ensemble",
        "thermostat",
        "box_geometry",
        "constraint",
        "backend",
        "omp_threads",
        "mpi_ranks",
        "comparison",
    }
    assert matrix.required_consecutive_production_runs == 3
    assert matrix.promotion_state == "shadow"
    executable = [
        scenario
        for scenario in matrix.scenarios
        if scenario.status == "executable"
    ]
    assert len(executable) == 12
    for scenario in executable:
        assert scenario.case_ids == (scenario.scenario_id,)
        runtime_case = runtime_cases[scenario.scenario_id]
        assert scenario.axis_values() == runtime_case.axis_values()
        assert scenario.barostat == runtime_case.barostat
        assert scenario.tier == runtime_case.tier


def test_execution_matrix_fixture_hashes_are_reviewed_and_pinned():
    manifest = json.loads(
        (MATRIX_FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["initial_velocity_seed"] == 20260709
    assert manifest["nonorthogonal_box_angles_degrees"] == [80.0, 100.0, 110.0]
    assert set(manifest["sha256"]) == {
        "common/mdin.bundled.spg.toml",
        "common/legacy_sidecars/LJ_in_file/tip3p_LJ.txt",
        "common/legacy_sidecars/bond_in_file/tip3p_bond.txt",
        "common/legacy_sidecars/charge_in_file/tip3p_charge.txt",
        "common/legacy_sidecars/exclude_in_file/tip3p_exclude.txt",
        "common/legacy_sidecars/mass_in_file/tip3p_mass.txt",
        "common/legacy_sidecars/residue_in_file/tip3p_residue.txt",
        "common/topology.spgt.h5",
        "common/protocol.spgp.h5",
        "orthogonal/restart.spgr.h5",
        "nonorthogonal/restart.spgr.h5",
        "initial_velocity.txt",
    }
    for relative_path, expected in manifest["sha256"].items():
        actual = hashlib.sha256(
            (MATRIX_FIXTURE_ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert actual == expected, relative_path


def test_execution_matrix_rejects_removed_axis_and_unmapped_combination():
    matrix = load_execution_matrix()
    axes = dict(matrix.required_axes)
    axes.pop("mpi_ranks")
    with pytest.raises(AssertionError, match="axes differ"):
        validate_execution_matrix(replace(matrix, required_axes=axes))

    impossible = {
        "ensemble": "npt",
        "thermostat": "nose_hoover_chain",
        "comparison": "deterministic",
    }
    with pytest.raises(AssertionError, match="has no scenario"):
        validate_execution_matrix(
            replace(
                matrix,
                required_combinations=(
                    *matrix.required_combinations,
                    impossible,
                ),
            )
        )

    unknown_case = replace(
        matrix.scenarios[0], case_ids=("missing_executable_case",)
    )
    with pytest.raises(AssertionError, match="references unknown cases"):
        validate_execution_matrix(
            replace(
                matrix,
                scenarios=(unknown_case, *matrix.scenarios[1:]),
            ),
            available_case_ids={case.name for case in _cases_for_profile()}
            | {case.scenario_id for case in MATRIX_RUNTIME_CASES},
        )


def test_current_execution_matrix_cannot_be_promoted_by_declarations_only():
    matrix = load_execution_matrix()
    decision = evaluate_promotion_readiness(
        matrix,
        load_contract_registry(),
        evidence_report=None,
        production_runs=(),
    )

    assert decision.ready is False
    assert any("promotion_state is shadow" in item for item in decision.blockers)
    assert any(
        "does not prove environment" in item for item in decision.blockers
    )
    assert any("contracts lack evidence" in item for item in decision.blockers)
    assert any("consecutive retry-free" in item for item in decision.blockers)


def test_promotion_requires_environment_metadata_for_every_scenario():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    assert evaluate_promotion_readiness(matrix, contracts, report, runs).ready
    contract_id = next(iter(contracts))
    contract_report = {
        "cases": {
            "contract_case": {
                "records": [
                    {
                        "contract_id": contract_id,
                        "evidence_level": "E3",
                        "status": "passed",
                    }
                ]
            }
        }
    }
    assert evaluate_promotion_readiness(
        matrix,
        contracts,
        contract_report,
        runs,
        scenario_evidence_report=report,
    ).ready

    first = matrix.scenarios[0]
    case_payload = report["cases"][first.case_ids[0]]
    case_payload["metadata"]["omp_num_threads"] = first.omp_threads + 1
    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(
        first.scenario_id in blocker and "does not prove environment" in blocker
        for blocker in decision.blockers
    )


def test_promotion_requires_three_consecutive_runs_without_retry():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    runs = (*runs[:-1], replace(runs[-1], retry_count=1))

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any("got 0" in blocker for blocker in decision.blockers)


@pytest.mark.parametrize(
    ("field", "budget_field", "token"),
    [
        ("runtime_ratio", "maximum_runtime_ratio", "runtime_ratio"),
        (
            "finalize_fraction",
            "maximum_finalize_fraction",
            "finalize_fraction",
        ),
        (
            "output_bytes_ratio",
            "maximum_output_bytes_ratio",
            "output_bytes_ratio",
        ),
    ],
)
def test_promotion_rejects_each_performance_budget_violation(
    field, budget_field, token
):
    matrix, contracts, report, runs = _ready_promotion_fixture()
    limit = getattr(matrix.performance_budgets, budget_field)
    runs = (*runs[:-1], replace(runs[-1], **{field: limit + 0.01}))

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any(token in blocker for blocker in decision.blockers)


def test_promotion_requires_all_comparator_mutations_to_be_rejected():
    matrix, contracts, report, runs = _ready_promotion_fixture()
    runs = (
        *runs[:-1],
        replace(runs[-1], comparator_mutations_rejected=False),
    )

    decision = evaluate_promotion_readiness(matrix, contracts, report, runs)

    assert decision.ready is False
    assert any("mutations were not all rejected" in item for item in decision.blockers)


def test_shadow_workflow_runs_tiers_without_becoming_a_release_gate():
    workflow = AB_SHADOW_WORKFLOW.read_text(encoding="utf-8")
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Bundled I/O A/B Shadow" in workflow
    assert "python -m benchmarks.bundled_io.execution_matrix" in workflow
    assert "smoke-bundled-io-contract" in workflow
    assert "ab-bundled-io-medium" in workflow
    assert "ab-bundled-io-production" in workflow
    assert "test_bundled_io_ab_execution_matrix.py" in workflow
    assert workflow.count("SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: cpu-rank1") == 2
    assert "SPONGE_BUNDLED_IO_AB_MATRIX_SCENARIOS: cpu-rank2" in workflow
    assert "pixi install -e dev-cpu-mpi" in workflow
    assert workflow.count("continue-on-error: true") == 3
    assert "schedule:" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "bundled-io-ab-shadow" not in release
    assert "ab-bundled-io-production" not in release
