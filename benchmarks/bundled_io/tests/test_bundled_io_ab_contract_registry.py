from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.bundled_io.ab_contracts import (
    AssertionEvidence,
    build_case_evidence,
    load_contract_registry,
    load_implementation_inventory,
    update_evidence_report,
    validate_complete_evidence_report,
    validate_contract_registry,
    validate_implementation_inventory,
)
from benchmarks.bundled_io.tests.test_bundled_io_ab_production import (
    _cases_for_profile,
)


def _synthetic_evidence_level(assertion_id: str) -> str:
    if assertion_id == "full_contract_input_inventory":
        return "E0"
    if assertion_id in {
        "restart_continuation_equivalence",
        "restart_dynamic_continuation_equivalence",
        "restart_protocol_full_continuation_equivalence",
    }:
        return "E4"
    if assertion_id == "stable_failure_semantics":
        return "F1"
    return "E3"


def test_real_registry_and_case_matrix_are_symmetric():
    contracts = load_contract_registry()
    summary = validate_contract_registry(contracts, _cases_for_profile())
    validate_implementation_inventory(
        contracts, load_implementation_inventory()
    )

    assert summary["contract_count"] == len(contracts)
    assert summary["status_counts"] == {
        "deferred": 0,
        "supported": 82,
        "unsupported": 1,
    }


def test_removing_a_registered_case_fails_the_gate():
    contracts = load_contract_registry()
    cases = _cases_for_profile()

    with pytest.raises(AssertionError, match="missing case"):
        validate_contract_registry(contracts, cases[:-1])


def test_unknown_contract_reference_fails_the_gate():
    contracts = load_contract_registry()
    cases = _cases_for_profile()
    broken = replace(
        cases[0], contract_ids=cases[0].contract_ids + ("unknown.contract",)
    )

    with pytest.raises(AssertionError, match="unknown contracts"):
        validate_contract_registry(contracts, [broken, *cases[1:]])


def test_missing_runtime_assertion_evidence_fails_the_gate():
    contracts = load_contract_registry()
    case = _cases_for_profile()[0]
    assertions = [
        AssertionEvidence(
            assertion_id="mdout_statistical_equivalence",
            evidence_level="E3",
            details={"method": "test"},
        )
    ]

    with pytest.raises(AssertionError, match="assertion evidence mismatch"):
        build_case_evidence(contracts, case, assertions)


def test_e0_cannot_satisfy_an_e3_contract():
    contracts = load_contract_registry()
    case = _cases_for_profile()[0]
    assertions = [
        AssertionEvidence(
            assertion_id=assertion_id,
            evidence_level="E0",
            details={"method": "path_presence_only"},
        )
        for assertion_id in case.assertion_ids
    ]

    with pytest.raises(AssertionError, match=r"requires E3, got \['E0'\]"):
        build_case_evidence(contracts, case, assertions)


def test_evidence_report_merges_cases_and_recomputes_coverage(tmp_path):
    contracts = load_contract_registry()
    cases = _cases_for_profile()[:2]
    report_path = tmp_path / "ab_evidence.json"

    for case in cases:
        assertions = [
            AssertionEvidence(
                assertion_id=assertion_id,
                evidence_level=_synthetic_evidence_level(assertion_id),
                details={"method": "unit_test"},
            )
            for assertion_id in case.assertion_ids
        ]
        records = build_case_evidence(contracts, case, assertions)
        report = update_evidence_report(
            report_path,
            contracts,
            case,
            records,
            {"profile": "unit"},
            "unit-run",
        )

    assert set(report["cases"]) == {case.name for case in cases}
    coverage = report["coverage"]
    assert coverage["covered_supported_contract_count"] > 0
    assert coverage["missing_supported_contracts"]
    assert 0.0 < coverage["supported_coverage_fraction"] < 1.0
    with pytest.raises(AssertionError, match="missing supported contracts"):
        validate_complete_evidence_report(report_path, contracts, "unit-run")


def test_failure_evidence_cannot_satisfy_behavior_evidence():
    contracts = load_contract_registry()
    case = _cases_for_profile()[0]
    assertions = [
        AssertionEvidence(
            assertion_id=assertion_id,
            evidence_level="F1",
            details={"method": "failure_contract"},
        )
        for assertion_id in case.assertion_ids
    ]

    with pytest.raises(AssertionError, match=r"requires E3, got \['F1'\]"):
        build_case_evidence(contracts, case, assertions)


def test_complete_report_requires_and_accepts_every_supported_contract(
    tmp_path,
):
    contracts = load_contract_registry()
    report_path = tmp_path / "complete_ab_evidence.json"

    for case in _cases_for_profile():
        assertions = [
            AssertionEvidence(
                assertion_id=assertion_id,
                evidence_level=_synthetic_evidence_level(assertion_id),
                details={"method": "complete_report_unit_test"},
            )
            for assertion_id in case.assertion_ids
        ]
        records = build_case_evidence(contracts, case, assertions)
        update_evidence_report(
            report_path,
            contracts,
            case,
            records,
            {"profile": "unit"},
            "complete-unit-run",
        )

    report = validate_complete_evidence_report(
        report_path, contracts, "complete-unit-run"
    )
    coverage = report["coverage"]
    assert coverage["supported_coverage_fraction"] == 1.0
    assert coverage["missing_supported_contracts"] == []
