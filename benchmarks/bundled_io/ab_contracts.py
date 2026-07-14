from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for local inspection
    fcntl = None


REGISTRY_PATH = Path(__file__).with_name("contracts") / "ab_contracts.json"
VALID_STATUSES = {"supported", "deferred", "unsupported"}
VALID_DIRECTIONS = {"input", "output", "runtime", "system"}
INVENTORY_SECTIONS = {
    "h5_input_keys",
    "h5_output_keys",
    "legacy_output_keys",
    "topology_sidecar_keys",
    "protocol_sidecar_keys",
    "rerun_control_keys",
}
EVIDENCE_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
VALID_EVIDENCE_LEVELS = set(EVIDENCE_RANK) | {"F1"}
DYNAMIC_RESTART_CONTRACT_STATUSES = {
    "input.restart.dynamic.integrator_state": "supported",
    "input.restart.dynamic.nose_hoover_chain": "supported",
    "input.restart.dynamic.bussi_thermostat": "deferred",
    "input.restart.dynamic.pressure_based_barostat": "deferred",
    "input.restart.dynamic.middle_langevin_rng": "unsupported",
    "input.restart.dynamic.andersen_rng": "unsupported",
    "input.restart.dynamic.monte_carlo_barostat_rng": "unsupported",
}


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    direction: str
    component: str
    status: str
    minimum_evidence: str
    legacy_surface: str
    bundled_surface: str
    case_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]
    inventory_refs: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class AssertionEvidence:
    assertion_id: str
    evidence_level: str
    details: Mapping[str, object]


@dataclass(frozen=True)
class EvidenceRecord:
    case_id: str
    contract_id: str
    assertion_id: str
    evidence_level: str
    status: str
    details: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "contract_id": self.contract_id,
            "assertion_id": self.assertion_id,
            "evidence_level": self.evidence_level,
            "status": self.status,
            "details": dict(self.details),
        }


def load_contract_registry(
    path: Path = REGISTRY_PATH,
) -> dict[str, ContractSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1":
        raise AssertionError("A/B contract registry schema_version must be '1'")
    raw_contracts = payload.get("contracts")
    if not isinstance(raw_contracts, list):
        raise AssertionError("A/B contract registry contracts must be a list")

    contracts: dict[str, ContractSpec] = {}
    for raw in raw_contracts:
        if not isinstance(raw, dict):
            raise AssertionError(
                "A/B contract registry entries must be objects"
            )
        spec = _parse_contract(raw)
        if spec.contract_id in contracts:
            raise AssertionError(
                f"duplicate A/B contract ID: {spec.contract_id}"
            )
        contracts[spec.contract_id] = spec
    if not contracts:
        raise AssertionError("A/B contract registry must not be empty")
    return contracts


def load_implementation_inventory(
    path: Path = REGISTRY_PATH,
) -> dict[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_inventory = payload.get("implementation_inventory")
    if not isinstance(raw_inventory, dict):
        raise AssertionError("A/B registry requires implementation_inventory")
    if set(raw_inventory) != INVENTORY_SECTIONS:
        raise AssertionError(
            "A/B implementation inventory sections differ: "
            f"expected={sorted(INVENTORY_SECTIONS)}, "
            f"actual={sorted(raw_inventory)}"
        )
    inventory = {}
    for section, raw_values in raw_inventory.items():
        values = _string_tuple(raw_values, section, "implementation_inventory")
        _require_unique(values, f"implementation_inventory {section}")
        inventory[section] = values
    return inventory


def validate_implementation_inventory(
    contracts: Mapping[str, ContractSpec],
    inventory: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    known_refs = {
        f"{section}:{value}"
        for section, values in inventory.items()
        for value in values
    }
    owners: dict[str, str] = {}
    for spec in contracts.values():
        for inventory_ref in spec.inventory_refs:
            if inventory_ref not in known_refs:
                raise AssertionError(
                    f"contract {spec.contract_id} owns unknown inventory ref "
                    f"{inventory_ref}"
                )
            if inventory_ref in owners:
                raise AssertionError(
                    f"inventory ref {inventory_ref} has multiple owners: "
                    f"{owners[inventory_ref]}, {spec.contract_id}"
                )
            owners[inventory_ref] = spec.contract_id
    missing = sorted(known_refs - set(owners))
    if missing:
        raise AssertionError(
            f"implementation inventory refs have no contract: {missing}"
        )
    return owners


def validate_contract_registry(
    contracts: Mapping[str, ContractSpec], cases: Sequence[object]
) -> dict[str, object]:
    _validate_dynamic_restart_contract_inventory(contracts)
    case_by_id: dict[str, object] = {}
    for case in cases:
        case_id = _case_id(case)
        if case_id in case_by_id:
            raise AssertionError(f"duplicate A/B case ID: {case_id}")
        case_by_id[case_id] = case

        contract_ids = _case_values(case, "contract_ids")
        assertion_ids = _case_values(case, "assertion_ids")
        _require_unique(contract_ids, f"{case_id} contract_ids")
        _require_unique(assertion_ids, f"{case_id} assertion_ids")
        if not contract_ids:
            raise AssertionError(f"{case_id} declares no A/B contracts")
        if not assertion_ids:
            raise AssertionError(f"{case_id} declares no A/B assertions")

        unknown = sorted(set(contract_ids) - set(contracts))
        if unknown:
            raise AssertionError(
                f"{case_id} references unknown contracts: {unknown}"
            )
        allowed_assertions = {
            assertion_id
            for contract_id in contract_ids
            for assertion_id in contracts[contract_id].assertion_ids
        }
        undeclared = sorted(set(assertion_ids) - allowed_assertions)
        if undeclared:
            raise AssertionError(
                f"{case_id} references assertions not owned by its contracts: "
                f"{undeclared}"
            )

    for spec in contracts.values():
        if spec.status == "supported":
            if not spec.case_ids:
                raise AssertionError(
                    f"supported contract {spec.contract_id} has no cases"
                )
            if not spec.assertion_ids:
                raise AssertionError(
                    f"supported contract {spec.contract_id} has no assertions"
                )
        elif not spec.reason:
            raise AssertionError(
                f"{spec.status} contract {spec.contract_id} requires a reason"
            )

        for case_id in spec.case_ids:
            if case_id not in case_by_id:
                raise AssertionError(
                    f"contract {spec.contract_id} references missing case {case_id}"
                )
            case = case_by_id[case_id]
            if spec.contract_id not in _case_values(case, "contract_ids"):
                raise AssertionError(
                    f"contract {spec.contract_id} and case {case_id} are not "
                    "mapped symmetrically"
                )
            if not set(spec.assertion_ids).intersection(
                _case_values(case, "assertion_ids")
            ):
                raise AssertionError(
                    f"contract {spec.contract_id} has no executable assertion "
                    f"in case {case_id}"
                )

    return registry_summary(contracts)


def _validate_dynamic_restart_contract_inventory(
    contracts: Mapping[str, ContractSpec],
) -> None:
    actual = {
        contract_id
        for contract_id in contracts
        if contract_id.startswith("input.restart.dynamic.")
    }
    expected = set(DYNAMIC_RESTART_CONTRACT_STATUSES)
    if actual != expected:
        raise AssertionError(
            "dynamic restart contract inventory differs: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    mismatched = {
        contract_id: {
            "expected": expected_status,
            "actual": contracts[contract_id].status,
        }
        for contract_id, expected_status in (
            DYNAMIC_RESTART_CONTRACT_STATUSES.items()
        )
        if contracts[contract_id].status != expected_status
    }
    if mismatched:
        raise AssertionError(
            f"dynamic restart contract statuses differ: {mismatched}"
        )


def build_case_evidence(
    contracts: Mapping[str, ContractSpec],
    case: object,
    assertions: Sequence[AssertionEvidence],
) -> list[EvidenceRecord]:
    case_id = _case_id(case)
    declared_assertions = _case_values(case, "assertion_ids")
    executed = {item.assertion_id: item for item in assertions}
    if len(executed) != len(assertions):
        raise AssertionError(f"{case_id} emitted duplicate assertion evidence")

    missing = sorted(set(declared_assertions) - set(executed))
    unexpected = sorted(set(executed) - set(declared_assertions))
    if missing or unexpected:
        raise AssertionError(
            f"{case_id} assertion evidence mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    records: list[EvidenceRecord] = []
    for contract_id in _case_values(case, "contract_ids"):
        if contract_id not in contracts:
            raise AssertionError(
                f"{case_id} references unknown contract {contract_id}"
            )
        spec = contracts[contract_id]
        candidates = [
            executed[assertion_id]
            for assertion_id in spec.assertion_ids
            if assertion_id in executed
        ]
        if not candidates:
            raise AssertionError(
                f"{case_id} emitted no evidence for contract {contract_id}"
            )
        satisfying = [
            item
            for item in candidates
            if _evidence_satisfies(item.evidence_level, spec.minimum_evidence)
        ]
        if spec.status == "supported" and not satisfying:
            actual = sorted({item.evidence_level for item in candidates})
            raise AssertionError(
                f"{case_id} contract {contract_id} requires "
                f"{spec.minimum_evidence}, got {actual}"
            )
        eligible = satisfying or candidates
        best = max(
            eligible, key=lambda item: _evidence_sort_rank(item.evidence_level)
        )
        records.append(
            EvidenceRecord(
                case_id=case_id,
                contract_id=contract_id,
                assertion_id=best.assertion_id,
                evidence_level=best.evidence_level,
                status="passed",
                details=best.details,
            )
        )
    return records


def registry_summary(
    contracts: Mapping[str, ContractSpec],
) -> dict[str, object]:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    minimum_levels: dict[str, int] = {}
    for spec in contracts.values():
        counts[spec.status] += 1
        minimum_levels[spec.minimum_evidence] = (
            minimum_levels.get(spec.minimum_evidence, 0) + 1
        )
    return {
        "contract_count": len(contracts),
        "status_counts": counts,
        "minimum_evidence_counts": dict(sorted(minimum_levels.items())),
    }


def update_evidence_report(
    path: Path,
    contracts: Mapping[str, ContractSpec],
    case: object,
    records: Sequence[EvidenceRecord],
    metadata: Mapping[str, object],
    run_id: str,
) -> dict[str, object]:
    if not run_id:
        raise AssertionError("A/B evidence report requires a non-empty run_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        report = _read_existing_report(path)
        if report.get("run_id") != run_id:
            report = {"schema_version": "1", "run_id": run_id, "cases": {}}
        cases = report.setdefault("cases", {})
        if not isinstance(cases, dict):
            raise AssertionError("A/B evidence report cases must be an object")
        cases[_case_id(case)] = {
            "metadata": dict(metadata),
            "records": [record.as_dict() for record in records],
        }
        report["coverage"] = _report_coverage(contracts, cases)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return report


def validate_complete_evidence_report(
    path: Path, contracts: Mapping[str, ContractSpec], run_id: str
) -> dict[str, object]:
    report = _read_existing_report(path)
    if report.get("run_id") != run_id:
        raise AssertionError(
            f"A/B evidence report run_id mismatch: expected={run_id!r}, "
            f"actual={report.get('run_id')!r}"
        )
    cases = report.get("cases")
    if not isinstance(cases, dict):
        raise AssertionError("A/B evidence report cases must be an object")
    coverage = _report_coverage(contracts, cases)
    if coverage["missing_supported_contracts"]:
        raise AssertionError(
            "A/B evidence report is missing supported contracts: "
            f"{coverage['missing_supported_contracts']}"
        )
    if coverage["supported_coverage_fraction"] != 1.0:
        raise AssertionError("A/B supported contract coverage must equal 1.0")
    return report


def _parse_contract(raw: Mapping[str, object]) -> ContractSpec:
    required_strings = (
        "contract_id",
        "direction",
        "component",
        "status",
        "minimum_evidence",
        "legacy_surface",
        "bundled_surface",
    )
    values: dict[str, str] = {}
    for key in required_strings:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise AssertionError(f"A/B contract entry requires non-empty {key}")
        values[key] = value
    if values["status"] not in VALID_STATUSES:
        raise AssertionError(
            f"invalid A/B contract status for {values['contract_id']}: "
            f"{values['status']}"
        )
    if values["direction"] not in VALID_DIRECTIONS:
        raise AssertionError(
            f"invalid A/B contract direction for {values['contract_id']}: "
            f"{values['direction']}"
        )
    _require_evidence_level(values["minimum_evidence"])
    case_ids = _string_tuple(
        raw.get("case_ids"), "case_ids", values["contract_id"]
    )
    assertion_ids = _string_tuple(
        raw.get("assertion_ids"), "assertion_ids", values["contract_id"]
    )
    inventory_refs = _string_tuple(
        raw.get("inventory_refs", []),
        "inventory_refs",
        values["contract_id"],
    )
    _require_unique(case_ids, f"{values['contract_id']} case_ids")
    _require_unique(assertion_ids, f"{values['contract_id']} assertion_ids")
    _require_unique(inventory_refs, f"{values['contract_id']} inventory_refs")
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise AssertionError(f"{values['contract_id']} reason must be a string")
    return ContractSpec(
        **values,
        case_ids=case_ids,
        assertion_ids=assertion_ids,
        inventory_refs=inventory_refs,
        reason=reason,
    )


def _string_tuple(
    value: object, field: str, contract_id: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AssertionError(f"{contract_id} {field} must be a string list")
    return tuple(value)


def _case_id(case: object) -> str:
    value = getattr(case, "name", None)
    if not isinstance(value, str) or not value:
        raise AssertionError("A/B cases require a non-empty name")
    return value


def _case_values(case: object, field: str) -> tuple[str, ...]:
    value = getattr(case, field, None)
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AssertionError(
            f"{_case_id(case)} requires a string tuple {field}"
        )
    return value


def _require_unique(values: Iterable[str], label: str) -> None:
    values = tuple(values)
    if len(values) != len(set(values)):
        raise AssertionError(f"{label} contains duplicates")


def _require_evidence_level(level: str) -> None:
    if level not in VALID_EVIDENCE_LEVELS:
        raise AssertionError(f"unknown A/B evidence level: {level}")


def _evidence_satisfies(actual: str, required: str) -> bool:
    _require_evidence_level(actual)
    _require_evidence_level(required)
    if actual == "F1" or required == "F1":
        return actual == required
    return EVIDENCE_RANK[actual] >= EVIDENCE_RANK[required]


def _evidence_sort_rank(level: str) -> int:
    _require_evidence_level(level)
    return EVIDENCE_RANK.get(level, 0)


def _read_existing_report(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": "1", "cases": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1":
        raise AssertionError("A/B evidence report schema_version must be '1'")
    return payload


def _report_coverage(
    contracts: Mapping[str, ContractSpec], cases: Mapping[str, object]
) -> dict[str, object]:
    passed_contracts = {
        record["contract_id"]
        for case_payload in cases.values()
        if isinstance(case_payload, dict)
        for record in case_payload.get("records", [])
        if isinstance(record, dict) and record.get("status") == "passed"
    }
    supported = {
        contract_id
        for contract_id, spec in contracts.items()
        if spec.status == "supported"
    }
    covered = sorted(supported.intersection(passed_contracts))
    missing = sorted(supported - passed_contracts)
    status_coverage = {}
    for status in sorted(VALID_STATUSES):
        status_contracts = {
            contract_id
            for contract_id, spec in contracts.items()
            if spec.status == status
        }
        evidenced = sorted(status_contracts.intersection(passed_contracts))
        status_coverage[status] = {
            "contract_count": len(status_contracts),
            "contract_ids": sorted(status_contracts),
            "evidenced_contract_count": len(evidenced),
            "evidenced_contracts": evidenced,
        }
    return {
        "supported_contract_count": len(supported),
        "covered_supported_contract_count": len(covered),
        "supported_coverage_fraction": (
            len(covered) / len(supported) if supported else 1.0
        ),
        "covered_supported_contracts": covered,
        "missing_supported_contracts": missing,
        "status_coverage": status_coverage,
    }
