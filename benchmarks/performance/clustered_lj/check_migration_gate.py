#!/usr/bin/env python3
"""Enforce the clustered-LJ migration performance acceptance matrix.

The gate is intentionally conjunctive.  It never averages results across
systems or ensembles, and it verifies the production LJ-parameter route so a
water-only fast-path result cannot hide a packed-AB regression.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

EXPECTED_SYSTEM_ROUTES = {
    "wat160k": "comb",
    "wat600k": "comb",
    "dna_cou": "packed-ab",
}
EXPECTED_ENSEMBLES = ("nvt", "npt")
EXPECTED_OUTPUT_MODES = ("force-only", "full")
MAX_REGRESSION = 0.03
MIN_MATRIX_RUNS = 3
MIN_MATRIX_STEPS = 10_000
MIN_REPLAY_RUNS = 3
MIN_REPLAY_ITERS = 2_000

EXPECTED_REPLAY_LAYOUT = {
    ("wat160k", "force-only"): (1, False),
    ("wat160k", "full"): (1, False),
    ("wat600k", "force-only"): (1, False),
    ("wat600k", "full"): (1, False),
    ("dna_cou", "force-only"): (3, True),
    ("dna_cou", "full"): (2, False),
}

MATRIX_FIELDS = {
    "cycle",
    "implementation",
    "system",
    "ensemble",
    "steps",
    "valid",
    "lj_mode",
    "force_s",
    "wall_s",
    "speed_ns_day",
}
REPLAY_FIELDS = {
    "cycle",
    "implementation",
    "system",
    "output_mode",
    "valid",
    "lj_mode",
    "sci_work_parts",
    "contiguous_sci_work",
    "iters",
    "sanity",
    "matched",
    "avg_ms",
}

TRUE_VALUES = {"1", "true", "yes", "pass", "valid", "ok"}
FALSE_VALUES = {"0", "false", "no", "fail", "invalid", "bad"}


class GateError(RuntimeError):
    """Raised when one or more acceptance cells fail."""


def _read_tsv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = sorted(required_fields - fields)
        if missing:
            raise GateError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise GateError(f"{path}: no result rows")
    return rows


def _parse_bool(value: str, context: str) -> bool:
    if value is None:
        raise GateError(f"{context}: missing boolean value")
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise GateError(f"{context}: expected boolean, got {value!r}")


def _parse_int(value: str, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise GateError(
            f"{context}: expected integer, got {value!r}"
        ) from error
    return parsed


def _parse_positive_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"{context}: expected number, got {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise GateError(
            f"{context}: expected finite positive value, got {value!r}"
        )
    return parsed


def _mean(rows: Iterable[dict[str, str]], field: str, context: str) -> float:
    return statistics.mean(
        _parse_positive_float(row[field], f"{context} {field}") for row in rows
    )


def _validate_domain(
    rows: list[dict[str, str]],
    *,
    labels: set[str],
    has_ensemble: bool,
) -> list[str]:
    failures = []
    allowed_systems = set(EXPECTED_SYSTEM_ROUTES)
    allowed_secondary = (
        set(EXPECTED_ENSEMBLES) if has_ensemble else set(EXPECTED_OUTPUT_MODES)
    )
    secondary_field = "ensemble" if has_ensemble else "output_mode"

    seen_labels = {row["implementation"] for row in rows}
    seen_systems = {row["system"] for row in rows}
    seen_secondary = {row[secondary_field] for row in rows}
    if seen_labels != labels:
        failures.append(
            f"implementations must be exactly {sorted(labels)}, got "
            f"{sorted(seen_labels)}"
        )
    if seen_systems != allowed_systems:
        failures.append(
            f"systems must be exactly {sorted(allowed_systems)}, got "
            f"{sorted(seen_systems)}"
        )
    if seen_secondary != allowed_secondary:
        failures.append(
            f"{secondary_field} values must be exactly "
            f"{sorted(allowed_secondary)}, got {sorted(seen_secondary)}"
        )
    return failures


def _validate_matrix(
    rows: list[dict[str, str]],
    *,
    baseline_label: str,
    candidate_label: str,
    threshold: float,
    min_runs: int,
    required_steps: int,
) -> tuple[list[str], list[tuple[str, str, float, float, float]]]:
    labels = {baseline_label, candidate_label}
    failures = _validate_domain(rows, labels=labels, has_ensemble=True)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen_cycles: set[tuple[str, str, str, int]] = set()

    for row_number, row in enumerate(rows, start=2):
        context = f"matrix row {row_number}"
        cycle = _parse_int(row["cycle"], f"{context} cycle")
        if cycle <= 0:
            failures.append(f"{context}: cycle={cycle}, expected positive ID")
        key = (
            row["implementation"],
            row["system"],
            row["ensemble"],
            cycle,
        )
        if key in seen_cycles:
            failures.append(f"{context}: duplicate cell/cycle {key}")
        seen_cycles.add(key)
        if not _parse_bool(row["valid"], f"{context} valid"):
            failures.append(f"{context}: row is marked invalid")
        steps = _parse_int(row["steps"], f"{context} steps")
        if steps != required_steps:
            failures.append(
                f"{context}: steps={steps}, expected {required_steps}"
            )
        expected_route = EXPECTED_SYSTEM_ROUTES.get(row["system"])
        if expected_route is not None and row["lj_mode"] != expected_route:
            failures.append(
                f"{context}: {row['system']} used lj_mode={row['lj_mode']!r}, "
                f"expected {expected_route!r}"
            )
        for field in ("force_s", "wall_s", "speed_ns_day"):
            _parse_positive_float(row[field], f"{context} {field}")
        groups[(row["implementation"], row["system"], row["ensemble"])].append(
            row
        )

    summaries = []
    for system in EXPECTED_SYSTEM_ROUTES:
        for ensemble in EXPECTED_ENSEMBLES:
            baseline = groups[(baseline_label, system, ensemble)]
            candidate = groups[(candidate_label, system, ensemble)]
            cell = f"{system}/{ensemble}"
            if len(baseline) < min_runs:
                failures.append(
                    f"{cell}: baseline has {len(baseline)} runs, "
                    f"requires at least {min_runs}"
                )
            if len(candidate) < min_runs:
                failures.append(
                    f"{cell}: candidate has {len(candidate)} runs, "
                    f"requires at least {min_runs}"
                )
            baseline_cycles = {
                _parse_int(row["cycle"], f"{cell} baseline cycle")
                for row in baseline
            }
            candidate_cycles = {
                _parse_int(row["cycle"], f"{cell} candidate cycle")
                for row in candidate
            }
            if baseline_cycles != candidate_cycles:
                failures.append(
                    f"{cell}: baseline/candidate cycles do not match "
                    f"({sorted(baseline_cycles)} vs "
                    f"{sorted(candidate_cycles)})"
                )
            if not baseline or not candidate:
                continue

            force_delta = (
                _mean(candidate, "force_s", cell)
                / _mean(baseline, "force_s", cell)
                - 1.0
            )
            wall_delta = (
                _mean(candidate, "wall_s", cell)
                / _mean(baseline, "wall_s", cell)
                - 1.0
            )
            speed_delta = (
                _mean(candidate, "speed_ns_day", cell)
                / _mean(baseline, "speed_ns_day", cell)
                - 1.0
            )
            summaries.append(
                (system, ensemble, force_delta, wall_delta, speed_delta)
            )
            if force_delta > threshold:
                failures.append(
                    f"{cell}: force regression {force_delta:+.3%} exceeds "
                    f"{threshold:.3%}"
                )
            if wall_delta > threshold:
                failures.append(
                    f"{cell}: wall regression {wall_delta:+.3%} exceeds "
                    f"{threshold:.3%}"
                )
            if speed_delta < -threshold:
                failures.append(
                    f"{cell}: speed regression {speed_delta:+.3%} exceeds "
                    f"{threshold:.3%}"
                )
    return failures, summaries


def _validate_replays(
    rows: list[dict[str, str]],
    *,
    baseline_label: str,
    candidate_label: str,
    threshold: float,
    min_runs: int,
    min_iters: int,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    labels = {baseline_label, candidate_label}
    failures = _validate_domain(rows, labels=labels, has_ensemble=False)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen_cycles: set[tuple[str, str, str, int]] = set()

    for row_number, row in enumerate(rows, start=2):
        context = f"replay row {row_number}"
        cycle = _parse_int(row["cycle"], f"{context} cycle")
        if cycle <= 0:
            failures.append(f"{context}: cycle={cycle}, expected positive ID")
        key = (
            row["implementation"],
            row["system"],
            row["output_mode"],
            cycle,
        )
        if key in seen_cycles:
            failures.append(f"{context}: duplicate cell/cycle {key}")
        seen_cycles.add(key)
        if not _parse_bool(row["valid"], f"{context} valid"):
            failures.append(f"{context}: row is marked invalid")
        if row["sanity"] != "ok":
            failures.append(
                f"{context}: sanity={row['sanity']!r}, expected 'ok'"
            )
        if row["output_mode"] == "full" and not _parse_bool(
            row["matched"], f"{context} matched"
        ):
            failures.append(f"{context}: strict full replay did not match")
        iters = _parse_int(row["iters"], f"{context} iters")
        if iters < min_iters:
            failures.append(
                f"{context}: iters={iters}, requires at least {min_iters}"
            )
        expected_route = EXPECTED_SYSTEM_ROUTES.get(row["system"])
        if expected_route is not None and row["lj_mode"] != expected_route:
            failures.append(
                f"{context}: {row['system']} used lj_mode={row['lj_mode']!r}, "
                f"expected {expected_route!r}"
            )
        expected_layout = EXPECTED_REPLAY_LAYOUT.get(
            (row["system"], row["output_mode"])
        )
        if expected_layout is not None:
            work_parts = _parse_int(
                row["sci_work_parts"], f"{context} sci_work_parts"
            )
            contiguous = _parse_bool(
                row["contiguous_sci_work"],
                f"{context} contiguous_sci_work",
            )
            if (work_parts, contiguous) != expected_layout:
                failures.append(
                    f"{context}: route layout {(work_parts, contiguous)} "
                    f"does not match expected {expected_layout}"
                )
        _parse_positive_float(row["avg_ms"], f"{context} avg_ms")
        groups[
            (row["implementation"], row["system"], row["output_mode"])
        ].append(row)

    summaries = []
    for system in EXPECTED_SYSTEM_ROUTES:
        for output_mode in EXPECTED_OUTPUT_MODES:
            baseline = groups[(baseline_label, system, output_mode)]
            candidate = groups[(candidate_label, system, output_mode)]
            cell = f"{system}/{output_mode}"
            if len(baseline) < min_runs:
                failures.append(
                    f"{cell}: baseline has {len(baseline)} replays, "
                    f"requires at least {min_runs}"
                )
            if len(candidate) < min_runs:
                failures.append(
                    f"{cell}: candidate has {len(candidate)} replays, "
                    f"requires at least {min_runs}"
                )
            baseline_cycles = {
                _parse_int(row["cycle"], f"{cell} baseline replay cycle")
                for row in baseline
            }
            candidate_cycles = {
                _parse_int(row["cycle"], f"{cell} candidate replay cycle")
                for row in candidate
            }
            if baseline_cycles != candidate_cycles:
                failures.append(
                    f"{cell}: baseline/candidate replay cycles do not match "
                    f"({sorted(baseline_cycles)} vs "
                    f"{sorted(candidate_cycles)})"
                )
            if (
                not baseline
                or not candidate
                or baseline_cycles != candidate_cycles
            ):
                continue
            baseline_by_cycle = {
                _parse_int(row["cycle"], f"{cell} baseline cycle"): row
                for row in baseline
            }
            candidate_by_cycle = {
                _parse_int(row["cycle"], f"{cell} candidate cycle"): row
                for row in candidate
            }
            paired_deltas = [
                _parse_positive_float(
                    candidate_by_cycle[cycle]["avg_ms"],
                    f"{cell} candidate cycle {cycle} avg_ms",
                )
                / _parse_positive_float(
                    baseline_by_cycle[cycle]["avg_ms"],
                    f"{cell} baseline cycle {cycle} avg_ms",
                )
                - 1.0
                for cycle in sorted(baseline_cycles)
            ]
            duration_delta = statistics.median(paired_deltas)
            summaries.append((system, output_mode, duration_delta))
            if duration_delta > threshold:
                failures.append(
                    f"{cell}: replay regression {duration_delta:+.3%} "
                    f"exceeds {threshold:.3%}"
                )
    return failures, summaries


def run_gate(
    matrix_path: Path,
    replay_path: Path,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "current",
    threshold: float = MAX_REGRESSION,
    min_runs: int = MIN_MATRIX_RUNS,
    required_steps: int = MIN_MATRIX_STEPS,
    min_replay_runs: int = MIN_REPLAY_RUNS,
    min_replay_iters: int = MIN_REPLAY_ITERS,
) -> tuple[
    list[tuple[str, str, float, float, float]],
    list[tuple[str, str, float]],
]:
    """Validate both result files or raise :class:`GateError`."""

    if baseline_label == candidate_label:
        raise GateError("baseline and candidate labels must differ")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= MAX_REGRESSION:
        raise GateError(
            f"threshold must be finite and no greater than {MAX_REGRESSION}"
        )
    if min_runs < MIN_MATRIX_RUNS:
        raise GateError(f"min_runs must be at least {MIN_MATRIX_RUNS}")
    if required_steps < MIN_MATRIX_STEPS:
        raise GateError(f"required_steps must be at least {MIN_MATRIX_STEPS}")
    if min_replay_runs < MIN_REPLAY_RUNS:
        raise GateError(f"min_replay_runs must be at least {MIN_REPLAY_RUNS}")
    if min_replay_iters < MIN_REPLAY_ITERS:
        raise GateError(f"min_replay_iters must be at least {MIN_REPLAY_ITERS}")

    matrix_rows = _read_tsv(Path(matrix_path), MATRIX_FIELDS)
    replay_rows = _read_tsv(Path(replay_path), REPLAY_FIELDS)
    matrix_failures, matrix_summary = _validate_matrix(
        matrix_rows,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        threshold=threshold,
        min_runs=min_runs,
        required_steps=required_steps,
    )
    replay_failures, replay_summary = _validate_replays(
        replay_rows,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        threshold=threshold,
        min_runs=min_replay_runs,
        min_iters=min_replay_iters,
    )
    failures = matrix_failures + replay_failures
    if failures:
        raise GateError("\n".join(f"- {failure}" for failure in failures))
    return matrix_summary, replay_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="current")
    parser.add_argument("--threshold", type=float, default=MAX_REGRESSION)
    parser.add_argument("--min-runs", type=int, default=MIN_MATRIX_RUNS)
    parser.add_argument("--steps", type=int, default=MIN_MATRIX_STEPS)
    parser.add_argument("--min-replay-runs", type=int, default=MIN_REPLAY_RUNS)
    parser.add_argument(
        "--min-replay-iters", type=int, default=MIN_REPLAY_ITERS
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        matrix_summary, replay_summary = run_gate(
            args.matrix,
            args.replays,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            threshold=args.threshold,
            min_runs=args.min_runs,
            required_steps=args.steps,
            min_replay_runs=args.min_replay_runs,
            min_replay_iters=args.min_replay_iters,
        )
    except (GateError, OSError) as error:
        print(f"clustered-LJ migration gate: FAIL\n{error}", file=sys.stderr)
        return 1

    print("system\tensemble\tforce_delta\twall_delta\tspeed_delta")
    for (
        system,
        ensemble,
        force_delta,
        wall_delta,
        speed_delta,
    ) in matrix_summary:
        print(
            system,
            ensemble,
            f"{force_delta:+.3%}",
            f"{wall_delta:+.3%}",
            f"{speed_delta:+.3%}",
            sep="\t",
        )
    print("\nsystem\toutput_mode\tpaired_median_replay_delta")
    for system, output_mode, duration_delta in replay_summary:
        print(system, output_mode, f"{duration_delta:+.3%}", sep="\t")
    print("clustered-LJ migration gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
