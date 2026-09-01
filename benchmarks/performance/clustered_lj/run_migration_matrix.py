#!/usr/bin/env python3
"""Run the reproducible three-system clustered-LJ migration matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from .check_migration_gate import (
        EXPECTED_ENSEMBLES,
        EXPECTED_SYSTEM_ROUTES,
        MIN_MATRIX_RUNS,
        MIN_MATRIX_STEPS,
    )
except ImportError:
    from check_migration_gate import (
        EXPECTED_ENSEMBLES,
        EXPECTED_SYSTEM_ROUTES,
        MIN_MATRIX_RUNS,
        MIN_MATRIX_STEPS,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_SOURCES = {
    "wat160k": REPO_ROOT / "benchmarks/performance/wat/SPONGE_water_160k",
    "wat600k": (
        REPO_ROOT / "benchmarks/performance/wat/SPONGE_water_600k_2x2x1"
    ),
    "dna_cou": (
        REPO_ROOT / "benchmarks/performance/sinkmeta/statics/dna_cou_sinkmeta"
    ),
}
WATER_INPUTS = {
    "wat160k": ("water.top", "water_npt_eq.gro"),
    "wat600k": ("water_2x2x1.top", "water_npt_eq_2x2x1.gro"),
}
DNA_AB_WARNING = (
    "[clustered gmxpacked lj comb] requested but LJ pair table is not "
    "compatible with geometric comb; using AB-table parameter path"
)
STALE_DIRECT_ENV = "SPONGE_CLUSTERED_USE_GMXPACKED_DIRECT"
ACTIVE_VIEW_ENV = "SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW"
LIFECYCLE_ENV = "SPONGE_CLUSTERED_GMXPACKED_LIFECYCLE_POLICY"
ROLLING_CACHE_ENV = (
    "SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ROLLING_SOURCE_CACHE"
)
HISTORICAL_BASELINE_DNA_PROBES = {
    "SPONGE_CLUSTERED_GMXPACKED_FORCE_SCI_SPLIT3_CONTIGUOUS_PROBE",
    "SPONGE_CLUSTERED_GMXPACKED_VIRIAL_SCI_SPLIT2_PROBE",
    "SPONGE_CLUSTERED_GMXPACKED_ENERGY_VIRIAL_SCI_SPLIT2_PROBE",
}
HISTORICAL_BASELINE_DNA_ONLY_ENV = HISTORICAL_BASELINE_DNA_PROBES | {
    "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING"
}
MATRIX_HEADER = (
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
)
ATTEMPT_HEADER = MATRIX_HEADER + (
    "pre_idle_sm_pct",
    "pre_idle_pclk_mhz",
    "post_idle_sm_pct",
    "post_idle_pclk_mhz",
    "returncode",
    "reason",
    "case_dir",
)


class MatrixRunError(RuntimeError):
    """Raised when the protocol cannot produce a qualifying row."""


@dataclass(frozen=True)
class MatrixCase:
    cycle: int
    implementation: str
    system: str
    ensemble: str
    steps: int

    @property
    def name(self) -> str:
        return (
            f"{self.cycle:02d}_{self.system}_{self.ensemble}_"
            f"{self.implementation}"
        )


@dataclass(frozen=True)
class GpuIdleSample:
    sm_pct: float
    pclk_mhz: float
    power_w: float


def _is_enabled(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _read_environment(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise MatrixRunError(f"{path}: environment JSON must be an object")

    environment = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise MatrixRunError(
                f"{path}: environment entries must be scalar key/value pairs"
            )
        environment[key] = str(value)
    return environment


def load_environment(path: Path) -> dict[str, str]:
    environment = _read_environment(path)
    validate_environment(environment)
    return environment


def load_baseline_environment(path: Path) -> dict[str, str]:
    environment = _read_environment(path)
    validate_baseline_environment(environment)
    return environment


def _validate_common_environment(environment: dict[str, str]) -> None:
    if not _is_enabled(environment.get(ACTIVE_VIEW_ENV, "")):
        raise MatrixRunError(f"{ACTIVE_VIEW_ENV}=1 is required")
    if environment.get(LIFECYCLE_ENV) not in {"outer", "outer-source"}:
        raise MatrixRunError(
            f"{LIFECYCLE_ENV} must be 'outer' or 'outer-source'"
        )
    if _is_enabled(environment.get(ROLLING_CACHE_ENV, "0")):
        raise MatrixRunError(f"{ROLLING_CACHE_ENV}=1 is unsupported")
    if not _is_enabled(
        environment.get("SPONGE_CLUSTERED_GMXPACKED_USE_LJ_COMB_KERNEL", "1")
    ):
        raise MatrixRunError(
            "SPONGE_CLUSTERED_GMXPACKED_USE_LJ_COMB_KERNEL may not be "
            "disabled in the migration matrix"
        )
    if not _is_enabled(
        environment.get("SPONGE_CLUSTERED_GMXPACKED_USE_FAST_KERNEL", "1")
    ):
        raise MatrixRunError(
            "SPONGE_CLUSTERED_GMXPACKED_USE_FAST_KERNEL may not be disabled "
            "in the migration matrix"
        )


def validate_environment(environment: dict[str, str]) -> None:
    if STALE_DIRECT_ENV in environment:
        raise MatrixRunError(
            f"{STALE_DIRECT_ENV} has no production reader in current and is "
            "forbidden"
        )
    probe_keys = sorted(key for key in environment if "_PROBE" in key)
    if probe_keys:
        raise MatrixRunError(
            "qualifying environment may not contain probe keys: "
            + ", ".join(probe_keys)
        )
    _validate_common_environment(environment)


def validate_baseline_environment(environment: dict[str, str]) -> None:
    probe_keys = {key for key in environment if "_PROBE" in key}
    unknown_probes = sorted(probe_keys - HISTORICAL_BASELINE_DNA_PROBES)
    if unknown_probes:
        raise MatrixRunError(
            "historical baseline environment contains unsupported probes: "
            + ", ".join(unknown_probes)
        )
    if probe_keys:
        missing = sorted(HISTORICAL_BASELINE_DNA_PROBES - probe_keys)
        disabled = sorted(
            key
            for key in HISTORICAL_BASELINE_DNA_PROBES
            if not _is_enabled(environment.get(key, ""))
        )
        if missing or disabled:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if disabled:
                details.append("disabled " + ", ".join(disabled))
            raise MatrixRunError(
                "historical baseline must select the complete frozen DNA "
                "split3/split2 probe set: " + "; ".join(details)
            )
        if not _is_enabled(
            environment.get("SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING", "")
        ):
            raise MatrixRunError(
                "historical baseline DNA probes require "
                "SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1"
            )
        if not _is_enabled(environment.get(STALE_DIRECT_ENV, "")):
            raise MatrixRunError(
                "historical baseline DNA probes require "
                f"{STALE_DIRECT_ENV}=1 so the baseline binary cannot fall "
                "back to native LJ"
            )
    elif STALE_DIRECT_ENV in environment and not _is_enabled(
        environment[STALE_DIRECT_ENV]
    ):
        raise MatrixRunError(
            f"historical baseline {STALE_DIRECT_ENV} must be enabled when "
            "present"
        )
    _validate_common_environment(environment)


def resolve_case_environment(
    case: MatrixCase,
    *,
    baseline_environment: dict[str, str],
    current_environment: dict[str, str],
) -> dict[str, str]:
    environment = dict(
        baseline_environment
        if case.implementation == "baseline"
        else current_environment
    )
    if case.implementation == "baseline" and case.system != "dna_cou":
        for key in HISTORICAL_BASELINE_DNA_ONLY_ENV:
            environment.pop(key, None)
    return environment


def build_process_environment(
    environment: dict[str, str],
) -> dict[str, str]:
    process_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SPONGE_")
    }
    process_environment.update(environment)
    return process_environment


def build_case_plan(runs: int, steps: int) -> list[MatrixCase]:
    if runs < MIN_MATRIX_RUNS:
        raise MatrixRunError(f"runs must be at least {MIN_MATRIX_RUNS}")
    if steps < MIN_MATRIX_STEPS:
        raise MatrixRunError(f"steps must be at least {MIN_MATRIX_STEPS}")

    cases = []
    systems = tuple(EXPECTED_SYSTEM_ROUTES)
    for cycle in range(1, runs + 1):
        ensembles = (
            EXPECTED_ENSEMBLES
            if cycle % 2 == 1
            else tuple(reversed(EXPECTED_ENSEMBLES))
        )
        implementations = (
            ("baseline", "current")
            if cycle % 2 == 1
            else ("current", "baseline")
        )
        for system in systems:
            for ensemble in ensembles:
                for implementation in implementations:
                    cases.append(
                        MatrixCase(
                            cycle,
                            implementation,
                            system,
                            ensemble,
                            steps,
                        )
                    )
    return cases


def _water_mdin(case: MatrixCase) -> str:
    topology, coordinate = WATER_INPUTS[case.system]
    npt = case.ensemble == "npt"
    lines = [
        f'md_name = "{case.system} clustered-LJ migration {case.ensemble}"',
        f'mode = "{case.ensemble}"',
        f"step_limit = {case.steps}",
        "dt = 0.001",
        "cutoff = 8.0",
        'thermostat = "middle_langevin"',
        "thermostat_seed = 20260727",
        "target_temperature = 300.0",
        'constrain_mode = "SETTLE"',
        f'gromacs_top = "{topology}"',
        f'gromacs_gro = "{coordinate}"',
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        "print_zeroth_frame = 0",
        f"print_pressure = {1 if npt else 0}",
        f"write_information_interval = {case.steps}",
        f"write_mdout_interval = {case.steps}",
        "write_trajectory_interval = 0",
        "write_restart_file_interval = 0",
        "dont_check_input = 1",
    ]
    if npt:
        lines.extend(
            [
                'barostat = "andersen_barostat"',
                "target_pressure = 1.0",
                "barostat_update_interval = 10",
                "barostat_tau = 1.0",
                "barostat_compressibility = 4.5e-5",
            ]
        )
    lines.extend(
        [
            "",
            "[LJ]",
            'direct_kernel = "clustered"',
            "",
            "[PM]",
            'backend = "pme"',
            "Direct_Tolerance = 1e-4",
            "print_detail = true",
        ]
    )
    return "\n".join(lines) + "\n"


def _dna_mdin(case: MatrixCase) -> str:
    npt = case.ensemble == "npt"
    lines = [
        f'md_name = "DNA_COU clustered-LJ migration {case.ensemble}"',
        f'mode = "{case.ensemble}"',
        f"step_limit = {case.steps}",
        "dt = 0.002",
        "cutoff = 8.0",
        'thermostat = "middle_langevin"',
        "thermostat_seed = 20260727",
        "target_temperature = 300.0",
        'default_in_file_prefix = "2m2c"',
        'coordinate_in_file = "Pmin_coordinate.txt"',
        'velocity_in_file = "Pmin_velocity.txt"',
        'constrain_mode = "SHAKE"',
        'mdout = "mdout.txt"',
        'mdinfo = "mdinfo.txt"',
        'box = "mdbox.txt"',
        "print_zeroth_frame = 0",
        f"print_pressure = {1 if npt else 0}",
        f"write_information_interval = {case.steps}",
        f"write_mdout_interval = {case.steps}",
        "write_trajectory_interval = 0",
        "write_restart_file_interval = 0",
        "dont_check_input = 1",
    ]
    if npt:
        lines.extend(
            [
                'barostat = "andersen_barostat"',
                "target_pressure = 1.0",
                "barostat_update_interval = 10",
                "barostat_tau = 1.0",
                "barostat_compressibility = 4.5e-5",
            ]
        )
    lines.extend(
        [
            "",
            "[LJ]",
            'direct_kernel = "clustered"',
            "",
            "[PM]",
            "MPI_size = 0",
        ]
    )
    return "\n".join(lines) + "\n"


def stage_case(case: MatrixCase, output_root: Path) -> Path:
    case_dir = output_root / "cases" / case.name
    case_dir.mkdir(parents=True, exist_ok=False)
    source_dir = SYSTEM_SOURCES[case.system]
    if case.system in WATER_INPUTS:
        source_names = WATER_INPUTS[case.system]
        mdin = _water_mdin(case)
    else:
        source_names = tuple(
            path.name for path in sorted(source_dir.iterdir()) if path.is_file()
        )
        mdin = _dna_mdin(case)
    for name in source_names:
        source = source_dir / name
        if not source.is_file():
            raise MatrixRunError(f"missing tracked input: {source}")
        (case_dir / name).symlink_to(source.resolve())
    (case_dir / "mdin.spg.toml").write_text(mdin, encoding="utf-8")
    return case_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_root: Path,
    *,
    binaries: dict[str, Path],
    baseline_environment: dict[str, str],
    current_environment: dict[str, str],
    cases: list[MatrixCase],
    staged: dict[str, Path],
    dry_run: bool,
) -> None:
    sources = {}
    for system, source_dir in SYSTEM_SOURCES.items():
        if system in WATER_INPUTS:
            paths = [source_dir / name for name in WATER_INPUTS[system]]
        else:
            paths = sorted(
                path for path in source_dir.iterdir() if path.is_file()
            )
        sources[system] = {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths
        }
    mdin_hashes = {
        case.name: _sha256(staged[case.name] / "mdin.spg.toml")
        for case in cases
    }
    for cycle in range(1, max(case.cycle for case in cases) + 1):
        for system in EXPECTED_SYSTEM_ROUTES:
            for ensemble in EXPECTED_ENSEMBLES:
                baseline_name = MatrixCase(
                    cycle,
                    "baseline",
                    system,
                    ensemble,
                    next(
                        case.steps
                        for case in cases
                        if case.cycle == cycle
                        and case.system == system
                        and case.ensemble == ensemble
                    ),
                ).name
                current_name = MatrixCase(
                    cycle,
                    "current",
                    system,
                    ensemble,
                    next(
                        case.steps
                        for case in cases
                        if case.cycle == cycle
                        and case.system == system
                        and case.ensemble == ensemble
                    ),
                ).name
                if mdin_hashes[baseline_name] != mdin_hashes[current_name]:
                    raise MatrixRunError(
                        f"paired mdin mismatch: {baseline_name} vs "
                        f"{current_name}"
                    )
    manifest = {
        "schema": 2,
        "mode": "dry-run" if dry_run else "execution",
        "repo_root": str(REPO_ROOT),
        "binaries": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in binaries.items()
        },
        "environment": current_environment,
        "environments": {
            "process_policy": {
                "inherit": "non-SPONGE variables",
                "clear_prefixes": ["SPONGE_"],
            },
            "baseline_configured": baseline_environment,
            "current_configured": current_environment,
            "resolved": {
                implementation: {
                    system: resolve_case_environment(
                        MatrixCase(
                            1,
                            implementation,
                            system,
                            EXPECTED_ENSEMBLES[0],
                            MIN_MATRIX_STEPS,
                        ),
                        baseline_environment=baseline_environment,
                        current_environment=current_environment,
                    )
                    for system in EXPECTED_SYSTEM_ROUTES
                }
                for implementation in ("baseline", "current")
            },
        },
        "sources": sources,
        "mdin_sha256": mdin_hashes,
        "cases": [asdict(case) | {"name": case.name} for case in cases],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def query_idle_gpu(nvidia_smi: str) -> GpuIdleSample:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=utilization.gpu,clocks.current.sm,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise MatrixRunError(
            f"nvidia-smi failed ({result.returncode}): {result.stderr.strip()}"
        )
    lines = result.stdout.splitlines()
    if not lines:
        raise MatrixRunError("nvidia-smi returned no GPU rows")
    first_line = lines[0]
    fields = [field.strip() for field in first_line.split(",")]
    if len(fields) != 3:
        raise MatrixRunError(f"unexpected nvidia-smi output: {first_line!r}")
    try:
        sample = GpuIdleSample(*(float(field) for field in fields))
    except ValueError as error:
        raise MatrixRunError(
            f"non-numeric nvidia-smi output: {first_line!r}"
        ) from error
    if not all(math.isfinite(value) for value in asdict(sample).values()):
        raise MatrixRunError(f"non-finite nvidia-smi output: {first_line!r}")
    return sample


def parse_timings(mdinfo_path: Path) -> tuple[float, float, float]:
    values = {}
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    patterns = {
        "force_s": re.compile(
            rf"\|\s*Calculate_Force\s*\|\s*{number}\s+(seconds|minutes)\s*$"
        ),
        "wall_s": re.compile(
            rf"^\s*Core Run Wall Time:\s*{number}\s+(seconds|minutes)"
            r"(?:\s+\([^)]*\))?\s*$"
        ),
        "speed_ns_day": re.compile(
            rf"^\s*Core Run Speed:\s*{number}\s+ns/day\s*$"
        ),
    }
    for line in mdinfo_path.read_text(encoding="utf-8").splitlines():
        for field, pattern in patterns.items():
            match = pattern.search(line)
            if match is not None:
                value = float(match.group(1))
                if field in {"force_s", "wall_s"} and match.group(2) == "minutes":
                    value *= 60.0
                values[field] = value
    missing = sorted(set(patterns) - set(values))
    if missing:
        raise MatrixRunError(
            f"{mdinfo_path}: missing timing fields: {', '.join(missing)}"
        )
    result = (
        values["force_s"],
        values["wall_s"],
        values["speed_ns_day"],
    )
    if not all(math.isfinite(value) and value > 0.0 for value in result):
        raise MatrixRunError(f"{mdinfo_path}: invalid timing values {result}")
    return result


def require_finite_mdout(mdout_path: Path) -> None:
    text = mdout_path.read_text(encoding="utf-8")
    if re.search(
        r"(?i)(?<![A-Za-z])[-+]?(?:nan(?:\(ind\))?|inf)(?![A-Za-z])",
        text,
    ):
        raise MatrixRunError(f"{mdout_path}: non-finite simulation output")


def infer_lj_mode(system: str, stderr: str) -> str:
    has_ab_warning = DNA_AB_WARNING in stderr
    if system == "dna_cou":
        if not has_ab_warning:
            raise MatrixRunError(
                "DNA_COU did not report the required packed-AB fallback"
            )
        return "packed-ab"
    if has_ab_warning:
        raise MatrixRunError(f"{system} unexpectedly selected packed-AB")
    return "comb"


def _write_row(
    path: Path, header: tuple[str, ...], row: dict[str, object]
) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=header,
            delimiter="\t",
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_case(
    case: MatrixCase,
    case_dir: Path,
    *,
    binary: Path,
    environment: dict[str, str],
    timeout: int,
    nvidia_smi: str,
    max_idle_sm: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        **asdict(case),
        "valid": 0,
        "lj_mode": EXPECTED_SYSTEM_ROUTES[case.system],
        "force_s": "nan",
        "wall_s": "nan",
        "speed_ns_day": "nan",
        "pre_idle_sm_pct": "nan",
        "pre_idle_pclk_mhz": "nan",
        "post_idle_sm_pct": "nan",
        "post_idle_pclk_mhz": "nan",
        "returncode": -1,
        "reason": "",
        "case_dir": str(case_dir),
    }
    try:
        pre = query_idle_gpu(nvidia_smi)
    except (MatrixRunError, OSError, subprocess.SubprocessError) as error:
        row["reason"] = f"pre-run GPU telemetry failed: {error}"
        return row
    row["pre_idle_sm_pct"] = pre.sm_pct
    row["pre_idle_pclk_mhz"] = pre.pclk_mhz
    if pre.sm_pct > max_idle_sm:
        row["reason"] = (
            f"pre-run idle SM {pre.sm_pct:.1f}% exceeds {max_idle_sm:.1f}%"
        )
        return row

    process_env = build_process_environment(environment)
    try:
        with (case_dir / "run.stdout").open("w", encoding="utf-8") as stdout:
            with (case_dir / "run.stderr").open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(
                    [str(binary), "-mdin", "mdin.spg.toml"],
                    cwd=case_dir,
                    env=process_env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
        row["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        row["returncode"] = -2
        row["reason"] = f"SPONGE exceeded timeout={timeout}s"
    time.sleep(2.0)
    try:
        post = query_idle_gpu(nvidia_smi)
    except (MatrixRunError, OSError, subprocess.SubprocessError) as error:
        row["reason"] = f"post-run GPU telemetry failed: {error}"
        return row
    row["post_idle_sm_pct"] = post.sm_pct
    row["post_idle_pclk_mhz"] = post.pclk_mhz
    if row["returncode"] == -2:
        return row
    if result.returncode != 0:
        row["reason"] = f"SPONGE exited with code {result.returncode}"
        return row
    if post.sm_pct > max_idle_sm:
        row["reason"] = (
            f"post-run idle SM {post.sm_pct:.1f}% exceeds {max_idle_sm:.1f}%"
        )
        return row

    try:
        force_s, wall_s, speed = parse_timings(case_dir / "mdinfo.txt")
        require_finite_mdout(case_dir / "mdout.txt")
        stderr_text = (case_dir / "run.stderr").read_text(encoding="utf-8")
        route = infer_lj_mode(case.system, stderr_text)
    except (MatrixRunError, OSError) as error:
        row["reason"] = str(error)
        return row

    row.update(
        {
            "valid": 1,
            "lj_mode": route,
            "force_s": force_s,
            "wall_s": wall_s,
            "speed_ns_day": speed,
            "reason": "ok",
        }
    )
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", required=True, type=Path)
    parser.add_argument("--current-bin", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--baseline-environment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=MIN_MATRIX_RUNS)
    parser.add_argument("--steps", type=int, default=MIN_MATRIX_STEPS)
    parser.add_argument("--timeout", type=int, default=7_200)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--max-idle-sm", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        current_environment = load_environment(args.environment)
        baseline_environment = load_baseline_environment(
            args.baseline_environment
        )
        cases = build_case_plan(args.runs, args.steps)
        binaries = {
            "baseline": args.baseline_bin,
            "current": args.current_bin,
        }
        for label, binary in binaries.items():
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise MatrixRunError(
                    f"{label} binary is missing or not executable: {binary}"
                )
        if args.timeout <= 0:
            raise MatrixRunError("timeout must be positive")
        if (
            not math.isfinite(args.max_idle_sm)
            or not 0.0 <= args.max_idle_sm <= 5.0
        ):
            raise MatrixRunError(
                "max-idle-sm must be finite and no greater than 5%"
            )
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise MatrixRunError(
                f"output root must be absent or empty: {args.output_root}"
            )
        args.output_root.mkdir(parents=True, exist_ok=True)
        staged = {
            case.name: stage_case(case, args.output_root) for case in cases
        }
        write_manifest(
            args.output_root,
            binaries=binaries,
            baseline_environment=baseline_environment,
            current_environment=current_environment,
            cases=cases,
            staged=staged,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                f"clustered-LJ matrix dry-run: staged {len(cases)} cases in "
                f"{args.output_root}"
            )
            return 0

        for case in cases:
            row = run_case(
                case,
                staged[case.name],
                binary=binaries[case.implementation],
                environment=resolve_case_environment(
                    case,
                    baseline_environment=baseline_environment,
                    current_environment=current_environment,
                ),
                timeout=args.timeout,
                nvidia_smi=args.nvidia_smi,
                max_idle_sm=args.max_idle_sm,
            )
            _write_row(args.output_root / "attempts.tsv", ATTEMPT_HEADER, row)
            _write_row(args.output_root / "matrix.tsv", MATRIX_HEADER, row)
            if not bool(row["valid"]):
                raise MatrixRunError(
                    f"{case.name}: invalid row recorded: {row['reason']}"
                )
    except (
        json.JSONDecodeError,
        MatrixRunError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"clustered-LJ matrix: FAIL\n{error}", file=sys.stderr)
        return 1

    print(
        f"clustered-LJ matrix: PASS ({len(cases)} valid rows in "
        f"{args.output_root / 'matrix.tsv'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
