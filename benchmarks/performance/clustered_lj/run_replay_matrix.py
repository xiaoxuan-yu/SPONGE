#!/usr/bin/env python3
"""Produce force-only/full replay rows for the clustered-LJ migration gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from .check_migration_gate import (
        EXPECTED_OUTPUT_MODES,
        EXPECTED_REPLAY_LAYOUT,
        EXPECTED_SYSTEM_ROUTES,
        MIN_REPLAY_ITERS,
        MIN_REPLAY_RUNS,
    )
    from .run_migration_matrix import (
        MatrixRunError,
        build_process_environment,
        query_idle_gpu,
    )
except ImportError:
    from check_migration_gate import (
        EXPECTED_OUTPUT_MODES,
        EXPECTED_REPLAY_LAYOUT,
        EXPECTED_SYSTEM_ROUTES,
        MIN_REPLAY_ITERS,
        MIN_REPLAY_RUNS,
    )
    from run_migration_matrix import (
        MatrixRunError,
        build_process_environment,
        query_idle_gpu,
    )


REPLAY_HEADER = (
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
)
ATTEMPT_HEADER = REPLAY_HEADER + (
    "pre_idle_sm_pct",
    "pre_idle_pclk_mhz",
    "post_idle_sm_pct",
    "post_idle_pclk_mhz",
    "returncode",
    "reason",
    "stdout_path",
    "stderr_path",
)
IMPLEMENTATION_LABELS = ("baseline", "current")
DEFAULT_SPONGE_LJ_MODE = "production-gmxpacked"
ALLOWED_SPONGE_LJ_MODES = {
    ("wat160k", "force-only"): {DEFAULT_SPONGE_LJ_MODE},
    ("wat160k", "full"): {DEFAULT_SPONGE_LJ_MODE},
    ("wat600k", "force-only"): {DEFAULT_SPONGE_LJ_MODE},
    ("wat600k", "full"): {DEFAULT_SPONGE_LJ_MODE},
    ("dna_cou", "force-only"): {DEFAULT_SPONGE_LJ_MODE},
    ("dna_cou", "full"): {DEFAULT_SPONGE_LJ_MODE},
}
RESERVED_REPLAY_ARGUMENTS = {
    "--iters",
    "--kernel",
    "--snapshot",
    "--sponge-lj-mode",
    "--warmup",
}


@dataclass(frozen=True)
class ReplayCase:
    implementation: str
    system: str
    output_mode: str
    cycle: int = 1

    @property
    def name(self) -> str:
        return (
            f"{self.cycle:02d}_{self.system}_{self.output_mode}_"
            f"{self.implementation}"
        )


@dataclass(frozen=True)
class ReplayImplementation:
    binary: Path
    snapshots: dict[str, dict[str, Path]]
    sponge_lj_modes: dict[str, dict[str, str]]
    environment: dict[str, str]
    arguments: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_snapshots(
    raw: object, *, context: str, path: Path
) -> dict[str, dict[str, Path]]:
    if not isinstance(raw, dict) or set(raw) != set(EXPECTED_SYSTEM_ROUTES):
        raise MatrixRunError(
            f"{path}: {context} snapshots must be exactly "
            f"{sorted(EXPECTED_SYSTEM_ROUTES)}"
        )
    snapshots = {}
    for system in EXPECTED_SYSTEM_ROUTES:
        modes = raw[system]
        if not isinstance(modes, dict) or set(modes) != set(
            EXPECTED_OUTPUT_MODES
        ):
            raise MatrixRunError(
                f"{path}: {context} {system} snapshots must be force-only "
                "and full"
            )
        if not all(
            isinstance(modes[mode], str) for mode in EXPECTED_OUTPUT_MODES
        ):
            raise MatrixRunError(
                f"{path}: {context} {system} snapshot paths must be strings"
            )
        snapshots[system] = {
            mode: Path(modes[mode]) for mode in EXPECTED_OUTPUT_MODES
        }
    return snapshots


def _default_sponge_lj_modes() -> dict[str, dict[str, str]]:
    return {
        system: {
            output_mode: DEFAULT_SPONGE_LJ_MODE
            for output_mode in EXPECTED_OUTPUT_MODES
        }
        for system in EXPECTED_SYSTEM_ROUTES
    }


def _parse_sponge_lj_modes(
    raw: object, *, context: str, path: Path
) -> dict[str, dict[str, str]]:
    modes = _default_sponge_lj_modes()
    if raw is None:
        return modes
    if not isinstance(raw, dict):
        raise MatrixRunError(
            f"{path}: {context} sponge_lj_modes must be an object"
        )
    unknown_systems = sorted(set(raw) - set(EXPECTED_SYSTEM_ROUTES))
    if unknown_systems:
        raise MatrixRunError(
            f"{path}: {context} sponge_lj_modes has unknown systems: "
            + ", ".join(unknown_systems)
        )
    for system, raw_system_modes in raw.items():
        if not isinstance(raw_system_modes, dict):
            raise MatrixRunError(
                f"{path}: {context} {system} sponge_lj_modes must be an object"
            )
        unknown_modes = sorted(
            set(raw_system_modes) - set(EXPECTED_OUTPUT_MODES)
        )
        if unknown_modes:
            raise MatrixRunError(
                f"{path}: {context} {system} sponge_lj_modes has unknown "
                "output modes: " + ", ".join(unknown_modes)
            )
        for output_mode, value in raw_system_modes.items():
            if not isinstance(value, str) or not value:
                raise MatrixRunError(
                    f"{path}: {context} {system}/{output_mode} "
                    "sponge_lj_mode must be a non-empty string"
                )
            allowed = ALLOWED_SPONGE_LJ_MODES[(system, output_mode)]
            if value not in allowed:
                raise MatrixRunError(
                    f"{path}: {context} {system}/{output_mode} "
                    f"sponge_lj_mode {value!r} is not an accepted production "
                    f"route; expected one of {sorted(allowed)}"
                )
            modes[system][output_mode] = value
    return modes


def _parse_environment(
    raw: object, *, context: str, path: Path
) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MatrixRunError(f"{path}: {context} environment must be an object")
    environment = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise MatrixRunError(
                f"{path}: {context} environment entries must be scalar "
                "key/value pairs"
            )
        if "_PROBE" in key:
            raise MatrixRunError(
                f"{path}: {context} replay environment may not contain probe "
                f"key {key}; select the exact replay implementation with "
                "sponge_lj_modes"
            )
        environment[key] = str(value)
    return environment


def _parse_arguments(
    raw: object, *, context: str, path: Path
) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(argument, str) and argument for argument in raw
    ):
        raise MatrixRunError(
            f"{path}: {context} arguments must be a list of non-empty strings"
        )
    for argument in raw:
        option = argument.split("=", 1)[0]
        if option in RESERVED_REPLAY_ARGUMENTS:
            raise MatrixRunError(
                f"{path}: {context} arguments may not override {option}"
            )
    return tuple(raw)


def load_replay_spec(path: Path) -> dict[str, ReplayImplementation]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise MatrixRunError(f"{path}: replay spec must be a JSON object")

    raw_implementations = raw.get("implementations")
    if raw_implementations is None:
        raw_binaries = raw.get("binaries")
        raw_snapshots = raw.get("snapshots")
        if not isinstance(raw_binaries, dict) or raw_snapshots is None:
            raise MatrixRunError(
                f"{path}: requires legacy binaries/snapshots or "
                "implementations objects"
            )
        if set(raw_binaries) != set(IMPLEMENTATION_LABELS) or not all(
            isinstance(raw_binaries[label], str)
            for label in IMPLEMENTATION_LABELS
        ):
            raise MatrixRunError(
                f"{path}: binaries must be exactly baseline and current"
            )
        snapshots = _parse_snapshots(raw_snapshots, context="shared", path=path)
        return {
            label: ReplayImplementation(
                binary=Path(raw_binaries[label]),
                snapshots=snapshots,
                sponge_lj_modes=_default_sponge_lj_modes(),
                environment={},
                arguments=(),
            )
            for label in IMPLEMENTATION_LABELS
        }

    if "binaries" in raw or "snapshots" in raw:
        raise MatrixRunError(
            f"{path}: implementations may not be combined with legacy "
            "binaries/snapshots"
        )
    if not isinstance(raw_implementations, dict) or set(
        raw_implementations
    ) != set(IMPLEMENTATION_LABELS):
        raise MatrixRunError(
            f"{path}: implementations must be exactly baseline and current"
        )

    implementations = {}
    for label in IMPLEMENTATION_LABELS:
        implementation = raw_implementations[label]
        context = f"implementations.{label}"
        if not isinstance(implementation, dict):
            raise MatrixRunError(f"{path}: {context} must be an object")
        binary = implementation.get("binary")
        if not isinstance(binary, str):
            raise MatrixRunError(f"{path}: {context}.binary must be a string")
        implementations[label] = ReplayImplementation(
            binary=Path(binary),
            snapshots=_parse_snapshots(
                implementation.get("snapshots"), context=context, path=path
            ),
            sponge_lj_modes=_parse_sponge_lj_modes(
                implementation.get("sponge_lj_modes"),
                context=context,
                path=path,
            ),
            environment=_parse_environment(
                implementation.get("environment"),
                context=context,
                path=path,
            ),
            arguments=_parse_arguments(
                implementation.get("arguments"), context=context, path=path
            ),
        )
    return implementations


def build_replay_plan(runs: int = MIN_REPLAY_RUNS) -> list[ReplayCase]:
    if runs < MIN_REPLAY_RUNS:
        raise MatrixRunError(f"runs must be at least {MIN_REPLAY_RUNS}")

    cases = []
    for cycle in range(1, runs + 1):
        systems = (
            tuple(EXPECTED_SYSTEM_ROUTES)
            if cycle % 2 == 1
            else tuple(reversed(EXPECTED_SYSTEM_ROUTES))
        )
        output_modes = (
            EXPECTED_OUTPUT_MODES
            if cycle % 2 == 1
            else tuple(reversed(EXPECTED_OUTPUT_MODES))
        )
        implementations = (
            IMPLEMENTATION_LABELS
            if cycle % 2 == 1
            else tuple(reversed(IMPLEMENTATION_LABELS))
        )
        for system in systems:
            for output_mode in output_modes:
                for implementation in implementations:
                    cases.append(
                        ReplayCase(
                            implementation,
                            system,
                            output_mode,
                            cycle,
                        )
                    )
    return cases


def parse_replay_output(
    text: str,
    *,
    expected_system: str,
    expected_output_mode: str,
    expected_iters: int,
) -> dict[str, object]:
    kernel_rows = []
    reference_rows = []

    def parse_fields(line: str) -> dict[str, str]:
        fields = {}
        for field in line.split()[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            if key in fields:
                raise MatrixRunError(f"duplicate output field: {key}")
            fields[key] = value
        return fields

    for line in text.splitlines():
        if line.startswith("gmxpacked_fulloutput_reference "):
            reference_rows.append(parse_fields(line))
        if line.startswith("kernel=sponge_production_gmxpacked "):
            kernel_rows.append(parse_fields("production " + line))
    if len(kernel_rows) != 1:
        raise MatrixRunError(
            f"microbench output has {len(kernel_rows)} production rows, "
            "expected exactly 1"
        )
    expected_reference_rows = 1 if expected_output_mode == "full" else 0
    if len(reference_rows) != expected_reference_rows:
        raise MatrixRunError(
            f"microbench output has {len(reference_rows)} strict reference rows "
            f"for strict full replay, expected exactly {expected_reference_rows}"
        )
    kernel_fields = kernel_rows[0]
    matched = reference_rows[0].get("matched", "0") if reference_rows else "NA"

    expected_route = EXPECTED_SYSTEM_ROUTES[expected_system]
    expected_parts, expected_contiguous = EXPECTED_REPLAY_LAYOUT[
        (expected_system, expected_output_mode)
    ]
    required = {
        "output_mode": expected_output_mode,
        "lj_mode": expected_route,
        "sci_work_parts": str(expected_parts),
        "contiguous_sci_work": "1" if expected_contiguous else "0",
        "iters": str(expected_iters),
        "sanity": "ok",
    }
    mismatches = [
        f"{key}={kernel_fields.get(key)!r}, expected {value!r}"
        for key, value in required.items()
        if kernel_fields.get(key) != value
    ]
    if mismatches:
        raise MatrixRunError("; ".join(mismatches))
    if expected_output_mode == "full" and matched != "1":
        raise MatrixRunError("strict full replay did not report matched=1")
    try:
        avg_ms = float(kernel_fields["avg_ms"])
    except (KeyError, ValueError) as error:
        raise MatrixRunError("microbench output has invalid avg_ms") from error
    if not math.isfinite(avg_ms) or avg_ms <= 0.0:
        raise MatrixRunError(f"microbench output has invalid avg_ms={avg_ms}")
    return {
        "lj_mode": expected_route,
        "sci_work_parts": expected_parts,
        "contiguous_sci_work": int(expected_contiguous),
        "iters": expected_iters,
        "sanity": "ok",
        "matched": matched,
        "avg_ms": avg_ms,
    }


def _timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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


def build_replay_command(
    *,
    binary: Path,
    snapshot: Path,
    sponge_lj_mode: str,
    warmup: int,
    iters: int,
    arguments: tuple[str, ...] = (),
) -> list[str]:
    return [
        str(binary),
        "--kernel",
        "sponge",
        "--snapshot",
        str(snapshot),
        "--sponge-lj-mode",
        sponge_lj_mode,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        *arguments,
    ]


def run_replay(
    case: ReplayCase,
    *,
    binary: Path,
    snapshot: Path,
    sponge_lj_mode: str = DEFAULT_SPONGE_LJ_MODE,
    environment: Optional[dict[str, str]] = None,
    arguments: tuple[str, ...] = (),
    output_root: Path,
    warmup: int,
    iters: int,
    timeout: int,
    nvidia_smi: str,
    max_idle_sm: float,
) -> dict[str, object]:
    stdout_path = output_root / f"{case.name}.stdout"
    stderr_path = output_root / f"{case.name}.stderr"
    expected_parts, expected_contiguous = EXPECTED_REPLAY_LAYOUT[
        (case.system, case.output_mode)
    ]
    row: dict[str, object] = {
        "cycle": case.cycle,
        **asdict(case),
        "valid": 0,
        "lj_mode": EXPECTED_SYSTEM_ROUTES[case.system],
        "sci_work_parts": expected_parts,
        "contiguous_sci_work": int(expected_contiguous),
        "iters": iters,
        "sanity": "bad",
        "matched": "NA",
        "avg_ms": "nan",
        "pre_idle_sm_pct": "nan",
        "pre_idle_pclk_mhz": "nan",
        "post_idle_sm_pct": "nan",
        "post_idle_pclk_mhz": "nan",
        "returncode": -1,
        "reason": "",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
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

    command = build_replay_command(
        binary=binary,
        snapshot=snapshot,
        sponge_lj_mode=sponge_lj_mode,
        warmup=warmup,
        iters=iters,
        arguments=arguments,
    )
    process_environment = build_process_environment(environment or {})
    try:
        result = subprocess.run(
            command,
            env=process_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        row["returncode"] = result.returncode
    except subprocess.TimeoutExpired as error:
        stdout_path.write_text(_timeout_text(error.stdout), encoding="utf-8")
        stderr_path.write_text(_timeout_text(error.stderr), encoding="utf-8")
        row["returncode"] = -2
        row["reason"] = f"microbench exceeded timeout={timeout}s"

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
        row["reason"] = f"microbench exited with code {result.returncode}"
        return row
    if post.sm_pct > max_idle_sm:
        row["reason"] = (
            f"post-run idle SM {post.sm_pct:.1f}% exceeds {max_idle_sm:.1f}%"
        )
        return row
    try:
        parsed = parse_replay_output(
            result.stdout,
            expected_system=case.system,
            expected_output_mode=case.output_mode,
            expected_iters=iters,
        )
    except MatrixRunError as error:
        row["reason"] = str(error)
        return row
    row.update(parsed)
    row["valid"] = 1
    row["reason"] = "ok"
    return row


def write_manifest(
    output_root: Path,
    *,
    spec_path: Path,
    implementations: dict[str, ReplayImplementation],
    plan: list[ReplayCase],
    warmup: int,
    iters: int,
    dry_run: bool,
) -> None:
    implementation_manifests = {}
    for label, implementation in implementations.items():
        implementation_manifests[label] = {
            "binary": {
                "path": str(implementation.binary.resolve()),
                "sha256": _sha256(implementation.binary),
            },
            "environment": implementation.environment,
            "arguments": list(implementation.arguments),
            "sponge_lj_modes": implementation.sponge_lj_modes,
            "snapshots": {
                system: {
                    output_mode: {
                        "path": str(snapshot.resolve()),
                        "sha256": _sha256(snapshot),
                    }
                    for output_mode, snapshot in modes.items()
                }
                for system, modes in implementation.snapshots.items()
            },
        }

    resolved_plan = []
    for case in plan:
        implementation = implementations[case.implementation]
        snapshot = implementation.snapshots[case.system][case.output_mode]
        sponge_lj_mode = implementation.sponge_lj_modes[case.system][
            case.output_mode
        ]
        resolved_plan.append(
            asdict(case)
            | {
                "name": case.name,
                "snapshot": {
                    "path": str(snapshot.resolve()),
                    "sha256": _sha256(snapshot),
                },
                "sponge_lj_mode": sponge_lj_mode,
                "command": build_replay_command(
                    binary=implementation.binary,
                    snapshot=snapshot,
                    sponge_lj_mode=sponge_lj_mode,
                    warmup=warmup,
                    iters=iters,
                    arguments=implementation.arguments,
                ),
                "environment": implementation.environment,
            }
        )

    manifest = {
        "schema": 2,
        "mode": "dry-run" if dry_run else "execution",
        "spec_path": str(spec_path.resolve()),
        "warmup": warmup,
        "iters": iters,
        "runs": max(case.cycle for case in plan),
        "process_environment_policy": {
            "inherit": "non-SPONGE variables",
            "clear_prefixes": ["SPONGE_"],
        },
        "binaries": {
            label: implementation_manifest["binary"]
            for label, implementation_manifest in (
                implementation_manifests.items()
            )
        },
        "implementations": implementation_manifests,
        "plan": resolved_plan,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=MIN_REPLAY_ITERS)
    parser.add_argument("--runs", type=int, default=MIN_REPLAY_RUNS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--max-idle-sm", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        implementations = load_replay_spec(args.spec)
        for label, implementation in implementations.items():
            if not implementation.binary.is_file() or not os.access(
                implementation.binary, os.X_OK
            ):
                raise MatrixRunError(
                    f"{label} binary is missing or not executable: "
                    f"{implementation.binary}"
                )
            for system, modes in implementation.snapshots.items():
                for output_mode, snapshot in modes.items():
                    if not snapshot.is_file():
                        raise MatrixRunError(
                            f"missing {label} {system}/{output_mode} "
                            f"snapshot: {snapshot}"
                        )
        if args.warmup < 0:
            raise MatrixRunError("warmup must be non-negative")
        if args.iters < MIN_REPLAY_ITERS:
            raise MatrixRunError(f"iters must be at least {MIN_REPLAY_ITERS}")
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
        plan = build_replay_plan(args.runs)
        write_manifest(
            args.output_root,
            spec_path=args.spec,
            implementations=implementations,
            plan=plan,
            warmup=args.warmup,
            iters=args.iters,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                f"clustered-LJ replay dry-run: staged {len(plan)} cells in "
                f"{args.output_root}"
            )
            return 0

        for case in plan:
            implementation = implementations[case.implementation]
            row = run_replay(
                case,
                binary=implementation.binary,
                snapshot=implementation.snapshots[case.system][
                    case.output_mode
                ],
                sponge_lj_mode=implementation.sponge_lj_modes[case.system][
                    case.output_mode
                ],
                environment=implementation.environment,
                arguments=implementation.arguments,
                output_root=args.output_root,
                warmup=args.warmup,
                iters=args.iters,
                timeout=args.timeout,
                nvidia_smi=args.nvidia_smi,
                max_idle_sm=args.max_idle_sm,
            )
            _write_row(args.output_root / "attempts.tsv", ATTEMPT_HEADER, row)
            _write_row(args.output_root / "replays.tsv", REPLAY_HEADER, row)
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
        print(f"clustered-LJ replay matrix: FAIL\n{error}", file=sys.stderr)
        return 1

    print(
        f"clustered-LJ replay matrix: PASS ({len(plan)} valid rows in "
        f"{args.output_root / 'replays.tsv'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
