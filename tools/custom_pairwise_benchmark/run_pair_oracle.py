#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


ORACLE_RESULT = re.compile(
    r"canonical_pair_oracle metadata_ready=(?P<metadata>[01]) "
    r"matched=(?P<matched>[01]) payload=(?P<payload>\d+) "
    r"oracle=(?P<oracle>\d+) duplicates=(?P<duplicates>\d+) "
    r"missing=(?P<missing>\d+) extra=(?P<extra>\d+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a clustered custom-pairwise fixture and validate its shared "
            "SCI/CJ payload with the independent canonical pair oracle."
        )
    )
    parser.add_argument("case", type=Path)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--microbench", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def run_checked(command, cwd, env, timeout):
    result = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{' '.join(str(value) for value in command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def validate_case(case):
    required = [
        case / "mdin.spg.toml",
        case / "zero_lj.txt",
        case / "pairwise_force.txt",
        case / "morse-force_in_file.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "pair-oracle case is incomplete; generate it with "
            "prepare_case.py --pair-oracle. Missing: " + ", ".join(missing)
        )
    mdin = (case / "mdin.spg.toml").read_text(encoding="utf-8")
    if 'LJ_in_file = "zero_lj.txt"' not in mdin or (
        'direct_kernel = "clustered"' not in mdin
    ):
        raise RuntimeError(
            "pair-oracle case must enable zero-coefficient clustered LJ"
        )


def main():
    args = parse_args()
    case = args.case.resolve()
    producer = args.producer.resolve()
    microbench = args.microbench.resolve()
    validate_case(case)
    if not producer.is_file() or not microbench.is_file():
        raise RuntimeError(
            "snapshot producer and NBNXM_MICROBENCH must be existing files"
        )

    with tempfile.TemporaryDirectory(
        prefix="sponge-custom-pair-oracle-"
    ) as temp_dir:
        dump_prefix = Path(temp_dir) / "custom_pairwise"
        env = os.environ.copy()
        run_checked(
            [
                producer,
                "--snapshot-prefix",
                dump_prefix,
                "mdin.spg.toml",
            ],
            case,
            env,
            args.timeout,
        )

        snapshot = Path(
            f"{dump_prefix}.sponge_gmxpacked_forceonly.bin"
        )
        if not snapshot.is_file():
            raise RuntimeError(
                "snapshot producer completed without writing snapshot: "
                f"{snapshot}"
            )
        oracle = run_checked(
            [
                microbench,
                "--kernel",
                "sponge",
                "--snapshot",
                snapshot,
                "--pair-oracle",
            ],
            case,
            env,
            args.timeout,
        )
        match = ORACLE_RESULT.search(oracle.stdout)
        if match is None:
            raise RuntimeError(
                "canonical pair oracle did not emit a parseable result\n"
                f"stdout:\n{oracle.stdout}\nstderr:\n{oracle.stderr}"
            )
        values = {key: int(value) for key, value in match.groupdict().items()}
        if (
            values["metadata"] != 1
            or values["matched"] != 1
            or values["payload"] != values["oracle"]
            or values["duplicates"] != 0
            or values["missing"] != 0
            or values["extra"] != 0
        ):
            raise RuntimeError(
                "canonical pair oracle mismatch\n"
                f"stdout:\n{oracle.stdout}\nstderr:\n{oracle.stderr}"
            )
        print(match.group(0))


if __name__ == "__main__":
    main()
