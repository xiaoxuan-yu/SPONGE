#!/usr/bin/env python3

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a synthetic clustered custom-Morse workload."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--side", type=int, default=20)
    parser.add_argument("--spacing", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--pair-oracle",
        action="store_true",
        help=(
            "also enable zero-coefficient clustered LJ so SPONGE can dump "
            "the shared SCI/CJ payload for the canonical pair oracle"
        ),
    )
    return parser.parse_args()


def require_new_case(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    (output / "system").mkdir(parents=True, exist_ok=True)


def write_coordinate(path, side, spacing):
    atom_numbers = side**3
    box_length = side * spacing
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{atom_numbers} 0.0\n")
        for ix in range(side):
            for iy in range(side):
                for iz in range(side):
                    handle.write(
                        f"{(ix + 0.5) * spacing:.12f} "
                        f"{(iy + 0.5) * spacing:.12f} "
                        f"{(iz + 0.5) * spacing:.12f}\n"
                    )
        handle.write(
            f"{box_length:.12f} {box_length:.12f} {box_length:.12f}\n"
        )
        handle.write("90.0 90.0 90.0\n")


def write_mass(path, atom_numbers):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{atom_numbers}\n")
        handle.writelines("12.0\n" for _ in range(atom_numbers))


def write_pairwise_parameters(output, atom_numbers):
    (output / "pairwise_force.txt").write_text(
        "\n".join(
            [
                "[[[ morse_force ]]]",
                "[[ parameters ]]",
                "float D0_ij, float alpha_ij, float r0_ij",
                "[[ potential ]]",
                (
                    "E = D0_ij * (expf(-2.0f * alpha_ij * "
                    "(r_ij - r0_ij)) - 2.0f * expf(-alpha_ij * "
                    "(r_ij - r0_ij)));"
                ),
                "[[ with_ele ]]",
                "false",
                "[[ end ]]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with (output / "morse-force_in_file.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(f"{atom_numbers} 1\n")
        handle.write("0.25\n1.35\n2.10\n")
        handle.writelines("0\n" for _ in range(atom_numbers))


def write_zero_lj_parameters(output, atom_numbers):
    with (output / "zero_lj.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{atom_numbers} 1\n")
        handle.write("0.0\n0.0\n")
        handle.writelines("0\n" for _ in range(atom_numbers))


def write_mdin(output, steps, pair_oracle):
    lines = [
        "# Synthetic custom-pairwise clustered microbenchmark",
        'mode = "nve"',
        "dt = 0",
        f"step_limit = {steps}",
        "cutoff = 8.0",
        "skin = 2.0",
        'default_in_file_prefix = "system/test"',
        "print_pressure = 0",
        "print_zeroth_frame = 0",
        f"write_trajectory_interval = {steps + 1}",
        f"write_mdout_interval = {steps + 1}",
        'pairwise_force_in_file = "pairwise_force.txt"',
        'morse_force_in_file = "morse-force_in_file.txt"',
    ]
    if pair_oracle:
        lines.extend(
            [
                'LJ_in_file = "zero_lj.txt"',
                "",
                "[LJ]",
                'direct_kernel = "clustered"',
                "clustered_rebuild_skin = 2.0",
            ]
        )
    (output / "mdin.spg.toml").write_text(
        "\n".join([*lines, ""]),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.side <= 0 or args.spacing <= 0.0 or args.steps <= 0:
        raise ValueError("side, spacing and steps must be positive")
    require_new_case(args.output)
    atom_numbers = args.side**3
    write_coordinate(
        args.output / "system" / "test_coordinate.txt",
        args.side,
        args.spacing,
    )
    write_mass(args.output / "system" / "test_mass.txt", atom_numbers)
    write_pairwise_parameters(args.output, atom_numbers)
    if args.pair_oracle:
        write_zero_lj_parameters(args.output, atom_numbers)
    write_mdin(args.output, args.steps, args.pair_oracle)
    print(
        f"prepared {atom_numbers} atoms in {args.output} "
        f"(side={args.side}, spacing={args.spacing}, steps={args.steps})"
    )


if __name__ == "__main__":
    main()
