import csv

import pytest

from benchmarks.performance.clustered_lj.check_migration_gate import (
    EXPECTED_ENSEMBLES,
    EXPECTED_OUTPUT_MODES,
    EXPECTED_REPLAY_LAYOUT,
    EXPECTED_SYSTEM_ROUTES,
    GateError,
    main,
    run_gate,
)

MATRIX_FIELDS = (
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
REPLAY_FIELDS = (
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


def write_tsv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_matrix_rows():
    rows = []
    for cycle in range(1, 4):
        for implementation in ("baseline", "current"):
            for system, route in EXPECTED_SYSTEM_ROUTES.items():
                for ensemble in EXPECTED_ENSEMBLES:
                    rows.append(
                        {
                            "cycle": cycle,
                            "implementation": implementation,
                            "system": system,
                            "ensemble": ensemble,
                            "steps": 10_000,
                            "valid": 1,
                            "lj_mode": route,
                            "force_s": 10.0,
                            "wall_s": 12.0,
                            "speed_ns_day": 100.0,
                        }
                    )
    return rows


def make_replay_rows():
    rows = []
    for cycle in range(1, 4):
        for implementation in ("baseline", "current"):
            for system, route in EXPECTED_SYSTEM_ROUTES.items():
                for output_mode in EXPECTED_OUTPUT_MODES:
                    work_parts, contiguous = EXPECTED_REPLAY_LAYOUT[
                        (system, output_mode)
                    ]
                    rows.append(
                        {
                            "cycle": cycle,
                            "implementation": implementation,
                            "system": system,
                            "output_mode": output_mode,
                            "valid": 1,
                            "lj_mode": route,
                            "sci_work_parts": work_parts,
                            "contiguous_sci_work": int(contiguous),
                            "iters": 2_000,
                            "sanity": "ok",
                            "matched": (1 if output_mode == "full" else "NA"),
                            "avg_ms": 1.0,
                        }
                    )
    return rows


def run_fixture(tmp_path, matrix_rows=None, replay_rows=None):
    matrix_path = tmp_path / "matrix.tsv"
    replay_path = tmp_path / "replays.tsv"
    write_tsv(
        matrix_path,
        MATRIX_FIELDS,
        make_matrix_rows() if matrix_rows is None else matrix_rows,
    )
    write_tsv(
        replay_path,
        REPLAY_FIELDS,
        make_replay_rows() if replay_rows is None else replay_rows,
    )
    return run_gate(matrix_path, replay_path)


def test_complete_three_system_and_gate_passes(tmp_path):
    matrix_summary, replay_summary = run_fixture(tmp_path)

    assert len(matrix_summary) == 6
    assert len(replay_summary) == 6


def test_missing_dna_cannot_be_hidden_by_water(tmp_path):
    matrix_rows = [
        row for row in make_matrix_rows() if row["system"] != "dna_cou"
    ]

    with pytest.raises(GateError, match="systems must be exactly"):
        run_fixture(tmp_path, matrix_rows=matrix_rows)


def test_one_regressing_cell_fails_without_cross_system_averaging(tmp_path):
    matrix_rows = make_matrix_rows()
    for row in matrix_rows:
        if row["implementation"] != "current":
            continue
        if row["system"] == "dna_cou" and row["ensemble"] == "npt":
            row["force_s"] = 10.31
        else:
            row["force_s"] = 5.0

    with pytest.raises(
        GateError, match=r"dna_cou/npt: force regression \+3.100%"
    ):
        run_fixture(tmp_path, matrix_rows=matrix_rows)


def test_wrong_dna_parameter_route_fails(tmp_path):
    replay_rows = make_replay_rows()
    for row in replay_rows:
        if row["implementation"] == "current" and row["system"] == "dna_cou":
            row["lj_mode"] = "comb"

    with pytest.raises(GateError, match="dna_cou used lj_mode='comb'"):
        run_fixture(tmp_path, replay_rows=replay_rows)


def test_missing_full_replay_fails(tmp_path):
    replay_rows = [
        row
        for row in make_replay_rows()
        if not (
            row["implementation"] == "current"
            and row["system"] == "wat600k"
            and row["output_mode"] == "full"
        )
    ]

    with pytest.raises(
        GateError, match="wat600k/full: candidate has 0 replays"
    ):
        run_fixture(tmp_path, replay_rows=replay_rows)


def test_replay_requires_three_matched_cycles_per_cell(tmp_path):
    replay_rows = [
        row
        for row in make_replay_rows()
        if not (
            row["implementation"] == "current"
            and row["system"] == "dna_cou"
            and row["output_mode"] == "force-only"
            and row["cycle"] == 3
        )
    ]

    with pytest.raises(GateError) as error:
        run_fixture(tmp_path, replay_rows=replay_rows)

    message = str(error.value)
    assert "dna_cou/force-only: candidate has 2 replays" in message
    assert "replay cycles do not match" in message


def test_replay_gate_uses_paired_cycle_median(tmp_path):
    replay_rows = make_replay_rows()
    for row in replay_rows:
        if row["system"] != "dna_cou" or row["output_mode"] != "full":
            continue
        if row["cycle"] == 2:
            row["avg_ms"] = 1000.0
        elif row["implementation"] == "current":
            row["avg_ms"] = 1.1

    with pytest.raises(
        GateError, match=r"dna_cou/full: replay regression \+10.000%"
    ):
        run_fixture(tmp_path, replay_rows=replay_rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps", 9_999, "steps=9999, expected 10000"),
        ("valid", 0, "row is marked invalid"),
    ],
)
def test_invalid_matrix_rows_fail(tmp_path, field, value, message):
    matrix_rows = make_matrix_rows()
    matrix_rows[0][field] = value

    with pytest.raises(GateError, match=message):
        run_fixture(tmp_path, matrix_rows=matrix_rows)


def test_replay_layout_and_strict_full_match_are_required(tmp_path):
    replay_rows = make_replay_rows()
    for row in replay_rows:
        if (
            row["implementation"] == "current"
            and row["system"] == "dna_cou"
            and row["output_mode"] == "full"
        ):
            row["sci_work_parts"] = 1
            row["matched"] = 0

    with pytest.raises(GateError) as error:
        run_fixture(tmp_path, replay_rows=replay_rows)

    message = str(error.value)
    assert "strict full replay did not match" in message
    assert "route layout (1, False)" in message


def test_cli_returns_zero_only_for_complete_gate(tmp_path, capsys):
    matrix_path = tmp_path / "matrix.tsv"
    replay_path = tmp_path / "replays.tsv"
    write_tsv(matrix_path, MATRIX_FIELDS, make_matrix_rows())
    write_tsv(replay_path, REPLAY_FIELDS, make_replay_rows())

    assert (
        main(["--matrix", str(matrix_path), "--replays", str(replay_path)]) == 0
    )
    assert "clustered-LJ migration gate: PASS" in capsys.readouterr().out

    incomplete = [
        row for row in make_matrix_rows() if row["system"] != "dna_cou"
    ]
    write_tsv(matrix_path, MATRIX_FIELDS, incomplete)

    assert (
        main(["--matrix", str(matrix_path), "--replays", str(replay_path)]) == 1
    )
    assert "clustered-LJ migration gate: FAIL" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", ["nan", "inf", "0.031"])
def test_cli_cannot_weaken_regression_threshold(tmp_path, capsys, threshold):
    matrix_path = tmp_path / "matrix.tsv"
    replay_path = tmp_path / "replays.tsv"
    write_tsv(matrix_path, MATRIX_FIELDS, make_matrix_rows())
    write_tsv(replay_path, REPLAY_FIELDS, make_replay_rows())

    assert (
        main(
            [
                "--matrix",
                str(matrix_path),
                "--replays",
                str(replay_path),
                "--threshold",
                threshold,
            ]
        )
        == 1
    )
    assert "threshold must be finite and no greater than 0.03" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_runs": 2}, "min_runs must be at least 3"),
        ({"required_steps": 9_999}, "required_steps must be at least 10000"),
        (
            {"min_replay_iters": 1_999},
            "min_replay_iters must be at least 2000",
        ),
        (
            {"min_replay_runs": 2},
            "min_replay_runs must be at least 3",
        ),
    ],
)
def test_gate_configuration_cannot_weaken_protocol(tmp_path, kwargs, message):
    matrix_path = tmp_path / "matrix.tsv"
    replay_path = tmp_path / "replays.tsv"
    write_tsv(matrix_path, MATRIX_FIELDS, make_matrix_rows())
    write_tsv(replay_path, REPLAY_FIELDS, make_replay_rows())

    with pytest.raises(GateError, match=message):
        run_gate(matrix_path, replay_path, **kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cycle", 0, "cycle=0, expected positive ID"),
        ("force_s", "", "expected number"),
    ],
)
def test_malformed_cells_fail_cleanly(tmp_path, field, value, message):
    matrix_rows = make_matrix_rows()
    matrix_rows[0][field] = value

    with pytest.raises(GateError, match=message):
        run_fixture(tmp_path, matrix_rows=matrix_rows)
