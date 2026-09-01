import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.performance.clustered_lj.run_migration_matrix import (
    GpuIdleSample,
)
from benchmarks.performance.clustered_lj.run_replay_matrix import (
    DEFAULT_SPONGE_LJ_MODE,
    MatrixRunError,
    ReplayCase,
    build_replay_plan,
    load_replay_spec,
    main,
    parse_replay_output,
    run_replay,
)


def make_spec(tmp_path):
    snapshots = {}
    for system in ("wat160k", "wat600k", "dna_cou"):
        snapshots[system] = {}
        for output_mode in ("force-only", "full"):
            path = tmp_path / f"{system}-{output_mode}.bin"
            path.write_bytes(f"{system}-{output_mode}".encode())
            snapshots[system][output_mode] = str(path)
    spec = {
        "binaries": {
            "baseline": sys.executable,
            "current": sys.executable,
        },
        "snapshots": snapshots,
    }
    path = tmp_path / "replay-spec.json"
    path.write_text(json.dumps(spec))
    return path


def make_v2_spec(tmp_path):
    implementations = {}
    for implementation in ("baseline", "current"):
        snapshots = {}
        for system in ("wat160k", "wat600k", "dna_cou"):
            snapshots[system] = {}
            for output_mode in ("force-only", "full"):
                snapshot = (
                    tmp_path / f"{implementation}-{system}-{output_mode}.bin"
                )
                snapshot.write_bytes(
                    f"{implementation}-{system}-{output_mode}".encode()
                )
                snapshots[system][output_mode] = str(snapshot)
        implementations[implementation] = {
            "binary": sys.executable,
            "snapshots": snapshots,
            "environment": {
                "CUDA_DEVICE_MAX_CONNECTIONS": (
                    "1" if implementation == "baseline" else "2"
                )
            },
            "arguments": ["--exact-imask-radius-scale", "1.0"],
        }
    spec_path = tmp_path / "replay-spec-v2.json"
    spec_path.write_text(json.dumps({"implementations": implementations}))
    return spec_path


def kernel_line(
    *,
    output_mode="force-only",
    lj_mode="packed-ab",
    work_parts=4,
    contiguous=1,
    iters=2000,
    sanity="ok",
    avg_ms=0.08,
):
    return (
        "kernel=sponge_production_gmxpacked snapshot=test "
        f"avg_ms={avg_ms} iters={iters} variant=split "
        f"output_mode={output_mode} lj_mode={lj_mode} "
        f"sci_work_parts={work_parts} "
        f"contiguous_sci_work={contiguous} sanity={sanity}"
    )


def test_replay_plan_has_both_modes_for_all_systems_and_binaries():
    plan = build_replay_plan()

    assert len(plan) == 36
    assert len({case.name for case in plan}) == 36
    assert {
        (
            case.cycle,
            case.implementation,
            case.system,
            case.output_mode,
        )
        for case in plan
    } == {
        (cycle, implementation, system, output_mode)
        for cycle in range(1, 4)
        for implementation in ("baseline", "current")
        for system in ("wat160k", "wat600k", "dna_cou")
        for output_mode in ("force-only", "full")
    }
    assert plan[0].implementation == "baseline"
    assert plan[1].implementation == "current"
    assert plan[12].cycle == 2
    assert plan[12].implementation == "current"
    assert plan[13].implementation == "baseline"


def test_replay_plan_cannot_weaken_minimum_cycles():
    with pytest.raises(MatrixRunError, match="runs must be at least 3"):
        build_replay_plan(2)


def test_parser_requires_exact_route_layout_and_strict_match():
    force = parse_replay_output(
        kernel_line(),
        expected_system="dna_cou",
        expected_output_mode="force-only",
        expected_iters=2000,
    )
    assert force["avg_ms"] == 0.08
    assert force["matched"] == "NA"

    full_text = (
        "gmxpacked_fulloutput_reference matched=1 tolerance=2.000e-05\n"
        + kernel_line(
            output_mode="full",
            work_parts=4,
            contiguous=0,
            avg_ms=0.12,
        )
    )
    full = parse_replay_output(
        full_text,
        expected_system="dna_cou",
        expected_output_mode="full",
        expected_iters=2000,
    )
    assert full["matched"] == "1"
    assert full["sci_work_parts"] == 4

    with pytest.raises(MatrixRunError, match="strict full replay"):
        parse_replay_output(
            kernel_line(
                output_mode="full",
                work_parts=4,
                contiguous=0,
            ),
            expected_system="dna_cou",
            expected_output_mode="full",
            expected_iters=2000,
        )
    with pytest.raises(MatrixRunError, match="expected 'packed-ab'"):
        parse_replay_output(
            kernel_line(lj_mode="comb"),
            expected_system="dna_cou",
            expected_output_mode="force-only",
            expected_iters=2000,
        )
    with pytest.raises(MatrixRunError, match="2 production rows"):
        parse_replay_output(
            kernel_line() + "\n" + kernel_line(),
            expected_system="dna_cou",
            expected_output_mode="force-only",
            expected_iters=2000,
        )
    with pytest.raises(MatrixRunError, match="2 strict reference rows"):
        parse_replay_output(
            "gmxpacked_fulloutput_reference matched=0\n"
            "gmxpacked_fulloutput_reference matched=1\n"
            + kernel_line(
                output_mode="full",
                work_parts=4,
                contiguous=0,
            ),
            expected_system="dna_cou",
            expected_output_mode="full",
            expected_iters=2000,
        )


def test_spec_requires_exact_three_system_two_mode_shape(tmp_path):
    spec_path = make_spec(tmp_path)
    raw = json.loads(spec_path.read_text())
    del raw["snapshots"]["dna_cou"]["full"]
    spec_path.write_text(json.dumps(raw))

    with pytest.raises(MatrixRunError, match="force-only and full"):
        load_replay_spec(spec_path)


def test_v2_spec_resolves_implementation_specific_inputs(tmp_path):
    spec_path = make_v2_spec(tmp_path)
    implementations = load_replay_spec(spec_path)

    assert set(implementations) == {"baseline", "current"}
    assert (
        implementations["baseline"].sponge_lj_modes["dna_cou"]["force-only"]
        == DEFAULT_SPONGE_LJ_MODE
    )
    assert (
        implementations["current"].sponge_lj_modes["dna_cou"]["force-only"]
        == DEFAULT_SPONGE_LJ_MODE
    )
    assert (
        implementations["baseline"].snapshots["dna_cou"]["full"]
        != implementations["current"].snapshots["dna_cou"]["full"]
    )
    assert (
        implementations["baseline"].environment["CUDA_DEVICE_MAX_CONNECTIONS"]
        == "1"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["implementations"]["current"]["environment"].update(
                {"SPONGE_CLUSTERED_EXPERIMENT_PROBE": "1"}
            ),
            "may not contain probe key",
        ),
        (
            lambda raw: raw["implementations"]["baseline"]["arguments"].append(
                "--iters=1"
            ),
            "may not override --iters",
        ),
        (
            lambda raw: raw["implementations"]["current"].update(
                {
                    "sponge_lj_modes": {
                        "wat160k": {
                            "force-only": (
                                "production-gmxpacked-sorted-force-sci-split3"
                            )
                        }
                    }
                }
            ),
            "is not an accepted production route",
        ),
    ],
)
def test_v2_spec_rejects_unverifiable_overrides(tmp_path, mutate, message):
    spec_path = make_v2_spec(tmp_path)
    raw = json.loads(spec_path.read_text())
    mutate(raw)
    spec_path.write_text(json.dumps(raw))

    with pytest.raises(MatrixRunError, match=message):
        load_replay_spec(spec_path)


def test_contention_records_invalid_without_launch(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.bin"
    snapshot.write_bytes(b"snapshot")
    case = ReplayCase("current", "dna_cou", "force-only")
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.query_idle_gpu",
        lambda _command: GpuIdleSample(26.0, 240.0, 32.0),
    )

    row = run_replay(
        case,
        binary=tmp_path / "must-not-run",
        snapshot=snapshot,
        output_root=tmp_path,
        warmup=200,
        iters=2000,
        timeout=1,
        nvidia_smi="nvidia-smi",
        max_idle_sm=5.0,
    )

    assert row["valid"] == 0
    assert "idle SM 26.0%" in row["reason"]
    assert not (tmp_path / f"{case.name}.stdout").exists()


def test_timeout_bytes_are_logged_and_return_invalid(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.bin"
    snapshot.write_bytes(b"snapshot")
    case = ReplayCase("current", "dna_cou", "force-only")
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.query_idle_gpu",
        lambda _command: GpuIdleSample(0.0, 2700.0, 30.0),
    )
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.time.sleep",
        lambda _seconds: None,
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd="microbench",
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.subprocess.run",
        timeout,
    )

    row = run_replay(
        case,
        binary=tmp_path / "fake",
        snapshot=snapshot,
        output_root=tmp_path,
        warmup=200,
        iters=2000,
        timeout=1,
        nvidia_smi="nvidia-smi",
        max_idle_sm=5.0,
    )

    assert row["valid"] == 0
    assert row["returncode"] == -2
    assert "exceeded timeout=1s" in row["reason"]
    assert (tmp_path / f"{case.name}.stdout").read_text() == "partial stdout"
    assert (tmp_path / f"{case.name}.stderr").read_text() == "partial stderr"


def test_replay_uses_resolved_mode_arguments_and_environment(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot.bin"
    snapshot.write_bytes(b"snapshot")
    case = ReplayCase("baseline", "dna_cou", "force-only")
    invocations = []
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.query_idle_gpu",
        lambda _command: GpuIdleSample(0.0, 2700.0, 30.0),
    )
    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.time.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setenv("SPONGE_CLUSTERED_EXPERIMENT_PROBE", "1")

    def capture(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(
            stdout=kernel_line(),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(
        "benchmarks.performance.clustered_lj.run_replay_matrix.subprocess.run",
        capture,
    )

    row = run_replay(
        case,
        binary=tmp_path / "baseline-microbench",
        snapshot=snapshot,
        sponge_lj_mode=DEFAULT_SPONGE_LJ_MODE,
        environment={"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        arguments=("--exact-imask-radius-scale", "1.0"),
        output_root=tmp_path,
        warmup=200,
        iters=2000,
        timeout=1,
        nvidia_smi="nvidia-smi",
        max_idle_sm=5.0,
    )

    assert row["valid"] == 1
    command, kwargs = invocations[0]
    assert command[command.index("--sponge-lj-mode") + 1] == (
        DEFAULT_SPONGE_LJ_MODE
    )
    assert command[-2:] == ["--exact-imask-radius-scale", "1.0"]
    assert kwargs["env"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "SPONGE_CLUSTERED_EXPERIMENT_PROBE" not in kwargs["env"]


def test_replay_dry_run_hashes_all_inputs_without_gpu(tmp_path):
    spec_path = make_spec(tmp_path)
    output_root = tmp_path / "output"

    assert (
        main(
            [
                "--spec",
                str(spec_path),
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["schema"] == 2
    assert manifest["mode"] == "dry-run"
    assert len(manifest["plan"]) == 36
    assert manifest["iters"] == 2000
    assert manifest["runs"] == 3
    assert set(manifest["implementations"]) == {"baseline", "current"}
    assert manifest["process_environment_policy"]["clear_prefixes"] == [
        "SPONGE_"
    ]
    assert (
        manifest["implementations"]["baseline"]["snapshots"]["dna_cou"]["full"][
            "sha256"
        ]
        == manifest["implementations"]["current"]["snapshots"]["dna_cou"][
            "full"
        ]["sha256"]
    )
    assert not (output_root / "replays.tsv").exists()


def test_v2_dry_run_records_resolved_provenance(tmp_path):
    spec_path = make_v2_spec(tmp_path)
    output_root = tmp_path / "output"

    assert (
        main(
            [
                "--spec",
                str(spec_path),
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((output_root / "manifest.json").read_text())
    baseline_dna_force = next(
        row
        for row in manifest["plan"]
        if row["implementation"] == "baseline"
        and row["system"] == "dna_cou"
        and row["output_mode"] == "force-only"
    )
    current_dna_force = next(
        row
        for row in manifest["plan"]
        if row["implementation"] == "current"
        and row["system"] == "dna_cou"
        and row["output_mode"] == "force-only"
    )
    assert baseline_dna_force["sponge_lj_mode"] == DEFAULT_SPONGE_LJ_MODE
    assert current_dna_force["sponge_lj_mode"] == DEFAULT_SPONGE_LJ_MODE
    assert (
        baseline_dna_force["snapshot"]["sha256"]
        != current_dna_force["snapshot"]["sha256"]
    )
    assert (
        baseline_dna_force["environment"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    )
    assert "--exact-imask-radius-scale" in baseline_dna_force["command"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--iters", "1999"], "iters must be at least 2000"),
        (["--runs", "2"], "runs must be at least 3"),
        (
            ["--max-idle-sm", "5.1"],
            "max-idle-sm must be finite and no greater than 5%",
        ),
    ],
)
def test_replay_cli_cannot_weaken_protocol(
    tmp_path, capsys, arguments, message
):
    spec_path = make_spec(tmp_path)
    output_root = tmp_path / "output"

    assert (
        main(
            [
                "--spec",
                str(spec_path),
                "--output-root",
                str(output_root),
                *arguments,
            ]
        )
        == 1
    )
    assert message in capsys.readouterr().err
