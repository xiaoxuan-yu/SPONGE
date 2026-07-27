import sys

import pytest

from benchmarks.utils import Runner


def run_python(case_dir, source, **kwargs):
    return Runner.run_sponge(
        case_dir,
        mdin_name=None,
        sponge_cmd=sys.executable,
        extra_args=("-c", source),
        **kwargs,
    )


def test_run_sponge_requires_outputs_from_current_run(tmp_path):
    stale = tmp_path / "frc.dat"
    stale.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="fresh output"):
        run_python(
            tmp_path,
            "print('Stop Wall Time: now')",
            fresh_outputs=("frc.dat",),
            require_completion=True,
        )

    assert not stale.exists()


def test_run_sponge_accepts_fresh_nonempty_outputs(tmp_path):
    output = run_python(
        tmp_path,
        (
            "from pathlib import Path; "
            "Path('mdout.txt').write_text('frame'); "
            "Path('frc.dat').write_bytes(b'force'); "
            "print('Stop Wall Time: now')"
        ),
        fresh_outputs=("mdout.txt", "frc.dat"),
        require_completion=True,
    )

    assert "Stop Wall Time:" in output
    assert (tmp_path / "mdout.txt").read_text() == "frame"
    assert (tmp_path / "frc.dat").read_bytes() == b"force"


def test_run_sponge_rejects_zero_status_error_marker(tmp_path):
    with pytest.raises(RuntimeError, match="zero return code"):
        run_python(tmp_path, "print('spongeErrorUnclassified')")


def test_run_sponge_rejects_fresh_output_outside_case(tmp_path):
    with pytest.raises(ValueError, match="escapes the case directory"):
        run_python(tmp_path, "pass", fresh_outputs=("../frc.dat",))
