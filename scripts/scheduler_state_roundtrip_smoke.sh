#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE"
SPONGE_BIN="${ROOT_DIR}/build-dev-cuda13/SPONGE"
FEP_ROOT="/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD"

if [[ ! -x "${SPONGE_BIN}" ]]; then
  echo "SPONGE executable not found: ${SPONGE_BIN}" >&2
  exit 1
fi

python3 - <<'PY'
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT_DIR = Path("/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE")
SPONGE_BIN = ROOT_DIR / "build-dev-cuda13/SPONGE"
FEP_ROOT = Path("/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD")

COMMON_ARGS = [
    "-mdin", str(FEP_ROOT / "step2_mdin.txt"),
    "-workspace", ".",
    "-default_in_file_prefix", "TMP",
    "-default_out_file_prefix", "scheduler_roundtrip",
    "-lambda_lj", "0.000000",
    "-step_limit", "2",
    "-write_information_interval", "1",
    "-dont_check_input", "1",
]


def write_bool(f, value):
    f.write(struct.pack("<?", bool(value)))


def read_bool(f):
    return struct.unpack("<?", f.read(1))[0]


def write_int(f, value):
    f.write(struct.pack("<i", int(value)))


def read_int(f):
    return struct.unpack("<i", f.read(4))[0]


def write_u64(f, value):
    f.write(struct.pack("<Q", int(value)))


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def write_float(f, value):
    f.write(struct.pack("<f", float(value)))


def read_float(f):
    return struct.unpack("<f", f.read(4))[0]


def write_double(f, value):
    f.write(struct.pack("<d", float(value)))


def read_double(f):
    return struct.unpack("<d", f.read(8))[0]


def write_string(f, value):
    data = value.encode("utf-8")
    write_u64(f, len(data))
    f.write(data)


def read_string(f):
    size = read_u64(f)
    return f.read(size).decode("utf-8")


def write_vec_atoms(f, atoms):
    write_u64(f, len(atoms))
    for atom in atoms:
        write_float(f, atom[0])
        write_float(f, atom[1])
        write_float(f, atom[2])


def read_vec_atoms(f):
    size = read_u64(f)
    return [(read_float(f), read_float(f), read_float(f)) for _ in range(size)]


def write_vec_floats(f, values):
    write_u64(f, len(values))
    for value in values:
        write_float(f, value)


def read_vec_floats(f):
    size = read_u64(f)
    return [read_float(f) for _ in range(size)]


def write_vec_bytes(f, values):
    write_u64(f, len(values))
    for value in values:
        f.write(struct.pack("<B", int(value)))


def read_vec_bytes(f):
    size = read_u64(f)
    return list(f.read(size))


def empty_runtime_state():
    return {
        "atom_count": 0,
        "step": 0,
        "step_limit": 0,
        "start_time_ps": 0.0,
        "current_time_ps": 0.0,
        "box_length": [0.0, 0.0, 0.0],
        "box_angle": [0.0, 0.0, 0.0],
        "coordinates": [],
        "velocities": [],
        "local_accelerations": [],
        "nhc_coordinates": [],
        "nhc_velocities": [],
        "settle_last_pair_ab": [],
        "settle_last_triangle_ba": [],
        "settle_last_triangle_ca": [],
        "shake_last_pair_dr": [],
        "pressure_barostat_g": [0.0] * 6,
        "pressure_barostat_v0": 0.0,
        "pressure_barostat_rng_state": "",
        "pressure_barostat_distribution_state": "",
        "mc_barostat_total_count": [0, 0, 0],
        "mc_barostat_accept_count": [0, 0, 0],
        "mc_barostat_accept_rate": [0.0, 0.0, 0.0],
        "mc_barostat_delta_box_length_max": [0.0, 0.0, 0.0],
        "mc_barostat_rng_state": "",
        "middle_langevin_rng_state": [],
        "andersen_rng_state": [],
        "bussi_rng_state": "",
        "bussi_distribution_state": "",
        "has_local_accelerations": False,
        "has_nhc_state": False,
        "has_settle_state": False,
        "has_shake_state": False,
        "has_pressure_barostat_state": False,
        "has_mc_barostat_state": False,
        "has_middle_langevin_rng_state": False,
        "has_andersen_rng_state": False,
        "has_bussi_rng_state": False,
        "valid": False,
    }


def write_runtime_state(f, state):
    write_int(f, state["atom_count"])
    write_int(f, state["step"])
    write_int(f, state["step_limit"])
    write_double(f, state["start_time_ps"])
    write_double(f, state["current_time_ps"])
    for value in state["box_length"]:
        write_float(f, value)
    for value in state["box_angle"]:
        write_float(f, value)
    write_vec_atoms(f, state["coordinates"])
    write_vec_atoms(f, state["velocities"])
    write_vec_atoms(f, state["local_accelerations"])
    write_vec_floats(f, state["nhc_coordinates"])
    write_vec_floats(f, state["nhc_velocities"])
    write_vec_atoms(f, state["settle_last_pair_ab"])
    write_vec_atoms(f, state["settle_last_triangle_ba"])
    write_vec_atoms(f, state["settle_last_triangle_ca"])
    write_vec_atoms(f, state["shake_last_pair_dr"])
    for value in state["pressure_barostat_g"]:
        write_float(f, value)
    write_float(f, state["pressure_barostat_v0"])
    write_string(f, state["pressure_barostat_rng_state"])
    write_string(f, state["pressure_barostat_distribution_state"])
    for value in state["mc_barostat_total_count"]:
        write_int(f, value)
    for value in state["mc_barostat_accept_count"]:
        write_int(f, value)
    for value in state["mc_barostat_accept_rate"]:
        write_float(f, value)
    for value in state["mc_barostat_delta_box_length_max"]:
        write_float(f, value)
    write_string(f, state["mc_barostat_rng_state"])
    write_vec_bytes(f, state["middle_langevin_rng_state"])
    write_vec_bytes(f, state["andersen_rng_state"])
    write_string(f, state["bussi_rng_state"])
    write_string(f, state["bussi_distribution_state"])
    write_bool(f, state["has_local_accelerations"])
    write_bool(f, state["has_nhc_state"])
    write_bool(f, state["has_settle_state"])
    write_bool(f, state["has_shake_state"])
    write_bool(f, state["has_pressure_barostat_state"])
    write_bool(f, state["has_mc_barostat_state"])
    write_bool(f, state["has_middle_langevin_rng_state"])
    write_bool(f, state["has_andersen_rng_state"])
    write_bool(f, state["has_bussi_rng_state"])
    write_bool(f, state["valid"])


def read_runtime_state(f):
    state = {
        "atom_count": read_int(f),
        "step": read_int(f),
        "step_limit": read_int(f),
        "start_time_ps": read_double(f),
        "current_time_ps": read_double(f),
        "box_length": [read_float(f) for _ in range(3)],
        "box_angle": [read_float(f) for _ in range(3)],
        "coordinates": read_vec_atoms(f),
        "velocities": read_vec_atoms(f),
        "local_accelerations": read_vec_atoms(f),
        "nhc_coordinates": read_vec_floats(f),
        "nhc_velocities": read_vec_floats(f),
        "settle_last_pair_ab": read_vec_atoms(f),
        "settle_last_triangle_ba": read_vec_atoms(f),
        "settle_last_triangle_ca": read_vec_atoms(f),
        "shake_last_pair_dr": read_vec_atoms(f),
        "pressure_barostat_g": [read_float(f) for _ in range(6)],
        "pressure_barostat_v0": read_float(f),
        "pressure_barostat_rng_state": read_string(f),
        "pressure_barostat_distribution_state": read_string(f),
        "mc_barostat_total_count": [read_int(f) for _ in range(3)],
        "mc_barostat_accept_count": [read_int(f) for _ in range(3)],
        "mc_barostat_accept_rate": [read_float(f) for _ in range(3)],
        "mc_barostat_delta_box_length_max": [read_float(f) for _ in range(3)],
        "mc_barostat_rng_state": read_string(f),
        "middle_langevin_rng_state": read_vec_bytes(f),
        "andersen_rng_state": read_vec_bytes(f),
        "bussi_rng_state": read_string(f),
        "bussi_distribution_state": read_string(f),
        "has_local_accelerations": read_bool(f),
        "has_nhc_state": read_bool(f),
        "has_settle_state": read_bool(f),
        "has_shake_state": read_bool(f),
        "has_pressure_barostat_state": read_bool(f),
        "has_mc_barostat_state": read_bool(f),
        "has_middle_langevin_rng_state": read_bool(f),
        "has_andersen_rng_state": read_bool(f),
        "has_bussi_rng_state": read_bool(f),
        "valid": read_bool(f),
    }
    return state


def write_request(path, *, steps, emit_output, probe_only, runtime_state):
    with open(path, "wb") as f:
        write_int(f, steps)
        write_bool(f, emit_output)
        write_bool(f, probe_only)
        write_bool(f, runtime_state is not None and runtime_state["valid"])
        write_runtime_state(f, runtime_state if runtime_state else empty_runtime_state())


def read_response(path):
    with open(path, "rb") as f:
        snapshot = {
            "next_step": read_int(f),
            "last_completed_step": read_int(f),
            "step_limit": read_int(f),
            "current_time_ps": read_double(f),
            "dt_ps": read_double(f),
            "temperature": read_float(f),
            "target_temperature": read_float(f),
            "pressure": read_float(f),
            "target_pressure": read_float(f),
            "total_potential": read_float(f),
            "effective_potential": read_float(f),
            "box_length": [read_float(f) for _ in range(3)],
            "initialized": read_bool(f),
            "finished": read_bool(f),
        }
        observable = {
            "step": read_int(f),
            "time_ps": read_double(f),
            "total_potential": read_float(f),
            "effective_potential": read_float(f),
            "temperature": read_float(f),
            "target_temperature": read_float(f),
            "pressure": read_float(f),
            "target_pressure": read_float(f),
            "volume": read_float(f),
        }
        runtime_state = read_runtime_state(f)
        finished = read_bool(f)
    return {
        "snapshot": snapshot,
        "observable": observable,
        "runtime_state": runtime_state,
        "finished": finished,
    }


def run_worker(case_dir, *, steps, runtime_state):
    request_path = case_dir / f"request_{steps}_{'import' if runtime_state else 'fresh'}.bin"
    response_path = case_dir / f"response_{steps}_{'import' if runtime_state else 'fresh'}.bin"
    write_request(
        request_path,
        steps=steps,
        emit_output=False,
        probe_only=False,
        runtime_state=runtime_state,
    )
    cmd = [
        str(SPONGE_BIN),
        *COMMON_ARGS,
        "--worker-request", str(request_path),
        "--worker-response", str(response_path),
    ]
    proc = subprocess.run(cmd, cwd=case_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker invocation failed in {case_dir} with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return read_response(response_path)


def compare_float(label, actual, expected, tol):
    diff = abs(actual - expected)
    if diff > tol:
        raise AssertionError(
            f"{label} mismatch: actual={actual} expected={expected} diff={diff} tol={tol}"
        )


def compare_state(lhs, rhs):
    if lhs["atom_count"] != rhs["atom_count"]:
        raise AssertionError("atom_count mismatch")
    if lhs["step"] != rhs["step"]:
        raise AssertionError(f"step mismatch: {lhs['step']} vs {rhs['step']}")
    compare_float("current_time_ps", lhs["current_time_ps"], rhs["current_time_ps"], 1e-9)
    for key in ("start_time_ps",):
        compare_float(key, lhs[key], rhs[key], 1e-9)
    for idx, (a, b) in enumerate(zip(lhs["box_length"], rhs["box_length"])):
        compare_float(f"box_length[{idx}]", a, b, 1e-6)
    for idx, (a, b) in enumerate(zip(lhs["coordinates"], rhs["coordinates"])):
        for comp, (va, vb) in zip("xyz", zip(a, b)):
            compare_float(f"coordinate[{idx}].{comp}", va, vb, 5e-4)
    for idx, (a, b) in enumerate(zip(lhs["velocities"], rhs["velocities"])):
        for comp, (va, vb) in zip("xyz", zip(a, b)):
            compare_float(f"velocity[{idx}].{comp}", va, vb, 5e-4)


def compare_observable(lhs, rhs):
    compare_float("observable.step", lhs["step"], rhs["step"], 0.0)
    compare_float("observable.time_ps", lhs["time_ps"], rhs["time_ps"], 1e-9)
    compare_float("observable.total_potential", lhs["total_potential"], rhs["total_potential"], 1e-3)
    compare_float("observable.temperature", lhs["temperature"], rhs["temperature"], 1e-3)
    compare_float("observable.volume", lhs["volume"], rhs["volume"], 1e-3)


work_root = Path(tempfile.mkdtemp(prefix="scheduler_roundtrip_"))
try:
    ref_dir = work_root / "ref"
    seg_dir = work_root / "seg"
    shutil.copytree(FEP_ROOT / "0", ref_dir)
    shutil.copytree(FEP_ROOT / "0", seg_dir)

    direct = run_worker(ref_dir, steps=2, runtime_state=None)
    mid = run_worker(seg_dir, steps=1, runtime_state=None)
    resumed = run_worker(seg_dir, steps=1, runtime_state=mid["runtime_state"])

    compare_state(direct["runtime_state"], resumed["runtime_state"])
    compare_observable(direct["observable"], resumed["observable"])

    print("scheduler runtime-state roundtrip matched direct execution")
    print(
        f"step={direct['runtime_state']['step']} "
        f"time_ps={direct['runtime_state']['current_time_ps']} "
        f"potential={direct['observable']['total_potential']} "
        f"temperature={direct['observable']['temperature']}"
    )
finally:
    shutil.rmtree(work_root, ignore_errors=True)
PY
