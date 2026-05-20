#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPONGE_BIN="${ROOT_DIR}/build-dev-cuda13/SPONGE"
SINKMETA_STATIC_CASE="${ROOT_DIR}/benchmarks/performance/sinkmeta/statics/dna_cou_sinkmeta"
SITS_STATIC_CASE="${ROOT_DIR}/benchmarks/performance/sits/statics/ala2_sits"

if [[ ! -x "${SPONGE_BIN}" ]]; then
  echo "SPONGE executable not found: ${SPONGE_BIN}" >&2
  exit 1
fi

WORK_ROOT="$(mktemp -d /tmp/remd_probe_guard.XXXXXX)"
trap 'rm -rf "${WORK_ROOT}"' EXIT
SINKMETA_CASE_DIR="${WORK_ROOT}/dna_cou_sinkmeta"
SITS_CASE_DIR="${WORK_ROOT}/ala2_sits"
cp -R "${SINKMETA_STATIC_CASE}" "${SINKMETA_CASE_DIR}"
cp -R "${SITS_STATIC_CASE}" "${SITS_CASE_DIR}"

cat > "${SINKMETA_CASE_DIR}/cv.txt" <<'EOF'
ML
{
    vatom_type = center_of_mass
    atom_in_file = ligand.list
}
cx
{
    CV_type = position_x
    atom = ML
}
cy
{
    CV_type = position_y
    atom = ML
}
cz
{
    CV_type = position_z
    atom = ML
}
meta
{
    Ndim = 3
    CV = cx cy cz
    CV_minimal = 20.0 20.0 25.0
    CV_maximum = 45.0 45.0 70.0
    CV_period = 0 0 0
    CV_grid = 50 50 90
    CV_sigma = 0.600000 0.600000 0.600000
    height = 0.200000
    potential_update_interval = 500
    welltemp_factor = 20
    sink = 1
    scatter_in_file = 14d5.txt
    kde = 1
    dip = 1.000000
    convmeta = 1
}
print
{
    CV = cx cy cz
}
EOF

cat > "${SINKMETA_CASE_DIR}/mdin.spg.toml" <<'EOF'
md_name = "performance DNA_COU sinkmeta"
mode = "nvt"
dt = 0.002
step_limit = 1
write_information_interval = 1
write_mdout_interval = 1
write_trajectory_interval = 1
write_restart_file_interval = 1
thermostat = "middle_langevin"
target_temperature = 300.0
default_in_file_prefix = "2m2c"
coordinate_in_file = "Pmin_coordinate.txt"
cv_in_file = "cv.txt"
mdout = "mdout.txt"
mdinfo = "mdinfo.txt"
box = "mdbox.txt"
crd = "mdcrd.dat"
rst = "restart"
print_zeroth_frame = 1
constrain_mode = "SHAKE"
restrain_atom_id = "restrain_dnaH.txt"
restrain_refcoord_scaling = "all"
restrain_single_weight = 10.0
dont_check_input = 1
EOF

cat > "${SITS_CASE_DIR}/cv.txt" <<'EOF'
print
{
    CV = phi psi
}
phi
{
    CV_type = dihedral
    atom = 4 6 8 14
}
psi
{
    CV_type = dihedral
    atom = 6 8 14 16
}
EOF

cat > "${SITS_CASE_DIR}/mdin.spg.toml" <<'EOF'
md_name = "benchmark alanine_dipeptide_tip3p_water SITS"
mode = "nvt"
step_limit = 1
dt = 0.002
cutoff = 8.0
thermostat = "middle_langevin"
thermostat_tau = 1.0
thermostat_seed = 2026
target_temperature = 300.0
default_in_file_prefix = "ALA"
print_zeroth_frame = 1
write_mdout_interval = 1
write_information_interval = 1
SITS_mode = "iteration"
SITS_atom_numbers = 22
SITS_k_numbers = 4
SITS_T_low = 273.0
SITS_T_high = 650.0
SITS_record_interval = 1
SITS_update_interval = 20
SITS_nk_fix = 0
SITS_pe_a = 1.0
SITS_pe_b = 34.23
constrain_mode = "SHAKE"
cv_in_file = "cv.txt"
EOF

REQ="${WORK_ROOT}/probe_request.bin"

python3 - <<'PY' "${REQ}"
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])

def w_bool(f, value):
    f.write(struct.pack("<?", bool(value)))

def w_int(f, value):
    f.write(struct.pack("<i", int(value)))

def w_u64(f, value):
    f.write(struct.pack("<Q", int(value)))

def w_float(f, value):
    f.write(struct.pack("<f", float(value)))

def w_double(f, value):
    f.write(struct.pack("<d", float(value)))

def w_string(f, value):
    data = value.encode("utf-8")
    w_u64(f, len(data))
    f.write(data)

def w_empty_vec(f):
    w_u64(f, 0)

with path.open("wb") as f:
    w_int(f, 0)
    w_bool(f, False)
    w_bool(f, True)
    w_bool(f, False)
    w_int(f, 0)
    w_int(f, 0)
    w_int(f, 0)
    w_double(f, 0.0)
    w_double(f, 0.0)
    for _ in range(3):
        w_float(f, 0.0)
    for _ in range(3):
        w_float(f, 0.0)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    w_empty_vec(f)
    for _ in range(6):
        w_float(f, 0.0)
    w_float(f, 0.0)
    w_string(f, "")
    w_string(f, "")
    for _ in range(3):
        w_int(f, 0)
    for _ in range(3):
        w_int(f, 0)
    for _ in range(3):
        w_float(f, 0.0)
    for _ in range(3):
        w_float(f, 0.0)
    w_string(f, "")
    w_empty_vec(f)
    w_empty_vec(f)
    w_string(f, "")
    w_string(f, "")
    for _ in range(10):
        w_bool(f, False)
PY

run_probe_guard() {
  local case_dir="$1"
  local expected_message="$2"
  local label="$3"
  local response_path="${case_dir}/probe_response.bin"
  local log_path="${case_dir}/probe_guard.log"

  set +e
  (
    cd "${case_dir}" && \
    "${SPONGE_BIN}" -mdin mdin.spg.toml \
      --worker-request "${REQ}" \
      --worker-response "${response_path}"
  ) > "${log_path}" 2>&1
  local status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "Expected ${label} foreign-state probe to fail closed, but it succeeded." >&2
    cat "${log_path}" >&2
    exit 1
  fi

  if [[ -e "${response_path}" ]]; then
    echo "Expected no worker response file on ${label} probe failure." >&2
    cat "${log_path}" >&2
    exit 1
  fi

  if ! grep -q "${expected_message}" "${log_path}"; then
    echo "Expected ${label} probe guard message not found." >&2
    cat "${log_path}" >&2
    exit 1
  fi

  echo "${label} probe guard triggered as expected"
  tail -n 8 "${log_path}"
}

run_probe_guard \
  "${SINKMETA_CASE_DIR}" \
  "foreign-state observable probing is not safe when sink metadynamics bias is enabled" \
  "sinkmeta"

run_probe_guard \
  "${SITS_CASE_DIR}" \
  "foreign-state observable probing is not safe when SITS/enhanced-sampling bias is enabled" \
  "SITS"
