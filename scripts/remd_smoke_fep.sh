#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/media/yuh/BCDC9249DC91FDB8/Software/SPONGE/SPONGE"
BUILD_DIR="${ROOT_DIR}/build-dev-cuda13"
MANAGER_BIN="${BUILD_DIR}/SPONGE_MANAGER"
FEP_ROOT="/media/yuh/BCDC9249DC91FDB8/Data/FEP_test_for_REMD"
STATE_IDS="0,1"
LAMBDA_LIST="0.0,0.333333"
CONFIG_FILE="$(mktemp /tmp/fep_manager_config.XXXXXX.toml)"
trap 'rm -f "${CONFIG_FILE}"' EXIT

if [[ ! -x "${MANAGER_BIN}" ]]; then
  echo "manager executable not found: ${MANAGER_BIN}" >&2
  exit 1
fi

cat > "${CONFIG_FILE}" <<EOF
[manager]
block_steps = 1
epochs = 1
emit_output = false
log_path = "${FEP_ROOT}/manager_exchange.log"

[exchange]
enabled = true
mode = "hremd"
start_round = 0

[worker_defaults]
launch = "child_process"
args = [
  "-mdin", "${FEP_ROOT}/step2_mdin.txt",
  "-workspace", ".",
  "-default_in_file_prefix", "TMP",
  "-step_limit", "1",
  "-write_information_interval", "1",
  "-dont_check_input", "1",
]

[[schedules]]
schedule_id = 0
label = "fep_state_0"
working_directory = "${FEP_ROOT}/0"

[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = 0
lambda_lj = 0.0
default_out_file_prefix = "manager_smoke"

[schedules.worker]
name = "worker_0"

[[schedules]]
schedule_id = 1
label = "fep_state_1"
working_directory = "${FEP_ROOT}/1"

[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = 1
lambda_lj = 0.333333
default_out_file_prefix = "manager_smoke"

[schedules.worker]
name = "worker_1"
EOF

run_case() {
  local name="$1"
  shift
  local logfile
  logfile="$(mktemp "/tmp/${name}.XXXXXX.log")"
  echo "== ${name} =="
  "${MANAGER_BIN}" "$@" >"${logfile}"
  tail -n 8 "${logfile}"
  echo "logfile=${logfile}"
  echo
}

run_case \
  "manager_no_exchange" \
  --fep-root "${FEP_ROOT}" \
  --state-ids "${STATE_IDS}" \
  --lambda-lj-list "${LAMBDA_LIST}" \
  --block-steps 1 \
  --emit-output 0

run_case \
  "manager_config_hremd" \
  --config "${CONFIG_FILE}"

run_case \
  "tremd_child" \
  --fep-root "${FEP_ROOT}" \
  --state-ids "${STATE_IDS}" \
  --lambda-lj-list "${LAMBDA_LIST}" \
  --thermo-temperatures 300,600 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode tremd \
  --exchange-round 0

run_case \
  "hremd_child" \
  --fep-root "${FEP_ROOT}" \
  --state-ids "${STATE_IDS}" \
  --lambda-lj-list "${LAMBDA_LIST}" \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode hremd \
  --exchange-round 0

run_case \
  "htremd_child" \
  --fep-root "${FEP_ROOT}" \
  --state-ids "${STATE_IDS}" \
  --lambda-lj-list "${LAMBDA_LIST}" \
  --thermo-temperatures 300,600 \
  --block-steps 1 \
  --epochs 1 \
  --emit-output 0 \
  --worker-launch child_process \
  --remd-mode htremd \
  --exchange-round 0
