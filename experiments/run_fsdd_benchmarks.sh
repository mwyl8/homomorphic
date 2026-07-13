#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:-/home/ec2-user/fsdd_results}"
mkdir -p "${LOG_DIR}"

run_and_log() {
  local label="$1"
  local cmd="$2"
  local logfile="${LOG_DIR}/fsdd_${label}.log"
  echo "==> Running ${label}"
  echo "Command: ${cmd}" | tee "${logfile}"
  echo "Started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${logfile}"
  ${cmd} >> "${logfile}" 2>&1
  echo "Finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${logfile}"
}

run_and_log "baseline_1024" "python3 /home/ec2-user/experiment/basetest_fsdd_1024.py"
run_and_log "aheeqv_1024" "python3 /home/ec2-user/experiment/aheeqv_fsdd_1024.py"
run_and_log "aheedb_1024" "python3 /home/ec2-user/experiment/aheedb_fsdd_1024.py"
run_and_log "fhe_1024" "python3 /home/ec2-user/experiment/fhe_fsdd_1024.py"

run_and_log "baseline_512" "python3 /home/ec2-user/experiment/basetest_fsdd_512.py"
run_and_log "aheeqv_512" "python3 /home/ec2-user/experiment/aheeqv_fsdd_512.py"
run_and_log "aheedb_512" "python3 /home/ec2-user/experiment/aheedb_fsdd_512.py"
run_and_log "fhe_512" "python3 /home/ec2-user/experiment/fhe_fsdd_512.py"

run_and_log "baseline_256" "python3 /home/ec2-user/experiment/basetest_fsdd_256.py"
run_and_log "aheeqv_256" "python3 /home/ec2-user/experiment/aheeqv_fsdd_256.py"
run_and_log "aheedb_256" "python3 /home/ec2-user/experiment/aheedb_fsdd_256.py"
run_and_log "fhe_256" "python3 /home/ec2-user/experiment/fhe_fsdd_256.py"

run_and_log "baseline_128" "python3 /home/ec2-user/experiment/basetest_fsdd_128.py"
run_and_log "aheeqv_128" "python3 /home/ec2-user/experiment/aheeqv_fsdd_128.py"
run_and_log "aheedb_128" "python3 /home/ec2-user/experiment/aheedb_fsdd_128.py"
run_and_log "fhe_128" "python3 /home/ec2-user/experiment/fhe_fsdd_128.py"
