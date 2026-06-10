#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

PASS_FILE="${RUN_DIR}/passwords_baseline.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-160}"

log "Scenario 01: baseline fast brute-force"
log "Question: Does the IDS detect and block a normal fast SSH brute-force?"
log "Target: ssh://${TARGET_HOST}:${TARGET_PORT}, user=${TARGET_USER}"

hydra \
  -l "$TARGET_USER" \
  -P "$PASS_FILE" \
  -s "$TARGET_PORT" \
  -t "${HYDRA_THREADS:-4}" \
  -w "${HYDRA_WAIT:-5}" \
  "ssh://${TARGET_HOST}" 2>&1 | tee "${RUN_DIR}/hydra_baseline.log" || true

sleep "$((IDS_POLL_SEC + IDS_WINDOW_SEC / 4))"
log "Expected result: at least one ALERT, then BLOCK for the attacker IP."
