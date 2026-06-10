#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

PASS_FILE="${RUN_DIR}/passwords_recovery.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-120}"

log "Scenario 06: recovery and block lifecycle"
log "Question: Does BLOCK happen, and can test IPs be cleanly unblocked between runs?"

hydra \
  -l "$TARGET_USER" \
  -P "$PASS_FILE" \
  -s "$TARGET_PORT" \
  -t "${HYDRA_THREADS:-4}" \
  -w "${HYDRA_WAIT:-5}" \
  "ssh://${TARGET_HOST}" 2>&1 | tee "${RUN_DIR}/hydra_recovery.log" || true

sleep "$((IDS_POLL_SEC + 10))"
show_blocks
cleanup_blocks
show_blocks
log "Expected result: DROP rule appears after BLOCK and is removed by cleanup_blocks."
