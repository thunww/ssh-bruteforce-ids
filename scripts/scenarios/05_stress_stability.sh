#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

PASS_FILE="${RUN_DIR}/passwords_stress.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-500}"
DURATION_SEC="${DURATION_SEC:-3600}"
END_AT=$(( $(date +%s) + DURATION_SEC ))
round=0

log "Scenario 05: stress and stability"
log "Question: Does the detector stay stable under repeated SSH failures?"
log "Duration=${DURATION_SEC}s, threads=${HYDRA_THREADS:-8}, workers=${IDS_WORKERS}"

while (( $(date +%s) < END_AT )); do
  round=$((round + 1))
  log "Stress round ${round}"
  hydra \
    -l "$TARGET_USER" \
    -P "$PASS_FILE" \
    -s "$TARGET_PORT" \
    -t "${HYDRA_THREADS:-8}" \
    -w "${HYDRA_WAIT:-5}" \
    "ssh://${TARGET_HOST}" >> "${RUN_DIR}/hydra_stress.log" 2>&1 || true
  sleep "${ROUND_GAP_SEC:-5}"
done

sleep "$((IDS_POLL_SEC + 10))"
log "Expected result: detector process alive, CPU/RAM stable, queues do not keep growing, no uncontrolled duplicate alerts."
