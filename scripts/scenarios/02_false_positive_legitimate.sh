#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

LEGIT_USER="${LEGIT_USER:-${SUDO_USER:-${USER:-$TARGET_USER}}}"
PASS_FILE="${RUN_DIR}/passwords_false_positive.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-8}"

log "Scenario 02: false positive / legitimate mistakes"
log "Question: Does the IDS block normal-looking failed login mistakes?"
log "This intentionally stays low-volume. Default: 8 attempts, 10s gap."
log "Using legitimate local user: ${LEGIT_USER}"

attempt=0
while read -r password; do
  attempt=$((attempt + 1))
  log "Attempt ${attempt}: one failed SSH login"
  hydra \
    -l "$LEGIT_USER" \
    -p "$password" \
    -s "$TARGET_PORT" \
    -t 1 \
    -w "${HYDRA_WAIT:-5}" \
    "ssh://${TARGET_HOST}" >> "${RUN_DIR}/hydra_false_positive.log" 2>&1 || true
  sleep "${ATTEMPT_GAP_SEC:-10}"
done < "$PASS_FILE"

sleep "$((IDS_POLL_SEC + 5))"
log "Expected result: ideally no BLOCK. ALERT may be acceptable only if risk remains low and no DROP rule is added."
