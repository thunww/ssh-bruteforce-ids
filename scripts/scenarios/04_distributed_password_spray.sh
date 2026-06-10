#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_hydra
start_detector

PASS_FILE="${RUN_DIR}/passwords_spray.txt"
make_password_file "$PASS_FILE" "${PASSWORD_COUNT:-20}"
DISTRIBUTED_HOSTS_FILE="${DISTRIBUTED_HOSTS_FILE:-}"

log "Scenario 04: distributed / password spraying"
log "Question: If each source IP sends only a few attempts, does per-IP detection miss the attack?"

if [[ -n "$DISTRIBUTED_HOSTS_FILE" && -f "$DISTRIBUTED_HOSTS_FILE" ]]; then
  log "Using distributed workers from $DISTRIBUTED_HOSTS_FILE"
  while read -r worker; do
    [[ -z "$worker" || "$worker" =~ ^# ]] && continue
    log "Launching worker $worker"
    ssh "$worker" \
      "for i in \$(seq 1 '${PASSWORD_COUNT:-20}'); do p=\$(printf 'ids_wrong_password_%04d' \"\$i\"); hydra -l '$TARGET_USER' -p \"\$p\" -s '$TARGET_PORT' -t 1 -w '${HYDRA_WAIT:-5}' 'ssh://${TARGET_HOST}' || true; sleep '${SPRAY_GAP_SEC:-6}'; done" \
      >> "${RUN_DIR}/distributed_workers.log" 2>&1 &
  done < "$DISTRIBUTED_HOSTS_FILE"
  wait || true
else
  log "No DISTRIBUTED_HOSTS_FILE provided. Running local spray simulation from one host."
  log "Note: this is not true distributed traffic; use multiple workers for final evidence."
  attempt=0
  while read -r password; do
    attempt=$((attempt + 1))
    hydra \
      -l "$TARGET_USER" \
      -p "$password" \
      -s "$TARGET_PORT" \
      -t 1 \
      -w "${HYDRA_WAIT:-5}" \
      "ssh://${TARGET_HOST}" >> "${RUN_DIR}/hydra_local_spray.log" 2>&1 || true
    sleep "${SPRAY_GAP_SEC:-6}"
  done < "$PASS_FILE"
fi

sleep "$((IDS_POLL_SEC + 10))"
log "Expected result: true distributed spray may avoid BLOCK because current state is per Src IP."
