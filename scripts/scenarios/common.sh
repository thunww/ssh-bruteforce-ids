#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCENARIO_DIR}/../.." && pwd)"

TARGET_HOST="${TARGET_HOST:-127.0.0.1}"
TARGET_PORT="${TARGET_PORT:-22}"
TARGET_USER="${TARGET_USER:-invalid_ids_user}"
TARGET_IP="${TARGET_IP:-$TARGET_HOST}"

IDS_WINDOW_SEC="${IDS_WINDOW_SEC:-60}"
IDS_POLL_SEC="${IDS_POLL_SEC:-5}"
IDS_WORKERS="${IDS_WORKERS:-4}"
IDS_MONITOR_INTERVAL="${IDS_MONITOR_INTERVAL:-30}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/scenario-runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
ALERTS_PATH="${ALERTS_PATH:-${RUN_DIR}/alerts.jsonl}"
DETECTOR_LOG="${DETECTOR_LOG:-${RUN_DIR}/detector.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/summary.txt}"
PID_FILE="${PID_FILE:-${RUN_DIR}/detector.pid}"

mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUMMARY_PATH"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: missing required command: $1"
    exit 2
  fi
}

require_hydra() {
  if ! command -v hydra >/dev/null 2>&1; then
    log "ERROR: hydra is required for this scenario."
    log "Install on Ubuntu/Kali: sudo apt-get update && sudo apt-get install -y hydra"
    exit 2
  fi
}

attacker_ips() {
  if [[ -n "${TEST_IPS:-}" ]]; then
    printf '%s\n' $TEST_IPS
    return
  fi

  if [[ "$TARGET_HOST" == "127.0.0.1" || "$TARGET_HOST" == "localhost" ]]; then
    printf '127.0.0.1\n'
    return
  fi

  hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF {print; exit}'
}

cleanup_blocks() {
  local ip
  log "Cleaning IDS iptables DROP rules for test IPs"
  while read -r ip; do
    [[ -z "$ip" ]] && continue
    while sudo iptables -C INPUT -s "$ip" -j DROP >/dev/null 2>&1; do
      sudo iptables -D INPUT -s "$ip" -j DROP || true
      log "Removed DROP rule for $ip"
    done
  done < <(attacker_ips)
}

show_blocks() {
  log "Current matching INPUT DROP rules"
  sudo iptables -L INPUT -n --line-numbers | tee -a "$SUMMARY_PATH" || true
}

make_password_file() {
  local path="$1"
  local count="${2:-120}"
  : > "$path"
  for i in $(seq 1 "$count"); do
    printf 'ids_wrong_password_%04d\n' "$i" >> "$path"
  done
}

start_detector() {
  if [[ "${USE_EXISTING_DETECTOR:-0}" == "1" ]]; then
    log "Using existing detector. ALERTS_PATH must point to the detector alert file."
    return
  fi

  require_cmd python3
  sudo -v
  cleanup_blocks
  : > "$ALERTS_PATH"
  : > "$DETECTOR_LOG"

  log "Starting detector"
  log "Alerts: $ALERTS_PATH"
  (
    cd "$PROJECT_ROOT"
    IDS_ALERTS_PATH="$ALERTS_PATH" \
    IDS_WINDOW_SEC="$IDS_WINDOW_SEC" \
    IDS_POLL_SEC="$IDS_POLL_SEC" \
    IDS_WORKERS="$IDS_WORKERS" \
    IDS_MONITOR_INTERVAL="$IDS_MONITOR_INTERVAL" \
    PYTHONPATH="$PROJECT_ROOT" \
    python3 scripts/09_realtime_detector.py
  ) > "$DETECTOR_LOG" 2>&1 &

  echo "$!" > "$PID_FILE"
  log "Detector PID: $(cat "$PID_FILE")"
  sleep "$((IDS_POLL_SEC + 3))"
}

stop_detector() {
  if [[ "${USE_EXISTING_DETECTOR:-0}" == "1" ]]; then
    return
  fi
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      log "Stopping detector PID $pid"
      kill "$pid" || true
      sleep 2
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

summarize_alerts() {
  log "Summarizing alerts"
  python3 "$SCENARIO_DIR/summarize_alerts.py" "$ALERTS_PATH" | tee -a "$SUMMARY_PATH"
}

finish_scenario() {
  summarize_alerts || true
  show_blocks || true
  cleanup_blocks || true
  stop_detector || true
  log "Scenario artifacts: $RUN_DIR"
}

trap finish_scenario EXIT
