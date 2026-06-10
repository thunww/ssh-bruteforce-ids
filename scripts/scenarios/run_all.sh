#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCENARIO_DIR/02_false_positive_legitimate.sh"
bash "$SCENARIO_DIR/01_baseline_fast_bruteforce.sh"
bash "$SCENARIO_DIR/03_evasion_low_and_slow.sh"
bash "$SCENARIO_DIR/04_distributed_password_spray.sh"
DURATION_SEC="${DURATION_SEC:-3600}" bash "$SCENARIO_DIR/05_stress_stability.sh"
bash "$SCENARIO_DIR/06_recovery_block_lifecycle.sh"
