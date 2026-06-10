# Scenario Testing Guide

This guide runs the real-time SSH IDS against repeatable lab scenarios.
Run these scripts on the Ubuntu VM where the detector and SSH server are installed.

## Safety

- Use only a lab VM or a machine you own.
- The IDS adds `iptables -A INPUT -s <ip> -j DROP` rules when it blocks.
- Every scenario calls `cleanup_blocks.sh` logic before and after the run.
- Cleanup removes only DROP rules for test IPs, not the whole firewall.
- If you attack from another machine, set `TEST_IPS` to that attacker IP so cleanup can remove the right rule.

## Requirements

```bash
sudo apt-get update
sudo apt-get install -y hydra openssh-server iptables
cd /home/user/ssh-bruteforce-ids
source .venv/bin/activate
```

If `python3 scripts/09_realtime_detector.py` uses project imports, run scripts from the project root.
The scenario scripts set `PYTHONPATH` automatically.

## Common variables

```bash
export TARGET_HOST=172.29.139.144
export TARGET_PORT=22
export TARGET_USER=invalid_ids_user
export LEGIT_USER=than
export TEST_IPS="172.29.139.144"
```

`TARGET_USER` is the fake/nonexistent account used for attack scenarios.
`LEGIT_USER` is the real local account used by the false-positive scenario.
When testing inside WSL, prefer the WSL interface IP from `ip a` instead of
`127.0.0.1` so the source IP shown in SSH logs is closer to a real network test.

For worker/attacker machines launched remotely by the detector VM, set `TEST_IPS`
to those source IPs so cleanup removes the right DROP rules on the detector VM:

```bash
export TARGET_HOST=192.168.1.105
export TARGET_USER=invalid_ids_user
export TEST_IPS="192.168.1.50 192.168.1.51"
```

Useful detector tuning variables:

```bash
export IDS_WINDOW_SEC=60
export IDS_POLL_SEC=5
export IDS_WORKERS=4
export IDS_MONITOR_INTERVAL=30
export IDS_LOW_SLOW_ALERT_WINDOW_SEC=300
export IDS_LOW_SLOW_BLOCK_WINDOW_SEC=900
export IDS_LOW_SLOW_ALERT_COUNT=12
export IDS_LOW_SLOW_BLOCK_COUNT=24
```

The low-and-slow defaults mean:

```text
12 failed logins within 5 minutes  -> ALERT, reason=LOW_AND_SLOW_5M
24 failed logins within 15 minutes -> BLOCK, reason=LOW_AND_SLOW_15M
```

When the detector starts, verify these lines:

```text
Low-and-slow thresholds: failed_300s>=12 ALERT, failed_900s>=24 BLOCK
Consumer started (window_sec=60, retained_window_sec=900)
```

## Cleanup only

```bash
bash scripts/scenarios/cleanup_blocks.sh
```

Use this before every manual retest, or if a previous run was interrupted.

## Scenario 01: Baseline fast brute-force

Question: can the IDS detect and block a normal fast SSH brute-force?

```bash
bash scripts/scenarios/01_baseline_fast_bruteforce.sh
```

Expected evidence:

- `actions` contains `ALERT` and then `BLOCK` or `BLOCKED`.
- `iptables -L INPUT -n --line-numbers` shows a DROP rule during the run.
- Cleanup removes that DROP rule before exit.

## Scenario 02: False positive

Question: does the IDS block normal failed login mistakes?

```bash
PASSWORD_COUNT=8 ATTEMPT_GAP_SEC=10 \
bash scripts/scenarios/02_false_positive_legitimate.sh
```

Expected evidence:

- Best result: no `BLOCK`.
- A few `ALERT` rows can be discussed if risk is low and no DROP rule remains.
- Any `BLOCK` here is a serious false-positive finding.

## Scenario 03: Evasion low-and-slow

Question: can a slow/noisy brute-force avoid detection?

```bash
PASSWORD_COUNT=36 MIN_GAP_SEC=8 MAX_GAP_SEC=25 NOISE_EVERY=6 \
bash scripts/scenarios/03_evasion_low_and_slow.sh
```

Expected evidence:

- Before the low-and-slow improvement, `total=0` was a valid finding and showed an evasion weakness.
- After the improvement, expect at least `ALERT` with `reason=LOW_AND_SLOW_5M`.
- If the attack lasts long enough to reach `failed_15m >= 24`, expect `BLOCK` with `reason=LOW_AND_SLOW_15M`.
- Report `failed_5m`, `failed_15m`, `reason`, detection time, and max risk score.

Useful result commands:

```bash
cat outputs/alerts.jsonl
python3 scripts/scenarios/summarize_alerts.py outputs/alerts.jsonl
sudo iptables -L INPUT -n --line-numbers
journalctl -u ssh --since "20 minutes ago" --no-pager | grep "Failed password" | tail -30
```

Expected alert example:

```json
{"ip": "172.29.139.144", "username": "than", "event_count": 3, "failed_5m": 12, "failed_15m": 12, "action": "ALERT", "reason": "LOW_AND_SLOW_5M"}
```

## Scenario 04: Distributed/password spraying

Question: does per-IP detection miss a distributed attack?

Local simulation:

```bash
PASSWORD_COUNT=20 SPRAY_GAP_SEC=6 \
bash scripts/scenarios/04_distributed_password_spray.sh
```

True distributed mode requires worker machines reachable by SSH. Put one SSH target per line:

```text
kali1
kali2
kali3
```

Then run:

```bash
export DISTRIBUTED_HOSTS_FILE=/tmp/ids_workers.txt
bash scripts/scenarios/04_distributed_password_spray.sh
```

Note: local simulation is not true distributed traffic because the source IP is still one host.

## Scenario 05: Stress and stability

Question: does the detector stay stable under long-running load?

Short smoke test:

```bash
DURATION_SEC=300 HYDRA_THREADS=8 IDS_WORKERS=4 \
bash scripts/scenarios/05_stress_stability.sh
```

Full report run:

```bash
DURATION_SEC=3600 HYDRA_THREADS=8 IDS_WORKERS=4 \
bash scripts/scenarios/05_stress_stability.sh
```

Expected evidence:

- Detector process stays alive.
- `detector.log` shows CPU/RAM without steady growth.
- `event_q` and `infer_q` do not keep growing.
- Duplicate alerts are explainable and bounded.

## Scenario 06: Recovery and block lifecycle

Question: does block cleanup work reliably between runs?

```bash
bash scripts/scenarios/06_recovery_block_lifecycle.sh
```

Expected evidence:

- DROP rule appears after a BLOCK.
- The script removes the DROP rule.
- Final `iptables` listing has no DROP rule for `TEST_IPS`.

## Run all

Recommended order:

```bash
bash scripts/scenarios/run_all.sh
```

For a faster full pass:

```bash
DURATION_SEC=300 bash scripts/scenarios/run_all.sh
```

## Output files

Each run creates:

```text
outputs/scenario-runs/<timestamp>/
  alerts.jsonl
  detector.log
  summary.txt
  hydra_*.log
```

Use `summary.txt` for the report. It includes:

- total alert rows
- action counts
- min/max risk score
- min/max model probability
- max event count
- action counts by source IP

For improved low-and-slow tests, also preserve raw `alerts.jsonl` rows because they contain:

- `reason`
- `username`
- `failed_5m`
- `failed_15m`
