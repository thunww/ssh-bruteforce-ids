# SSH IDS Scenario Test Results

This document records scenario-based validation results for the real-time SSH brute-force IDS.
It is intended to be updated after each scenario run.

## Test Environment

| Item | Value |
|---|---|
| Test date | 2026-06-10 |
| Runtime | WSL / Linux |
| Target host | `172.29.139.144` |
| SSH port | `22` |
| Test user | `than` |
| Detector script | `scripts/09_realtime_detector.py` |
| Real-time model | `models/xgb_realtime_model.joblib` |
| Alert output | `outputs/alerts.jsonl` |
| Blocking mechanism | `iptables` DROP rule |

The detector was started manually with:

```bash
PYTHONPATH=/mnt/d/ssh-bruteforce-ids \
python3 scripts/09_realtime_detector.py
```

Detector startup evidence:

```text
Model loaded from models/xgb_realtime_model.joblib
Expected features: ['flow_rate_per_window', 'interarrival_mean', 'interarrival_std', 'rst_flow_ratio', 'short_flow_ratio']
=== REALTIME SSH IDS STARTED ===
Producer started (poll_sec=5)
Consumer started (window_sec=60)
Worker pool started (workers=4)
Monitor started (interval=30s)
```

SSH failed-login log evidence before scenario testing:

```text
Jun 10 21:12:33 GiaThan sshd[3062]: Failed password for than from 172.29.139.144 port 57536 ssh2
Jun 10 21:12:37 GiaThan sshd[3062]: Failed password for than from 172.29.139.144 port 57536 ssh2
Jun 10 21:12:40 GiaThan sshd[3062]: Failed password for than from 172.29.139.144 port 57536 ssh2
```

## Scenario Summary

| Scenario | Purpose | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| 1. Baseline fast brute-force | Verify detection and blocking of high-rate SSH brute-force | `ALERT` then `BLOCK` | 10 `ALERT`, 1 `BLOCK`, 87 `BLOCKED` | PASS |
| 2. False positive / legitimate mistakes | Verify legitimate low-rate failed logins are not blocked | No `BLOCK`; ideally no `ALERT` | 0 alerts, 0 blocks | PASS |
| 3. Evasion low-and-slow | Verify whether slow/noisy brute-force can evade detection | At least `ALERT`; `BLOCK` if attack persists | 69 `ALERT`, 2 `BLOCK`, 277 `BLOCKED` | PASS |
| 4. Distributed/password spraying | Verify per-IP detection weakness under distributed attempts | Analyze distributed behavior | Not run | Pending |
| 5. Stress and stability | Verify detector stability under sustained load | Stable CPU/RAM/queues | Not run | Pending |
| 6. Recovery/block lifecycle | Verify block and cleanup behavior | DROP rule appears, then cleanup removes it | Not run | Pending |

## Scenario 1: Baseline Fast Brute-force

### Objective

This scenario verifies whether the IDS can detect and block a normal high-rate SSH brute-force attack.
The attacker repeatedly tries many wrong passwords against the same real SSH user.

### Test Command

```bash
hydra -l than -P /tmp/ids_pass_baseline.txt -s 22 -t 4 -w 5 ssh://172.29.139.144
```

### Test Configuration

| Item | Value |
|---|---|
| Target user | `than` |
| Password attempts | `160` |
| Hydra threads | `4` |
| Target | `ssh://172.29.139.144:22` |

### Hydra Evidence

```text
Hydra v9.5
[DATA] max 4 tasks per 1 server, overall 4 tasks, 160 login tries (l:1/p:160), ~40 tries per task
[DATA] attacking ssh://172.29.139.144:22/
[STATUS] 64.00 tries/min, 64 tries in 00:01h, 96 to do in 00:02h, 4 active
[STATUS] 58.00 tries/min, 116 tries in 00:02h, 44 to do in 00:01h, 4 active
```

The run was interrupted manually after enough evidence had been collected, but the IDS had already generated `ALERT`, `BLOCK`, and `BLOCKED` actions.

### Alert Evidence

First low-risk alerts:

```json
{"ip": "172.29.139.144", "now": "2026-06-10 21:20:29.353712", "event_count": 20, "model_prob": 0.0027, "risk_score": 0.237, "action": "ALERT", "consecutive_suspicious": 1}
{"ip": "172.29.139.144", "now": "2026-06-10 21:20:30.355638", "event_count": 20, "model_prob": 0.0027, "risk_score": 0.237, "action": "ALERT", "consecutive_suspicious": 2}
```

Block event:

```json
{"ip": "172.29.139.144", "now": "2026-06-10 21:20:44.404857", "event_count": 44, "model_prob": 0.9973, "risk_score": 0.8762, "action": "BLOCK", "consecutive_suspicious": 11}
```

Subsequent blocked state:

```json
{"ip": "172.29.139.144", "now": "2026-06-10 21:20:45.407712", "event_count": 44, "model_prob": 0.9973, "risk_score": 0.8762, "action": "BLOCKED", "consecutive_suspicious": 11}
{"ip": "172.29.139.144", "now": "2026-06-10 21:20:49.917172", "event_count": 52, "model_prob": 0.9973, "risk_score": 0.8745, "action": "BLOCKED", "consecutive_suspicious": 11}
```

### Summary Output

```text
alerts_file=outputs/alerts.jsonl
total=98
actions={"ALERT": 10, "BLOCK": 1, "BLOCKED": 87}
bad_lines=0
risk_min=0.2370
risk_max=0.8762
model_prob_min=0.0027
model_prob_max=0.9973
max_event_count=80
by_ip={"172.29.139.144": {"ALERT": 10, "BLOCK": 1, "BLOCKED": 87}}
```

### Analysis

At the beginning of the attack, the IDS generated `ALERT` records when the event count was still relatively low.
The initial model probability was low (`0.0027`), but the combined risk score exceeded the alert threshold because the failed-login activity was already suspicious.

When the number of events increased to `44`, the model probability rose sharply to `0.9973`.
At that point, the risk score reached `0.8762`, exceeding the block threshold and producing a `BLOCK` action.
Subsequent records were marked as `BLOCKED`, confirming that the detector entered a blocked state for the attacking IP.

### Result

```text
PASS
```

The IDS successfully detected and blocked a high-rate SSH brute-force attack.

## Scenario 2: False Positive / Legitimate Mistakes

### Objective

This scenario verifies whether the IDS blocks legitimate users by mistake.
It simulates a real local user entering the wrong SSH password a small number of times with delays between attempts.

### Test Command

```bash
USE_EXISTING_DETECTOR=1 \
ALERTS_PATH=outputs/alerts.jsonl \
PASSWORD_COUNT=8 \
ATTEMPT_GAP_SEC=10 \
bash scripts/scenarios/02_false_positive_legitimate.sh
```

### Test Configuration

| Item | Value |
|---|---|
| Legitimate user | `than` |
| Failed attempts | `8` |
| Delay between attempts | `10` seconds |
| Target | `ssh://172.29.139.144:22` |

### Script Evidence

```text
[2026-06-10 21:26:39] Using existing detector. ALERTS_PATH must point to the detector alert file.
[2026-06-10 21:26:39] Scenario 02: false positive / legitimate mistakes
[2026-06-10 21:26:39] Question: Does the IDS block normal-looking failed login mistakes?
[2026-06-10 21:26:39] This intentionally stays low-volume. Default: 8 attempts, 10s gap.
[2026-06-10 21:26:39] Using legitimate local user: than
[2026-06-10 21:26:39] Attempt 1: one failed SSH login
[2026-06-10 21:27:05] Attempt 2: one failed SSH login
[2026-06-10 21:27:17] Attempt 3: one failed SSH login
[2026-06-10 21:27:31] Attempt 4: one failed SSH login
[2026-06-10 21:27:45] Attempt 5: one failed SSH login
[2026-06-10 21:28:00] Attempt 6: one failed SSH login
[2026-06-10 21:28:15] Attempt 7: one failed SSH login
[2026-06-10 21:28:30] Attempt 8: one failed SSH login
[2026-06-10 21:28:55] Expected result: ideally no BLOCK. ALERT may be acceptable only if risk remains low and no DROP rule is added.
```

### Alert Summary Evidence

```text
alerts_file=outputs/alerts.jsonl
total=0
actions={}
bad_lines=0
max_event_count=0
by_ip={}
```

### Firewall Evidence

```text
Chain INPUT (policy ACCEPT)
num  target     prot opt source               destination
```

No DROP rule was added for the test IP.

### Analysis

The detector produced no `ALERT`, `BLOCK`, or `BLOCKED` records.
This is the expected behavior for the false-positive scenario.
The `alerts.jsonl` file is empty because the detector only writes records for non-normal actions.
Since the login attempts were low-volume and spaced apart, the traffic stayed below alert and block thresholds.

The firewall state also confirms that no blocking rule was added.
Therefore, the IDS did not incorrectly block a legitimate user after a small number of failed login attempts.

### Result

```text
PASS
```

The IDS did not generate a false positive for low-rate failed logins from a legitimate user.

## Scenario 3: Evasion Low-and-Slow

### Objective

This scenario checks whether a slow and noisy brute-force pattern can evade detection.
The attack spreads failed SSH login attempts over time and inserts longer pauses after a small number of attempts.

### Test Command

```bash
USE_EXISTING_DETECTOR=1 \
ALERTS_PATH=outputs/alerts.jsonl \
PASSWORD_COUNT=36 \
MIN_GAP_SEC=8 \
MAX_GAP_SEC=25 \
NOISE_EVERY=6 \
bash scripts/scenarios/03_evasion_low_and_slow.sh
```

### Test Configuration

| Item | Value |
|---|---|
| Target user | `than` |
| Failed attempts | `36` |
| Gap between attempts | Random `8-25` seconds |
| Noise pause | Every `6` attempts |
| Low-and-slow alert threshold | `failed_5m >= 12` |
| Low-and-slow block threshold | `failed_15m >= 24` |

### Script Evidence

The scenario ran all 36 attempts and inserted a pause after every 6 attempts:

```text
[2026-06-10 21:58:53] Scenario 03: evasion low-and-slow with jitter/noise
[2026-06-10 21:58:53] Attempts=36, gap=8-25s, noise_every=6
[2026-06-10 21:58:53] Evasion attempt 1
[2026-06-10 22:00:51] Evasion attempt 6
[2026-06-10 22:00:56] Noise pause inserted
[2026-06-10 22:03:42] Evasion attempt 12
[2026-06-10 22:03:45] Noise pause inserted
[2026-06-10 22:06:28] Evasion attempt 18
[2026-06-10 22:06:31] Noise pause inserted
[2026-06-10 22:09:24] Evasion attempt 24
[2026-06-10 22:09:27] Noise pause inserted
[2026-06-10 22:12:11] Evasion attempt 30
[2026-06-10 22:12:13] Noise pause inserted
[2026-06-10 22:14:55] Evasion attempt 36
[2026-06-10 22:15:00] Noise pause inserted
```

### Alert Summary Evidence

```text
alerts_file=outputs/alerts.jsonl
total=348
actions={"ALERT": 69, "BLOCK": 2, "BLOCKED": 277}
bad_lines=0
risk_min=0.0974
risk_max=0.1774
model_prob_min=0.0027
model_prob_max=0.0541
max_event_count=4
by_ip={"172.29.139.144": {"ALERT": 69, "BLOCK": 2, "BLOCKED": 277}}
```

### Firewall Evidence

During the run, a DROP rule was present:

```text
Chain INPUT (policy ACCEPT)
num  target     prot opt source               destination
1    DROP       0    --  172.29.139.144       0.0.0.0/0
```

The scenario cleanup then removed the rule:

```text
[2026-06-10 22:16:47] Removed DROP rule for 172.29.139.144
```

### Detector Runtime Log Evidence

The detector runtime log confirms that the block was caused by the long-window low-and-slow rule, not by the original short-window risk threshold.

Representative `BLOCKED` records:

```text
2026-06-10T22:09:56 [worker-pool] ERROR — BLOCK  | ip=172.29.139.144 events=3 failed_5m=12 failed_15m=32 p=0.054 risk=0.107 action=BLOCKED reason=LOW_AND_SLOW_15M
2026-06-10T22:10:08 [worker-pool] ERROR — BLOCK  | ip=172.29.139.144 events=1 failed_5m=11 failed_15m=32 p=0.003 risk=0.157 action=BLOCKED reason=LOW_AND_SLOW_15M
2026-06-10T22:10:43 [worker-pool] ERROR — BLOCK  | ip=172.29.139.144 events=1 failed_5m=9 failed_15m=33 p=0.003 risk=0.157 action=BLOCKED reason=LOW_AND_SLOW_15M
```

Representative `BLOCK` decision:

```text
2026-06-10T22:10:50 [worker-pool] ERROR — BLOCK  | ip=172.29.139.144 events=1 failed_5m=9 failed_15m=33 p=0.003 risk=0.157 action=BLOCK reason=LOW_AND_SLOW_15M
```

This is the key evidence line for the mitigation:

```text
failed_15m=33
risk=0.157
action=BLOCK
reason=LOW_AND_SLOW_15M
```

It shows that the short-window risk score was still below the old `ALERT_THRESHOLD=0.20`, but the long-window `failed_15m` counter exceeded the low-and-slow block threshold.

iptables block add evidence:

```text
2026-06-10T22:12:59 [worker-pool] ERROR — [BLOCK] iptables DROP added for 172.29.139.144
```

Monitor evidence during and after the run:

```text
2026-06-10T22:10:18 [monitor] INFO — OVERHEAD | CPU=0.0%  RAM=296.5MB  event_q=0  infer_q=0
2026-06-10T22:11:19 [monitor] INFO — OVERHEAD | CPU=0.0%  RAM=296.5MB  event_q=0  infer_q=19
2026-06-10T22:12:21 [monitor] INFO — OVERHEAD | CPU=1.0%  RAM=296.5MB  event_q=0  infer_q=60
2026-06-10T22:13:22 [monitor] INFO — OVERHEAD | CPU=0.0%  RAM=296.5MB  event_q=0  infer_q=0
2026-06-10T22:18:28 [monitor] INFO — OVERHEAD | CPU=0.0%  RAM=296.5MB  event_q=0  infer_q=0
```

The temporary `infer_q` increase happened while the detector process was waiting for the sudo password to add the iptables rule. After the password was entered and the queued results were processed, `infer_q` returned to `0`.

### Analysis

This result proves that the low-and-slow mitigation worked.

The most important observation is that the short-window risk score stayed below the original alert threshold:

```text
risk_max=0.1774
model_prob_max=0.0541
max_event_count=4
```

Before the long-window improvement, this pattern produced no alerts because every 60-second window looked too small to be suspicious.
After the improvement, the detector accumulated failed-login evidence across longer windows and generated long-window actions.

The `BLOCK` action means the detector made a new block decision.
The `BLOCKED` action means later events arrived while the IP was already in the blocked state.
In this run, the detector produced both:

```text
2 BLOCK
277 BLOCKED
```

Two `BLOCK` records are acceptable in this run because `BLOCK_SECONDS` is 300 seconds.
The scenario lasted longer than 15 minutes, so the first block could expire while the same IP still had enough long-window failed attempts to trigger another block.

### Result

```text
PASS
```

After the mitigation, the IDS no longer misses the low-and-slow evasion scenario.
It first alerts on accumulated long-window failures and then blocks the source IP when the behavior persists.

## Scenario 4: Distributed / Password Spraying

Pending.

## Scenario 5: Stress and Stability

Pending.

## Scenario 6: Recovery and Block Lifecycle

Pending.

## Current Findings

1. The IDS detects and blocks fast SSH brute-force attacks.
2. The IDS does not block low-rate failed logins from a legitimate user.
3. The original low-and-slow evasion weakness was mitigated with long-window counters.
4. After mitigation, Scenario 3 generated 69 `ALERT`, 2 `BLOCK`, and 277 `BLOCKED` records even though the short-window risk score stayed below 0.20.

## Low-and-Slow Mitigation Plan and Implementation

### Problem

Scenario 3 showed a likely evasion path: SSH failed-login events were present in `journalctl`, but no alert was written to `outputs/alerts.jsonl`.
This means the detector could see the traffic source, but the short 60-second window did not accumulate enough risk to cross the alert threshold.

The original real-time detector was strong against fast bursts, but weak when the attacker spread attempts over time.

### Implemented Changes

The detector was extended with long-term behavior tracking while preserving the existing short-window model path.

Changed files:

| File | Change |
|---|---|
| `src/realtime/collector.py` | Parse SSH username and event type from `Failed password` / `Invalid user` logs |
| `scripts/09_realtime_detector.py` | Retain per-IP events for a longer window, deduplicate overlapped journalctl reads, compute `failed_5m` and `failed_15m` |
| `src/detection/early_stop.py` | Add low-and-slow decision rules and alert `reason` |

New default thresholds:

| Counter | Threshold | Action | Reason |
|---|---:|---|---|
| Failed logins in 5 minutes | `>= 12` | `ALERT` | `LOW_AND_SLOW_5M` |
| Failed logins in 15 minutes | `>= 24` | `BLOCK` | `LOW_AND_SLOW_15M` |

The existing risk-score path remains active:

| Condition | Action |
|---|---|
| `risk_score >= 0.20` | `ALERT` |
| `risk_score >= 0.40` twice | `BLOCK` |

### Expected Post-Fix Behavior

| Scenario | Expected after mitigation |
|---|---|
| Scenario 1: Fast brute-force | Still produces `ALERT` then `BLOCK`, usually with reason `RISK_THRESHOLD` |
| Scenario 2: False positive | Still produces no alert/block for 8 spaced failed attempts |
| Scenario 3: Low-and-slow | Should produce `ALERT` after enough long-window failures; may produce `BLOCK` if attempts reach the 15-minute threshold |

New alert fields:

```json
{
  "ip": "172.29.139.144",
  "username": "than",
  "event_count": 3,
  "failed_5m": 12,
  "failed_15m": 12,
  "risk_score": 0.15,
  "action": "ALERT",
  "reason": "LOW_AND_SLOW_5M"
}
```

### Validation Required

After restarting the detector, rerun:

1. Scenario 2 to make sure false positives are still controlled.
2. Scenario 3 to confirm low-and-slow attempts now generate at least `ALERT`.
3. Scenario 1 to ensure fast brute-force detection still works.
