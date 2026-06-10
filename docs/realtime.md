# Real-time Detector

## Architecture

```text
journalctl -u ssh
    |
    v
Producer thread
    - polls SSH logs every IDS_POLL_SEC seconds
    - extracts failed SSH login events
    - parses source IP and username
    |
    v
event_queue
    |
    v
Consumer thread
    - deduplicates overlapped journalctl reads
    - keeps a per-IP event buffer for 15 minutes by default
    - builds 60-second real-time ML features
    - computes long-window counters for low-and-slow detection
    |
    v
inference_queue
    |
    v
Worker pool
    - runs XGBoost real-time model
    - computes risk_score
    - applies early-stop decision logic
    |
    v
NORMAL / ALERT / BLOCK / BLOCKED
```

## Log Parsing

The collector reads SSH logs from `journalctl -u ssh` and recognizes both failed-password and invalid-user events.

Supported log patterns include:

```text
Failed password for than from 172.29.139.144 port 57536 ssh2
Failed password for invalid user admin from 172.29.139.144 port 57536 ssh2
Invalid user admin from 172.29.139.144 port 57536
```

Each event contains:

| Field | Meaning |
|---|---|
| `Src IP` | Source IP from SSH log |
| `Username` | Parsed SSH username |
| `Timestamp` | Event timestamp |
| `event_type` | `failed_password` or `invalid_user` |
| `raw` | Original log line |

## Short-Window ML Path

The XGBoost real-time model still uses 5 features computed from the short window.

| Feature | Computation |
|---|---|
| `flow_rate_per_window` | Failed attempts / `IDS_WINDOW_SEC` |
| `interarrival_mean` | Mean time between failed attempts |
| `interarrival_std` | Standard deviation of interarrival time |
| `rst_flow_ratio` | Ratio of reset-like events, currently `0` from SSH logs |
| `short_flow_ratio` | Ratio of short events, currently `1` from SSH logs |

Risk score:

```python
risk_score = (
    0.60 * model_prob +
    0.15 * flow_rate_norm +
    0.10 * inv_interarrival_std +
    0.10 * rst_flow_ratio +
    0.05 * short_flow_ratio
)
```

Short-window decisions:

| Condition | Action | Reason |
|---|---|---|
| `risk_score < 0.20` | `NORMAL` | `RISK_NORMAL` |
| `0.20 <= risk_score < 0.40` | `ALERT` | `RISK_THRESHOLD` |
| `risk_score >= 0.40` twice | `BLOCK` | `RISK_THRESHOLD` |
| IP already blocked | `BLOCKED` | Last block reason |

## Low-and-Slow Detection

The original detector used a 60-second sliding window only. That worked for fast brute-force attacks but could miss low-and-slow attacks where failed attempts are spread over time.

The detector now also keeps long-window counters:

| Counter | Default Window | Default Threshold | Action | Reason |
|---|---:|---:|---|---|
| `failed_5m` | 300 seconds | `>= 12` | `ALERT` | `LOW_AND_SLOW_5M` |
| `failed_15m` | 900 seconds | `>= 24` | `BLOCK` | `LOW_AND_SLOW_15M` |

This means:

```text
Fast brute-force       -> detected by risk_score path.
Slow repeated failures -> detected by failed_5m / failed_15m counters.
Light user mistakes    -> should stay NORMAL if below long-window thresholds.
```

## Environment Variables

| Variable | Default | Meaning |
|---|---:|---|
| `IDS_MODEL_PATH` | `models/xgb_realtime_model.joblib` | Real-time XGBoost model |
| `IDS_META_PATH` | `models/xgb_realtime_features.json` | Expected feature metadata |
| `IDS_ALERTS_PATH` | `outputs/alerts.jsonl` | Alert output path |
| `IDS_WINDOW_SEC` | `60` | Short ML feature window |
| `IDS_POLL_SEC` | `5` | journalctl polling interval |
| `IDS_WORKERS` | `4` | Worker threads |
| `IDS_MONITOR_INTERVAL` | `30` | CPU/RAM monitor interval |
| `IDS_LOW_SLOW_ALERT_WINDOW_SEC` | `300` | Low-and-slow alert window |
| `IDS_LOW_SLOW_BLOCK_WINDOW_SEC` | `900` | Low-and-slow block window |
| `IDS_LOW_SLOW_ALERT_COUNT` | `12` | Failed attempts needed for low-and-slow alert |
| `IDS_LOW_SLOW_BLOCK_COUNT` | `24` | Failed attempts needed for low-and-slow block |

Startup evidence should include:

```text
Low-and-slow thresholds: failed_300s>=12 ALERT, failed_900s>=24 BLOCK
Consumer started (window_sec=60, retained_window_sec=900)
```

## Alert Format

Example fast brute-force block:

```json
{
  "ip": "172.29.139.144",
  "username": "than",
  "now": "2026-06-10 21:20:44.404857",
  "event_count": 44,
  "failed_5m": 44,
  "failed_15m": 44,
  "model_prob": 0.9973,
  "risk_score": 0.8762,
  "action": "BLOCK",
  "reason": "RISK_THRESHOLD",
  "consecutive_suspicious": 11
}
```

Example low-and-slow alert:

```json
{
  "ip": "172.29.139.144",
  "username": "than",
  "now": "2026-06-10 22:05:00.000000",
  "event_count": 3,
  "failed_5m": 12,
  "failed_15m": 12,
  "model_prob": 0.0027,
  "risk_score": 0.15,
  "action": "ALERT",
  "reason": "LOW_AND_SLOW_5M",
  "consecutive_suspicious": 1
}
```

## Monitoring

The monitor thread logs overhead every `IDS_MONITOR_INTERVAL` seconds:

```text
OVERHEAD | CPU=0.0%  RAM=266.4MB  event_q=0  infer_q=0
```

Use this during stress tests to confirm that CPU, memory, and queue sizes remain stable.
