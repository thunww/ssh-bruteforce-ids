# Real-time Detector

## Kiến trúc Producer-Consumer

```
journalctl -u ssh (poll 5s)
        │
        ▼
┌──────────────────┐
│  Producer Thread  │ → event_queue (maxsize=10,000)
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Consumer Thread  │ → per-IP sliding buffer (60s)
└──────────────────┘       → feature extraction (5 features)
        │
        ▼
  inference_queue (maxsize=1,000)
        │
        ▼
┌────────────────────────────┐
│ Worker Pool (4 threads)    │ → XGBoost inference → Risk scoring
└────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
ALERT      BLOCK
log file   iptables DROP
```

## Risk scoring

```python
risk_score = (
    0.60 * model_prob          # XGBoost confidence
    0.15 * flow_rate_norm      # Tốc độ kết nối
    0.10 * inv_interarrival    # Timing đều đặn (tấn công tự động)
    0.10 * rst_flow_ratio      # Tỉ lệ kết nối bị từ chối
    0.05 * short_flow_ratio    # Tỉ lệ flow ngắn
)
```

## Ngưỡng quyết định

| Ngưỡng | Giá trị | Hành động |
|---|---|---|
| `risk_score < 0.20` | NORMAL | Bỏ qua |
| `0.20 ≤ risk_score < 0.40` | ALERT | Ghi vào alerts.jsonl |
| `risk_score ≥ 0.40` (2 lần liên tiếp) | BLOCK | iptables DROP + ghi alert |
| IP đang bị block | BLOCKED | Giữ nguyên 5 phút |

## 5 Features real-time

| Feature | Cách tính |
|---|---|
| `flow_rate_per_window` | Số failed attempts / 60s |
| `interarrival_mean` | Thời gian trung bình giữa các lần thử |
| `interarrival_std` | Độ lệch chuẩn interarrival |
| `rst_flow_ratio` | Tỉ lệ connections có RST flag |
| `short_flow_ratio` | Tỉ lệ connections ngắn |

## Monitor overhead

Thread monitor log mỗi 30 giây:
```
[monitor] OVERHEAD | CPU=1.0%  RAM=274MB  event_q=0  infer_q=0
```

## Alert format (outputs/alerts.jsonl)

```json
{
  "ip": "192.168.1.10",
  "now": "2026-05-21 22:44:08",
  "event_count": 44,
  "model_prob": 0.997,
  "risk_score": 0.804,
  "action": "BLOCK",
  "consecutive_suspicious": 2
}
```
