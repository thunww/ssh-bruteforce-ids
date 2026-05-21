# Data Pipeline

## Tổng quan

```
data/raw/tuesday.csv
    ↓ 01_audit_tuesday.py
    ↓ 02_filter_ssh.py         → Lọc Port 22, TCP, label SSH-Patator
    ↓ 03_clean_and_prune.py    → Parse timestamp, xóa cột thừa, xử lý NaN
data/interim/tuesday_ssh_clean.parquet
    ↓ 04_build_windows.py      → Sliding window 60s theo Src IP, 75 features
data/processed/tuesday_ssh_windows.parquet  (620 windows)
    ↓ 05_prepare_train_data.py → Time-based split 70/15/15
data/processed/splits/
    ├── X_train.parquet  (433 rows, 75 features)
    ├── X_val.parquet    (93 rows)
    └── X_test.parquet   (94 rows)
```

## Features (75 tổng)

| Nhóm | Features | Mô tả |
|---|---|---|
| Flow rate | `flow_rate_per_window` | Số flow / 60s |
| Interarrival | `interarrival_mean/std/min/max` | Thống kê thời gian giữa các flow |
| Packet stats | `total_fwd/bwd_packet_*` | Thống kê số packet |
| Byte stats | `fwd/bwd_bytes_*` | Thống kê số byte |
| Flag stats | `syn/rst/ack/psh_flag_*` | Thống kê TCP flags |
| Derived | `byte_ratio`, `syn_ack_ratio`, `packet_rate_per_window` | Features mới |
| Ratio | `rst_flow_ratio`, `short_flow_ratio`, `high_rate_flow_ratio` | Tỉ lệ flow đặc trưng |

## Class imbalance

- Window-level: 556 benign / 64 attack (tỉ lệ 8.7:1)
- Xử lý: `class_weight='balanced'` (RF), `scale_pos_weight=8.84` (XGB), `pos_weight` (LSTM)
- Không dùng SMOTE vì đây là time-series data (tạo synthetic samples vi phạm temporal ordering)

## Time-based split

Sử dụng `classwise_time_split`: tách attack và benign riêng theo thời gian, đảm bảo không có data leakage.
