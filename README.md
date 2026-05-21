# SSH Brute-force Intrusion Detection System

Hệ thống phát hiện tấn công SSH brute-force theo thời gian thực, sử dụng Machine Learning kết hợp phân tích log. Khi phát hiện tấn công, hệ thống tự động chặn IP bằng `iptables`.

---

## Mục lục

1. [Kiến trúc hệ thống](#1-kiến-trúc-hệ-thống)
2. [Dataset](#2-dataset)
3. [Cài đặt](#3-cài-đặt)
4. [Chạy offline pipeline](#4-chạy-offline-pipeline)
5. [Kết quả mô hình](#5-kết-quả-mô-hình)
6. [Chạy real-time detector](#6-chạy-real-time-detector)
7. [Deploy trên Ubuntu VM](#7-deploy-trên-ubuntu-vm)
8. [Giám sát và xem alert](#8-giám-sát-và-xem-alert)
9. [Rollback và xử lý sự cố](#9-rollback-và-xử-lý-sự-cố)

---

## 1. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                      OFFLINE PIPELINE                        │
│                                                             │
│  CICIDS2017 CSV → Lọc SSH → Làm sạch → Window Features     │
│  → Train/Val/Test Split → Train RF / XGB / LSTM             │
│  → Đánh giá (F1, ROC-AUC, PR-AUC, Precision@k)             │
│  → Lưu models/                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                    models/*.joblib / *.pt
                            │
┌─────────────────────────────────────────────────────────────┐
│                    REAL-TIME PIPELINE                        │
│                                                             │
│  [Producer]  journalctl SSH log → event_queue               │
│  [Consumer]  Gom events theo IP (cửa sổ 60s) → features    │
│  [Workers]   4 threads song song → XGBoost inference        │
│  [Monitor]   Log CPU/RAM mỗi 30 giây                        │
│                                                             │
│  NORMAL → bỏ qua                                            │
│  ALERT  → ghi vào outputs/alerts.jsonl                      │
│  BLOCK  → iptables DROP + ghi alert                         │
└─────────────────────────────────────────────────────────────┘
```

### Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Dataset | CICIDS2017 Tuesday (SSH-Patator) |
| Feature engineering | Sliding window 60s theo Src IP |
| Models | Random Forest, XGBoost, LSTM (PyTorch) |
| Real-time | Python threading, queue.Queue, ThreadPoolExecutor |
| Blocking | iptables |
| Monitoring | psutil |

---

## 2. Dataset

**Nguồn:** [CICIDS2017 - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

**File cần tải:** `Tuesday-WorkingHours.pcap_ISCX.csv`

**Cách chuẩn bị:**
```bash
# Đổi tên và đặt vào thư mục data/raw/
mv Tuesday-WorkingHours.pcap_ISCX.csv data/raw/tuesday.csv
```

**Thống kê dataset:**

| Chỉ số | Giá trị |
|---|---|
| Tổng số flow | 322,078 |
| SSH flows (Port 22, TCP) | 4,031 |
| Attack (SSH-Patator) | 2,988 |
| Benign | 1,043 |
| Windows sau xử lý | 620 |
| Train / Val / Test | 433 / 93 / 94 |

---

## 3. Cài đặt

### Yêu cầu

- Python 3.10+
- Linux (Ubuntu 20.04+) — để chạy real-time detector
- SSH server đang chạy — để test

### Cài đặt dependencies

```bash
git clone https://github.com/thunww/ssh-bruteforce-ids.git
cd ssh-bruteforce-ids
pip install -r requirements.txt
```

---

## 4. Chạy offline pipeline

Chạy lần lượt các script theo thứ tự:

```bash
# Bước 1: Kiểm tra cấu trúc dataset
python scripts/01_audit_tuesday.py

# Bước 2: Lọc traffic SSH (Port 22, TCP)
python scripts/02_filter_ssh.py

# Bước 3: Làm sạch dữ liệu
python scripts/03_clean_and_prune.py

# Bước 4: Tạo time-window features (cửa sổ 60s theo Src IP)
python scripts/04_build_windows.py

# Bước 5: Chia train/val/test (time-based, không random)
python scripts/05_prepare_train_data.py

# Bước 6a: Train Random Forest + XGBoost
python scripts/06_train_models.py

# Bước 6b: Ablation study và so sánh với rule-based baseline
python scripts/06b_rule_and_ablation.py

# Bước 6c: Train LSTM (chuỗi 5 window liên tiếp theo IP)
python scripts/06c_train_lstm.py

# Bước 6d: Train model nhỏ cho real-time (5 features từ log)
python scripts/06_retrain_realtime_model.py

# Bước 6e: Cross-validation 5-fold (đánh giá độ tin cậy)
python scripts/06e_cross_validate.py

# Bước 7: Mô phỏng early-stop detection trên test set
python scripts/07_simulate_early_stop.py

# Bước 8: Đánh giá detection delay và false positive rate
python scripts/08_evaluate_early_stop.py
```

> **Lưu ý:** Các model đã được train sẵn trong `models/`. Không cần chạy lại trừ khi muốn thay đổi hyperparameter.

---

## 5. Kết quả mô hình

### So sánh 3 mô hình (Test set: 94 windows, 10 attack)

| Mô hình | F1 | ROC-AUC | PR-AUC | Precision@10 |
|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 0.10 |
| XGBoost | 0.824 | 0.950 | 0.911 | 0.10 |
| LSTM | 1.000 | 1.000 | 1.000 | 0.10 |

### Cross-validation 5-fold (620 windows, đáng tin hơn)

| Mô hình | F1 trung bình | Độ lệch chuẩn |
|---|---|---|
| Random Forest | 1.000 | ±0.000 |
| XGBoost | 0.964 | ±0.073 |
| LSTM | 1.000 | ±0.000 |

### Chi tiết từng mô hình

**Random Forest**
- GridSearchCV 5-fold, best: `max_depth=8, n_estimators=200`
- `class_weight='balanced'` để xử lý mất cân bằng dữ liệu

**XGBoost**
- Early stopping tại iteration 97 (từ max 500)
- `scale_pos_weight=8.84` (tỉ lệ benign/attack)

**LSTM**
- Input: chuỗi 5 window liên tiếp theo Src IP
- 2-layer LSTM, hidden=64, Dropout=0.3
- Early stop tại epoch 16

> **Kết quả thực tế:** SSH-Patator tạo pattern rất đặc trưng (flow rate cao, timing đều, toàn short flows) nên mọi model đều phân tách tốt trên dataset này.

### Figures

Tất cả biểu đồ lưu tại `outputs/figures/`:
- Confusion matrix, PR curve, ROC curve cho từng model
- Feature importance cho Random Forest và XGBoost

---

## 6. Chạy real-time detector

### Yêu cầu

- Đang chạy trên Linux có SSH server
- Quyền `sudo` để gọi `iptables`

### Khởi động

```bash
sudo PYTHONPATH=/path/to/ssh-bruteforce-ids \
  .venv/bin/python scripts/09_realtime_detector.py
```

### Luồng hoạt động

```
journalctl (SSH log)
    ↓ poll mỗi 5 giây
Producer thread → event_queue
    ↓
Consumer thread → gom events theo IP (cửa sổ 60s)
    ↓
Worker pool (4 threads) → XGBoost inference → Risk scoring
    ↓
NORMAL  → bỏ qua
ALERT   → ghi outputs/alerts.jsonl
BLOCK   → iptables DROP + ghi alert
BLOCKED → IP đã bị chặn, giữ nguyên 5 phút
```

### Cấu hình qua biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `IDS_WINDOW_SEC` | `60` | Kích thước cửa sổ (giây) |
| `IDS_POLL_SEC` | `5` | Tần suất đọc log (giây) |
| `IDS_WORKERS` | `4` | Số worker threads |
| `IDS_MONITOR_INTERVAL` | `30` | Chu kỳ log CPU/RAM (giây) |
| `IDS_ALERTS_PATH` | `outputs/alerts.jsonl` | File ghi alert |

Ví dụ chạy với config tùy chỉnh:
```bash
sudo IDS_WINDOW_SEC=30 IDS_WORKERS=8 \
  PYTHONPATH=/path/to/project \
  .venv/bin/python scripts/09_realtime_detector.py
```

### Ngưỡng phát hiện (chỉnh trong `src/detection/early_stop.py`)

| Ngưỡng | Giá trị | Ý nghĩa |
|---|---|---|
| `ALERT_THRESHOLD` | 0.20 | Risk score ≥ 0.20 → ALERT |
| `BLOCK_THRESHOLD` | 0.40 | Risk score ≥ 0.40 liên tiếp 2 lần → BLOCK |
| `BLOCK_SECONDS` | 300 | Thời gian block IP (giây) |

---

## 7. Deploy trên Ubuntu VM

### Chuẩn bị VM

- VMware / VirtualBox, Ubuntu 22.04 LTS
- 2 CPU, 4GB RAM, 20GB disk
- Network: **Bridged** (để có IP riêng trong LAN)
- Ghi lại IP của VM, ví dụ: `192.168.1.105`

### Copy project lên VM

Từ máy Windows, chạy trong PowerShell:
```powershell
scp -r d:\ssh-bruteforce-ids user@192.168.1.105:/home/user/
```

### Cài đặt trên VM

```bash
# SSH vào VM
ssh user@192.168.1.105

# Vào thư mục project
cd /home/user/ssh-bruteforce-ids

# Cài SSH server (nếu chưa có)
sudo apt-get install -y openssh-server
sudo systemctl enable ssh && sudo systemctl start ssh

# Tạo môi trường Python
python3 -m venv .venv
source .venv/bin/activate

# Cài packages cho real-time detector
pip install pandas numpy scikit-learn xgboost joblib psutil
```

### Chạy detector

```bash
source .venv/bin/activate
sudo PYTHONPATH=/home/user/ssh-bruteforce-ids \
  .venv/bin/python scripts/09_realtime_detector.py
```

### Chạy như background service (không cần giữ terminal)

```bash
nohup sudo PYTHONPATH=/home/user/ssh-bruteforce-ids \
  .venv/bin/python scripts/09_realtime_detector.py \
  > outputs/detector.log 2>&1 &

echo "PID: $!"
```

### Test tấn công từ máy khác

```bash
# Tạo file password
seq 1 200 > /tmp/pass.txt

# Tấn công bằng hydra (từ máy Windows/Kali)
hydra -l root -P /tmp/pass.txt -t 4 ssh://192.168.1.105
```

---

## 8. Giám sát và xem alert

### Xem alerts real-time

```bash
tail -f outputs/alerts.jsonl
```

Output mẫu:
```json
{"ip": "192.168.1.10", "now": "2026-05-21 22:44:08", "event_count": 44, "model_prob": 0.997, "risk_score": 0.804, "action": "BLOCK", "consecutive_suspicious": 2}
```

### Xem log detector

```bash
# Nếu chạy nền
tail -f outputs/detector.log

# Nếu chạy systemd
sudo journalctl -u ssh-ids -f
```

### Xem các IP đang bị chặn

```bash
sudo iptables -L INPUT -n --line-numbers
```

### Monitor overhead

Detector tự log mỗi 30 giây:
```
[monitor] OVERHEAD | CPU=1.0%  RAM=274MB  event_q=0  infer_q=0
```

---

## 9. Rollback và xử lý sự cố

### Bỏ chặn một IP

```bash
# Xem danh sách rules
sudo iptables -L INPUT -n --line-numbers

# Xóa rule theo IP
sudo iptables -D INPUT -s 192.168.1.10 -j DROP
```

### Xóa tất cả rules do IDS tạo

```bash
sudo iptables -F INPUT
```

### Dừng detector

```bash
# Nếu đang chạy foreground
Ctrl+C

# Nếu đang chạy nền
pkill -f 09_realtime_detector.py
```

### Xử lý lỗi thường gặp

**`ModuleNotFoundError: No module named 'src'`**
```bash
# Thêm PYTHONPATH khi chạy
sudo PYTHONPATH=/path/to/project .venv/bin/python scripts/09_realtime_detector.py
```

**`journalctl: command not found`**
```bash
sudo apt-get install -y systemd
```

**`iptables: Permission denied`**
```bash
# Chạy với sudo
sudo PYTHONPATH=... .venv/bin/python scripts/09_realtime_detector.py
```

**Detector không phát hiện tấn công (stuck ở ALERT)**
- Tăng số lần thử của hydra: `-t 8` thay vì `-t 4`
- Hoặc hạ `BLOCK_THRESHOLD` trong `src/detection/early_stop.py` từ `0.40` xuống `0.25`

---

## Cấu trúc thư mục

```
ssh-bruteforce-ids/
├── config/
│   └── settings.yaml          # Cấu hình ngưỡng, hyperparameter
├── data/
│   ├── raw/tuesday.csv        # Dataset gốc (không commit lên git)
│   ├── interim/               # Dữ liệu sau lọc và làm sạch
│   └── processed/             # Windows features và train/val/test splits
├── docs/                      # Tài liệu chi tiết từng bước
├── models/
│   ├── rf_model.joblib        # Random Forest (75 features)
│   ├── xgb_model.joblib       # XGBoost (75 features)
│   ├── lstm_model.pt          # LSTM (75 features, seq_len=5)
│   ├── lstm_meta.json         # Config LSTM
│   ├── xgb_realtime_model.joblib  # XGBoost real-time (5 features)
│   ├── xgb_realtime_features.json # Features cho real-time model
│   └── scaler.joblib          # StandardScaler (cho LSTM)
├── outputs/
│   ├── alerts.jsonl           # Alert real-time (append)
│   ├── figures/               # Biểu đồ confusion matrix, PR, ROC
│   └── metrics/               # JSON metrics từng model
├── scripts/
│   ├── 01_audit_tuesday.py    # Kiểm tra dataset
│   ├── 02_filter_ssh.py       # Lọc SSH traffic
│   ├── 03_clean_and_prune.py  # Làm sạch dữ liệu
│   ├── 04_build_windows.py    # Tạo time-window features
│   ├── 05_prepare_train_data.py   # Chia train/val/test
│   ├── 06_train_models.py     # Train RF + XGBoost
│   ├── 06b_rule_and_ablation.py   # Ablation study
│   ├── 06c_train_lstm.py      # Train LSTM
│   ├── 06_retrain_realtime_model.py  # Train model real-time
│   ├── 06e_cross_validate.py  # Cross-validation 5-fold
│   ├── 07_simulate_early_stop.py    # Mô phỏng early-stop
│   ├── 08_evaluate_early_stop.py    # Đánh giá detection delay
│   └── 09_realtime_detector.py      # Real-time detector (chính)
├── src/
│   ├── data/                  # Load, filter, clean, split, preprocess
│   ├── detection/             # Early-stop logic, risk scoring
│   ├── features/              # Window aggregator, feature selector
│   ├── models/                # RF, XGB, LSTM, evaluate
│   ├── realtime/              # Collector, feature builder, blocker
│   └── utils/                 # IO helpers
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tác giả

**Than** — Đồ án môn học: Phát hiện SSH Brute-force dựa trên phân tích gói tin và AI thời gian thực
