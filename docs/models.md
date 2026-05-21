# Model Training & Evaluation

## Ba mô hình

### 1. Random Forest (`models/rf_model.joblib`)
- **Script:** `scripts/06_train_models.py`
- **Tuning:** GridSearchCV 5-fold, scoring=F1
- **Best params:** `max_depth=8, n_estimators=200, min_samples_leaf=1`
- **Imbalance:** `class_weight='balanced'`
- **Features:** 75 (toàn bộ window features)

### 2. XGBoost (`models/xgb_model.joblib`)
- **Script:** `scripts/06_train_models.py`
- **Tuning:** Early stopping, `early_stopping_rounds=30`
- **Best iteration:** 97 / 500
- **Imbalance:** `scale_pos_weight=8.84`
- **Features:** 75

### 3. LSTM (`models/lstm_model.pt`)
- **Script:** `scripts/06c_train_lstm.py`
- **Input:** Chuỗi 5 window liên tiếp theo Src IP
- **Architecture:** 2-layer LSTM (hidden=64) → Dropout(0.3) → Linear(1)
- **Imbalance:** `pos_weight=8.84` trong BCEWithLogitsLoss
- **Early stop:** Epoch 16 / 60

### 4. XGBoost Real-time (`models/xgb_realtime_model.joblib`)
- **Script:** `scripts/06_retrain_realtime_model.py`
- **Features:** 5 (chỉ những gì đọc được từ journalctl log)
- **Mục đích:** Dùng trong `09_realtime_detector.py`

## Kết quả (Test set)

| Mô hình | F1 | ROC-AUC | PR-AUC | Precision@10 |
|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 | 0.10 |
| XGBoost | 0.824 | 0.950 | 0.911 | 0.10 |
| LSTM | 1.000 | 1.000 | 1.000 | 0.10 |

## Cross-validation 5-fold

Dùng `StratifiedKFold(shuffle=False)` — đảm bảo mỗi fold có đủ attack samples.

> **Tại sao không dùng TimeSeriesSplit?** Attack tập trung ở rows 341-559/620. TimeSeriesSplit cho 2/5 fold không có attack → F1 undefined.

| Mô hình | F1 mean | F1 std |
|---|---|---|
| Random Forest | 1.000 | ±0.000 |
| XGBoost | 0.964 | ±0.073 |
| LSTM | 1.000 | ±0.000 |

## Metrics

Tất cả kết quả lưu tại `outputs/metrics/`:
- `model_summary.json` — tổng hợp 3 models
- `cross_validation_results.json` — kết quả CV 5-fold
- `{model}_{val|test}_metrics.json` — chi tiết từng model

Biểu đồ lưu tại `outputs/figures/`:
- Confusion matrix, PR curve, ROC curve cho từng model
- Feature importance cho RF và XGB
