"""
Preprocessing pipeline: StandardScaler only.
SMOTE không dùng vì đây là time-series window data — synthetic samples
tạo ra bởi SMOTE không có temporal validity.
Xử lý imbalance bằng class_weight='balanced' (RF) và scale_pos_weight (XGB).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    arr = scaler.transform(X)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> StandardScaler:
    return joblib.load(path)
