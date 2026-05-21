"""
Xây dựng sequence dataset cho LSTM.
Với mỗi Src IP, sắp xếp windows theo thời gian và tạo sliding sequences.
Label của sequence = label của window cuối cùng trong sequence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    X_list, y_list = [], []

    for ip, grp in df.groupby("Src IP"):
        grp = grp.sort_values("window_start").reset_index(drop=True)
        feats = grp[feature_cols].values.astype(np.float32)
        labels = grp["target"].values.astype(np.float32)

        if len(grp) < seq_len:
            # pad with zeros at the beginning
            pad = np.zeros((seq_len - len(grp), feats.shape[1]), dtype=np.float32)
            feats_padded = np.vstack([pad, feats])
            X_list.append(feats_padded)
            y_list.append(labels[-1])
        else:
            for i in range(len(grp) - seq_len + 1):
                X_list.append(feats[i: i + seq_len])
                y_list.append(labels[i + seq_len - 1])

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.stack(X_list), np.array(y_list, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
