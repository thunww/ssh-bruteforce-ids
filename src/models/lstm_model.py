"""
LSTM model cho chuỗi SSH brute-force attempts theo Src IP.
Input: sequence of window feature vectors per IP.
Output: binary classification (attack=1 / benign=0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        last = self.dropout(last)
        logit = self.fc(last)
        return logit.squeeze(-1)
