import json
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess_pipeline import load_scaler, apply_scaler
from src.models.lstm_model import LSTMClassifier
from src.data.sequence_builder import build_sequences
import torch

SPLITS = ROOT / "data/processed/splits"
MODELS = ROOT / "models"
OUT    = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

X_test   = pd.read_parquet(SPLITS / "X_test.parquet")
y_test   = pd.read_parquet(SPLITS / "y_test.parquet").squeeze()
meta_test = pd.read_parquet(SPLITS / "test_windows_with_meta.parquet")
feature_cols = X_test.columns.tolist()

fig, ax = plt.subplots(figsize=(7, 6))

# --- Random Forest ---
rf = joblib.load(MODELS / "rf_model.joblib")
rf_prob = rf.predict_proba(X_test)[:, 1]
prec, rec, _ = precision_recall_curve(y_test, rf_prob)
ax.plot(rec, prec, label=f"Random Forest (AUC = {auc(rec, prec):.3f})",
        color="#4C9BE8", lw=2, linestyle="--")

# --- XGBoost ---
xgb = joblib.load(MODELS / "xgb_model.joblib")
xgb_prob = xgb.predict_proba(X_test)[:, 1]
prec, rec, _ = precision_recall_curve(y_test, xgb_prob)
ax.plot(rec, prec, label=f"XGBoost (AUC = {auc(rec, prec):.3f})",
        color="#E8A24C", lw=2)

# --- LSTM ---
with open(MODELS / "lstm_meta.json") as f:
    meta = json.load(f)
scaler = load_scaler(MODELS / "scaler.joblib")
X_scaled = apply_scaler(scaler, X_test)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=meta_test.index)
combined = meta_test[["Src IP", "window_start", "target"]].join(X_scaled_df)
X_seq, y_seq = build_sequences(combined, feature_cols, seq_len=meta["seq_len"])
model = LSTMClassifier(
    input_size=meta["input_size"],
    hidden_size=meta["hidden_size"],
    num_layers=meta["num_layers"],
    dropout=meta["dropout"],
)
model.load_state_dict(torch.load(MODELS / "lstm_model.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_seq, dtype=torch.float32)).squeeze()
    lstm_prob = torch.sigmoid(logits).numpy()
prec, rec, _ = precision_recall_curve(y_seq, lstm_prob)
ax.plot(rec, prec, label=f"LSTM (AUC = {auc(rec, prec):.3f})",
        color="#6DBE6D", lw=2, linestyle="-.")

# --- Baseline (random) ---
baseline = y_test.mean()
ax.axhline(y=baseline, color="k", linestyle=":", lw=1.2,
           label=f"Random (Precision = {baseline:.2f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.set_title("Precision-Recall Curve — RF vs XGBoost vs LSTM (Test Set)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "combined_pr_curve.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/combined_pr_curve.png")
plt.show()
