"""
Train LSTM model cho SSH brute-force detection.
Input: chuỗi window features theo Src IP (seq_len=5).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from _bootstrap import add_project_root
add_project_root()

from src.data.sequence_builder import build_sequences, SequenceDataset
from src.data.preprocess_pipeline import load_scaler
from src.models.lstm_model import LSTMClassifier
from src.models.evaluate import (
    evaluate_binary_classifier,
    precision_at_k,
    save_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

DATA_DIR = Path("data/processed/splits")
MODELS_DIR = Path("models")
METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")

SEQ_LEN = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    train_meta = pd.read_parquet(DATA_DIR / "train_windows_with_meta.parquet")
    val_meta = pd.read_parquet(DATA_DIR / "val_windows_with_meta.parquet")
    test_meta = pd.read_parquet(DATA_DIR / "test_windows_with_meta.parquet")

    X_train_raw = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_val_raw = pd.read_parquet(DATA_DIR / "X_val.parquet")
    X_test_raw = pd.read_parquet(DATA_DIR / "X_test.parquet")

    feature_cols = list(X_train_raw.columns)

    # Scale features (LSTM needs normalized inputs)
    scaler = load_scaler(MODELS_DIR / "scaler.joblib")

    X_train_sc = pd.DataFrame(
        scaler.transform(X_train_raw), columns=feature_cols, index=X_train_raw.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val_raw), columns=feature_cols, index=X_val_raw.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test_raw), columns=feature_cols, index=X_test_raw.index
    )

    # Attach metadata for sequence building
    train_df = train_meta[["Src IP", "window_start", "target"]].copy().reset_index(drop=True)
    val_df = val_meta[["Src IP", "window_start", "target"]].copy().reset_index(drop=True)
    test_df = test_meta[["Src IP", "window_start", "target"]].copy().reset_index(drop=True)

    for col in feature_cols:
        train_df[col] = X_train_sc[col].values
        val_df[col] = X_val_sc[col].values
        test_df[col] = X_test_sc[col].values

    return train_df, val_df, test_df, feature_cols


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.extend(prob.tolist())
    return np.array(probs)


def choose_best_threshold(y_true, y_prob):
    best_th, best_f1 = 0.5, -1.0
    for th in [i / 100 for i in range(5, 96, 5)]:
        m = evaluate_binary_classifier(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
    return best_th, best_f1


def main():
    for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_df, val_df, test_df, feature_cols = load_data()

    # Build sequences
    X_train, y_train = build_sequences(train_df, feature_cols, seq_len=SEQ_LEN)
    X_val, y_val = build_sequences(val_df, feature_cols, seq_len=SEQ_LEN)
    X_test, y_test = build_sequences(test_df, feature_cols, seq_len=SEQ_LEN)

    print(f"Sequences — train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")
    print(f"train attack: {int(y_train.sum())} / {len(y_train)}")
    print(f"val   attack: {int(y_val.sum())} / {len(y_val)}")
    print(f"test  attack: {int(y_test.sum())} / {len(y_test)}")

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # pos_weight for imbalanced training
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    patience = 15

    print("\nTraining LSTM...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        val_prob = predict_proba(model, val_loader)
        val_th, val_f1 = choose_best_threshold(y_val, val_prob)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    val_prob = predict_proba(model, val_loader)
    best_threshold, _ = choose_best_threshold(y_val, val_prob)
    test_prob = predict_proba(model, test_loader)

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    # precision@k — map sequences back to IPs
    test_ip_series = pd.Series(
        [ip for ip, grp in test_df.groupby("Src IP", sort=False)
         for _ in range(max(1, len(grp) - SEQ_LEN + 1))]
    )
    if len(test_ip_series) == len(test_prob):
        p10 = precision_at_k(y_test, test_prob, test_ip_series.values, k=10)
        p50 = precision_at_k(y_test, test_prob, test_ip_series.values, k=50)
    else:
        p10, p50 = float("nan"), float("nan")

    test_metrics["precision_at_10"] = p10
    test_metrics["precision_at_50"] = p50

    # Save model
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
    model_meta = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "seq_len": SEQ_LEN,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold,
    }
    with open(MODELS_DIR / "lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    save_metrics(val_metrics, METRICS_DIR / "lstm_val_metrics.json")
    save_metrics(test_metrics, METRICS_DIR / "lstm_test_metrics.json")

    plot_confusion_matrix(val_metrics, FIGURES_DIR / "lstm_val_confusion_matrix.png", "LSTM Val Confusion Matrix")
    plot_confusion_matrix(test_metrics, FIGURES_DIR / "lstm_test_confusion_matrix.png", "LSTM Test Confusion Matrix")
    plot_pr_curve(y_val, val_prob, FIGURES_DIR / "lstm_val_pr_curve.png", "LSTM Val PR Curve")
    plot_pr_curve(y_test, test_prob, FIGURES_DIR / "lstm_test_pr_curve.png", "LSTM Test PR Curve")
    plot_roc_curve(y_val, val_prob, FIGURES_DIR / "lstm_val_roc_curve.png", "LSTM Val ROC Curve")
    plot_roc_curve(y_test, test_prob, FIGURES_DIR / "lstm_test_roc_curve.png", "LSTM Test ROC Curve")

    print("\n=== LSTM RESULTS ===")
    print(f"Best threshold: {best_threshold}")
    print(f"Val  — F1={val_metrics['f1']:.4f}  ROC={val_metrics['roc_auc']:.4f}  PR={val_metrics['pr_auc']:.4f}")
    print(f"Test — F1={test_metrics['f1']:.4f}  ROC={test_metrics['roc_auc']:.4f}  PR={test_metrics['pr_auc']:.4f}")
    print(f"precision@10={p10}  precision@50={p50}")
    print("LSTM model saved to models/lstm_model.pt")


if __name__ == "__main__":
    main()
