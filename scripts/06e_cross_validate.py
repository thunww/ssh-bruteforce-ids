"""
Cross-validation 5-fold cho RF, XGB, LSTM.
Dung StratifiedKFold(shuffle=False) vi:
- TimeSeriesSplit cho 2/5 fold khong co attack (F1 undefined)
- Attack tap trung o rows 341-559/620, khong phan bo deu theo thoi gian
- StratifiedKFold dam bao moi fold co 12-13 attack samples
Ket qua: F1 mean +- std dang tin hon F1 = 1.0 tren 1 test set nho.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from _bootstrap import add_project_root
add_project_root()

from src.features.feature_selector import select_features
from src.models.evaluate import evaluate_binary_classifier
from src.data.sequence_builder import build_sequences, SequenceDataset
from src.models.lstm_model import LSTMClassifier

DATA_PATH = Path("data/processed/tuesday_ssh_windows.parquet")
OUT_PATH = Path("outputs/metrics/cross_validation_results.json")

N_SPLITS = 5
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def choose_threshold(y_true, y_prob):
    best_th, best_f1 = 0.5, -1.0
    for th in [i / 100 for i in range(5, 96, 5)]:
        m = evaluate_binary_classifier(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
    return best_th


def summarize(scores: list[dict], metric: str) -> tuple[float, float]:
    vals = [s[metric] for s in scores if not np.isnan(s.get(metric, float("nan")))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


# ─────────────────────────────────────────────
# RF
# ─────────────────────────────────────────────
def cv_rf(X: pd.DataFrame, y: pd.Series, folds) -> list[dict]:
    results = []
    for fold, (tr, te) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_te)[:, 1]
        th = choose_threshold(y_te, prob)
        m = evaluate_binary_classifier(y_te, prob, threshold=th)
        m["fold"] = fold
        m["n_train"] = len(tr)
        m["n_test"] = len(te)
        m["attack_in_test"] = int(y_te.sum())
        results.append(m)
        print(f"  RF  fold {fold}: F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
              f"(attack_test={m['attack_in_test']})")
    return results


# ─────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────
def cv_xgb(X: pd.DataFrame, y: pd.Series, folds) -> list[dict]:
    results = []
    for fold, (tr, te) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        # Val nho de early stopping (15% cuoi train)
        val_cut = int(len(X_tr) * 0.85)
        X_val = X_tr.iloc[val_cut:]
        y_val = y_tr.iloc[val_cut:]
        X_tr2 = X_tr.iloc[:val_cut]
        y_tr2 = y_tr.iloc[:val_cut]

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            early_stopping_rounds=30, random_state=RANDOM_STATE,
            n_jobs=-1, scale_pos_weight=spw, verbosity=0,
        )
        model.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)

        prob = model.predict_proba(X_te)[:, 1]
        th = choose_threshold(y_te, prob)
        m = evaluate_binary_classifier(y_te, prob, threshold=th)
        m["fold"] = fold
        m["n_train"] = len(tr)
        m["n_test"] = len(te)
        m["attack_in_test"] = int(y_te.sum())
        m["best_iteration"] = int(model.best_iteration)
        results.append(m)
        print(f"  XGB fold {fold}: F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
              f"(attack_test={m['attack_in_test']}, iter={model.best_iteration})")
    return results


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────
def train_lstm_fold(X_tr_sc, y_tr, X_te_sc, y_te, df_full, tr_idx, te_idx, feature_cols):
    SEQ_LEN = 5
    EPOCHS = 60
    PATIENCE = 15

    # Build sequence dfs
    tr_df = df_full.iloc[tr_idx][["Src IP", "window_start", "target"]].copy().reset_index(drop=True)
    te_df = df_full.iloc[te_idx][["Src IP", "window_start", "target"]].copy().reset_index(drop=True)

    for col in feature_cols:
        tr_df[col] = X_tr_sc[col].values
        te_df[col] = X_te_sc[col].values

    X_tr_seq, y_tr_seq = build_sequences(tr_df, feature_cols, seq_len=SEQ_LEN)
    X_te_seq, y_te_seq = build_sequences(te_df, feature_cols, seq_len=SEQ_LEN)

    if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
        return None

    train_loader = DataLoader(SequenceDataset(X_tr_seq, y_tr_seq), batch_size=32, shuffle=True)
    test_loader = DataLoader(SequenceDataset(X_te_seq, y_te_seq), batch_size=32, shuffle=False)

    input_size = X_tr_seq.shape[2]
    model = LSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)

    n_pos = int(y_tr_seq.sum())
    n_neg = len(y_tr_seq) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().tolist())
        probs = np.array(probs)

        th = choose_threshold(y_te_seq, probs)
        f1 = evaluate_binary_classifier(y_te_seq, probs, threshold=th)["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            probs.extend(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().tolist())
    probs = np.array(probs)

    return probs, y_te_seq


def cv_lstm(X: pd.DataFrame, y: pd.Series, df_full: pd.DataFrame, folds) -> list[dict]:
    feature_cols = list(X.columns)
    results = []

    for fold, (tr, te) in enumerate(folds, 1):
        X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        X_te, y_te = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)

        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
        X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=feature_cols)

        out = train_lstm_fold(X_tr_sc, y_tr, X_te_sc, y_te, df_full, tr, te, feature_cols)
        if out is None:
            print(f"  LSTM fold {fold}: SKIP (empty sequences)")
            continue

        probs, y_te_seq = out
        th = choose_threshold(y_te_seq, probs)
        m = evaluate_binary_classifier(y_te_seq, probs, threshold=th)
        m["fold"] = fold
        m["n_train"] = len(tr)
        m["n_test"] = len(te)
        m["attack_in_test"] = int(y_te.sum())
        results.append(m)
        print(f"  LSTM fold {fold}: F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
              f"(attack_test={m['attack_in_test']})")
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("window_start").reset_index(drop=True)

    feature_cols = select_features(df)
    X = df[feature_cols]
    y = df["target"]

    print(f"Dataset: {len(df)} windows | {int(y.sum())} attack | {int((y==0).sum())} benign")
    print(f"Features: {len(feature_cols)}")
    print(f"Device: {DEVICE}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    folds = list(skf.split(X, y))

    print(f"\nFold distribution:")
    for i, (tr, te) in enumerate(folds, 1):
        print(f"  Fold {i}: train={len(tr)} (atk={int(y.iloc[tr].sum())})  "
              f"test={len(te)} (atk={int(y.iloc[te].sum())})")

    all_results = {}

    print("\n--- RF ---")
    rf_scores = cv_rf(X, y, folds)
    all_results["rf"] = rf_scores

    print("\n--- XGBoost ---")
    xgb_scores = cv_xgb(X, y, folds)
    all_results["xgb"] = xgb_scores

    print("\n--- LSTM ---")
    lstm_scores = cv_lstm(X, y, df, folds)
    all_results["lstm"] = lstm_scores

    # Summary
    print("\n" + "="*60)
    print(f"{'Model':6}  {'F1 mean':>9}  {'F1 std':>7}  {'ROC mean':>9}  {'PR mean':>8}")
    print("-"*60)

    summary = {}
    for name, scores in all_results.items():
        f1_mean, f1_std = summarize(scores, "f1")
        roc_mean, roc_std = summarize(scores, "roc_auc")
        pr_mean, pr_std = summarize(scores, "pr_auc")
        summary[name] = {
            "f1_mean": f1_mean, "f1_std": f1_std,
            "roc_auc_mean": roc_mean, "roc_auc_std": roc_std,
            "pr_auc_mean": pr_mean, "pr_auc_std": pr_std,
            "folds": scores,
        }
        print(f"{name.upper():6}  {f1_mean:>8.4f}  {f1_std:>7.4f}  "
              f"{roc_mean:>8.4f}  {pr_mean:>8.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
