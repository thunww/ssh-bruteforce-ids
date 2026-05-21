"""
Train RF + XGB với đầy đủ: GridSearchCV (RF), early stopping (XGB),
ROC-AUC, PR-AUC, precision@k=10/50, confusion matrix, ROC + PR curves.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from _bootstrap import add_project_root
add_project_root()

from src.models.evaluate import (
    evaluate_binary_classifier,
    precision_at_k,
    save_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_feature_importance,
    plot_feature_importance,
)

DATA_DIR = Path("data/processed/splits")
MODELS_DIR = Path("models")
METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")


def load_split():
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["target"]
    X_val = pd.read_parquet(DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(DATA_DIR / "y_val.parquet")["target"]
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["target"]
    train_meta = pd.read_parquet(DATA_DIR / "train_windows_with_meta.parquet")
    val_meta = pd.read_parquet(DATA_DIR / "val_windows_with_meta.parquet")
    test_meta = pd.read_parquet(DATA_DIR / "test_windows_with_meta.parquet")
    return X_train, y_train, X_val, y_val, X_test, y_test, train_meta, val_meta, test_meta


def choose_best_threshold(y_true, y_prob):
    best_threshold, best_f1 = 0.5, -1.0
    for th in [i / 100 for i in range(5, 96, 5)]:
        m = evaluate_binary_classifier(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = th
    return best_threshold, best_f1


# ─────────────────────────────────────────────
# RF with GridSearchCV
# ─────────────────────────────────────────────
def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, test_meta):
    print("\n=== TRAIN RF (GridSearchCV) ===")

    # Combine train+val for GridSearch with time-respecting CV
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 2],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    gs = GridSearchCV(
        base_rf,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_tv, y_tv)

    print("Best params:", gs.best_params_)
    model = gs.best_estimator_
    # Refit on train only for threshold selection
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = choose_best_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    p10 = precision_at_k(y_test.values, test_prob, test_meta["Src IP"].values, k=10)
    p50 = precision_at_k(y_test.values, test_prob, test_meta["Src IP"].values, k=50)
    test_metrics["precision_at_10"] = p10
    test_metrics["precision_at_50"] = p50

    joblib.dump(model, MODELS_DIR / "rf_model.joblib")

    save_metrics(val_metrics, METRICS_DIR / "rf_val_metrics.json")
    save_metrics(test_metrics, METRICS_DIR / "rf_test_metrics.json")

    plot_confusion_matrix(val_metrics, FIGURES_DIR / "rf_val_confusion_matrix.png", "RF Val Confusion Matrix")
    plot_confusion_matrix(test_metrics, FIGURES_DIR / "rf_test_confusion_matrix.png", "RF Test Confusion Matrix")
    plot_pr_curve(y_val, val_prob, FIGURES_DIR / "rf_val_pr_curve.png", "RF Val PR Curve")
    plot_pr_curve(y_test, test_prob, FIGURES_DIR / "rf_test_pr_curve.png", "RF Test PR Curve")
    plot_roc_curve(y_val, val_prob, FIGURES_DIR / "rf_val_roc_curve.png", "RF Val ROC Curve")
    plot_roc_curve(y_test, test_prob, FIGURES_DIR / "rf_test_roc_curve.png", "RF Test ROC Curve")

    fi_df = save_feature_importance(
        list(X_train.columns), model.feature_importances_,
        METRICS_DIR / "rf_feature_importance.csv",
    )
    plot_feature_importance(fi_df, FIGURES_DIR / "rf_feature_importance.png", "RF Feature Importance")

    return {
        "best_params": gs.best_params_,
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


# ─────────────────────────────────────────────
# XGBoost with early stopping
# ─────────────────────────────────────────────
def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, test_meta):
    print("\n=== TRAIN XGBoost (early stopping) ===")

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight={scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration}")

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = choose_best_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    p10 = precision_at_k(y_test.values, test_prob, test_meta["Src IP"].values, k=10)
    p50 = precision_at_k(y_test.values, test_prob, test_meta["Src IP"].values, k=50)
    test_metrics["precision_at_10"] = p10
    test_metrics["precision_at_50"] = p50

    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")

    save_metrics(val_metrics, METRICS_DIR / "xgb_val_metrics.json")
    save_metrics(test_metrics, METRICS_DIR / "xgb_test_metrics.json")

    plot_confusion_matrix(val_metrics, FIGURES_DIR / "xgb_val_confusion_matrix.png", "XGB Val Confusion Matrix")
    plot_confusion_matrix(test_metrics, FIGURES_DIR / "xgb_test_confusion_matrix.png", "XGB Test Confusion Matrix")
    plot_pr_curve(y_val, val_prob, FIGURES_DIR / "xgb_val_pr_curve.png", "XGB Val PR Curve")
    plot_pr_curve(y_test, test_prob, FIGURES_DIR / "xgb_test_pr_curve.png", "XGB Test PR Curve")
    plot_roc_curve(y_val, val_prob, FIGURES_DIR / "xgb_val_roc_curve.png", "XGB Val ROC Curve")
    plot_roc_curve(y_test, test_prob, FIGURES_DIR / "xgb_test_roc_curve.png", "XGB Test ROC Curve")

    fi_df = save_feature_importance(
        list(X_train.columns), model.feature_importances_,
        METRICS_DIR / "xgb_feature_importance.csv",
    )
    plot_feature_importance(fi_df, FIGURES_DIR / "xgb_feature_importance.png", "XGB Feature Importance")

    return {
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": model.best_iteration,
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_meta, val_meta, test_meta,
    ) = load_split()

    print("=== DATA SHAPES ===")
    print(f"train: {X_train.shape}  attack={y_train.sum()}")
    print(f"val  : {X_val.shape}   attack={y_val.sum()}")
    print(f"test : {X_test.shape}  attack={y_test.sum()}")

    rf_result = train_rf(X_train, y_train, X_val, y_val, X_test, y_test, test_meta)
    xgb_result = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, test_meta)

    summary = {"rf": rf_result, "xgb": xgb_result}
    with open(METRICS_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, res in summary.items():
        tm = res["test_metrics"]
        print(
            f"{name.upper():5s}  F1={tm['f1']:.4f}  ROC={tm['roc_auc']:.4f}"
            f"  PR={tm['pr_auc']:.4f}  P@10={tm.get('precision_at_10','N/A')}"
        )
    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    main()
