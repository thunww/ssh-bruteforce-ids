from pathlib import Path
import json
import joblib
import pandas as pd

from src.models.train_rf import build_rf_model
from src.models.train_xgb import build_xgb_model
from src.models.evaluate import (
    evaluate_binary_classifier,
    save_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
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

    return X_train, y_train, X_val, y_val, X_test, y_test


def choose_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = -1.0

    for th in [i / 100 for i in range(10, 91, 5)]:
        metrics = evaluate_binary_classifier(y_true, y_prob, threshold=th)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = th

    return best_threshold, best_f1


def train_and_eval_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    model = build_rf_model()
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = choose_best_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    joblib.dump(model, MODELS_DIR / "rf_model.joblib")

    save_metrics(val_metrics, METRICS_DIR / "rf_val_metrics.json")
    save_metrics(test_metrics, METRICS_DIR / "rf_test_metrics.json")

    plot_confusion_matrix(val_metrics, FIGURES_DIR / "rf_val_confusion_matrix.png", "RF Validation Confusion Matrix")
    plot_confusion_matrix(test_metrics, FIGURES_DIR / "rf_test_confusion_matrix.png", "RF Test Confusion Matrix")

    plot_pr_curve(y_val, val_prob, FIGURES_DIR / "rf_val_pr_curve.png", "RF Validation PR Curve")
    plot_pr_curve(y_test, test_prob, FIGURES_DIR / "rf_test_pr_curve.png", "RF Test PR Curve")

    fi_df = save_feature_importance(
        list(X_train.columns),
        model.feature_importances_,
        METRICS_DIR / "rf_feature_importance.csv",
    )
    plot_feature_importance(fi_df, FIGURES_DIR / "rf_feature_importance.png", "RF Feature Importance")

    return {
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def train_and_eval_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = build_xgb_model(scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = choose_best_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")

    save_metrics(val_metrics, METRICS_DIR / "xgb_val_metrics.json")
    save_metrics(test_metrics, METRICS_DIR / "xgb_test_metrics.json")

    plot_confusion_matrix(val_metrics, FIGURES_DIR / "xgb_val_confusion_matrix.png", "XGB Validation Confusion Matrix")
    plot_confusion_matrix(test_metrics, FIGURES_DIR / "xgb_test_confusion_matrix.png", "XGB Test Confusion Matrix")

    plot_pr_curve(y_val, val_prob, FIGURES_DIR / "xgb_val_pr_curve.png", "XGB Validation PR Curve")
    plot_pr_curve(y_test, test_prob, FIGURES_DIR / "xgb_test_pr_curve.png", "XGB Test PR Curve")

    fi_df = save_feature_importance(
        list(X_train.columns),
        model.feature_importances_,
        METRICS_DIR / "xgb_feature_importance.csv",
    )
    plot_feature_importance(fi_df, FIGURES_DIR / "xgb_feature_importance.png", "XGB Feature Importance")

    return {
        "scale_pos_weight": scale_pos_weight,
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()

    print("=== DATA SHAPES ===")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    print("\n=== TRAIN RF ===")
    rf_result = train_and_eval_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    print(json.dumps(rf_result, indent=2))

    print("\n=== TRAIN XGB ===")
    xgb_result = train_and_eval_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    print(json.dumps(xgb_result, indent=2))

    summary = {
        "rf": rf_result,
        "xgb": xgb_result,
    }
    with open(METRICS_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Models, metrics, and figures saved.")


if __name__ == "__main__":
    main()
