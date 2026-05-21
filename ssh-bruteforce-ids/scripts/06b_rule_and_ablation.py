from pathlib import Path
import json
import itertools
import pandas as pd

from xgboost import XGBClassifier

from src.models.evaluate import evaluate_binary_classifier, save_metrics

DATA_DIR = Path("data/processed/splits")
OUT_DIR = Path("outputs/metrics")


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


def build_xgb_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def run_rule_baseline(X_val, y_val, X_test, y_test):
    required = ["flow_rate_per_window", "interarrival_std"]
    missing = [c for c in required if c not in X_val.columns]
    if missing:
        raise ValueError(f"Missing required columns for rule baseline: {missing}")

    # Quét ngưỡng trên validation
    fr_values = sorted(set(X_val["flow_rate_per_window"].quantile([0.5, 0.6, 0.7, 0.8, 0.9]).round(6)))
    ia_values = sorted(set(X_val["interarrival_std"].quantile([0.1, 0.2, 0.3, 0.4, 0.5]).round(6)))

    best = None
    best_f1 = -1.0

    for fr_th, ia_th in itertools.product(fr_values, ia_values):
        y_val_pred = (
            (X_val["flow_rate_per_window"] >= fr_th) &
            (X_val["interarrival_std"] <= ia_th)
        ).astype(int)

        # rule output là nhị phân, convert thành pseudo-prob để dùng evaluator thống nhất
        y_val_prob = y_val_pred.astype(float).to_numpy()
        metrics = evaluate_binary_classifier(y_val, y_val_prob, threshold=0.5)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best = {
                "flow_rate_threshold": float(fr_th),
                "interarrival_std_threshold": float(ia_th),
                "val_metrics": metrics,
            }

    # Eval on test với ngưỡng tốt nhất
    y_test_pred = (
        (X_test["flow_rate_per_window"] >= best["flow_rate_threshold"]) &
        (X_test["interarrival_std"] <= best["interarrival_std_threshold"])
    ).astype(int)

    y_test_prob = y_test_pred.astype(float).to_numpy()
    test_metrics = evaluate_binary_classifier(y_test, y_test_prob, threshold=0.5)
    best["test_metrics"] = test_metrics

    return best


def run_xgb_ablation(
    X_train, y_train, X_val, y_val, X_test, y_test,
    drop_features: list[str],
    name: str,
):
    used_cols = [c for c in X_train.columns if c not in drop_features]

    X_train_sub = X_train[used_cols]
    X_val_sub = X_val[used_cols]
    X_test_sub = X_test[used_cols]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = build_xgb_model(scale_pos_weight=scale_pos_weight)
    model.fit(X_train_sub, y_train)

    val_prob = model.predict_proba(X_val_sub)[:, 1]
    best_threshold, _ = choose_best_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test_sub)[:, 1]

    val_metrics = evaluate_binary_classifier(y_val, val_prob, threshold=best_threshold)
    test_metrics = evaluate_binary_classifier(y_test, test_prob, threshold=best_threshold)

    result = {
        "name": name,
        "drop_features": drop_features,
        "n_features": len(used_cols),
        "best_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()

    print("=== RULE BASELINE ===")
    rule_result = run_rule_baseline(X_val, y_val, X_test, y_test)
    print(json.dumps(rule_result, indent=2))
    save_metrics(rule_result, OUT_DIR / "rule_baseline_metrics.json")

    print("\n=== XGB ABLATION: FULL ===")
    full_result = run_xgb_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test,
        drop_features=[],
        name="xgb_full"
    )
    print(json.dumps(full_result, indent=2))
    save_metrics(full_result, OUT_DIR / "xgb_ablation_full.json")

    print("\n=== XGB ABLATION: DROP flow_rate_per_window ===")
    drop_flow_rate_result = run_xgb_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test,
        drop_features=["flow_rate_per_window"],
        name="xgb_drop_flow_rate"
    )
    print(json.dumps(drop_flow_rate_result, indent=2))
    save_metrics(drop_flow_rate_result, OUT_DIR / "xgb_ablation_drop_flow_rate.json")

    print("\n=== XGB ABLATION: DROP flow_rate_per_window + interarrival_std ===")
    drop_two_result = run_xgb_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test,
        drop_features=["flow_rate_per_window", "interarrival_std"],
        name="xgb_drop_flow_rate_and_interarrival_std"
    )
    print(json.dumps(drop_two_result, indent=2))
    save_metrics(drop_two_result, OUT_DIR / "xgb_ablation_drop_two.json")

    summary = {
        "rule_baseline": rule_result,
        "xgb_full": full_result,
        "xgb_drop_flow_rate": drop_flow_rate_result,
        "xgb_drop_flow_rate_and_interarrival_std": drop_two_result,
    }

    with open(OUT_DIR / "rule_and_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Rule baseline and ablation results saved.")


if __name__ == "__main__":
    main()
