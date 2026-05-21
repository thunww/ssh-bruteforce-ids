from pathlib import Path
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

DATA_PATH = Path("data/processed/tuesday_ssh_windows.parquet")
OUT_MODEL = Path("models/xgb_realtime_model.joblib")
OUT_META = Path("models/xgb_realtime_features.json")

REALTIME_FEATURES = [
    "flow_rate_per_window",
    "interarrival_mean",
    "interarrival_std",
    "rst_flow_ratio",
    "short_flow_ratio",
]


def classwise_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    df = df.sort_values("window_start").reset_index(drop=True)

    attack_df = df[df["target"] == 1].sort_values("window_start").reset_index(drop=True)
    benign_df = df[df["target"] == 0].sort_values("window_start").reset_index(drop=True)

    def _split_one(sub_df: pd.DataFrame):
        n = len(sub_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return sub_df.iloc[:train_end], sub_df.iloc[train_end:val_end], sub_df.iloc[val_end:]

    atk_train, atk_val, atk_test = _split_one(attack_df)
    ben_train, ben_val, ben_test = _split_one(benign_df)

    train_df = pd.concat([atk_train, ben_train]).sort_values("window_start").reset_index(drop=True)
    val_df = pd.concat([atk_val, ben_val]).sort_values("window_start").reset_index(drop=True)
    test_df = pd.concat([atk_test, ben_test]).sort_values("window_start").reset_index(drop=True)

    return train_df, val_df, test_df


def eval_at_threshold(y_true, y_prob, th=0.1):
    y_pred = (y_prob >= th).astype(int)
    return {
        "threshold": th,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def main():
    df = pd.read_parquet(DATA_PATH)

    missing = [c for c in REALTIME_FEATURES + ["target", "window_start"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    train_df, val_df, test_df = classwise_time_split(df)

    X_train = train_df[REALTIME_FEATURES]
    y_train = train_df["target"]

    X_val = val_df[REALTIME_FEATURES]
    y_val = val_df["target"]

    X_test = test_df[REALTIME_FEATURES]
    y_test = test_df["target"]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics = eval_at_threshold(y_val, val_prob, th=0.1)
    test_metrics = eval_at_threshold(y_test, test_prob, th=0.1)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODEL)

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": REALTIME_FEATURES,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved realtime model:", OUT_MODEL)
    print("Saved feature meta:", OUT_META)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()