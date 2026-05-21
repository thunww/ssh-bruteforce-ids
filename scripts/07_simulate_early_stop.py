from pathlib import Path
import joblib
import pandas as pd

from _bootstrap import add_project_root
add_project_root()

from src.detection.risk_scoring import compute_risk_score
from src.detection.early_stop import EarlyStopDetector

DATA_DIR = Path("data/processed/splits")
MODEL_PATH = Path("models/xgb_model.joblib")


def main():
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["target"]

    test_meta_path = Path("data/processed/splits/test_windows_with_meta.parquet")
    if not test_meta_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/splits/test_windows_with_meta.parquet. "
            "Need to save test split with metadata in step 5."
        )

    test_meta = pd.read_parquet(test_meta_path).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = joblib.load(MODEL_PATH)

    detector = EarlyStopDetector()

    probs = model.predict_proba(X_test)[:, 1]

    results = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]
        meta = test_meta.iloc[i]

        comp = compute_risk_score(
            model_prob=float(probs[i]),
            flow_rate_per_window=float(row.get("flow_rate_per_window", 0.0)),
            interarrival_std=float(row.get("interarrival_std", 0.0)),
            rst_flow_ratio=float(row.get("rst_flow_ratio", 0.0)),
            short_flow_ratio=float(row.get("short_flow_ratio", 0.0)),
        )

        decision = detector.decide(
            src_ip=str(meta["Src IP"]),
            now=pd.to_datetime(meta["window_start"]).to_pydatetime(),
            risk_score=float(comp["risk_score"]),
        )

        results.append({
            "Src IP": meta["Src IP"],
            "window_start": meta["window_start"],
            "target": int(y_test.iloc[i]),
            "model_prob": float(probs[i]),
            "risk_score": float(comp["risk_score"]),
            "action": decision["action"],
            "consecutive_suspicious": int(decision["consecutive_suspicious"]),
        })

    result_df = pd.DataFrame(results)
    out_path = Path("outputs/metrics/early_stop_simulation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print("Simulation done.")
    print(result_df.head(30))
    print("\nAction counts:")
    print(result_df["action"].value_counts())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
