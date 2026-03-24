from pathlib import Path
import pandas as pd

from src.utils.io import ensure_dir, save_df
from src.features.feature_selector import select_features
from src.data.split_dataset import classwise_time_split
IN_PATH = Path("data/processed/tuesday_ssh_windows.parquet")
OUT_DIR = Path("data/processed/splits")


def main():
    ensure_dir(OUT_DIR)

    df = pd.read_parquet(IN_PATH)

    # ===== SPLIT =====
    train_df, val_df, test_df = classwise_time_split(df)

    print("=== SPLIT SHAPES ===")
    print("train:", train_df.shape)
    print("val  :", val_df.shape)
    print("test :", test_df.shape)

    # ===== FEATURE SELECT =====
    feature_cols = select_features(df)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # ===== SAVE =====
    save_df(X_train, OUT_DIR / "X_train.parquet")
    save_df(y_train.to_frame(), OUT_DIR / "y_train.parquet")

    save_df(X_val, OUT_DIR / "X_val.parquet")
    save_df(y_val.to_frame(), OUT_DIR / "y_val.parquet")

    save_df(X_test, OUT_DIR / "X_test.parquet")
    save_df(y_test.to_frame(), OUT_DIR / "y_test.parquet")

    # save raw splits with metadata for simulation / analysis
    save_df(train_df, OUT_DIR / "train_windows_with_meta.parquet")
    save_df(val_df, OUT_DIR / "val_windows_with_meta.parquet")
    save_df(test_df, OUT_DIR / "test_windows_with_meta.parquet")

    summary = pd.DataFrame([
        {"split": "train", "rows": len(train_df), "attack_ratio": y_train.mean(), "attack_count": int(y_train.sum())},
        {"split": "val", "rows": len(val_df), "attack_ratio": y_val.mean(), "attack_count": int(y_val.sum())},
        {"split": "test", "rows": len(test_df), "attack_ratio": y_test.mean(), "attack_count": int(y_test.sum())},
    ])
    save_df(summary, OUT_DIR / "split_summary.csv")

    print("\n=== SUMMARY ===")
    print(summary)

    print("\n=== FEATURE COUNT ===")
    print(len(feature_cols))

    print("\nDone. Data ready for training.")


if __name__ == "__main__":
    main()