from pathlib import Path
import json
import pandas as pd

from src.utils.io import ensure_dir, save_df
from src.data.clean_data import clean_ssh_dataframe

IN_PATH = Path("data/interim/tuesday_ssh_filtered.parquet")
OUT_PATH = Path("data/interim/tuesday_ssh_clean.parquet")
REPORT_DIR = Path("outputs/reports")


def main():
    ensure_dir(OUT_PATH.parent)
    ensure_dir(REPORT_DIR)

    df = pd.read_parquet(IN_PATH)
    clean_df, report = clean_ssh_dataframe(df)

    save_df(clean_df, OUT_PATH)

    with open(REPORT_DIR / "tuesday_ssh_clean_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    col_rows = []
    for col in clean_df.columns:
        col_rows.append({
            "column": col,
            "dtype": str(clean_df[col].dtype),
            "missing_count": int(clean_df[col].isna().sum()),
            "missing_ratio": float(clean_df[col].isna().mean()),
            "nunique": int(clean_df[col].nunique(dropna=False)),
        })
    col_df = pd.DataFrame(col_rows)
    save_df(col_df, REPORT_DIR / "tuesday_ssh_clean_columns.csv")

    summary = pd.DataFrame([
        {"metric": "rows_input", "value": len(df)},
        {"metric": "rows_clean", "value": len(clean_df)},
        {"metric": "cols_clean", "value": len(clean_df.columns)},
        {"metric": "attack_count", "value": int((clean_df["target"] == 1).sum())},
        {"metric": "benign_count", "value": int((clean_df["target"] == 0).sum())},
        {"metric": "timestamp_parse_failed", "value": report["timestamp_parse_failed"]},
        {"metric": "dropped_rows_invalid_timestamp", "value": report["dropped_rows_invalid_timestamp"]},
    ])
    save_df(summary, REPORT_DIR / "tuesday_ssh_clean_summary.csv")

    print("Clean done.")
    print(f"Input shape : {df.shape}")
    print(f"Output shape: {clean_df.shape}")
    print("Dropped immediate:", report["dropped_immediate"])
    print("Dropped high missing:", report["dropped_high_missing"])
    print("Dropped constant:", report["dropped_constant"])
    print("Timestamp parse failed:", report["timestamp_parse_failed"])
    print("Dropped rows invalid timestamp:", report["dropped_rows_invalid_timestamp"])
    print("Saved clean parquet:", OUT_PATH)
    print("Saved reports in:", REPORT_DIR)


if __name__ == "__main__":
    main()