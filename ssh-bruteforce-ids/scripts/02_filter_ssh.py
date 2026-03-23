from pathlib import Path
import pandas as pd

from src.data.load_csv import load_csv
from src.data.filter_ssh import filter_ssh_flows
from src.utils.io import ensure_dir, save_df

RAW_PATH = Path("data/raw/tuesday.csv")
OUT_PATH = Path("data/interim/tuesday_ssh_filtered.parquet")
REPORT_PATH = Path("outputs/reports/tuesday_ssh_filtered_summary.csv")


def main():
    ensure_dir(OUT_PATH.parent)
    ensure_dir(REPORT_PATH.parent)

    df = load_csv(RAW_PATH)
    ssh_df = filter_ssh_flows(df)

    summary = pd.DataFrame([
        {"metric": "rows_after_filter", "value": len(ssh_df)},
        {"metric": "benign_count", "value": int((ssh_df["target"] == 0).sum())},
        {"metric": "attack_count", "value": int((ssh_df["target"] == 1).sum())},
        {"metric": "benign_ratio", "value": float((ssh_df["target"] == 0).mean())},
        {"metric": "attack_ratio", "value": float((ssh_df["target"] == 1).mean())},
    ])

    save_df(ssh_df, OUT_PATH)
    save_df(summary, REPORT_PATH)

    print(f"Filtered rows: {len(ssh_df)}")
    print(summary)
    print(f"Saved filtered data to: {OUT_PATH}")
    print(f"Saved summary to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
