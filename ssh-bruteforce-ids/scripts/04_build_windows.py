from pathlib import Path
import pandas as pd

from src.utils.io import ensure_dir, save_df
from src.features.window_aggregator import build_ip_time_windows

IN_PATH = Path("data/interim/tuesday_ssh_clean.parquet")
OUT_PATH = Path("data/processed/tuesday_ssh_windows.parquet")
REPORT_PATH = Path("outputs/reports/tuesday_ssh_windows_summary.csv")

WINDOW_SEC = 60
STEP_SEC = 60
MIN_FLOWS_PER_WINDOW = 1


def main():
    ensure_dir(OUT_PATH.parent)
    ensure_dir(REPORT_PATH.parent)

    df = pd.read_parquet(IN_PATH)

    required_cols = ["Src IP", "ParsedTime", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_windows: list[dict] = []

    for src_ip, ip_df in df.groupby("Src IP", sort=False):
        windows = build_ip_time_windows(
            ip_df=ip_df,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            min_flows_per_window=MIN_FLOWS_PER_WINDOW,
        )
        all_windows.extend(windows)

    windows_df = pd.DataFrame(all_windows)

    if len(windows_df) == 0:
        raise ValueError("No windows created. Check timestamp range or window config.")

    windows_df = windows_df.sort_values(["window_start", "Src IP"]).reset_index(drop=True)
    save_df(windows_df, OUT_PATH)

    summary = pd.DataFrame([
        {"metric": "window_sec", "value": WINDOW_SEC},
        {"metric": "step_sec", "value": STEP_SEC},
        {"metric": "min_flows_per_window", "value": MIN_FLOWS_PER_WINDOW},
        {"metric": "num_src_ip", "value": int(df["Src IP"].nunique())},
        {"metric": "num_windows", "value": int(len(windows_df))},
        {"metric": "attack_windows", "value": int((windows_df["target"] == 1).sum())},
        {"metric": "benign_windows", "value": int((windows_df["target"] == 0).sum())},
        {"metric": "attack_window_ratio", "value": float((windows_df["target"] == 1).mean())},
    ])
    save_df(summary, REPORT_PATH)

    print("Window build done.")
    print(f"Input flows: {df.shape}")
    print(f"Output windows: {windows_df.shape}")
    print(f"Unique Src IP: {df['Src IP'].nunique()}")
    print(f"Attack windows: {(windows_df['target'] == 1).sum()}")
    print(f"Benign windows: {(windows_df['target'] == 0).sum()}")
    print(f"Saved windows parquet: {OUT_PATH}")
    print(f"Saved summary: {REPORT_PATH}")


if __name__ == "__main__":
    main()
