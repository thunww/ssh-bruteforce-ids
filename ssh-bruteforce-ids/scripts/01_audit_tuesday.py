from pathlib import Path
import pandas as pd

from src.data.load_csv import load_csv
from src.utils.io import ensure_dir, save_df

RAW_PATH = Path("data/raw/tuesday.csv")
REPORT_DIR = Path("outputs/reports")


def main():
    ensure_dir(REPORT_DIR)

    df = load_csv(RAW_PATH)

    print(f"Rows: {len(df)}")
    print(f"Cols: {len(df.columns)}")
    print("Columns:")
    for c in df.columns:
        print(f" - {c}")

    audit_rows = []
    for col in df.columns:
        audit_rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_ratio": float(df[col].isna().mean())
        })

    audit_df = pd.DataFrame(audit_rows)
    save_df(audit_df, REPORT_DIR / "tuesday_initial_audit.csv")

    if "Label" in df.columns:
        label_summary = (
            df["Label"]
            .value_counts(dropna=False)
            .rename_axis("Label")
            .reset_index(name="count")
        )
        save_df(label_summary, REPORT_DIR / "tuesday_label_summary.csv")

    if "Dst Port" in df.columns:
        port22_df = df[df["Dst Port"] == 22].copy()

        if "Label" in port22_df.columns:
            port22_label_summary = (
                port22_df["Label"]
                .value_counts(dropna=False)
                .rename_axis("Label")
                .reset_index(name="count")
            )
            save_df(port22_label_summary, REPORT_DIR / "tuesday_label_port22_summary.csv")

        port_summary = (
            df["Dst Port"]
            .value_counts(dropna=False)
            .head(50)
            .rename_axis("Dst Port")
            .reset_index(name="count")
        )
        save_df(port_summary, REPORT_DIR / "tuesday_top_dst_ports.csv")

    print("Audit done.")
    print(f"Saved reports to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
