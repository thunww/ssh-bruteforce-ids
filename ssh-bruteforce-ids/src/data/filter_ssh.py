import pandas as pd

ALLOWED_LABELS = {"BENIGN", "SSH-Patator", "SSH-Patator - Attempted"}


def filter_ssh_flows(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Label", "Dst Port", "Protocol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[
        (df["Label"].isin(ALLOWED_LABELS)) &
        (df["Dst Port"] == 22) &
        (df["Protocol"] == 6)
    ].copy()

    out["target"] = out["Label"].map({
        "BENIGN": 0,
        "SSH-Patator": 1,
        "SSH-Patator - Attempted": 1
    })

    return out