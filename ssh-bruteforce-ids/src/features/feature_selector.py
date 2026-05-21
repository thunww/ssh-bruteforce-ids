import pandas as pd

DROP_COLUMNS = [
    # metadata
    "Src IP",
    "window_start",
    "window_end",
    "target",

    # leakage trực tiếp
    "attack_flow_count",
    "benign_flow_count",
    "attack_ratio",

    # leakage mạnh / gần-direct label
    "flow_count",
]

def select_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    return feature_cols