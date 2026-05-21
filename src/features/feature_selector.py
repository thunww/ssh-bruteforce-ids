import pandas as pd

DROP_COLUMNS = [
    # metadata
    "Src IP",
    "window_start",
    "window_end",
    "target",

    # direct leakage
    "attack_flow_count",
    "benign_flow_count",
    "attack_ratio",

    # strong leakage / near-direct label
    "flow_count",

    # constant (always = window_sec, no information)
    "window_duration_sec",
]


def select_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    return feature_cols
