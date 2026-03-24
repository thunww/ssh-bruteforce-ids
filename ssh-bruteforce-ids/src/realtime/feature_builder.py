import numpy as np
import pandas as pd

REALTIME_FEATURES = [
    "flow_rate_per_window",
    "interarrival_mean",
    "interarrival_std",
    "rst_flow_ratio",
    "short_flow_ratio",
]


def build_realtime_features(event_times, rst_flags=None, short_flags=None, window_sec=60):
    if not event_times:
        return None

    ts = pd.Series(pd.to_datetime(list(event_times))).sort_values().reset_index(drop=True)

    if len(ts) == 1:
        interarrival_mean = 0.0
        interarrival_std = 0.0
    else:
        diffs = ts.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        interarrival_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        interarrival_std = float(np.std(diffs)) if len(diffs) > 0 else 0.0

    duration = (ts.max() - ts.min()).total_seconds()
    flow_rate_per_window = len(ts) / max(window_sec, 1)

    if rst_flags is None:
        rst_flow_ratio = 0.0
    else:
        rst_arr = np.asarray(list(rst_flags), dtype=float)
        rst_flow_ratio = float(np.mean(rst_arr > 0)) if len(rst_arr) > 0 else 0.0

    if short_flags is None:
        short_flow_ratio = 0.0
    else:
        short_arr = np.asarray(list(short_flags), dtype=float)
        short_flow_ratio = float(np.mean(short_arr > 0)) if len(short_arr) > 0 else 0.0

    return {
        "flow_rate_per_window": flow_rate_per_window,
        "interarrival_mean": interarrival_mean,
        "interarrival_std": interarrival_std,
        "rst_flow_ratio": rst_flow_ratio,
        "short_flow_ratio": short_flow_ratio,
    }