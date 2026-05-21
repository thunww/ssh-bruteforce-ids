from __future__ import annotations

import numpy as np
import pandas as pd


WINDOW_NUMERIC_COLUMNS = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packet": "total_fwd_packet",
    "Total Bwd packets": "total_bwd_packet",
    "Total Length of Fwd Packet": "fwd_bytes",
    "Total Length of Bwd Packet": "bwd_bytes",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "SYN Flag Count": "syn_flag",
    "RST Flag Count": "rst_flag",
    "ACK Flag Count": "ack_flag",
    "PSH Flag Count": "psh_flag",
    "Average Packet Size": "avg_packet_size",
    "Packet Length Std": "packet_length_std",
    "Down/Up Ratio": "down_up_ratio",
}


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    value = series.std(ddof=0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    value = series.mean()
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_min(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    value = series.min()
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_max(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    value = series.max()
    if pd.isna(value):
        return 0.0
    return float(value)


def _compute_interarrivals(time_series: pd.Series) -> np.ndarray:
    if len(time_series) <= 1:
        return np.array([], dtype=float)

    ts = pd.to_datetime(time_series).sort_values()
    diffs = ts.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    return diffs


def aggregate_window_features(
    window_df: pd.DataFrame,
    src_ip: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    short_flow_threshold: float = 1.0,
    high_rate_threshold: float = 100.0,
) -> dict:
    row: dict = {}

    # metadata
    row["Src IP"] = src_ip
    row["window_start"] = window_start
    row["window_end"] = window_end
    row["window_duration_sec"] = float((window_end - window_start).total_seconds())

    # label / counts
    row["flow_count"] = int(len(window_df))
    row["attack_flow_count"] = int((window_df["target"] == 1).sum())
    row["benign_flow_count"] = int((window_df["target"] == 0).sum())
    row["target"] = 1 if row["attack_flow_count"] >= 1 else 0
    row["attack_ratio"] = (
        row["attack_flow_count"] / row["flow_count"] if row["flow_count"] > 0 else 0.0
    )

    # flow rate in window
    row["flow_rate_per_window"] = (
        row["flow_count"] / row["window_duration_sec"] if row["window_duration_sec"] > 0 else 0.0
    )

    # inter-arrivals
    interarrivals = _compute_interarrivals(window_df["ParsedTime"])
    if len(interarrivals) == 0:
        row["interarrival_mean"] = 0.0
        row["interarrival_std"] = 0.0
        row["interarrival_min"] = 0.0
        row["interarrival_max"] = 0.0
    else:
        row["interarrival_mean"] = float(np.mean(interarrivals))
        row["interarrival_std"] = float(np.std(interarrivals))
        row["interarrival_min"] = float(np.min(interarrivals))
        row["interarrival_max"] = float(np.max(interarrivals))

    # per-column aggregates
    for raw_col, prefix in WINDOW_NUMERIC_COLUMNS.items():
        if raw_col not in window_df.columns:
            row[f"{prefix}_mean"] = 0.0
            row[f"{prefix}_std"] = 0.0
            row[f"{prefix}_min"] = 0.0
            row[f"{prefix}_max"] = 0.0
            continue

        series = pd.to_numeric(window_df[raw_col], errors="coerce").dropna()
        row[f"{prefix}_mean"] = _safe_mean(series)
        row[f"{prefix}_std"] = _safe_std(series)
        row[f"{prefix}_min"] = _safe_min(series)
        row[f"{prefix}_max"] = _safe_max(series)

    # derived ratios
    duration_series = pd.to_numeric(window_df["Flow Duration"], errors="coerce").fillna(0.0)
    row["short_flow_ratio"] = float((duration_series <= short_flow_threshold).mean())

    rst_series = pd.to_numeric(window_df["RST Flag Count"], errors="coerce").fillna(0.0)
    row["rst_flow_ratio"] = float((rst_series > 0).mean())

    rate_series = pd.to_numeric(window_df["Flow Packets/s"], errors="coerce").fillna(0.0)
    row["high_rate_flow_ratio"] = float((rate_series >= high_rate_threshold).mean())

    return row


def build_ip_time_windows(
    ip_df: pd.DataFrame,
    window_sec: int = 60,
    step_sec: int = 10,
    min_flows_per_window: int = 1,
) -> list[dict]:
    if len(ip_df) == 0:
        return []

    ip_df = ip_df.sort_values("ParsedTime").reset_index(drop=True)

    src_ip = str(ip_df["Src IP"].iloc[0])
    t_min = ip_df["ParsedTime"].min()
    t_max = ip_df["ParsedTime"].max()

    windows: list[dict] = []
    current_start = t_min
    delta_window = pd.Timedelta(seconds=window_sec)
    delta_step = pd.Timedelta(seconds=step_sec)

    while current_start <= t_max:
        current_end = current_start + delta_window

        mask = (ip_df["ParsedTime"] >= current_start) & (ip_df["ParsedTime"] < current_end)
        window_df = ip_df.loc[mask].copy()

        if len(window_df) >= min_flows_per_window:
            row = aggregate_window_features(
                window_df=window_df,
                src_ip=src_ip,
                window_start=current_start,
                window_end=current_end,
            )
            windows.append(row)

        current_start = current_start + delta_step

    return windows
