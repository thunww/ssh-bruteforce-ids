from __future__ import annotations

import numpy as np
import pandas as pd


DROP_IMMEDIATE = [
    "id",
    "Flow ID",
    "Src Port",
    "Dst IP",
    "Attempted Category",
    "Column1",
    "ICMP Code",
    "ICMP Type",
    "Dst Port",
    "Protocol",
]

METADATA_KEEP = [
    "Src IP",
    "Timestamp",
    "ParsedTime",
    "Label",
    "target",
]


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Timestamp" not in out.columns:
        raise ValueError("Missing Timestamp column")

    ts_raw = out["Timestamp"].astype(str).str.strip()

    # Thử parse theo dayfirst trước vì dữ liệu kiểu 04/07/2017 11:59:40
    parsed = pd.to_datetime(ts_raw, errors="coerce", dayfirst=True)

    # Fallback nếu còn fail
    if parsed.isna().any():
        fallback_mask = parsed.isna()
        parsed_fallback = pd.to_datetime(ts_raw[fallback_mask], errors="coerce", dayfirst=False)
        parsed.loc[fallback_mask] = parsed_fallback

    out["ParsedTime"] = parsed
    return out


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def drop_immediate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    to_drop = [c for c in DROP_IMMEDIATE if c in out.columns]
    out = out.drop(columns=to_drop, errors="ignore")
    return out, to_drop


def find_constant_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    const_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].nunique(dropna=False) <= 1:
            const_cols.append(col)

    return const_cols


def find_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.95,
    exclude: list[str] | None = None
) -> list[str]:
    exclude = exclude or []
    cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].isna().mean() >= threshold:
            cols.append(col)

    return cols


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col in METADATA_KEEP:
            continue
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def clean_ssh_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    report = {}

    out = parse_timestamp_column(out)
    report["timestamp_parse_failed"] = int(out["ParsedTime"].isna().sum())
    report["rows_before_timestamp_filter"] = int(len(out))

    out = out[out["ParsedTime"].notna()].copy()
    out = out.sort_values("ParsedTime").reset_index(drop=True)

    report["rows_after_timestamp_filter"] = int(len(out))
    report["dropped_rows_invalid_timestamp"] = (
        report["rows_before_timestamp_filter"] - report["rows_after_timestamp_filter"]
    )

    out, dropped_immediate = drop_immediate_columns(out)
    report["dropped_immediate"] = dropped_immediate

    out = replace_inf_with_nan(out)
    out = coerce_numeric_columns(out)

    high_missing = find_high_missing_columns(out, threshold=0.95, exclude=METADATA_KEEP)
    out = out.drop(columns=high_missing, errors="ignore")
    report["dropped_high_missing"] = high_missing

    constant_cols = find_constant_columns(out, exclude=METADATA_KEEP)
    out = out.drop(columns=constant_cols, errors="ignore")
    report["dropped_constant"] = constant_cols

    return out, report