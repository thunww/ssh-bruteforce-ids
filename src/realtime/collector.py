from __future__ import annotations

import re
import subprocess
import pandas as pd


FAILED_RE = re.compile(r"Failed password.*from ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)")
INVALID_RE = re.compile(r"Invalid user .* from ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)")


def _to_naive_timestamp(ts_text: str) -> pd.Timestamp:
    ts = pd.to_datetime(ts_text, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now().tz_localize(None)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def collect_failed_ssh_events_journalctl(since: str = "10 seconds ago") -> list[dict]:
    cmd = [
        "journalctl",
        "-u",
        "ssh",
        "--since",
        since,
        "--no-pager",
        "-o",
        "short-iso",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.splitlines()

    events = []
    for line in lines:
        ip = None
        m1 = FAILED_RE.search(line)
        m2 = INVALID_RE.search(line)
        if m1:
            ip = m1.group(1)
        elif m2:
            ip = m2.group(1)

        if ip is None:
            continue

        ts_text = line[:25].strip()
        ts = _to_naive_timestamp(ts_text)

        events.append({
            "Src IP": ip,
            "Timestamp": ts,
            "rst_flag": 0,
            "short_flag": 1,
            "raw": line,
        })

    return events