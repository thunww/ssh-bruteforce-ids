#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/alerts.jsonl")
    if not path.exists():
        print(f"alerts_file={path}")
        print("total=0")
        return 0

    actions: Counter[str] = Counter()
    by_ip: dict[str, Counter[str]] = defaultdict(Counter)
    risks: list[float] = []
    probs: list[float] = []
    max_events = 0
    bad_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            action = str(row.get("action", "UNKNOWN"))
            ip = str(row.get("ip") or row.get("src_ip") or "UNKNOWN")
            actions[action] += 1
            by_ip[ip][action] += 1
            if "risk_score" in row:
                risks.append(float(row["risk_score"]))
            if "model_prob" in row:
                probs.append(float(row["model_prob"]))
            if "event_count" in row:
                max_events = max(max_events, int(row["event_count"]))

    print(f"alerts_file={path}")
    print(f"total={sum(actions.values())}")
    print("actions=" + json.dumps(dict(actions), sort_keys=True))
    print(f"bad_lines={bad_lines}")
    if risks:
        print(f"risk_min={min(risks):.4f}")
        print(f"risk_max={max(risks):.4f}")
    if probs:
        print(f"model_prob_min={min(probs):.4f}")
        print(f"model_prob_max={max(probs):.4f}")
    print(f"max_event_count={max_events}")
    print("by_ip=" + json.dumps({ip: dict(c) for ip, c in by_ip.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
