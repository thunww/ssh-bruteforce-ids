from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd

from src.detection.risk_scoring import compute_risk_score
from src.detection.early_stop import EarlyStopDetector
from src.realtime.collector import collect_failed_ssh_events_journalctl
from src.realtime.feature_builder import build_realtime_features, REALTIME_FEATURES
from src.realtime.blocker import block_ip_iptables
from src.realtime.notifier import send_telegram, telegram_is_enabled

MODEL_PATH = Path("models/xgb_realtime_model.joblib")
META_PATH = Path("models/xgb_realtime_features.json")

WINDOW_SEC = 60
POLL_SEC = 5
MODEL_THRESHOLD = 0.05

event_buffers = defaultdict(lambda: deque())
last_sent_state = {}  # ip -> last action sent to telegram


def trim_old_events(buf: deque, now: pd.Timestamp, window_sec: int):
    while buf and (now - buf[0]["Timestamp"]).total_seconds() > window_sec:
        buf.popleft()


def maybe_notify(ip: str, action: str, model_prob: float, risk_score: float, events: int):
    """
    Tránh spam Telegram: chỉ gửi khi action đổi trạng thái đáng kể.
    Gửi cho ALERT và BLOCK.
    """
    if action not in {"ALERT", "BLOCK"}:
        return

    prev = last_sent_state.get(ip)
    if prev == action:
        return

    icon = "⚠️" if action == "ALERT" else "🚫"
    msg = (
        f"{icon} SSH IDS {action}\n"
        f"IP: {ip}\n"
        f"Events(60s): {events}\n"
        f"Model prob: {model_prob:.3f}\n"
        f"Risk score: {risk_score:.3f}"
    )
    ok = send_telegram(msg)
    if ok:
        last_sent_state[ip] = action


def main():
    model = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected_features = meta["features"]
    else:
        expected_features = REALTIME_FEATURES

    detector = EarlyStopDetector()

    print("=== REALTIME SSH DETECTOR STARTED ===")
    print("Expected features:", expected_features)
    print("Telegram enabled:", telegram_is_enabled())

    while True:
        now = pd.Timestamp.now().tz_localize(None)
        events = collect_failed_ssh_events_journalctl(since=f"{POLL_SEC + 1} seconds ago")

        for ev in events:
            ip = ev["Src IP"]
            event_buffers[ip].append(ev)

        all_ips = list(event_buffers.keys())

        for ip in all_ips:
            trim_old_events(event_buffers[ip], now, WINDOW_SEC)

            if len(event_buffers[ip]) == 0:
                continue

            timestamps = [x["Timestamp"] for x in event_buffers[ip]]
            rst_flags = [x.get("rst_flag", 0) for x in event_buffers[ip]]
            short_flags = [x.get("short_flag", 1) for x in event_buffers[ip]]

            feats = build_realtime_features(
                event_times=timestamps,
                rst_flags=rst_flags,
                short_flags=short_flags,
                window_sec=WINDOW_SEC,
            )

            if feats is None:
                continue

            X = pd.DataFrame([feats])[expected_features]
            model_prob = float(model.predict_proba(X)[0][1])

            risk = compute_risk_score(
                model_prob=model_prob,
                flow_rate_per_window=float(feats["flow_rate_per_window"]),
                interarrival_std=float(feats["interarrival_std"]),
                rst_flow_ratio=float(feats["rst_flow_ratio"]),
                short_flow_ratio=float(feats["short_flow_ratio"]),
            )

            decision = detector.decide(
                src_ip=ip,
                now=now.to_pydatetime(),
                risk_score=float(risk["risk_score"]),
            )

            action = decision["action"]

            print(
                f"[{now}] ip={ip} events={len(timestamps)} "
                f"p={model_prob:.3f} risk={risk['risk_score']:.3f} action={action}"
            )

            if action == "ALERT":
                maybe_notify(
                    ip=ip,
                    action=action,
                    model_prob=model_prob,
                    risk_score=float(risk["risk_score"]),
                    events=len(timestamps),
                )

            if action == "BLOCK":
                block_ip_iptables(ip)
                print(f"[BLOCK] iptables DROP added for {ip}")
                maybe_notify(
                    ip=ip,
                    action=action,
                    model_prob=model_prob,
                    risk_score=float(risk["risk_score"]),
                    events=len(timestamps),
                )

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
