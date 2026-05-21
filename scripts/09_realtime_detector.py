"""
Real-time SSH brute-force detector — kiến trúc Producer-Consumer đa luồng.

Luồng dữ liệu:
  Producer thread  →  event_queue  →  Consumer thread  →  inference_queue
  →  Worker pool (ThreadPoolExecutor, 4 workers)  →  Blocker / Alert

Mỗi 30s: monitor thread log CPU/RAM overhead.
Alerts ghi ra outputs/alerts.jsonl (append).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import joblib
import pandas as pd
import psutil

from src.detection.risk_scoring import compute_risk_score
from src.detection.early_stop import EarlyStopDetector
from src.realtime.collector import collect_failed_ssh_events_journalctl
from src.realtime.feature_builder import build_realtime_features, REALTIME_FEATURES
from src.realtime.blocker import block_ip_iptables

# ──────────────────────────────────────────────
# Config (overridable via env vars)
# ──────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("IDS_MODEL_PATH", "models/xgb_realtime_model.joblib"))
META_PATH = Path(os.getenv("IDS_META_PATH", "models/xgb_realtime_features.json"))
ALERTS_PATH = Path(os.getenv("IDS_ALERTS_PATH", "outputs/alerts.jsonl"))

WINDOW_SEC = int(os.getenv("IDS_WINDOW_SEC", "60"))
POLL_SEC = int(os.getenv("IDS_POLL_SEC", "5"))
WORKER_THREADS = int(os.getenv("IDS_WORKERS", "4"))
MONITOR_INTERVAL = int(os.getenv("IDS_MONITOR_INTERVAL", "30"))

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ids")

# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
event_queue: Queue = Queue(maxsize=10_000)
inference_queue: Queue = Queue(maxsize=1_000)

# Per-IP sliding window buffer — protected by a lock
_buffer_lock = threading.Lock()
event_buffers: dict[str, deque] = defaultdict(deque)

detector = EarlyStopDetector()
_detector_lock = threading.Lock()

_stop_event = threading.Event()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _trim_old_events(buf: deque, now: pd.Timestamp, window_sec: int) -> None:
    while buf and (now - buf[0]["Timestamp"]).total_seconds() > window_sec:
        buf.popleft()


def _write_alert(record: dict) -> None:
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ──────────────────────────────────────────────
# Producer thread: poll journalctl → event_queue
# ──────────────────────────────────────────────
def producer_thread() -> None:
    log.info("Producer started (poll_sec=%d)", POLL_SEC)
    while not _stop_event.is_set():
        try:
            events = collect_failed_ssh_events_journalctl(
                since=f"{POLL_SEC + 1} seconds ago"
            )
            for ev in events:
                event_queue.put(ev, block=False)
        except Exception as exc:
            log.warning("Producer error: %s", exc)
        time.sleep(POLL_SEC)
    log.info("Producer stopped")


# ──────────────────────────────────────────────
# Consumer thread: event_queue → per-IP buffers → inference_queue
# ──────────────────────────────────────────────
def consumer_thread() -> None:
    log.info("Consumer started (window_sec=%d)", WINDOW_SEC)
    while not _stop_event.is_set():
        # Drain all available events first
        drained = False
        while True:
            try:
                ev = event_queue.get(timeout=1.0)
                ip = ev["Src IP"]
                with _buffer_lock:
                    event_buffers[ip].append(ev)
                drained = True
            except Empty:
                break

        now = pd.Timestamp.now().tz_localize(None)

        with _buffer_lock:
            all_ips = list(event_buffers.keys())

        for ip in all_ips:
            with _buffer_lock:
                _trim_old_events(event_buffers[ip], now, WINDOW_SEC)
                buf_snapshot = list(event_buffers[ip])

            if not buf_snapshot:
                continue

            timestamps = [x["Timestamp"] for x in buf_snapshot]
            rst_flags = [x.get("rst_flag", 0) for x in buf_snapshot]
            short_flags = [x.get("short_flag", 1) for x in buf_snapshot]

            feats = build_realtime_features(
                event_times=timestamps,
                rst_flags=rst_flags,
                short_flags=short_flags,
                window_sec=WINDOW_SEC,
            )
            if feats is not None:
                inference_queue.put({
                    "ip": ip,
                    "now": now,
                    "feats": feats,
                    "event_count": len(buf_snapshot),
                })

        if not drained:
            time.sleep(0.5)

    log.info("Consumer stopped")


# ──────────────────────────────────────────────
# Worker: inference on one IP task
# ──────────────────────────────────────────────
def run_inference(task: dict, model, expected_features: list[str]) -> dict:
    ip = task["ip"]
    now = task["now"]
    feats = task["feats"]

    X = pd.DataFrame([feats])[expected_features]
    model_prob = float(model.predict_proba(X)[0][1])

    risk = compute_risk_score(
        model_prob=model_prob,
        flow_rate_per_window=float(feats["flow_rate_per_window"]),
        interarrival_std=float(feats["interarrival_std"]),
        rst_flow_ratio=float(feats["rst_flow_ratio"]),
        short_flow_ratio=float(feats["short_flow_ratio"]),
    )

    with _detector_lock:
        decision = detector.decide(
            src_ip=ip,
            now=now.to_pydatetime(),
            risk_score=float(risk["risk_score"]),
        )

    return {
        "ip": ip,
        "now": str(now),
        "event_count": task["event_count"],
        "model_prob": round(model_prob, 4),
        "risk_score": round(float(risk["risk_score"]), 4),
        "action": decision["action"],
        "consecutive_suspicious": decision["consecutive_suspicious"],
    }


# ──────────────────────────────────────────────
# Worker pool thread
# ──────────────────────────────────────────────
def worker_pool_thread(model, expected_features: list[str]) -> None:
    log.info("Worker pool started (workers=%d)", WORKER_THREADS)
    with ThreadPoolExecutor(max_workers=WORKER_THREADS, thread_name_prefix="worker") as pool:
        while not _stop_event.is_set():
            tasks = []
            # Batch up to WORKER_THREADS tasks
            while len(tasks) < WORKER_THREADS:
                try:
                    task = inference_queue.get(timeout=0.5)
                    tasks.append(task)
                except Empty:
                    break

            if not tasks:
                continue

            futures = {
                pool.submit(run_inference, t, model, expected_features): t
                for t in tasks
            }
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    _handle_result(result)
                except Exception as exc:
                    log.error("Inference error: %s", exc)

    log.info("Worker pool stopped")


def _handle_result(result: dict) -> None:
    action = result["action"]
    ip = result["ip"]

    log_line = (
        f"ip={ip} events={result['event_count']} "
        f"p={result['model_prob']:.3f} risk={result['risk_score']:.3f} "
        f"action={action}"
    )

    if action == "NORMAL":
        log.debug(log_line)
        return

    if action == "ALERT":
        log.warning("ALERT  | %s", log_line)
    elif action in ("BLOCK", "BLOCKED"):
        log.error("BLOCK  | %s", log_line)

    _write_alert(result)

    if action == "BLOCK":
        block_ip_iptables(ip)
        log.error("[BLOCK] iptables DROP added for %s", ip)


# ──────────────────────────────────────────────
# Monitor thread: log CPU/RAM every 30s
# ──────────────────────────────────────────────
def monitor_thread() -> None:
    proc = psutil.Process()
    log.info("Monitor started (interval=%ds)", MONITOR_INTERVAL)
    while not _stop_event.is_set():
        cpu = proc.cpu_percent(interval=1.0)
        mem_mb = proc.memory_info().rss / 1024 / 1024
        eq_size = event_queue.qsize()
        iq_size = inference_queue.qsize()
        log.info(
            "OVERHEAD | CPU=%.1f%%  RAM=%.1fMB  event_q=%d  infer_q=%d",
            cpu, mem_mb, eq_size, iq_size,
        )
        time.sleep(MONITOR_INTERVAL - 1)
    log.info("Monitor stopped")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected_features = meta["features"]
    else:
        expected_features = REALTIME_FEATURES

    log.info("Expected features: %s", expected_features)
    log.info("=== REALTIME SSH IDS STARTED ===")
    log.info("Alerts → %s", ALERTS_PATH)

    threads = [
        threading.Thread(target=producer_thread, name="producer", daemon=True),
        threading.Thread(target=consumer_thread, name="consumer", daemon=True),
        threading.Thread(
            target=worker_pool_thread,
            args=(model, expected_features),
            name="worker-pool",
            daemon=True,
        ),
        threading.Thread(target=monitor_thread, name="monitor", daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        _stop_event.set()
        for t in threads:
            t.join(timeout=5.0)
        log.info("Stopped.")


if __name__ == "__main__":
    main()
