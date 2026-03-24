import time
import subprocess
from collections import defaultdict, deque
import pandas as pd
import joblib

MODEL_PATH = "outputs/models/xgb_model.pkl"

WINDOW_SIZE = 60
BLOCK_THRESHOLD = 2
PROB_THRESHOLD = 0.1

# load model
model = joblib.load(MODEL_PATH)

# buffer per IP
buffers = defaultdict(lambda: deque())

# state
suspicious_count = defaultdict(int)
blocked_ips = set()


def get_ssh_connections():
    cmd = "ss -tn state established '( dport = :22 or sport = :22 )'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lines = result.stdout.split("\n")[1:]

    connections = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        src = parts[4]
        ip = src.split(":")[0]

        connections.append({
            "Src IP": ip,
            "Timestamp": pd.Timestamp.now()
        })

    return connections


def build_features(df):
    if len(df) == 0:
        return None

    df = df.sort_values("Timestamp")

    duration = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds()

    flow_rate = len(df) / max(duration, 1)

    inter = df["Timestamp"].diff().dt.total_seconds().dropna()

    return {
        "flow_rate_per_window": flow_rate,
        "interarrival_std": inter.std() if len(inter) > 0 else 0,
        "flow_duration_mean": duration
    }


def block_ip(ip):
    if ip in blocked_ips:
        return

    print(f"[BLOCK] {ip}")

    subprocess.run(f"sudo iptables -A INPUT -s {ip} -j DROP", shell=True)
    blocked_ips.add(ip)


def main():
    print("=== REALTIME DETECTOR START ===")

    while True:
        conns = get_ssh_connections()

        now = pd.Timestamp.now()

        for c in conns:
            ip = c["Src IP"]

            buffers[ip].append(now)

            # remove old
            while buffers[ip] and (now - buffers[ip][0]).total_seconds() > WINDOW_SIZE:
                buffers[ip].popleft()

            df = pd.DataFrame({
                "Timestamp": list(buffers[ip])
            })

            feats = build_features(df)

            if feats is None:
                continue

            X = pd.DataFrame([feats])

            prob = model.predict_proba(X)[0][1]

            if prob > PROB_THRESHOLD:
                suspicious_count[ip] += 1

                if suspicious_count[ip] == 1:
                    print(f"[ALERT] {ip} prob={prob:.3f}")

                elif suspicious_count[ip] >= BLOCK_THRESHOLD:
                    block_ip(ip)

            else:
                suspicious_count[ip] = 0

        time.sleep(2)


if __name__ == "__main__":
    main()
