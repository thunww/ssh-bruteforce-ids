import re
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Parse detector.log for CPU/RAM ---
cpu_times, cpu_vals, ram_vals = [], [], []
pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*OVERHEAD.*CPU=([\d.]+)%.*RAM=([\d.]+)MB")
with open(ROOT / "outputs/detector.log") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            cpu_times.append(datetime.fromisoformat(m.group(1)))
            cpu_vals.append(float(m.group(2)))
            ram_vals.append(float(m.group(3)))

# --- Parse alerts.jsonl for action timeline ---
alerts = []
with open(ROOT / "outputs/alerts.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            alerts.append(json.loads(line))

df = pd.DataFrame(alerts)
df["now"] = pd.to_datetime(df["now"])
# Only today's session
df = df[df["now"].dt.date == pd.Timestamp("2026-05-23").date()].copy()

# Color map for actions
action_colors = {"ALERT": "#E8A24C", "BLOCK": "#E85C4C", "BLOCKED": "#9B59B6", "NORMAL": "#4C9BE8"}

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
fig.suptitle("Real-time Detector — System Overhead & Detection Timeline", fontsize=13, fontweight="bold")

# --- Plot 1: Risk Score timeline ---
ax1 = axes[0]
colors = [action_colors.get(a, "gray") for a in df["action"]]
ax1.scatter(df["now"], df["risk_score"], c=colors, s=20, zorder=3)
ax1.axhline(0.20, color="#E8A24C", linestyle="--", lw=1, label="ALERT threshold (0.20)")
ax1.axhline(0.40, color="#E85C4C", linestyle="--", lw=1, label="BLOCK threshold (0.40)")
ax1.set_ylabel("Risk Score")
ax1.set_ylim(0, 1.0)
ax1.set_title("Risk Score theo thời gian")
patches = [mpatches.Patch(color=v, label=k) for k, v in action_colors.items() if k in df["action"].values]
ax1.legend(handles=patches + [
    plt.Line2D([0],[0], color="#E8A24C", linestyle="--", label="ALERT (0.20)"),
    plt.Line2D([0],[0], color="#E85C4C", linestyle="--", label="BLOCK (0.40)"),
], fontsize=8, loc="upper right")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Plot 2: CPU% ---
ax2 = axes[1]
if cpu_times:
    ax2.plot(cpu_times, cpu_vals, color="#4C9BE8", lw=2, marker="o", markersize=4)
    ax2.set_ylabel("CPU (%)")
    ax2.set_title("CPU Overhead")
    ax2.set_ylim(0, max(cpu_vals) * 2 + 1)
else:
    ax2.text(0.5, 0.5, "Không có dữ liệu CPU trong log", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("CPU Overhead")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --- Plot 3: RAM MB ---
ax3 = axes[2]
if ram_vals:
    ax3.plot(cpu_times, ram_vals, color="#6DBE6D", lw=2, marker="o", markersize=4)
    ax3.set_ylabel("RAM (MB)")
    ax3.set_title("RAM Overhead")
    ax3.set_ylim(min(ram_vals) - 10, max(ram_vals) + 20)
else:
    ax3.text(0.5, 0.5, "Không có dữ liệu RAM trong log", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("RAM Overhead")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "detector_overhead.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/detector_overhead.png")
plt.show()
