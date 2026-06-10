import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

labels = ["Benign", "Attack"]
counts = [556, 64]
colors = ["#4C9BE8", "#E85C4C"]
total = sum(counts)
pcts = [c / total * 100 for c in counts]

fig = plt.figure(figsize=(12, 5))
fig.suptitle("Class Distribution of SSH Traffic Windows", fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Bar chart
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
ax1.set_title("Number of Windows per Class", fontsize=12)
ax1.set_ylabel("Count")
ax1.set_ylim(0, 650)
for bar, count, pct in zip(bars, counts, pcts):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        f"{count}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Pie chart
ax2 = fig.add_subplot(gs[1])
wedges, texts, autotexts = ax2.pie(
    counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Class Ratio (620 total windows)", fontsize=12)

plt.savefig("outputs/figures/class_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/figures/class_distribution.png")
plt.show()
