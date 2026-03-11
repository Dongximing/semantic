import matplotlib.pyplot as plt

datasets = {
    "Math-500": {"acc": [0.814, 0.910, 0.928],
                 "len": [4832.45, 4162.91, 3525.23],
                 "time": [12.70, 35.47, 89.91]},
    "AMC23": {"acc": [0.658, 0.892, 0.925],
              "len": [7410.83, 5943.30, 5368.62],
              "time": [19.59, 49.29, 154.34]},
    "AIME24": {"acc": [0.233, 0.553, 0.621],
               "len": [11210.73, 10090.74, 8811.20],
               "time": [29.45, 86.75, 214.76]},
    "GPQA-D": {"acc": [0.338, 0.573, 0.593],
               "len": [7922.03, 5653.57, 6945.10],
               "time": [21.30, 66.28, 170.87]},
}
# datasets = { 
#      "Math-500": {"acc": [0.814, 0.922, 0.935], "len": [4832.45, 4220.06, 4031.7],"time": [12.70,43.53,99.04]}, 
#      "AMC23": {"acc": [0.658, 0.892, 0.925], "len": [7410.83, 6410.88, 6702.81],"time": [19.59,75.42,165.02]}, 
#      "AIME24": {"acc": [0.233, 0.610, 0.630], "len": [11210.73, 10333.39, 10186.7],"time": [29.45,119.06,250.75]}, 
#      "GPQA-D": {"acc": [0.338, 0.552, 0.595], "len": [7922.03, 7235.10, 7002.30],"time": [21.30,98.68,176.25]} }

models = ["1.5B", "1.5B+32B (ours)", "32B"]
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

# Label offsets for each task (acc / len / time).
offsets = {
    "Math-500": {"acc": +0.05, "len": +0.15, "time": -0.30},
    "AMC23": {"acc": -0.15, "len": +0.15, "time": -0.35},
    "AIME24": {"acc": -0.3,  "len": +0.2,  "time": -0.30},
    "GPQA-D": {"acc": -0.20, "len": +0.20, "time": -0.35},
}

fig, axes = plt.subplots(4, 3, figsize=(15, 10), dpi=300)
plt.subplots_adjust(wspace=0.4, hspace=0.55)

for i, (name, data) in enumerate(datasets.items()):
    acc, length, time = data["acc"], data["len"], data["time"]

    # --- Accuracy ---
    ax = axes[i, 0]
    bars = ax.bar(models, acc, color=colors, edgecolor="black", width=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name} - Accuracy", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.015, f"{h:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Arrow 1.5B → Ours
    x0, y0, x1, y1 = 0, acc[0], 1, acc[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.0,
                                connectionstyle="arc3,rad=-0.25"))
    xm, ym = (x0+x1)/2 + offsets[name]["acc"], (y0+y1)/2 + 0.10
    acc_change = (acc[1] - acc[0]) / acc[0] * 100
    ax.text(xm, ym, f"+{acc_change:.1f}%", color='red',
            fontsize=12, fontweight='bold', ha='center')

    # --- Length ---
    ax = axes[i, 1]
    bars = ax.bar(models, length, color=colors, edgecolor="black", width=0.6)
    ymin, ymax = min(length)*0.8, max(length)*1.25
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{name} - Length", fontsize=14, fontweight='bold')
    ax.set_ylabel("Avg. Length", fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+(ymax-ymin)*0.02, f"{h:.0f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Arrow 1.5B → Ours
    x0, y0, x1, y1 = 0, length[0], 1, length[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.0,
                                connectionstyle="arc3,rad=0.25"))
    xm, ym = (x0+x1)/2 + offsets[name]["len"], (y0+y1)/2 + (ymax-ymin)*0.1
    len_change = (length[1] - length[0]) / length[0] * 100
    ax.text(xm, ym, f"-{abs(len_change):.1f}%", color='red',
            fontsize=12, fontweight='bold', ha='center')

    # --- Time ---
    ax = axes[i, 2]
    bars = ax.bar(models, time, color=colors, edgecolor="black", width=0.6)
    ymin, ymax = min(time)*0.8, max(time)*1.25
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{name} - Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+(ymax-ymin)*0.02, f"{h:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Arrow 32B -> Ours to indicate the speedup.
    x0, y0, x1, y1 = 2, time[2], 1, time[1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.0,
                                connectionstyle="arc3,rad=0.25"))
    xm, ym = (x0+x1)/2 + offsets[name]["time"], (y0+y1)/2 + (ymax-ymin)*0.1
    time_reduction = (time[2] - time[1]) / time[2] * 100
    ax.text(xm, ym, f"-{abs(time_reduction):.1f}%", color='blue',
            fontsize=12, fontweight='bold', ha='center')

# --- Global label ---
fig.text(0.5, -0.02, "Target Model = DeepseekR1-32B", 
         ha='center', fontsize=16, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig("acc_len_time_3x4_with_offsets.png", bbox_inches="tight")
plt.show()
