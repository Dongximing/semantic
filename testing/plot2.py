import matplotlib.pyplot as plt
import numpy as np

# === Data from your new table ===
datasets = ["AMC23", "AIME", "Math-500", "GPQA"]

# combine (red, left table)
combine_acc = [0.8583, 0.5443, 0.909, 0.572]
combine_len = [7572.55, 11491.56, 5000.18, 8488.84]
combine_time = [73.95, 125.01, 44.23, 133.23]

# new code improvement (blue, right table)
new_acc = [0.892, 0.61, 0.9217, 0.552]
new_len = [6410.88, 10333.39, 4220.06, 7235.10]
new_time = [75.42, 119.06, 43.53, 98.68]

x = np.arange(len(datasets))
width = 0.48  # 柱子更粗

def add_labels(ax, rects, integer=False, offset=0):
    for rect in rects:
        height = rect.get_height()
        label = f"{int(height)}" if integer else f"{height:.2f}"
        ax.text(
            rect.get_x() + rect.get_width()/2.,
            height * (1 + offset),
            label,
            ha='center', va='bottom', fontsize=11
        )

# === Plot all in one figure ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [
    ("Accuracy", new_acc, combine_acc, False),
    ("Output Tokens", new_len, combine_len, True),
    ("Time (s)", new_time, combine_time, False)
]

for i, (ylabel, ours, combine, is_int) in enumerate(metrics):
    ax = axes[i]
    rects1 = ax.bar(x - width/2, ours, width, label='Ours', color='blue')
    rects2 = ax.bar(x + width/2, combine, width, label='Combine', color='orange')
    ax.set_title(ylabel, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=15)
    add_labels(ax, rects1, integer=is_int)
    add_labels(ax, rects2, integer=is_int)

# === 全局 legend 和模型说明放一排 ===
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.1, -0.09),
    fontsize=15,
    ncol=2,
    frameon=False
)

fig.text(
      0.5, -0.03,
    "Draft Model: DeepSeek-R1-1.5B    |    Target Model: QwQ-32B",
    ha='center', va='center', fontsize=15, style='italic'
)

plt.tight_layout(rect=[0, 0, 1.01, 1])
plt.savefig("ours_vs_combine_comparison_new_qwq.png", dpi=300, bbox_inches='tight')
plt.show()
