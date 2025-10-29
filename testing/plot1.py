import matplotlib.pyplot as plt
import numpy as np

# === Data ===
datasets = ["AMC23", "AIME", "Math-500", "GPQA"]
combine_acc = [0.8917, 0.543, 0.9067, 0.564]
new_acc = [0.892, 0.553, 0.910, 0.5725]

combine_len = [5979, 10148, 4278, 6688]
new_len = [5943, 10090, 4162, 5653]

combine_time = [51.01, 83.90, 36.7, 89.57]
new_time = [49.29, 86.7, 35.47, 66.28]

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
    ax.set_xticklabels(datasets,fontsize=15)
    add_labels(ax, rects1, integer=is_int)
    add_labels(ax, rects2, integer=is_int)

# === 全局 legend 放到底部一排 ===
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.1, -0.09),
    fontsize=15,
    ncol=2,
    frameon=False
)

# === 添加 Draft/Target 模型说明文字 ===
fig.text(
    0.5, -0.03,
    "Draft Model: DeepSeek-R1-1.5B    |    Target Model: DeepSeek-R1-32B",
    ha='center', va='center', fontsize=15, style='italic'
)

plt.tight_layout(rect=[0, 0, 1.01, 1])  # 给底部留空间
plt.savefig("ours_vs_combine_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
