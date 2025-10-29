import matplotlib.pyplot as plt

# ================================
# 数据定义
# ================================
datasets_r1 = {
    "AMC23": {
        "Draft": [0.658, 19.59],
        "Target": [0.925, 154.34],
        "SpecReason": [0.775, 49.98],
        "SpecThinking": [0.650, 24.99],
        "SpecSampling": [0.842, 91.40],
        "Ours": [0.892, 49.29],
    },
    "AIME24": {
        "Draft": [0.233, 29.45],
        "Target": [0.621, 214.76],
        "SpecReason": [0.470, 124.61],
        "SpecThinking": [0.241, 37.09],
        "SpecSampling": [0.443, 191.45],
        "Ours": [0.553, 86.75],
    },
    "Math-500": {
        "Draft": [0.8142, 12.70],
        "Target": [0.928, 89.91],
        "SpecReason": [0.840, 31.92],
        "SpecThinking": [0.823, 15.94],
        "SpecSampling": [0.880, 59.32],
        "Ours": [0.910, 35.47],
    },
    "GPQA-D": {
        "Draft": [0.338, 21.30],
        "Target": [0.593, 170.87],
        "SpecReason": [0.281, 159.33],
        "SpecThinking": [0.315, 44.88],
        "SpecSampling": [0.424, 82.78],
        "Ours": [0.5725, 66.28],
    }
}

datasets_qwq = {
    "AMC23": {
        "Draft": [0.658, 19.59],
        "Target": [0.925, 165.02],
        "SpecReason": [0.857, 98.59],
        "SpecThinking": [0.614, 30.20],
        "SpecSampling": [0.825, 120.39],
        "Ours": [0.892, 75.42],
    },
    "AIME24": {
        "Draft": [0.233, 29.45],
        "Target": [0.630, 250.75],
        "SpecReason": [0.610, 205.70],
        "SpecThinking": [0.204, 45.87],
        "SpecSampling": [0.540, 316.19],
        "Ours": [0.610, 119.06],
    },
    "Math-500": {
        "Draft": [0.8142, 12.70],
        "Target": [0.935, 99.04],
        "SpecReason": [0.770, 53.66],
        "SpecThinking": [0.710, 27.99],
        "SpecSampling": [0.895, 90.33],
        "Ours": [0.9217, 43.53],
    },
    "GPQA-D": {
        "Draft": [0.338, 21.30],
        "Target": [0.5959, 176.25],
        "SpecReason": [0.310, 109.64],
        "SpecThinking": [0.310, 44.88],
        "SpecSampling": [0.5250, 194.01],
        "Ours": [0.552, 98.68],
    }
}

# ================================
# 样式定义
# ================================
markers = {
    "Draft": "x",
    "Target": "o",
    "SpecReason": "v",
    "SpecThinking": "s",
    "SpecSampling": "^",
    "Ours": "D"
}
colors = {
    "Draft": "#d62728",
    "Target": "black",
    "SpecReason": "#ff7f0e",
    "SpecThinking": "#1f77b4",
    "SpecSampling": "#2ca02c",
    "Ours": "#9467bd"
}

# ================================
# 绘图函数
# ================================
def plot_pair(ax_row, datasets, title_prefix):
    for i, (name, data) in enumerate(datasets.items()):
        ax = ax_row[i]
        for method, (acc, time) in data.items():
            ax.scatter(time, acc, marker=markers[method],
                       color=colors[method], s=80, edgecolors='black', linewidths=0.9)
        ax.set_title(f"{title_prefix} - {name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Latency (s)", fontsize=15)
        ax.set_ylabel("Pass@1", fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.0)

        xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
        # ↑ Pass@1 Better（左上）
        ax.annotate("", xy=(xmax * 0.08, ymax * 0.95),
                    xytext=(xmax * 0.08, ymax * 0.75),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.text(xmax * 0.02, ymax * 0.90, "Better", color='green',
                fontsize=10, fontweight='bold', rotation=90, va='center')
        # ← Latency Better（左下）
        ax.annotate("", xy=(xmax * 0.25, ymax * 0.08),
                    xytext=(xmax * 0.45, ymax * 0.08),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.text(xmax * 0.35, ymax * 0.12, "Better", color='green',
                fontsize=10, fontweight='bold', ha='center')

# ================================
# 绘制 2x4 图
# ================================
fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
plt.subplots_adjust(wspace=0.3, hspace=0.4)

plot_pair(axes[0], datasets_r1, "DeepSeekR1-32B Pair")
plot_pair(axes[1], datasets_qwq, "QwQ-32B Pair")

# 图例
handles = [plt.Line2D([], [], marker=markers[m], color=colors[m],
                      linestyle='', markersize=14, label=m)
           for m in datasets_r1["AMC23"].keys()]
fig.legend(handles=handles, loc='upper center', ncol=6,
           fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("latency_vs_accuracy_2x4_pairs.png", bbox_inches="tight")
plt.show()
