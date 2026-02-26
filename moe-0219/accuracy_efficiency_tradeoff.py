import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paper Style Config
# =========================
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# =========================
# Raw Data — Line series: A4, A6, A8
# =========================

line_methods = np.array(["4 Experts", "6 Experts", "8 Experts"])
line_tps     = np.array([52, 38, 29])
line_mmlu    = np.array([73.65, 79.26, 79.95])
line_gsm8k   = np.array([77.94, 86.28, 87.49])
line_ifeval  = np.array([61.03, 67.39, 70.50])
line_mbpp    = np.array([60.80, 72.80, 75.60])

sort_idx = np.argsort(-line_tps)
line_methods = line_methods[sort_idx]
line_tps     = line_tps[sort_idx]
line_mmlu    = line_mmlu[sort_idx]
line_gsm8k   = line_gsm8k[sort_idx]
line_ifeval  = line_ifeval[sort_idx]
line_mbpp    = line_mbpp[sort_idx]

# Ours (Ab3)
ours_tps    = 44
ours_mmlu   = 78.24
ours_gsm8k  = 86.66
ours_ifeval = 69.66
ours_mbpp   = 74.20

# =========================
# Plot
# =========================

fig, ax = plt.subplots(figsize=(7.2, 5.2))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Thin, light highlight bands for each method (A4, A6, A8, Ours)
band_hw = 0.8  # half-width
band_color_baseline = '#B0B0B0'
band_color_ours     = '#FF9999'

for t in line_tps:
    ax.axvspan(t - band_hw, t + band_hw, alpha=0.10, color=band_color_baseline, zorder=0)

ax.axvspan(ours_tps - band_hw, ours_tps + band_hw, alpha=0.15, color=band_color_ours, zorder=0)

# Lines
ax.plot(line_tps, line_mmlu,   marker='o', linewidth=2.2, markersize=6, label="MMLU",  color=colors[0])
ax.plot(line_tps, line_gsm8k,  marker='s', linewidth=2.2, markersize=6, label="GSM8K", color=colors[1])
ax.plot(line_tps, line_ifeval, marker='^', linewidth=2.2, markersize=6, label="IFEval",color=colors[2])
ax.plot(line_tps, line_mbpp,   marker='D', linewidth=2.2, markersize=6, label="MBPP",  color=colors[3])

# Ours scatter points (same markers/colors, black edge to distinguish)
ours_vals = [ours_mmlu, ours_gsm8k, ours_ifeval, ours_mbpp]
markers   = ['o', 's', '^', 'D']
for val, c, m in zip(ours_vals, colors[:4], markers):
    ax.scatter(ours_tps, val, marker=m, s=70, zorder=5, color=c,
              edgecolors='black', linewidths=0.8)

# Horizontal dashed lines at each Ours point
for val, c in zip(ours_vals, colors[:4]):
    ax.axhline(y=val, color=c, linestyle='--', linewidth=0.7, alpha=0.3, zorder=1)

# Annotate method names at top of each band
bench_line = [line_mmlu, line_gsm8k, line_ifeval, line_mbpp]
bench_ours = [ours_mmlu, ours_gsm8k, ours_ifeval, ours_mbpp]

for i in range(len(line_methods)):
    top_val = max(v[i] for v in bench_line)
    ax.annotate(line_methods[i],
                (line_tps[i], top_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10,
                fontweight='bold')

top_ours = max(bench_ours)
ax.annotate("Ours (Avg.5)",
            (ours_tps, top_ours),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10.5,
            fontweight='bold')

# Labels
ax.set_xlabel("TPS (Q8)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy–Efficiency Trade-off Across Benchmarks")

# Disable background grid
ax.grid(False)

# Put faster models on the right
ax.invert_xaxis()

# Legend
ax.legend(frameon=True, loc="best")

plt.tight_layout()

# =========================
# Save High-Quality PDF
# =========================
plt.savefig("accuracy_efficiency_tradeoff.pdf", dpi=600, bbox_inches="tight")
plt.show()
