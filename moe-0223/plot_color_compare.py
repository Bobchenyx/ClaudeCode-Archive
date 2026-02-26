import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

layers = list(range(48))

e4 = np.array([6280.98, 13205.59, 47142.12, 26244.67, 27442.96, 28914.61, 34422.37, 35360.88,
      36927.95, 37782.45, 37039.48, 40929.94, 39394.42, 44903.50, 45357.40, 50010.32,
      53172.57, 53447.63, 68615.32, 65357.21, 62556.80, 60833.99, 65807.50, 69611.33,
      69697.13, 72940.80, 72407.84, 79957.95, 83108.48, 85377.05, 103192.72, 98508.44,
      94121.32, 96608.35, 106690.87, 111940.34, 110676.49, 127606.99, 131890.50, 159453.83,
      187460.78, 219497.86, 260408.20, 295916.69, 326253.75, 368626.88, 428352.97, 746723.12]) / 1e3

e6 = np.array([6230.39, 12525.30, 41016.10, 25226.22, 26329.02, 27875.36, 33184.84, 33986.09,
      35505.19, 36030.99, 35636.84, 38990.57, 37877.01, 43541.49, 45280.82, 48322.66,
      51074.00, 51219.64, 65971.61, 62765.73, 59991.43, 58120.04, 62475.36, 66290.31,
      66523.25, 69894.48, 69645.14, 76936.57, 80058.67, 81859.20, 99185.18, 94893.25,
      90609.25, 92658.23, 101877.96, 107468.59, 106268.41, 123144.49, 127518.85, 153591.05,
      181469.08, 211910.80, 252488.20, 284977.53, 314086.44, 352832.50, 408981.12, 703102.75]) / 1e3

e8 = np.array([6098.62, 11580.93, 33523.19, 24011.89, 24686.67, 26302.39, 31671.19, 31831.60,
      32872.34, 33647.50, 32996.82, 35900.48, 35627.92, 41166.59, 42295.42, 45205.18,
      47605.66, 47908.50, 61691.12, 59190.73, 55731.16, 53655.03, 57024.24, 60615.56,
      60982.43, 64979.34, 65663.16, 71357.46, 74737.94, 76226.40, 93018.28, 88718.61,
      84520.76, 85764.34, 93225.51, 97885.98, 98245.05, 115823.87, 119462.08, 141957.83,
      168875.95, 196522.73, 235109.34, 262873.41, 286325.44, 320658.28, 366078.88, 608497.25]) / 1e3

e48res = np.array([6098.62, 12853.60, 55324.90, 25117.41, 25497.36, 27200.73, 32574.88, 33440.60,
          34597.51, 35243.34, 34937.21, 39321.41, 37548.80, 42547.53, 42966.46, 47848.01,
          50018.82, 50587.84, 65457.82, 62717.12, 60017.92, 57299.91, 62743.43, 66150.26,
          65073.77, 68836.45, 68942.34, 76386.00, 78622.47, 80673.37, 97597.11, 93160.73,
          89105.85, 89991.20, 99569.89, 104823.49, 102631.61, 119578.63, 123764.72, 150321.22,
          175596.88, 204552.70, 245070.09, 276549.41, 297719.03, 335537.56, 385185.97, 664886.88]) / 1e3

sample_idx = sorted(set([i for i in range(48) if i % 4 == 0] + [47]))
split = 36
left_idx = [i for i in sample_idx if i < split]
right_idx = [i for i in sample_idx if i >= split]

# Six color schemes
color_schemes = {
    'A: Tableau Muted': {
        'Top-4':          {'color': '#86BCB6'},
        'Top-6':          {'color': '#F28E2B'},
        'Default (Top-8)':{'color': '#76B7B2'},
        'Ours':           {'color': '#E15759'},
    },
    'B: Nature': {
        'Top-4':          {'color': '#E64B35'},
        'Top-6':          {'color': '#4DBBD5'},
        'Default (Top-8)':{'color': '#3C5488'},
        'Ours':           {'color': '#F39B7F'},
    },
    'C: Pastel Academic': {
        'Top-4':          {'color': '#8DA0CB'},
        'Top-6':          {'color': '#66C2A5'},
        'Default (Top-8)':{'color': '#FC8D62'},
        'Ours':           {'color': '#E78AC3'},
    },
    'D: Deep Jewel': {
        'Top-4':          {'color': '#6A3D9A'},
        'Top-6':          {'color': '#1F78B4'},
        'Default (Top-8)':{'color': '#33A02C'},
        'Ours':           {'color': '#E31A1C'},
    },
    'E: Warm Sunset': {
        'Top-4':          {'color': '#FDAE61'},
        'Top-6':          {'color': '#D73027'},
        'Default (Top-8)':{'color': '#4575B4'},
        'Ours':           {'color': '#1A9850'},
    },
    'F: Flat Modern': {
        'Top-4':          {'color': '#95A5A6'},
        'Top-6':          {'color': '#3498DB'},
        'Default (Top-8)':{'color': '#2C3E50'},
        'Ours':           {'color': '#E74C3C'},
    },
}

markers = {
    'Top-4': '-^',
    'Top-6': '-s',
    'Default (Top-8)': '-o',
    'Ours': '-D',
}

data = {'Top-4': e4, 'Top-6': e6, 'Default (Top-8)': e8, 'Ours': e48res}

fig, axes = plt.subplots(6, 2, figsize=(10, 24),
                         gridspec_kw={'width_ratios': [2, 1]})

for row, (scheme_name, colors) in enumerate(color_schemes.items()):
    ax1, ax2 = axes[row]

    for name, vals in data.items():
        c = colors[name]['color']
        fmt = markers[name]
        ms = 5.5 if name == 'Ours' else 5
        lw = 2.0 if name == 'Ours' else 1.5
        zorder = 4 if name == 'Ours' else 3
        ax1.plot(left_idx, vals[left_idx], fmt, color=c,
                 markersize=ms, linewidth=lw, label=name, zorder=zorder)
        ax2.plot(right_idx, vals[right_idx], fmt, color=c,
                 markersize=ms, linewidth=lw, label=name, zorder=zorder)

    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel(r'Cumulative $\Sigma(\mathrm{Act}^2)$ ($\times 10^3$)')
    ax1.set_xlim(-1, 33)
    ax1.set_xticks(left_idx)
    ax1.tick_params(axis='x', pad=1, length=2)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax1.set_title(scheme_name, fontsize=13, loc='left')

    ax2.set_xlabel('Layer Index')
    ax2.set_xlim(split - 1, 48)
    ax2.set_xticks(right_idx)
    ax2.tick_params(axis='x', pad=1, length=2)

fig.suptitle('Qwen3-30B-A3B: Cumulative Activation Squared — Color Comparison',
             fontsize=14, y=1.0)

plt.tight_layout()
plt.savefig('/Users/bobchenyx/Downloads/Claude/moe-0223/color_compare.png')
plt.savefig('/Users/bobchenyx/Downloads/Claude/moe-0223/color_compare.pdf')
print("Saved to color_compare.png and color_compare.pdf")
