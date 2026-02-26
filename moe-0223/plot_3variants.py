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

style = {
    'Top-4':          {'fmt': '-^', 'color': '#33A02C', 'ms': 5, 'lw': 1.5, 'zorder': 3},
    'Top-6':          {'fmt': '-s', 'color': '#1F78B4', 'ms': 5, 'lw': 1.5, 'zorder': 3},
    'Default (Top-8)':{'fmt': '-o', 'color': '#6A3D9A', 'ms': 5, 'lw': 1.5, 'zorder': 3},
    'Ours':           {'fmt': '-D', 'color': '#E31A1C', 'ms': 5.5, 'lw': 2.0, 'zorder': 4},
}
data = {'Top-4': e4, 'Top-6': e6, 'Default (Top-8)': e8, 'Ours': e48res}

HEIGHT = 3.5

# ============================================================
# Variant A: ylim cropped from bottom
# ============================================================
fig_a, (a1, a2) = plt.subplots(1, 2, figsize=(10, HEIGHT), gridspec_kw={'width_ratios': [2, 1]})

for name, vals in data.items():
    s = style[name]
    a1.plot(left_idx, vals[left_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])
    a2.plot(right_idx, vals[right_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])

# Crop y from nonzero to zoom into difference range
a1.set_ylim(bottom=0)
a1.set_xlabel('Layer Index')
a1.set_ylabel(r'Cumulative $\Sigma(\mathrm{Act}^2)$ ($\times 10^3$)')
a1.set_xlim(-1, 33)
a1.set_xticks(left_idx)
a1.tick_params(axis='x', pad=1, length=2)
a1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')

a2.set_xlabel('Layer Index')
a2.set_xlim(split - 1, 48)
a2.set_xticks(right_idx)
a2.tick_params(axis='x', pad=1, length=2)

fig_a.suptitle('A: Y-axis cropped  |  Qwen3-30B-A3B-Instruct-2507', fontsize=13, y=0.97)
fig_a.tight_layout()
fig_a.savefig('/Users/bobchenyx/Downloads/Claude/moe-0223/variant_A_ylim.png')

# ============================================================
# Variant B: Ratio relative to Default (Top-8)
# ============================================================
fig_b, (b1, b2) = plt.subplots(1, 2, figsize=(10, HEIGHT), gridspec_kw={'width_ratios': [2, 1]})

ratio_data = {
    'Top-4': e4 / e8,
    'Top-6': e6 / e8,
    'Default (Top-8)': e8 / e8,
    'Ours': e48res / e8,
}

for name, vals in ratio_data.items():
    s = style[name]
    b1.plot(left_idx, vals[left_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])
    b2.plot(right_idx, vals[right_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])

b1.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, zorder=1)
b2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, zorder=1)

b1.set_xlabel('Layer Index')
b1.set_ylabel(r'$\Sigma(\mathrm{Act}^2)$ / Default (Top-8)')
b1.set_xlim(-1, 33)
b1.set_xticks(left_idx)
b1.tick_params(axis='x', pad=1, length=2)
b1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')

b2.set_xlabel('Layer Index')
b2.set_xlim(split - 1, 48)
b2.set_xticks(right_idx)
b2.tick_params(axis='x', pad=1, length=2)

fig_b.suptitle('B: Ratio to Default  |  Qwen3-30B-A3B-Instruct-2507', fontsize=13, y=0.97)
fig_b.tight_layout()
fig_b.savefig('/Users/bobchenyx/Downloads/Claude/moe-0223/variant_B_ratio.png')

# ============================================================
# Variant C: fill_between min/max band
# ============================================================
fig_c, (c1, c2) = plt.subplots(1, 2, figsize=(10, HEIGHT), gridspec_kw={'width_ratios': [2, 1]})

all_vals = np.stack([e4, e6, e8, e48res])
vmin = all_vals.min(axis=0)
vmax = all_vals.max(axis=0)

c1.fill_between(left_idx, vmin[left_idx], vmax[left_idx], alpha=0.12, color='gray', zorder=1)
c2.fill_between(right_idx, vmin[right_idx], vmax[right_idx], alpha=0.12, color='gray', zorder=1)

for name, vals in data.items():
    s = style[name]
    c1.plot(left_idx, vals[left_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])
    c2.plot(right_idx, vals[right_idx], s['fmt'], color=s['color'],
            markersize=s['ms'], linewidth=s['lw'], label=name, zorder=s['zorder'])

c1.set_xlabel('Layer Index')
c1.set_ylabel(r'Cumulative $\Sigma(\mathrm{Act}^2)$ ($\times 10^3$)')
c1.set_xlim(-1, 33)
c1.set_xticks(left_idx)
c1.tick_params(axis='x', pad=1, length=2)
c1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')

c2.set_xlabel('Layer Index')
c2.set_xlim(split - 1, 48)
c2.set_xticks(right_idx)
c2.tick_params(axis='x', pad=1, length=2)

fig_c.suptitle('C: Fill between  |  Qwen3-30B-A3B-Instruct-2507', fontsize=13, y=0.97)
fig_c.tight_layout()
fig_c.savefig('/Users/bobchenyx/Downloads/Claude/moe-0223/variant_C_fill.png')

print("Saved variant_A_ylim.png, variant_B_ratio.png, variant_C_fill.png")
