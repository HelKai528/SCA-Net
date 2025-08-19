
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# optional: try use scipy KDE, otherwise fallback
try:
    from scipy.stats import gaussian_kde
    _have_scipy = True
except Exception:
    _have_scipy = False

df = pd.read_csv('/home81/haokai/lamost/result/data2_top2w_6_1c_0.15_nox.csv', sep=',')

params = [
    ('teff', 'T$_{\\rm eff}$ [K]'),
    ('logg', '$\\lg\\,\\left[g/\\rm (cm\\cdot s^{-2})\\right]$'),
    ('feh', '[Fe/H] [dex]')
]

plt.style.use('seaborn-whitegrid')
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=13, labelsize=12)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
fig = plt.figure(figsize=(6.5, 10))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1],
                       width_ratios=[3.5, 1.6], hspace=0.35, wspace=0.15)
c_line = plt.cm.viridis(0.85)
c_fill = plt.cm.viridis(0.60)

for i, (param, label) in enumerate(params):
    ax = fig.add_subplot(gs[i, 0])
    ax_kde = fig.add_subplot(gs[i, 1], sharey=None)

    x = df[f'{param}_true'].values
    y = df[f'{param}_pred'].values
    hb = ax.hexbin(
        x, y,
        gridsize=200,
        cmap='viridis',
        mincnt=1,
        linewidths=0.2,
        edgecolors='none'
    )
    mn = min(x.min(), y.min()) * 0.98
    mx = max(x.max(), y.max()) * 1.02
    ax.plot([mn, mx], [mn, mx], '--', color='0.3', linewidth=1)

    ax.set_xlabel(f'{label} (LAMOST)')
    ax.set_ylabel(f'{label} (SCA-Net)')
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    res = y - x
    mu, sigma = res.mean(), res.std(ddof=0)
    stats_text = f'μ = {mu:.3f}\nσ = {sigma:.3f}'
    # ax.text(
    #     0.98, 0.98, stats_text,
    #     transform=ax.transAxes,
    #     ha='right', va='top',
    #     fontsize=10,
    #     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, linewidth=0.5)
    # )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, linewidth=0.5)
    )
    ymin, ymax = res.min(), res.max()
    ax_kde.set_ylim(ymin, ymax)

    xs = np.linspace(ymin, ymax, 300)
    if _have_scipy:
        kde = gaussian_kde(res)
        dens = kde(xs)
    else:
        # histogram -> smooth approximation without scipy.ndimage
        counts, edges = np.histogram(res, bins=60, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens = np.interp(xs, centers, counts, left=0, right=0)
        # Gaussian-like smoothing via convolution kernel
        kernel_x = np.arange(-3, 4)
        kernel = np.exp(-0.5 * (kernel_x / 1.0) ** 2)
        kernel = kernel / kernel.sum()
        dens = np.convolve(dens, kernel, mode='same')
    width_scale = 0.9
    dens = dens / dens.max()
    dens = dens * width_scale
    ax_kde.clear()
    ax_kde.plot(dens, xs, lw=1.25, color=c_line)
    ax_kde.fill_betweenx(xs, 0, dens, alpha=0.25, color=c_fill)
    ax_kde.set_xlim(0, width_scale * 1.05)
    ax_kde.set_xlabel('Density', fontsize=9)
    ax_kde.xaxis.set_label_position('top')
    ax_kde.xaxis.tick_top()
    ax_kde.tick_params(axis='x', labelsize=8)
    ax_kde.tick_params(axis='y', labelsize=8)

    for spine in ['right', 'bottom']:
        ax_kde.spines[spine].set_visible(False)
    ax_kde.spines['left'].set_visible(False)

    plt.setp(ax_kde.get_yticklabels(), visible=True)

# plt.tight_layout()
fig.subplots_adjust(top=0.96, bottom=0.05)
plt.tight_layout()  
plt.savefig('/home81/haokai/lamost/result/data2_nox_best_with_kde.png', dpi=300)
# plt.show()

