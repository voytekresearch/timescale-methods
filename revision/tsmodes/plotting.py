"""Shared plotting style and reusable panels."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Estimators are colour-coded by domain (time series / autocorrelation /
# spectrum) and shaded by model order (light = AR(1), dark = AR(p)).  Six
# methods with an arbitrary palette are unreadable; six methods on a 3 x 2 grid
# of domain and order are not.
# Hue carries the domain, lightness the model order.  Lightness survives
# every common form of colour blindness, and the hues are taken from the
# Okabe-Ito set, which is designed to stay separable under all of them.  The
# light and dark ends are pushed further apart than is pretty, because these
# pairs are read as two overlaid curves in the spectrum and autocorrelation
# panels, not just as adjacent boxes.
DOMAIN_COLORS = {'time': '#0072B2', 'acf': '#D55E00', 'psd': '#009E73'}
METHOD_COLORS = {
    'ar1_direct': '#9ECAE9',
    'arp_direct': '#08519C',
    'ar1_acf': '#FDBF6F',
    'arp_acf': '#A63603',
    'ar1_psd': '#7FD0B0',
    'arp_psd': '#00441B',
}
METHOD_DOMAIN = {'ar1_direct': 'time', 'arp_direct': 'time',
                 'ar1_acf': 'acf', 'arp_acf': 'acf',
                 'ar1_psd': 'psd', 'arp_psd': 'psd'}
METHOD_ORDER = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf',
                'ar1_psd', 'arp_psd']
METHOD_TICKS = {'ar1_direct': 'AR(1)', 'arp_direct': 'AR($p$)',
                'ar1_acf': 'AR(1)', 'arp_acf': 'AR($p$)',
                'ar1_psd': 'AR(1)', 'arp_psd': 'AR($p$)'}
DOMAIN_LABELS = {'time': 'time series', 'acf': 'autocorrelation',
                 'psd': 'spectrum'}
MODE_COLORS = {'relaxation': '#4C72B0', 'oscillatory': '#C44E52',
               'nyquist': '#937860'}
GREY = '#555555'
LIGHT = '#BBBBBB'


def use_style():
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
        'errorbar.capsize': 2,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def panel_label(ax, letter, dx=-0.16, dy=1.06, **kwargs):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left', **kwargs)


def save(fig, path, also_png=True):
    fig.savefig(path, format='svg')
    if also_png:
        fig.savefig(str(path).replace('.svg', '.png'), format='png')
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# Reusable panels
# ---------------------------------------------------------------------------

def plot_pole_diagram(ax, model, annotate=True, max_annotate=4, size_scale=180):
    """Unit-circle pole diagram with marker area scaled by mode importance."""
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=LIGHT, lw=0.8))
    ax.axhline(0, color=LIGHT, lw=0.5)
    ax.axvline(0, color=LIGHT, lw=0.5)
    imp = np.array([m.importance for m in model.modes])
    imp_norm = imp / imp.max() if len(imp) and imp.max() > 0 else imp
    for i, mode in enumerate(model.modes):
        color = MODE_COLORS.get(mode.kind, GREY)
        for lam in mode.poles:
            ax.scatter(lam.real, lam.imag, s=20 + size_scale * imp_norm[i],
                       facecolor=color, edgecolor='white', linewidth=0.5,
                       zorder=3, alpha=0.85)
        if annotate and i < max_annotate:
            lam = mode.poles[0]
            label = (f'$\\tau$={mode.tau*1e3:.0f} ms'
                     + (f'\n{mode.freq:.1f} Hz' if mode.kind != 'relaxation' else ''))
            ax.annotate(label, (lam.real, lam.imag),
                        textcoords='offset points', xytext=(6, 6), fontsize=6,
                        color=color)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('Re $\\lambda$')
    ax.set_ylabel('Im $\\lambda$')
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def plot_mode_factor_spectra(ax, model, freqs, show_total=True):
    """Per-mode factor spectra whose product is the full model spectrum."""
    total = model.psd(freqs)
    for i, mode in enumerate(model.modes):
        color = MODE_COLORS.get(mode.kind, GREY)
        ax.loglog(freqs, mode.psd(freqs), color=color, lw=0.9,
                  alpha=0.9 if i < 3 else 0.35,
                  label=(f'mode {i}: {mode.kind[:5]}' if i < 3 else None))
    if show_total:
        ax.loglog(freqs, total / model.offset, color='k', lw=1.6,
                  label='product = model')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Factor spectrum')


def plot_mode_map(ax, models, labels=None, colors=None, fmax=None):
    """Timescale vs oscillation frequency, marker area = mode importance."""
    for j, model in enumerate(models):
        color = None if colors is None else colors[j]
        for mode in model.modes:
            f = mode.freq if mode.kind != 'relaxation' else 0.0
            x = max(f, 0.4)
            ax.scatter(x, mode.tau, s=8 + 220 * mode.importance,
                       facecolor=color or MODE_COLORS.get(mode.kind, GREY),
                       edgecolor='white', lw=0.4, alpha=0.8,
                       label=(labels[j] if labels and mode is model.modes[0] else None))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mode frequency (Hz)')
    ax.set_ylabel('Mode $\\tau$ (s)')
    if fmax:
        ax.set_xlim(0.3, fmax)


def violin_by_method(ax, df, value_col, methods=None, positions=None,
                     truth=None, log=True, widths=0.7):
    """Distribution of an estimate per method."""
    methods = [m for m in METHOD_ORDER if m in set(df['method'])] if methods is None else methods
    data, pos, colors = [], [], []
    for i, method in enumerate(methods):
        vals = df.loc[df['method'] == method, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)] if log else vals[np.isfinite(vals)]
        if len(vals) < 3:
            continue
        data.append(np.log10(vals) if log else vals)
        pos.append(i if positions is None else positions[i])
        colors.append(METHOD_COLORS.get(method, GREY))
    if not data:
        return methods
    parts = ax.violinplot(data, positions=pos, widths=widths, showextrema=False,
                          showmedians=True)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor('none')
    parts['cmedians'].set_color('k')
    parts['cmedians'].set_linewidth(1.0)
    if truth is not None:
        ax.axhline(np.log10(truth) if log else truth, color='k', ls='--', lw=0.9,
                   zorder=0)
    return methods


def log_tick_labels(ax, axis='y'):
    """Label a log10-valued linear axis with plain values."""
    getattr(ax, f'{axis}axis').set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{10**v:g}'))
