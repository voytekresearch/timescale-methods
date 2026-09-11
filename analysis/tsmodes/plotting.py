"""Shared plotting style and reusable panels."""

import matplotlib as mpl
import matplotlib.pyplot as plt

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
               'floor': '#937860'}
GREY = '#555555'
LIGHT = '#BBBBBB'


def use_style():
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # Matplotlib under-measures mathtext extents when computing the
        # tight bounding box, which clips the last character of labels
        # like '$\\tau_{env}$ (s)'.  Pad enough to cover the shortfall.
        'savefig.pad_inches': 0.25,
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


def panel_label(ax, letter, dx=-0.16, dy=1.06, fontsize=10, **kwargs):
    # usetex ignores fontweight -- LaTeX takes the weight from the markup --
    # so under usetex the letter has to carry its own \textbf.
    if mpl.rcParams['text.usetex']:
        letter = r'\textbf{%s}' % letter
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
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

def log_tick_labels(ax, axis='y'):
    """Label a log10-valued linear axis with plain values."""
    getattr(ax, f'{axis}axis').set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{10**v:g}'))
