"""Everything the figure scripts share: paths, the print-size style, colours,
the constructed AR(5) and a few helpers.

Each ``fig_*.py`` imports what it needs from here and can be run on its own,
from any directory::

    python scripts/figures/fig_03.py

``plot_figures.py`` runs them all in order (the DVC ``plot_figures`` stage).
"""
# %%
import os, sys, json, shutil, copy

# for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
#     os.environ.setdefault(_v, '1')

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]          # revision/
sys.path.insert(0, str(ROOT))
FIG, RES = ROOT / 'figures', ROOT / 'results'
FIG.mkdir(exist_ok=True)

from tsmodes import plotting as P
from tsmodes import datasets as D
from tsmodes.estimators import ESTIMATOR_LABELS, estimate, prepare, spectrum_acov
from tsmodes.modes import (ARModes, ar_acf, ar_acf_fft, ar_psd, poles_to_ar,
                           psd_from_poles)
from tsmodes.sims import (AR1, AR2_RHYTHM, AR2Damped, ARP_MODES, ARpModes,
                          PowerLaw, sim_ar_process)
from tsmodes.finite import phi_of_tau, tau_of_phi

P.use_style()
# Figures are drawn at the size they are printed.  sn-jnl's single-column
# page is text={31pc,194.25mm}, and \includesvg at natural size (or at
# width=\linewidth, which then rescales by at most a few percent) keeps every
# point size below the printed one: 7 pt labels, 6 pt ticks and legends, 8 pt
# titles, 9 pt panel letters.  A figure drawn wider and shrunk by LaTeX
# shrinks its text with it, which is how 9 pt ticks printed at 4 pt.
TEXT_WIDTH = 31 * 12 / 72.27          # 5.15 in
TEXT_HEIGHT = 194.25 / 25.4           # 7.65 in
mpl.rcParams.update({
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'legend.frameon': False,
    'savefig.pad_inches': 0.02,       # crop to the drawing; the page adds its own margin
    # Deterministic SVG ids, so a rerun writes byte-identical files and
    # neither git nor DVC sees an unchanged figure as changed.
    'svg.hashsalt': 'tsmodes',
})
FS = 500.0            # sampling rate of every simulation
TAU = 0.05            # the constructed timescale of the AR(1) simulations, s

# Colours shared by every figure.  Data is black and solid; an AR fit or the
# expected AR(1) form is this blue and dashed; modes are coloured by importance
# rank.  The two mode-kind colours (relaxation blue, oscillatory red) live in
# tsmodes.plotting.
DATA_COLOR = '#000000'
FIT_COLOR = '#0072B2'
FIT_STYLE = (0, (4, 2))
FIT_LW = 1.6
RANK_COLORS = ['#D55E00', '#009E73', '#CC79A7']

# The constructed AR(5) read mode by mode in fig_modes and used as the
# model-mismatch column of fig_bias: a 250 ms relaxation, a 5 Hz mode ringing
# half a cycle and a 50 Hz rhythm ringing one cycle.  It is the one AR(p)
# every simulation in the paper uses (tsmodes.sims.ARP_MODES, the benchmark's
# ARpModes process); PHI_ARP is its coefficients at FS.  The AR(2) oscillator
# of fig_bias and the benchmark is AR2_RHYTHM, a 10 Hz rhythm ringing 1.5
# cycles.
TRUE_MODES = [{'kind': 'relaxation' if freq == 0 else 'oscillatory',
               'tau': tau, 'freq': freq} for tau, freq in ARP_MODES]
ARP = ARpModes()
PHI_ARP = ARP.phi(FS)
AR2_OSC = AR2Damped(tau=AR2_RHYTHM[0], f_osc=AR2_RHYTHM[1])


def proxy(**kw):
    """A legend handle that is not tied to a drawn line."""
    return Line2D([0], [0], **kw)


def save(fig, name):
    """Write SVG and PNG into figures/, and leave the figure open so Jupyter
    shows it under the cell."""
    fig.savefig(FIG / name, format='svg', metadata={'Date': None})
    fig.savefig(FIG / name.replace('.svg', '.png'), format='png')
    print('wrote', FIG / name)
    return fig


def decade_ticks(ax, vals, pad=1.7):
    """Two or three labelled ticks on a log axis, whatever the span."""
    lo = float(np.min(vals)) / pad
    hi = float(np.max(vals)) * pad
    ax.set_ylim(lo, hi)
    best = None
    for mant in ([1.0], [1.0, 3.0], [1.0, 2.0, 5.0],
                 [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]):
        ticks = np.array(sorted(m * 10.0 ** e
                                for e in range(-9, 10) for m in mant))
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        if len(ticks) > 3:
            ticks = ticks[::int(np.ceil(len(ticks) / 3.0))]
        if best is None or len(ticks) > len(best):
            best = ticks
        if len(best) >= 3:
            break
    if best is not None and len(best) >= 2:
        ax.set_yticks(best)
        ax.set_yticklabels([f'{t:g}' for t in best])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
