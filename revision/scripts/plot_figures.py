"""Every figure of the paper, as a script.

GENERATED from figures.ipynb by scripts/run_notebook.py; edit the notebook,
not this file.  Run from revision/ with PYTHONPATH set to it.
"""

# ==========================================================================
# notebook cell 1
# ==========================================================================
# ---- Setup: run first ------------------------------------------------------
import os, sys, json, shutil, copy
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Works whether the kernel starts in revision/ or revision/notebooks/.
ROOT = Path.cwd().resolve()
if not (ROOT / 'tsmodes').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
FIG, RES = ROOT / 'figures', ROOT / 'results'
FIG.mkdir(exist_ok=True)

from tsmodes import plotting as P
from tsmodes import datasets as D
from tsmodes.estimators import ESTIMATOR_LABELS, estimate, prepare, spectrum_acov
from tsmodes.modes import (ARModes, ar_acf, ar_acf_fft, ar_psd, poles_to_ar,
                           psd_from_poles)
from tsmodes.sims import (AR1, AR1PlusSine, AR2Damped, ARP_MODES, ARpModes,
                          PowerLaw, sim_ar_process)
from tsmodes.finite import phi_of_tau, tau_of_phi

P.use_style()
# Larger text than the library default: these figures are typeset at column
# width, and 8 pt text at 7.4 inches shrinks below 6 pt on the page.
mpl.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'legend.frameon': False,
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

# The constructed AR(7) read mode by mode in fig_modes and used as the
# model-mismatch column of fig_bias: two relaxations of comparable importance
# an order of magnitude apart, two rhythms that ring for more than a cycle,
# and one fast relaxation that does not matter.  It is the one AR(p) every
# simulation in the paper uses (tsmodes.sims.ARP_MODES, the benchmark's
# ARpModes process); PHI7 is its coefficients at FS.
TRUE_MODES = [{'kind': 'relaxation' if freq == 0 else 'oscillatory',
               'tau': tau, 'freq': freq} for tau, freq in ARP_MODES]
PHI7 = ARpModes().phi(FS)


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


print('root:', ROOT)


# ==========================================================================
# notebook cell 3
# ==========================================================================
# ---- Fig. 1: fig_ar1 -------------------------------------------------------
# A user-space TeX Live (install-tl into ~/texlive) is found even when the
# shell that started the kernel does not have it on PATH.
import os
_TEX_BIN = Path.home() / 'texlive' / 'bin' / 'x86_64-linux'
if _TEX_BIN.is_dir() and str(_TEX_BIN) not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{_TEX_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
HAVE_LATEX = shutil.which('latex') is not None and (
    shutil.which('dvipng') is not None or shutil.which('dvisvgm') is not None)
if not HAVE_LATEX:
    print('No LaTeX found: equations rendered with mathtext instead of usetex.')

rng = np.random.default_rng(0)
phi = 0.95
fs = 1000
n_seconds = 1

sig = sim_ar_process(np.array([phi]), int(n_seconds * fs), rng)
sig = (sig - sig.mean()) / sig.std()
time = np.linspace(0, 1, len(sig))

lags = np.arange(101)
corrs = phi ** lags

freqs = np.linspace(.5, fs / 2, 1000)
powers = 1 / (1 + phi ** 2 - 2 * phi * np.cos(2 * np.pi * freqs * (1 / fs)))

# Jupyter draws the figure after the cell has finished, outside the context
# below, and each text keeps the usetex flag it was created with.  The
# preamble it needs is therefore set for good (it is inert while usetex is
# off), so the inline display does not fail with `align*` undefined.
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
with plt.rc_context({'text.usetex': HAVE_LATEX, 'font.family': 'sans-serif'}):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), height_ratios=[.7, .3])
    fig.subplots_adjust(wspace=.2, hspace=0.5)

    st = 16
    axes[0, 0].set_title("Autoregressive: AR(1)", size=st)
    axes[0, 1].set_title("Autocorrelation", size=st)
    axes[0, 2].set_title("Power Spectral Density", size=st)

    axes[0, 0].plot(time[:500], sig[:500], color=DATA_COLOR)
    axes[0, 1].plot(lags, corrs, color=DATA_COLOR)
    axes[0, 2].loglog(freqs, powers, color=DATA_COLOR)

    size = 15
    if HAVE_LATEX:
        s0 = r"\begin{align*}"
        s0 += r"x_t &= [ x_0, x_1, x_2, \dots, x_{n-1} ]^{\top}\\"
        s0 += r"x_{t+1} &= [ x_1, x_2, x_3, \dots, x_{n} ]^{\top}\\"
        s0 += r"x_{t+1} &= \varphi x_{t} + \epsilon"
        s0 += r"\end{align*}"
        axes[1, 0].text(.05, .6, s0, size=size)

        s1 = r"\begin{align*}"
        s1 += r"A(k)&= e^{\frac{-k}{\tau f_s}}\\"
        s1 += r"&= \varphi^k"
        s1 += r"\end{align*}"
        axes[1, 1].text(.3, .6, s1, size=size)

        s2 = r"\begin{align*}"
        s2 += r"P(f) &= \frac{1}{1 - 2 \varphi \cos{(2 \pi \frac{f}{f_s})} + \varphi^2}"
        s2 += r"\end{align*}"
        axes[1, 2].text(.03, .6, s2, size=size)
    else:
        axes[1, 0].text(.05, .85, r"$x_t = [\,x_0, x_1, x_2, \dots, x_{n-1}\,]^{\top}$", size=size)
        axes[1, 0].text(.05, .55, r"$x_{t+1} = [\,x_1, x_2, x_3, \dots, x_{n}\,]^{\top}$", size=size)
        axes[1, 0].text(.05, .25, r"$x_{t+1} = \varphi\, x_{t} + \epsilon$", size=size)
        axes[1, 1].text(.3, .55, r"$A(k) = e^{-k/(\tau f_s)} = \varphi^{k}$", size=size)
        axes[1, 2].text(.03, .55,
                        r"$P(f) = \dfrac{1}{1 - 2\varphi\cos(2\pi f/f_s) + \varphi^{2}}$",
                        size=size)

    sl = 11
    axes[0, 0].set_xlabel("Time (seconds)", size=sl)
    axes[0, 0].set_ylabel(r"Voltage ($x$, a.u.)", size=sl)
    axes[0, 1].set_ylabel(r"$A(k)$ (correlation)", size=sl)
    axes[0, 1].set_xlabel(r"Lag ($k$, samples)", size=sl)
    axes[0, 2].set_xlabel(r"Frequency ($f$, Hertz)", size=sl)
    axes[0, 2].set_ylabel(r"$P(f)$ (a.u.)", size=sl)

    for ax in axes[1].flatten():
        ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    save(fig, 'fig_ar1.svg')


# ==========================================================================
# notebook cell 5
# ==========================================================================
# ---- Fig. 2: fig_bias ------------------------------------------------------
rng = np.random.default_rng(1)
T_SHOW = 5.0                                   # seconds drawn in the signal row
NL = 150                                       # lags drawn in the ACF row
ks = np.arange(NL + 1)
lags = ks / FS
N_ACF = int(4 * FS)                            # lags used to transform ACF -> PSD


def acf_to_psd(corrs, fs):
    """Spectrum as the Fourier transform of an autocorrelation, exactly as in
    new_figures.ipynb (no normalization)."""
    powers = np.abs(np.fft.fft(corrs)) ** 2
    freqs = np.fft.fftfreq(len(corrs), 1 / fs)
    keep = slice(1, len(corrs) // 2)
    return freqs[keep], powers[keep]


def sample_acf(x, nlags):
    """Plain sample autocorrelation, normalized to one at lag zero."""
    x = x - x.mean()
    full = np.correlate(x, x, mode='full')[len(x) - 1:]
    return full[:nlags + 1] / full[0]


def ar1_acf(tau, n=N_ACF):
    return phi_of_tau(tau, FS) ** np.arange(n + 1)


# a) model mismatch: the AR(7) of fig_modes, the same model object and the
#    same exact spectrum as its panel b, against the AR(1) of its dominant
#    relaxation (200 ms) at the same innovation variance.
model7 = ARModes.from_phi(PHI7, FS, offset=1.0, freqs=np.linspace(0, FS / 2, 1025))
FREQS_A = np.logspace(np.log10(0.25), np.log10(FS / 2), 500)
acf_a = model7.acf(NL)
psd_a = model7.psd(FREQS_A)
TAU_A = 0.200
acf_a_ar1 = ar1_acf(TAU_A, NL)
psd_a_ar1 = ar_psd([phi_of_tau(TAU_A, FS)], FREQS_A, FS, 1.0)
sig_a = sim_ar_process(PHI7, int(T_SHOW * FS), rng)

# b) oscillations, as in new_figures.ipynb: an AR(1) plus a cosine carrying a
#    fifth of the aperiodic variance; the AR(1) form is the AR(1)'s own share
#    of the mixture, corrs_ap / (1 + osc_amp), transformed the same way.
OSC_AMP, F_OSC = 0.2, 20.0
acf_b_ar1 = ar1_acf(TAU)
acf_b = (acf_b_ar1 + OSC_AMP * np.cos(2 * np.pi * F_OSC * np.arange(N_ACF + 1) / FS)) / (1 + OSC_AMP)
f_b, psd_b = acf_to_psd(acf_b, FS)
_, psd_b_ar1 = acf_to_psd(acf_b_ar1 / (1 + OSC_AMP), FS)
t_b = np.arange(int(T_SHOW * FS)) / FS
sig_b_ap = sim_ar_process(np.array([phi_of_tau(TAU, FS)]), len(t_b), rng)
sig_b_ap = (sig_b_ap - sig_b_ap.mean()) / sig_b_ap.std()
sig_b = sig_b_ap + np.sqrt(2 * OSC_AMP) * np.cos(2 * np.pi * F_OSC * t_b)

# c) finite samples: one second of an AR(1), observed rather than theoretical;
#    the theoretical form is transformed at the same number of lags so the
#    two spectra share a grid and a scale.
T_FINITE = 1.0
sig_c = sim_ar_process(np.array([phi_of_tau(TAU, FS)]), int(T_FINITE * FS), rng)
sig_c = (sig_c - sig_c.mean()) / sig_c.std()
acf_c = sample_acf(sig_c, len(sig_c) - 1)
acf_c_ar1 = ar1_acf(TAU, len(acf_c) - 1)
f_c, psd_c = acf_to_psd(acf_c, FS)
_, psd_c_ar1 = acf_to_psd(acf_c_ar1, FS)

# d) dynamic timescales: an AR(1) whose coefficient rises over the recording,
#    with evenly spaced windows and the theoretical AR(1) of each window
n_d = int(T_SHOW * FS)
tau_t = np.linspace(0.010, 0.200, n_d)             # 10 ms -> 200 ms
phi_t = phi_of_tau(tau_t, FS)
eps = rng.normal(size=n_d)
sig_d = np.zeros(n_d)
for i in range(1, n_d):
    sig_d[i] = phi_t[i] * sig_d[i - 1] + eps[i]
WINDOWS = [(0.2, 1.0), (1.4, 2.2), (2.6, 3.4), (3.8, 4.6)]      # s, evenly spaced
WIN_COLORS = plt.cm.viridis(np.linspace(0.05, 0.85, len(WINDOWS)))
win_tau = [float(np.mean(tau_t[int(a * FS):int(b * FS)])) for a, b in WINDOWS]
win_curves = []
for tw, wc in zip(win_tau, WIN_COLORS):
    acf_w = ar1_acf(tw)
    f_w, psd_w = acf_to_psd(acf_w, FS)
    win_curves.append((acf_w[:NL + 1], f_w, psd_w, wc, f'$\\tau$ = {tw * 1e3:.0f} ms'))

fig = plt.figure(figsize=(8.4, 6.2))
gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.38, height_ratios=[0.7, 1.0, 1.0])
titles = ['Model mismatch', 'Oscillations', 'Finite samples', 'Dynamic timescales']
signals = [sig_a, sig_b, sig_c, sig_d]
for j in range(4):
    dynamic = j == 3

    # -- row 1: the signal ---------------------------------------------------
    ax = fig.add_subplot(gs[0, j])
    sig = signals[j]
    t = np.arange(len(sig)) / FS
    ax.plot(t, sig / np.std(sig), color=DATA_COLOR, lw=0.4)
    if dynamic:
        for (a, b), wc in zip(WINDOWS, WIN_COLORS):
            ax.axvspan(a, b, color=wc, alpha=0.28, lw=0)
    ax.set_xlim(0, T_SHOW)
    ax.set_ylim(-4, 4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_title(titles[j])
    ax.set_xlabel('Time (s)')
    if j == 0:
        ax.set_ylabel('Signal')
    P.panel_label(ax, 'abcd'[j], dx=-0.12, dy=1.45, fontsize=12)

    # -- row 2: the autocorrelation ------------------------------------------
    ax = fig.add_subplot(gs[1, j])
    if j == 0:
        ax.plot(lags, acf_a, color=DATA_COLOR, lw=1.4, label='AR(7)', zorder=2)
        ax.plot(lags, acf_a_ar1, color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, label='AR(1) form', zorder=3)
    elif j == 1:
        ax.plot(lags, acf_b[:NL + 1], color=DATA_COLOR, lw=1.4, label='AR(1) + osc.', zorder=2)
        ax.plot(lags, acf_b_ar1[:NL + 1], color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, label='AR(1) form', zorder=3)
    elif j == 2:
        ax.plot(lags, acf_c[:NL + 1], color=DATA_COLOR, lw=1.4, label='observed, 1 s', zorder=2)
        ax.plot(lags, acf_c_ar1[:NL + 1], color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, label='AR(1) form', zorder=3)
    else:
        for acf_w, _, _, wc, label in win_curves:
            ax.plot(lags, acf_w, color=wc, lw=1.4, label=label, zorder=2)
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_ylim(-0.45, 1.05)
    ax.set_yticks([0, 1])
    ax.set_xlabel('Lag (s)')
    if j == 0:
        ax.set_ylabel('Autocorrelation')
    ax.legend(loc='upper right', fontsize=8, handlelength=1.6, labelspacing=0.25)

    # -- row 3: the power spectrum -------------------------------------------
    ax = fig.add_subplot(gs[2, j])
    if j == 0:
        ax.loglog(FREQS_A, psd_a, color=DATA_COLOR, lw=1.2, zorder=2)
        ax.loglog(FREQS_A, psd_a_ar1, color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, zorder=3)
    elif j == 1:
        ax.loglog(f_b, psd_b, color=DATA_COLOR, lw=1.2, zorder=2)
        ax.loglog(f_b, psd_b_ar1, color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, zorder=3)
    elif j == 2:
        ax.loglog(f_c, psd_c, color=DATA_COLOR, lw=1.2, zorder=2)
        ax.loglog(f_c, psd_c_ar1, color=FIT_COLOR, ls=FIT_STYLE, lw=FIT_LW, zorder=3)
    else:
        for _, f_w, psd_w, wc, _ in win_curves:
            ax.loglog(f_w, psd_w, color=wc, lw=1.4, zorder=2)
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Frequency (Hz)')
    if j == 0:
        ax.set_ylabel('Power')

save(fig, 'fig_bias.svg')


# ==========================================================================
# notebook cell 7
# ==========================================================================
# ---- Fig. 3: fig_modes -----------------------------------------------------
model = ARModes.from_phi(PHI7, FS, offset=1.0, freqs=np.linspace(0, FS / 2, 1025))

RECOMBINED_COLOR = '#FFB000'     # product of modes (b) and sum of modes (c)
MODE_STYLE = (0, (2, 1.5))       # every individual mode

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw=dict(wspace=0.45))
freqs = np.logspace(np.log10(0.5), np.log10(150), 600)
order = np.argsort([-m.tau for m in model.modes])       # slowest first
shown = [model.modes[i] for i in order]
colours = [P.MODE_COLORS[m.kind] for m in shown]
nlags = 150
lags = np.arange(nlags + 1) / model.fs
mode_handles = [
    proxy(color=P.MODE_COLORS['relaxation'], lw=1.2, ls=MODE_STYLE, label='relaxation mode'),
    proxy(color=P.MODE_COLORS['oscillatory'], lw=1.2, ls=MODE_STYLE, label='oscillatory mode'),
]
LEG = dict(fontsize=8, handlelength=1.8, labelspacing=0.25)

# (a) the modes themselves ----------------------------------------------------
ax = axes[0]
n, step = 125, 1.35
t = np.arange(n) / model.fs
comps = model.impulse_components(n)
total = comps.sum(axis=0)
top = len(shown) * step
ax.plot(t, 0.5 * total / np.max(np.abs(total)) + top, color='k', lw=1.4)
ax.text(t[-1], top + 0.6, 'response to an input', fontsize=8, ha='right', va='bottom')
for i, mode in enumerate(shown):
    base = (len(shown) - 1 - i) * step
    y = 0.5 * comps[order[i]] / np.max(np.abs(comps[order[i]])) + base
    ax.axhline(base, color=P.LIGHT, lw=0.5, zorder=0)
    ax.plot(t, y, color=colours[i], lw=1.3)
    label = (f'$\\tau$ = {mode.tau * 1e3:.0f} ms'
             + ('' if mode.kind == 'relaxation' else f', {mode.freq:.0f} Hz'))
    ax.text(t[-1], base + 0.6, label, fontsize=8, ha='right', va='bottom', color=colours[i])
ax.set_yticks([])
ax.set_ylim(-0.75, top + 1.15)
ax.set_xlabel('Time after input (s)')
ax.set_title('Modes')
ax.spines['left'].set_visible(False)
P.panel_label(ax, 'a', dx=-0.10, dy=1.18, fontsize=12)

# (b) the modes multiply in the spectrum --------------------------------------
ax = axes[1]
centre = lambda power: power / np.exp(np.mean(np.log(power)))
for mode, c in zip(shown, colours):
    ax.loglog(freqs, centre(mode.psd(freqs)), color=c, lw=1.2, ls=MODE_STYLE, zorder=1)
product = psd_from_poles(np.concatenate([m.poles for m in model.modes]),
                         freqs, model.fs, model.offset)
ax.loglog(freqs, centre(model.psd(freqs)), color='k', lw=1.8, zorder=3)
ax.loglog(freqs, centre(product), color=RECOMBINED_COLOR, lw=1.2, ls='--', zorder=4)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Relative power')
ax.set_title('Spectrum: modes multiply')
ax.legend(handles=[proxy(color='k', lw=1.8, label='generating spectrum'),
                   proxy(color=RECOMBINED_COLOR, lw=1.2, ls='--', label='product of modes')]
          + mode_handles, loc='lower left', **LEG)
P.panel_label(ax, 'b', dx=-0.30, dy=1.18, fontsize=12)

# (c) the modes add in the autocorrelation ------------------------------------
ax = axes[2]
acf_comps = model.acf_components(nlags)
for i, c in enumerate(colours):
    ax.plot(lags, acf_comps[order[i]], color=c, lw=1.2, ls=MODE_STYLE, zorder=1)
ax.plot(lags, model.acf(nlags), color='k', lw=1.8, zorder=3)
ax.plot(lags, acf_comps.sum(axis=0), color=RECOMBINED_COLOR, lw=1.2, ls='--', zorder=4)
ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
ax.set_xlim(0, 0.3)
ax.set_xlabel('Lag (s)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation: modes add')
ax.legend(handles=[proxy(color='k', lw=1.8, label='generating ACF'),
                   proxy(color=RECOMBINED_COLOR, lw=1.2, ls='--', label='sum of modes')]
          + mode_handles, loc='upper right', **LEG)
P.panel_label(ax, 'c', dx=-0.30, dy=1.18, fontsize=12)

save(fig, 'fig_modes.svg')


# ==========================================================================
# notebook cell 9
# ==========================================================================
# ---- Fig. 4: fig_benchmark -------------------------------------------------
DURATION, WELCH_WIN_S = 100.0, 4.0
T_FINITE = 1.0
bench = pd.read_csv(RES / 'benchmark_main.csv.gz')
finite = pd.read_csv(RES / 'finite_sampling.csv.gz')
finite = finite[(finite['process'] == 'ar1') & np.isclose(finite['duration'], T_FINITE)]

# (process, label, table, robust fit?)
COLS = [
    (AR1(tau=TAU), 'AR(1)', bench[bench['process'] == 'ar1'], False),
    # The same added rhythm as fig_bias: 20 Hz, a fifth of the aperiodic variance.
    (AR1PlusSine(tau=TAU, f_osc=20.0, snr=0.2), 'AR(1) + sine', bench[bench['process'] == 'ar1_sine'], True),
    (AR2Damped(tau=TAU, f_osc=10.0), 'AR(2) oscillator', bench[bench['process'] == 'ar2_damped'], True),
    (ARpModes(), ARpModes().label, bench[bench['process'] == 'arp_modes'], False),
    (AR1(tau=TAU), f'Finite sample\n({T_FINITE:g} s of AR(1))', finite, False),
]
METHODS = [('arp_acf', 'ACF'), ('arp_psd', 'PSD')]
OSCILLATORY_TRUTH = {'ar2_damped'}


def target(df, proc):
    """The timescale each AR(p) fit is asked for: the envelope decay for the
    oscillator, the most important relaxation mode otherwise."""
    col = 'top_osc_tau' if proc.name in OSCILLATORY_TRUTH else 'tau_relaxation'
    return df[col].to_numpy(float)


n = len(COLS)
fig = plt.figure(figsize=(8.4, 7.4))
gs = fig.add_gridspec(4, n, hspace=0.6, wspace=0.32, height_ratios=[0.7, 1.0, 1.0, 1.15])
box_vals, tau_axes = [], []
for j, (proc, label, df, robust) in enumerate(COLS):
    rng = np.random.default_rng(3)
    finite_col = j == n - 1
    seconds = T_FINITE if finite_col else DURATION
    signal = prepare(proc.simulate(seconds, FS, rng), FS,
                     welch_win_s=0.5 if finite_col else WELCH_WIN_S)
    fit_psd = estimate(signal, 'arp_psd', robust=robust)
    fit_acf = estimate(signal, 'arp_acf', robust=robust)

    # -- row a: the signal ---------------------------------------------------
    ax = fig.add_subplot(gs[0, j])
    t = np.arange(int(min(1.5, seconds) * FS)) / FS
    y = signal.sig[:len(t)]
    ax.plot(t, y / np.std(y), color=DATA_COLOR, lw=0.5)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-4, 4)
    ax.set_xticks([0, 1])
    ax.set_yticks([])
    ax.set_title(label)
    ax.spines['left'].set_visible(False)
    if j == 0:
        ax.set_ylabel('Signal')
        P.panel_label(ax, 'a', dx=-0.32, dy=1.35, fontsize=12)
    if j == n // 2:
        ax.set_xlabel('Time (s)')

    # -- row b: the spectrum and its AR(p) fit -------------------------------
    ax = fig.add_subplot(gs[1, j])
    m = signal.fit_mask
    f = signal.freqs[m]
    ax.loglog(f, signal.powers[m], color=DATA_COLOR, lw=1.0)
    ax.loglog(f, ar_psd(fit_psd.phi, f, FS, fit_psd.offset), color=FIT_COLOR,
              lw=FIT_LW, ls=FIT_STYLE)
    ax.set_xticks([1, 10, 100])
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if j == 0:
        ax.set_ylabel('Power')
        ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='data'),
                           proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                  loc='lower left', fontsize=8, handlelength=1.2, handletextpad=0.5,
                  borderaxespad=0.2)
        P.panel_label(ax, 'b', dx=-0.32, dy=1.18, fontsize=12)
    if j == n // 2:
        ax.set_xlabel('Frequency (Hz)')

    # -- row c: the autocorrelation and its AR(p) fit ------------------------
    ax = fig.add_subplot(gs[2, j])
    nlags = min(len(signal.corrs) - 1, 150)
    lags = np.arange(nlags + 1) / FS
    ax.plot(lags, signal.corrs[:nlags + 1], color=DATA_COLOR, lw=1.0)
    ax.plot(lags, fit_acf.params['height'] * ar_acf_fft(fit_acf.phi, nlags)
            + fit_acf.params['acf_offset'], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE)
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.6, 1.05)
    ax.set_yticks([0, 1])
    if j == 0:
        ax.set_ylabel('Autocorrelation')
        ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='data'),
                           proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                  loc='upper right', fontsize=8, handlelength=1.2, handletextpad=0.5,
                  borderaxespad=0.2)
        P.panel_label(ax, 'c', dx=-0.32, dy=1.18, fontsize=12)
    else:
        ax.set_yticklabels([])
    if j == n // 2:
        ax.set_xlabel('Lag (s)')

    # -- row d: what each AR(p) fit reports ----------------------------------
    ax = fig.add_subplot(gs[3, j])
    for i, (method, name) in enumerate(METHODS):
        vals = target(df[df['method'] == method], proc) * 1e3
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if not len(vals):
            continue
        box_vals.append(np.percentile(vals, [10, 90]))
        colour = P.METHOD_COLORS[method]
        bp = ax.boxplot([vals], positions=[i], widths=0.6, whis=(10, 90),
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', lw=1.2),
                        whiskerprops=dict(color=colour, lw=0.9),
                        capprops=dict(color=colour, lw=0.9))
        for box in bp['boxes']:
            box.set(facecolor=colour, alpha=0.6, lw=0.7, edgecolor=colour)
    truth = proc.tau_true * 1e3
    if np.isfinite(truth):
        ax.axhline(truth, color=P.MODE_COLORS['oscillatory'] if proc.name in OSCILLATORY_TRUTH else 'k',
                   ls='--', lw=1.1)
    ax.set_yscale('log')
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([name for _, name in METHODS])
    ax.set_xlabel('AR($p$) fit to')
    if j == 0:
        ax.set_ylabel('Estimated $\\tau$ (ms)')
        ax.legend(handles=[proxy(color='k', ls='--', lw=1.1, label='true $\\tau$')],
                  loc='upper right', fontsize=8, handlelength=1.8)
        P.panel_label(ax, 'd', dx=-0.32, dy=1.14, fontsize=12)
    else:
        ax.tick_params(labelleft=False)
    tau_axes.append(ax)

# One y-range for every box panel, covering the 10-90% whiskers of all of
# them, and never below 5 ms: the few fits that collapse to a sample or two
# are real, but they should not set the axis for every other panel.
lo, hi = max(np.min(box_vals) / 1.6, 5.0), np.max(box_vals) * 1.6
for ax in tau_axes:
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))

save(fig, 'fig_benchmark.svg')


# ==========================================================================
# notebook cell 11
# ==========================================================================
# ---- Fig. 5: fig_finite ----------------------------------------------------
from scipy.signal import lfilter
from scipy.stats import gaussian_kde

rng = np.random.default_rng(5)
PHIS = np.linspace(0.5, 0.99, 25)
N_SIMS = 1000
BURN = 200


def phi_hat_ls(x):
    """Mean-corrected least-squares AR(1) coefficient for each row of x."""
    x = x - x.mean(axis=1, keepdims=True)
    return np.sum(x[:, 1:] * x[:, :-1], axis=1) / np.sum(x[:, :-1] ** 2, axis=1)


def ci_panel(ax, n, letter):
    mean_hat, lo, hi = [], [], []
    for phi in PHIS:
        eps = rng.normal(size=(N_SIMS, n + BURN))
        x = lfilter([1.0], [1.0, -phi], eps, axis=1)[:, BURN:]
        ph = phi_hat_ls(x)
        # Least-squares interval, eq. (var) of the paper, around each estimate.
        half = 1.96 * np.sqrt(np.maximum(1.0 - ph ** 2, 0.0) / (n - 1))
        mean_hat.append(ph.mean())
        lo.append((ph - half).mean())
        hi.append((ph + half).mean())
    ax.fill_between(PHIS, lo, hi, color=P.LIGHT, alpha=0.6, lw=0, label='95% CI')
    ax.plot(PHIS, PHIS, color='k', ls=':', lw=1.2, label='true')
    ax.plot(PHIS, mean_hat, color=FIT_COLOR, lw=1.8, label=r'mean $\hat\varphi$')
    ax.set_xlabel(r'True $\varphi$')
    ax.set_ylabel(r'Estimated $\varphi$')
    ax.set_title(f'n = {n} samples')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, 1.05)
    ax.set_box_aspect(1)
    P.panel_label(ax, letter, dx=-0.30, fontsize=12)


fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw=dict(wspace=0.5))
ci_panel(axes[0], 100, 'a')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3,
               handlelength=1.6, columnspacing=1.2)
ci_panel(axes[1], 1000, 'b')

# (c) the sampling density of tau-hat at three recording lengths ------------
sim = pd.read_csv(RES / 'finite_sampling.csv.gz')
durations = np.sort(sim['duration'].unique())
ax = axes[2]
grid = np.linspace(0, 2, 400)
for tot, colour in zip([10, 100, 1000], ['#D55E00', '#E69F00', '#0072B2']):
    near = durations[np.argmin(np.abs(durations - TAU * tot))]
    vals = sim[(sim['process'] == 'ar1') & (sim['method'] == 'ar1_direct')
               & (sim['duration'] == near)]['tau'].to_numpy(float) / TAU
    vals = vals[np.isfinite(vals)]
    ax.plot(grid, gaussian_kde(vals)(grid), color=colour, lw=1.6, label=rf'$T = {tot}\tau$')
ax.axvline(1.0, color='k', ls='--', lw=1.1)
ax.set_xlim(-0.1, 2.1)
ax.set_xlabel(r'$\hat\tau\,/\,\tau$')
ax.set_ylabel('Density')
ax.set_title('Sampling distribution')
ax.set_box_aspect(1)
ax.legend(loc='upper right', handlelength=1.6)
P.panel_label(ax, 'c', dx=-0.30, fontsize=12)

save(fig, 'fig_finite.svg')


# ==========================================================================
# notebook cell 13
# ==========================================================================
# ---- Fig. 6: fig_realdata --------------------------------------------------
fits = pd.read_csv(RES / 'real_data_fits.csv')
keys = D.available()
cache = {k: D.load(k) for k in keys}
N_MODES_SHOWN = 3


def resolvable_modes(fit, signal):
    """The fit's modes, most important first, dropping those slower than the
    fitted band can resolve: a mode with tau > 1 / (2 pi f_lo) has its knee
    below the band, so its timescale is an extrapolation, not a measurement."""
    if fit.modes is None:
        return []
    tau_max = 1.0 / (2.0 * np.pi * signal.f_range[0])
    keep = [m for m in fit.modes.modes if m.tau <= tau_max]
    return sorted(keep, key=lambda m: -m.importance)


n = len(keys)
fig = plt.figure(figsize=(8.4, 1.3 * n + 1.2))
gs = fig.add_gridspec(n, 4, hspace=0.5, wspace=0.42, width_ratios=[1, 1, 1, 0.6])
med = lambda a: np.median(np.vstack(a), axis=0)

for i, key in enumerate(keys):
    data = cache[key]
    sub = fits[fits['dataset'] == key]
    signal = D.prepare_trace(data['X'][0], data)
    mask = signal.fit_mask & signal.exclude_mask
    first, last = i == 0, i == n - 1

    # Every panel comes from the spectral AR(p) fit, which describes these
    # recordings best.  Panel d boxes the modes of every trace's fit.  Panels
    # a-c draw the median spectrum and autocorrelation across traces with ONE
    # fit to the median spectrum: its coefficients give the analytic AR(p)
    # autocorrelation in b and the modes that autocorrelation is made of in c.
    powers_all, corrs_all = [], []
    mode_taus = [[] for _ in range(N_MODES_SHOWN)]
    nlags = len(signal.corrs) - 1
    for trace in data['X']:
        sig = D.prepare_trace(trace, data)
        if len(sig.corrs) - 1 != nlags:
            continue
        powers_all.append(sig.powers)
        corrs_all.append(sig.corrs)
        fp_i = estimate(sig, 'arp_psd', robust=True)
        if fp_i.success and fp_i.phi is not None:
            for r, mode in enumerate(resolvable_modes(fp_i, sig)[:N_MODES_SHOWN]):
                mode_taus[r].append(mode.tau * 1e3)
    median_signal = copy.copy(signal)       # keeps the masks prepare() attaches
    median_signal.powers, median_signal.corrs = med(powers_all), med(corrs_all)
    psd_data, acf_data = median_signal.powers[mask], median_signal.corrs
    fp = estimate(median_signal, 'arp_psd', robust=True)
    psd_fit = ar_psd(fp.phi, signal.freqs, signal.fs, fp.offset)
    # The data's autocorrelation is the inverse transform of its spectrum over
    # the analysis band (prepare), so the fit is translated the same way: its
    # spectrum on the same grid, over the same band.  The model's analytic
    # full-band autocorrelation would instead carry its extrapolation below
    # the band's lower edge, where most of a 1/f-like model's variance sits,
    # and miss the data even where the fit is exact.  The components are the
    # partial fractions of the fitted spectrum, transformed the same way; all
    # of them together add up to the fit.
    acov_fit = spectrum_acov(median_signal, psd_fit, nlags)
    acf_fit = acov_fit / acov_fit[0]
    comps = spectrum_acov(median_signal, fp.modes.psd_components(signal.freqs),
                          nlags) / acov_fit[0]
    index = {id(m): j for j, m in enumerate(fp.modes.modes)}
    shown = resolvable_modes(fp, median_signal)[:N_MODES_SHOWN]
    mode_curves = [comps[index[id(m)]] for m in shown]
    # c and d show the same components: the ranks the fit to the median
    # spectrum resolves, boxed over the traces that resolve them.
    n_shown = len(shown)
    lags = np.arange(nlags + 1) / signal.fs
    taus = [t for t in (sub['tau'].median(), sub['tau_relaxation'].median())
            if np.isfinite(t) and t > 0]
    span = float(np.clip(max(6 * max(taus), 12 / signal.fs), 4 / signal.fs, lags[-1]))
    # Only the lags up to the span are drawn and the axes keep their default
    # limits, so lag zero sits at the same place in every row.
    keep = lags <= span

    # -- a: spectrum -------------------------------------------------------
    ax = fig.add_subplot(gs[i, 0])
    ax.loglog(signal.freqs[mask], psd_data, color=DATA_COLOR, lw=1.1, zorder=1)
    ax.loglog(signal.freqs[mask], psd_fit[mask], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
    ax.set_ylabel(data['panel_name'], labelpad=4)
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if first:
        ax.set_title('Power spectrum')
        P.panel_label(ax, 'a', dx=-0.26, dy=1.45, fontsize=12)
    if last:
        ax.set_xlabel('Frequency (Hz)')

    # -- b: autocorrelation and the fit's analytic autocorrelation ----------
    axb = fig.add_subplot(gs[i, 1])
    axb.plot(lags[keep], acf_data[keep], color=DATA_COLOR, lw=1.1, zorder=1)
    axb.plot(lags[keep], acf_fit[keep], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
    axb.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    if first:
        axb.set_title('Autocorrelation')
        P.panel_label(axb, 'b', dx=-0.24, dy=1.45, fontsize=12)
    if last:
        axb.set_xlabel('Lag (s)')

    # -- c: the modes that autocorrelation is made of, sharing b's axes ------
    axc = fig.add_subplot(gs[i, 2], sharex=axb, sharey=axb)
    for r, curve in enumerate(mode_curves):
        axc.plot(lags[keep], curve[keep], color=RANK_COLORS[r], lw=1.3, zorder=2)
    axc.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    axc.tick_params(labelleft=False)
    if first:
        axc.set_title('ACF decomposition')
        P.panel_label(axc, 'c', dx=-0.12, dy=1.45, fontsize=12)
    if last:
        axc.set_xlabel('Lag (s)')

    # -- d: the timescale of each component, over all traces -----------------
    ax = fig.add_subplot(gs[i, 3])
    drawn = []
    for r in range(n_shown):
        vals = np.array([v for v in mode_taus[r] if np.isfinite(v) and v > 0])
        if not len(vals):
            continue
        drawn.append(vals)
        colour = RANK_COLORS[r]
        bp = ax.boxplot([vals], positions=[r], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='k', lw=1.2),
                        whiskerprops=dict(color=colour, lw=0.9),
                        capprops=dict(color=colour, lw=0.9))
        for box in bp['boxes']:
            box.set(facecolor=colour, alpha=0.55, lw=0.7, edgecolor=colour)
    ax.set_yscale('log')
    ax.set_xlim(-0.7, N_MODES_SHOWN - 0.3)
    ax.set_xticks(np.arange(N_MODES_SHOWN))
    ax.set_xticklabels([f'{r + 1}' for r in range(N_MODES_SHOWN)] if last else [])
    ax.set_ylabel('ms', labelpad=1)
    if drawn:
        decade_ticks(ax, np.concatenate(drawn))
    if first:
        ax.set_title('Timescales')
        P.panel_label(ax, 'd', dx=-0.55, dy=1.45, fontsize=12)
    if last:
        ax.set_xlabel('Component')

handles = ([proxy(color=DATA_COLOR, lw=1.2, label='data'),
            proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')]
           + [proxy(color=RANK_COLORS[r], lw=1.4, label=f'component {r + 1}')
              for r in range(N_MODES_SHOWN)])
fig.subplots_adjust(bottom=0.06)
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0),
           handlelength=2.0, columnspacing=1.6)

save(fig, 'fig_realdata.svg')


# ==========================================================================
# notebook cell 15
# ==========================================================================
# ---- Fig. S1: figS_oscillation_grid ----------------------------------------
sweep = pd.read_csv(RES / 'oscillation_sweep.csv.gz')
SNRS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FREQS = [4.0, 6.0, 10.0, 16.0, 25.0, 40.0]
show = ['ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

fig, axes = plt.subplots(1, len(show), figsize=(8.4, 2.5))
for ax, method in zip(axes, show):
    grid = np.full((len(SNRS), len(FREQS)), np.nan)
    for i, snr in enumerate(SNRS):
        for j, f_osc in enumerate(FREQS):
            sel = ((sweep['method'] == method) & np.isclose(sweep['param_snr'], snr)
                   & np.isclose(sweep['param_f_osc'], f_osc))
            vals = sweep.loc[sel, 'tau_relaxation'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                grid[i, j] = np.median(vals) / TAU
    # Colour and number are the same quantity: estimate / truth, log-spaced so
    # a factor of two too big and too small are equally far from white.
    im = ax.imshow(np.log10(grid), origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(SNRS)):
        for j in range(len(FREQS)):
            value = grid[i, j]
            txt = f'{value:.2f}' if np.isfinite(value) else 'none'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                    color='k' if abs(np.log10(max(value, 1e-9))) < 0.6 else 'white')
    ax.set_xticks(range(len(FREQS)))
    ax.set_xticklabels([f'{f:g}' for f in FREQS])
    ax.set_yticks(range(len(SNRS)))
    ax.set_yticklabels([f'{v:g}' for v in SNRS] if ax is axes[0] else [])
    ax.set_xlabel('Oscillation freq. (Hz)')
    ax.set_title(ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'), fontsize=10)
axes[0].set_ylabel('Oscillation /\naperiodic power')
ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=np.log10(ticks))
cb.ax.set_yticklabels([f'{t:g}' for t in ticks])
cb.set_label('estimated $\\tau$ / true $\\tau$')

save(fig, 'figS_oscillation_grid.svg')


# ==========================================================================
# notebook cell 17
# ==========================================================================
# ---- Fig. S2: figS_windows -------------------------------------------------
emp = pd.read_csv(RES / 'empirical_truncation_summary.csv')
keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(wspace=0.55))

# (a) the median estimate against window length -------------------------------
ax = axes[0]
for key in keys:
    sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_acf')]
    ax.plot(sub['duration'] / sub['reference_tau'], sub['rel_to_longest'], 'o-',
            ms=3, lw=1.1, color=colours[key], label=D.SHORT_NAMES.get(key, key))
ax.axhline(1.0, color='k', ls=':', lw=1.0)
ax.axhspan(0.8, 1.25, color=P.LIGHT, alpha=0.35, lw=0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.8, 3e4); ax.set_ylim(0.2, 3.2)
ax.set_yticks([0.25, 0.5, 1, 2])
ax.set_yticklabels(['0.25', '0.5', '1', '2'])
ax.set_xlabel(r'Window length / $\hat\tau$')
ax.set_ylabel(r'$\hat\tau(T)\,/\,\hat\tau$(longest)')
ax.set_title('Window dependence')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.28),
          columnspacing=0.8, handlelength=1.2, fontsize=8)
P.panel_label(ax, 'a', dx=-0.24, fontsize=12)

# (b) the spread between windows, against the sampling bound ------------------
ax = axes[1]
height = 0.36
for i, method in enumerate(('ar1_acf', 'ar1_psd')):
    vals = [emp[(emp['dataset'] == k) & (emp['method'] == method)]['excess_sd'].iloc[-1]
            for k in keys]
    ax.barh(np.arange(len(keys)) + (i - 0.5) * height, vals, height=height,
            color=P.METHOD_COLORS[method], label=ESTIMATOR_LABELS[method])
ax.axvline(1.0, color='k', ls='--', lw=1.1)
ax.set_xscale('log')
ax.set_yticks(np.arange(len(keys)))
ax.set_yticklabels([D.SHORT_NAMES.get(k, k) for k in keys], fontsize=8)
ax.set_xlabel('Observed spread / sampling bound')
ax.set_title('Beyond sampling noise')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=1.2,
          fontsize=8)
P.panel_label(ax, 'b', dx=-0.46, fontsize=12)

save(fig, 'figS_windows.svg')

