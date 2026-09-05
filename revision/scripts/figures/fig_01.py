"""Fig. 1 -- an AR(1) as a signal, an autocorrelation and a spectrum (fig_ar1.svg).

Run on its own from any directory: python scripts/figures/fig_01.py
"""
# %%
from pathlib import Path
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import shutil

from fig_common import (
    DATA_COLOR, TEXT_WIDTH, save, sim_ar_process)

# %%
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
# with plt.rc_context({'text.usetex': HAVE_LATEX}):
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(TEXT_WIDTH, TEXT_WIDTH * 5 / 12), height_ratios=[.6, .4])
fig.subplots_adjust(wspace=.5, hspace=0.75)

st = 8
axes[0, 0].set_title("Autoregressive: AR(1)", size=st)
axes[0, 1].set_title("Autocorrelation", size=st)
axes[0, 2].set_title("Power Spectral Density", size=st)

axes[0, 0].plot(time[:500], sig[:500], color=DATA_COLOR)
axes[0, 1].plot(lags, corrs, color=DATA_COLOR)
axes[0, 2].loglog(freqs, powers, color=DATA_COLOR)

size = 7

def eq_text(ax, x, y, s):
    # Figure-level text: LaTeX text placed through an axes is clipped to
    # the axes box, and these equations are wider than their column.
    b = ax.get_position()
    # Hung from the top of the bottom-row axes so the block cannot reach
    # the x-labels of the row above.
    fig.text(b.x0 + x * b.width, b.y1, s, size=size, va='top')

if HAVE_LATEX:
    s0 = r"\begin{align*}"
    s0 += r"x_t &= [ x_0, x_1, x_2, \dots, x_{n-1} ]^{\top}\\"
    s0 += r"x_{t+1} &= [ x_1, x_2, x_3, \dots, x_{n} ]^{\top}\\"
    s0 += r"x_{t+1} &= \varphi x_{t} + \epsilon"
    s0 += r"\end{align*}"
    eq_text(axes[1, 0], .05, .6, s0)

    s1 = r"\begin{align*}"
    s1 += r"A(k)&= e^{\frac{-k}{\tau f_s}}\\"
    s1 += r"&= \varphi^k"
    s1 += r"\end{align*}"
    eq_text(axes[1, 1], .3, .6, s1)

    s2 = r"\begin{align*}"
    s2 += r"P(f) &= \frac{1}{1 - 2 \varphi \cos{(2 \pi \frac{f}{f_s})} + \varphi^2}"
    s2 += r"\end{align*}"
    eq_text(axes[1, 2], .03, .6, s2)
else:
    axes[1, 0].text(.02, .85, r"$x_t = [\,x_0, x_1, x_2, \dots, x_{n-1}\,]^{\top}$", size=size)
    axes[1, 0].text(.02, .55, r"$x_{t+1} = [\,x_1, x_2, x_3, \dots, x_{n}\,]^{\top}$", size=size)
    axes[1, 0].text(.02, .25, r"$x_{t+1} = \varphi\, x_{t} + \epsilon$", size=size)
    axes[1, 1].text(.3, .55, r"$A(k) = e^{-k/(\tau f_s)} = \varphi^{k}$", size=size)
    axes[1, 2].text(.02, .55,
                    r"$P(f) = \dfrac{1}{1 - 2\varphi\cos(2\pi f/f_s) + \varphi^{2}}$",
                    size=size)

sl = 7
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
