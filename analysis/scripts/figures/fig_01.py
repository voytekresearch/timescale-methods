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
time = np.linspace(0, len(sig)/fs, len(sig))

lags = np.arange(101)
corrs = phi ** lags

freqs = np.linspace(.5, fs / 2, 1000)
powers = 1 / (1 + phi ** 2 - 2 * phi * np.cos(2 * np.pi * freqs * (1 / fs)))

with plt.rc_context({'text.usetex': HAVE_LATEX}):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(TEXT_WIDTH, (TEXT_WIDTH * 5 / 12) * 0.5))

    fig.subplots_adjust(wspace=.5, hspace=0.75)
    st = 8
    axes[0].set_title("Autoregressive: AR(1)", size=st)
    axes[1].set_title("Autocorrelation", size=st)
    axes[2].set_title("Power Spectral Density", size=st)

    axes[0].plot(time[:500], sig[:500], color=DATA_COLOR)
    axes[1].plot(lags, corrs, color=DATA_COLOR)
    axes[2].loglog(freqs, powers, color=DATA_COLOR)

    sl = 7
    axes[0].set_xlabel("Time (seconds)", size=sl)
    axes[0].set_ylabel(r"Voltage ($x$, a.u.)", size=sl)
    axes[1].set_ylabel(r"$A(k)$ (correlation)", size=sl)
    axes[1].set_xlabel(r"Lag ($k$, samples)", size=sl)
    axes[2].set_xlabel(r"Frequency ($f$, Hertz)", size=sl)
    axes[2].set_ylabel(r"$P(f)$ (a.u.)", size=sl)


    save(fig, 'fig_ar1.svg')
