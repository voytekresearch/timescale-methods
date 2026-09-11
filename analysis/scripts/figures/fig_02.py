"""Fig. 2 -- four ways a timescale estimate goes wrong (fig_bias.svg).

Run on its own from any directory: python scripts/figures/fig_02.py
"""
# %%
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    AR2_OSC, ARModes, ARP, DATA_COLOR, FS, P, PHI_ARP, TAU, TEXT_WIDTH,
    phi_of_tau, save, sim_ar_process)

# %%
rng = np.random.default_rng(1)
T_SHOW = 5.0                                   # seconds simulated for the signal row
# Seconds drawn per column.  One second is enough to see the shape of the
# AR(7) and the 10 Hz rhythm, which five seconds of a 10 Hz rhythm on a
# panel an inch wide reduce to a solid band; the finite-sample column keeps
# the five-second axis because its point is how little of it is filled, and
# the dynamic column needs all five seconds for its windows.
T_DRAW = [2.0, 1.0, 5.0, 5.0]
NL = 250                                       # lags drawn in the ACF row: 0.5 s
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


# a) model mismatch: the AR(p) of fig_modes, the same model object and the
#    same exact spectrum as its panel b.
model_arp = ARModes.from_phi(PHI_ARP, FS, offset=1.0, freqs=np.linspace(0, FS / 2, 1025))
FREQS_A = np.logspace(np.log10(0.25), np.log10(FS / 2), 500)
acf_a = model_arp.acf(NL)
psd_a = model_arp.psd(FREQS_A)
sig_a = sim_ar_process(PHI_ARP, int(T_SHOW * FS), rng)
LABEL_A = ARP.label.split(',')[0]                  # 'AR(7)' or 'AR(5)'
# b) oscillations: the AR(2) stochastic oscillator of the benchmark, a 10 Hz
#    rhythm whose envelope decays in 150 ms, 1.5 cycles.  Its exact
#    autocorrelation is a damped cosine and its spectrum a peak twenty times
#    its low-frequency plateau.
acf_b = AR2_OSC.acf(np.arange(N_ACF + 1), FS)
f_b, psd_b = acf_to_psd(acf_b, FS)
sig_b = AR2_OSC.simulate(T_SHOW, FS, rng)

# c) finite samples: one second of an AR(1), observed rather than theoretical.
T_FINITE = 1.0
sig_c = sim_ar_process(np.array([phi_of_tau(TAU, FS)]), int(T_FINITE * FS), rng)
sig_c = (sig_c - sig_c.mean()) / sig_c.std()
acf_c = sample_acf(sig_c, len(sig_c) - 1)
f_c, psd_c = acf_to_psd(acf_c, FS)

# d) dynamic timescales: an AR(1) whose coefficient rises over the recording,
#    with evenly spaced windows and the theoretical AR(1) of each window.
#    The innovation is scaled by sqrt(1 - phi^2) so the process keeps unit
#    variance as the timescale changes: with a fixed innovation the variance
#    of an AR(1) grows as 1 / (1 - phi^2), seventeen-fold from 10 to 200 ms,
#    and the slow end of the recording swamped the fast end once the trace
#    was scaled to fit, hiding the change in timescale that the panel is
#    about.  Only the timescale changes here, not the amplitude.  The
#    timescale rises geometrically rather than linearly, so the recording
#    spends as long at the fast end, where the texture of the trace changes
#    most per millisecond of tau, as at the slow end.
n_d = int(T_SHOW * FS)
tau_t = np.geomspace(0.010, 0.200, n_d)            # 10 ms -> 200 ms
phi_t = phi_of_tau(tau_t, FS)
eps = rng.normal(size=n_d) * np.sqrt(1.0 - phi_t ** 2)
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

with plt.rc_context({'text.usetex': True}):
    fig = plt.figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 6.2 / 8.4))
    gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.38, height_ratios=[0.7, 1.0, 1.0])
    titles = ['Model\nmismatch', 'Oscillations\n', 'Finite\nsamples', 'Dynamic\ntimescales']
    signals = [sig_a, sig_b, sig_c, sig_d]
    abcd = ["a", "b", "c", "d"]

    for j in range(4):
        dynamic = j == 3

        # -- row 1: the signal ---------------------------------------------------
        ax = fig.add_subplot(gs[0, j])
        sig = signals[j]
        t = np.arange(len(sig)) / FS
        shown = t <= T_DRAW[j]
        ax.plot(t[shown], sig[shown] / np.std(sig), color=DATA_COLOR, lw=0.4)
        if dynamic:
            for (a, b), wc in zip(WINDOWS, WIN_COLORS):
                ax.axvspan(a, b, color=wc, alpha=0.28, lw=0)
        ax.set_xlim(0, T_DRAW[j])
        ax.set_xticks({1.0: [0, 0.5, 1], 2.0: [0, 1, 2]}.get(T_DRAW[j], [0, 2, 4]))
        ax.set_ylim(-4, 4)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_title(titles[j])
        ax.set_xlabel('Time (s)')
        if j == 0:
            ax.set_ylabel('Signal')

        P.panel_label(ax, abcd[j], dx=-0.22, dy=1.55, fontsize=9)

        # -- row 2: the autocorrelation ------------------------------------------
        ax = fig.add_subplot(gs[1, j])
        if j == 0:
            ax.plot(lags, acf_a, color=DATA_COLOR, lw=1.4, label=LABEL_A, zorder=2)
        elif j == 1:
            ax.plot(lags, acf_b[:NL + 1], color=DATA_COLOR, lw=1.4, label='AR(2)', zorder=2)
        elif j == 2:
            ax.plot(lags, acf_c[:NL + 1], color=DATA_COLOR, lw=1.4, label='Observed', zorder=2)
        else:
            for acf_w, _, _, wc, label in win_curves:
                ax.plot(lags, acf_w, color=wc, lw=1.4, label=label, zorder=2)
        ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        ax.set_ylim(-0.8, 1.05)      # the AR(2)'s first trough reaches -0.7
        ax.set_yticks([0, 1])
        ax.set_xlabel('Lag (s)')
        if j == 0:
            ax.set_ylabel('Autocorrelation')
        if dynamic:      # four entries: below the curves, which never go negative
            ax.legend(loc='lower right', ncol=1, fontsize=4.25, handlelength=1.0,
                    handletextpad=0.2, columnspacing=0.6, labelspacing=0.2,
                    borderaxespad=0.1)
        else:
            ax.legend(loc='upper right', fontsize=6, handlelength=1.0, borderaxespad=0.0, labelspacing=0.25)

        # -- row 3: the power spectrum -------------------------------------------
        ax = fig.add_subplot(gs[2, j])
        if j == 0:
            ax.loglog(FREQS_A, psd_a, color=DATA_COLOR, lw=1.2, zorder=2)
        elif j == 1:
            ax.loglog(f_b, psd_b, color=DATA_COLOR, lw=1.2, zorder=2)
        elif j == 2:
            ax.loglog(f_c, psd_c, color=DATA_COLOR, lw=1.2, zorder=2)
        else:
            for _, f_w, psd_w, wc, _ in win_curves:
                ax.loglog(f_w, psd_w, color=wc, lw=1.4, zorder=2)
        ax.set_yticks([])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel('Frequency (Hz)')
        if j == 0:
            ax.set_ylabel('Power')

    save(fig, 'fig_bias.svg')
