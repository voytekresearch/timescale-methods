"""Generative processes with closed-form spectra used in the benchmark.

Every process exposes the same interface:

``simulate(n_seconds, fs, rng)``  a realization
``psd(freqs, fs)``                the theoretical continuous one-sided PSD
``lines``                         discrete spectral lines, as (freq, power)
``acf(lags, fs)``                 the theoretical autocorrelation
``tau_true``                      the aperiodic relaxation time, when defined
``tau_integrated(fs)``            the integrated timescale, int_0^inf rho(t) dt

The integrated timescale is the model-free scalar summary that every estimator
is implicitly targeting: it equals ``tau`` for an AR(1), is a variance-weighted
mean of component timescales for a mixture, and diverges for a long-memory
process -- which is precisely when a scalar timescale stops being a property of
the signal.  It is evaluated with the trapezoid convention,
``(1/fs) (sum_{k>=0} rho(k) - rho(0)/2)``, so that a sampled exponential returns
its continuous-time ``tau`` rather than ``tau + 1/(2 fs)``.

The one-sided density convention is ``S(f) = (2 sigma^2 / fs) / |A(f)|^2`` so
that ``int_0^{fs/2} S(f) df = Var(x)``, matching ``scipy.signal.welch``.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from .modes import ARModes, ar_acf, ar_psd, ar_to_poles, poles_to_ar

_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tau_to_phi(tau, fs):
    return float(np.exp(-1.0 / (tau * fs)))


def phi_to_tau(phi, fs):
    return float(-1.0 / (fs * np.log(phi)))


def sim_ar_process(phi, n, rng, sigma=1.0, burn_in=None):
    """Simulate ``x_t = sum_k phi_k x_{t-k} + eps_t`` with burn-in."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = len(phi)
    if burn_in is None:
        radius = np.max(np.abs(ar_to_poles(phi))) if p else 0.0
        radius = min(radius, 0.999999)
        burn_in = int(np.clip(20.0 / max(1e-9, -np.log(max(radius, 1e-12))), 500, 200000))
    total = n + burn_in
    eps = rng.standard_normal(total) * sigma
    x = np.zeros(total)
    for i in range(p, total):
        x[i] = phi @ x[i - p:i][::-1] + eps[i]
    return x[burn_in:]


def ar_psd_density(phi, freqs, fs, sigma2=1.0):
    """One-sided AR(p) spectral density in variance-per-Hz units."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    k = np.arange(1, len(phi) + 1)
    z_inv = np.exp(-2j * np.pi * np.outer(freqs, k) / fs)
    return (2.0 * sigma2 / fs) / np.abs(1.0 - z_inv @ phi) ** 2


def davies_harte(acov, n, rng):
    """Exact circulant-embedding simulation of a stationary Gaussian process.

    ``acov`` must hold gamma(0), ..., gamma(n).
    """
    acov = np.asarray(acov, dtype=float)
    if len(acov) < n + 1:
        raise ValueError('Need autocovariance up to lag n.')
    c = np.concatenate([acov[: n + 1], acov[n - 1:0:-1]])
    lam = np.fft.fft(c).real
    if lam.min() < -1e-8 * lam.max():
        lam = np.clip(lam, 0.0, None)  # nearest PSD embedding
    m = len(c)

    z = rng.standard_normal(m)
    y = np.zeros(m, dtype=complex)
    y[0] = np.sqrt(lam[0]) * z[0]
    y[n] = np.sqrt(lam[n]) * z[n]
    half = np.arange(1, n)
    y[half] = np.sqrt(lam[half] / 2.0) * (z[half] + 1j * z[m - half])
    y[m - half] = np.conj(y[half])
    x = np.fft.fft(y).real / np.sqrt(m)
    return x[:n]


def integrated_timescale_from_acf(rho, fs):
    """``int_0^inf rho(t) dt`` by trapezoid on a sampled ACF."""
    return float(_trapz(rho, dx=1.0 / fs))


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class Process:
    """Base class; subclasses fill in the analytic forms.

    Fields are keyword-only on purpose.  ``Process`` carries its own fields
    (``name``, ``label``, ``tau_true``, ...) ahead of every subclass's, so
    positional construction bound ``AR1PlusSine(0.05, 4.0, 4.0)`` to
    ``name``/``label``/``tau_true`` and silently used the default tau,
    frequency and amplitude -- and ``__post_init__`` then overwrote those
    fields, erasing the evidence.  Keyword-only makes the mistake a TypeError.
    """

    name: str = 'process'
    label: str = 'process'
    tau_true: float = np.nan
    lines: tuple = ()
    params: dict = field(default_factory=dict)
    # Whether the process contains a rhythm.  Robust regression is only
    # appropriate when there is a narrowband feature to treat as an outlier.
    has_oscillation: bool = False

    def simulate(self, n_seconds, fs, rng):
        raise NotImplementedError

    def psd(self, freqs, fs):
        raise NotImplementedError

    def acf(self, lags, fs):
        raise NotImplementedError

    def tau_integrated(self, fs, max_lag_s=None):
        max_lag_s = 200 * (self.tau_true if np.isfinite(self.tau_true) else 1.0) \
            if max_lag_s is None else max_lag_s
        lags = np.arange(0, int(max_lag_s * fs) + 1)
        return integrated_timescale_from_acf(self.acf(lags, fs), fs)

    def psd_with_lines(self, freqs, fs, enbw):
        """Continuous PSD plus discrete lines smeared over the analysis bandwidth."""
        out = self.psd(freqs, fs).astype(float)
        for f0, power in self.lines:
            idx = int(np.argmin(np.abs(freqs - f0)))
            out[idx] += power / enbw
        return out


@dataclass(kw_only=True)
class AR1(Process):
    """Positive control: a single relaxation timescale."""

    tau: float = 0.05

    def __post_init__(self):
        self.name = 'ar1'
        self.label = f'AR(1), $\\tau$={self.tau*1e3:g} ms'
        self.tau_true = self.tau
        self.params = {'tau': self.tau}

    def phi(self, fs):
        return np.array([tau_to_phi(self.tau, fs)])

    def simulate(self, n_seconds, fs, rng):
        return sim_ar_process(self.phi(fs), int(round(n_seconds * fs)), rng)

    def psd(self, freqs, fs):
        return ar_psd_density(self.phi(fs), freqs, fs)

    def acf(self, lags, fs):
        return tau_to_phi(self.tau, fs) ** np.asarray(lags, dtype=float)

    def tau_integrated(self, fs, max_lag_s=None):
        phi = tau_to_phi(self.tau, fs)
        return float((1.0 / fs) * (1.0 / (1.0 - phi) - 0.5))


@dataclass(kw_only=True)
class AR1PlusSine(Process):
    """Oscillatory contamination: AR(1) plus an additive sinusoid."""

    tau: float = 0.05
    f_osc: float = 10.0
    snr: float = 1.0   # sinusoid variance / AR(1) variance

    def __post_init__(self):
        self.name = 'ar1_sine'
        self.has_oscillation = True
        self.label = (f'AR(1)+sine, $\\tau$={self.tau*1e3:g} ms, '
                      f'{self.f_osc:g} Hz, SNR={self.snr:g}')
        self.tau_true = self.tau
        self.params = {'tau': self.tau, 'f_osc': self.f_osc, 'snr': self.snr}

    def phi(self, fs):
        return np.array([tau_to_phi(self.tau, fs)])

    def _var_ar(self, fs):
        phi = tau_to_phi(self.tau, fs)
        return 1.0 / (1.0 - phi ** 2)

    def amplitude(self, fs):
        return float(np.sqrt(2.0 * self.snr * self._var_ar(fs)))

    def simulate(self, n_seconds, fs, rng):
        n = int(round(n_seconds * fs))
        x = sim_ar_process(self.phi(fs), n, rng)
        t = np.arange(n) / fs
        phase = rng.uniform(0, 2 * np.pi)
        return x + self.amplitude(fs) * np.cos(2 * np.pi * self.f_osc * t + phase)

    def psd(self, freqs, fs):
        return ar_psd_density(self.phi(fs), freqs, fs)

    def line_power(self, fs):
        return self.amplitude(fs) ** 2 / 2.0

    def psd_with_lines(self, freqs, fs, enbw):
        out = self.psd(freqs, fs).astype(float)
        idx = int(np.argmin(np.abs(freqs - self.f_osc)))
        out[idx] += self.line_power(fs) / enbw
        return out

    def acf(self, lags, fs):
        lags = np.asarray(lags, dtype=float)
        var_ar = self._var_ar(fs)
        var_osc = self.line_power(fs)
        rho = (var_ar * tau_to_phi(self.tau, fs) ** lags
               + var_osc * np.cos(2 * np.pi * self.f_osc * lags / fs))
        return rho / (var_ar + var_osc)

    def tau_integrated(self, fs, max_lag_s=None):
        # The sinusoid contributes no convergent integral; report the
        # aperiodic component's integrated timescale.
        return AR1(tau=self.tau).tau_integrated(fs)


@dataclass(kw_only=True)
class AR2Damped(Process):
    """A damped stochastic oscillator: one complex-conjugate pole pair."""

    tau: float = 0.05
    f_osc: float = 10.0

    def __post_init__(self):
        self.name = 'ar2_damped'
        self.has_oscillation = True
        self.label = (f'AR(2) damped osc., $\\tau$={self.tau*1e3:g} ms, '
                      f'{self.f_osc:g} Hz')
        self.tau_true = self.tau
        self.params = {'tau': self.tau, 'f_osc': self.f_osc}

    def phi(self, fs):
        r = tau_to_phi(self.tau, fs)
        theta = 2 * np.pi * self.f_osc / fs
        return np.array([2 * r * np.cos(theta), -r ** 2])

    def poles(self, fs):
        r = tau_to_phi(self.tau, fs)
        theta = 2 * np.pi * self.f_osc / fs
        return np.array([r * np.exp(1j * theta), r * np.exp(-1j * theta)])

    def simulate(self, n_seconds, fs, rng):
        return sim_ar_process(self.phi(fs), int(round(n_seconds * fs)), rng)

    def psd(self, freqs, fs):
        return ar_psd_density(self.phi(fs), freqs, fs)

    def acf(self, lags, fs):
        nlags = int(np.max(lags))
        full = ar_acf(self.phi(fs), nlags)
        return full[np.asarray(lags, dtype=int)]

    def tau_integrated(self, fs, max_lag_s=None):
        # With S(f) = (2/fs) sum_k gamma(k) e^{-i w k}, S(0) = 4 gamma(0) tau_int.
        phi = self.phi(fs)
        s0 = ar_psd_density(phi, np.array([0.0]), fs)[0]
        var = ar_acf(phi, 1, normalize=False)[0]
        return float(s0 / (4.0 * var))


@dataclass(kw_only=True)
class SumAR1(Process):
    """Two *independent* AR(1) processes added together.

    The sum is an ARMA(2,1): the spectral denominator keeps both latent poles,
    but the numerator gains a moving-average zero, so a pure AR(p) fit cannot
    reproduce it exactly and its poles need not land on the latent taus.
    """

    tau0: float = 0.01
    tau1: float = 0.20
    weight: float = 0.5   # fraction of total variance carried by the slow mode

    def __post_init__(self):
        self.name = 'sum_ar1'
        self.label = (f'Sum of 2 AR(1), $\\tau$={self.tau0*1e3:g}/'
                      f'{self.tau1*1e3:g} ms, w={self.weight:g}')
        self.tau_true = np.nan
        self.params = {'tau0': self.tau0, 'tau1': self.tau1, 'weight': self.weight}

    def _scales(self, fs):
        """Innovation scales giving the requested variance split."""
        phi0, phi1 = tau_to_phi(self.tau0, fs), tau_to_phi(self.tau1, fs)
        v0, v1 = 1.0 / (1 - phi0 ** 2), 1.0 / (1 - phi1 ** 2)
        a0 = np.sqrt((1 - self.weight) / v0)
        a1 = np.sqrt(self.weight / v1)
        return (phi0, phi1), (a0, a1)

    def simulate(self, n_seconds, fs, rng):
        n = int(round(n_seconds * fs))
        (phi0, phi1), (a0, a1) = self._scales(fs)
        x0 = sim_ar_process([phi0], n, rng)
        x1 = sim_ar_process([phi1], n, rng)
        return a0 * x0 + a1 * x1

    def psd(self, freqs, fs):
        (phi0, phi1), (a0, a1) = self._scales(fs)
        return (a0 ** 2 * ar_psd_density([phi0], freqs, fs)
                + a1 ** 2 * ar_psd_density([phi1], freqs, fs))

    def acf(self, lags, fs):
        lags = np.asarray(lags, dtype=float)
        (phi0, phi1), (a0, a1) = self._scales(fs)
        v0, v1 = a0 ** 2 / (1 - phi0 ** 2), a1 ** 2 / (1 - phi1 ** 2)
        return (v0 * phi0 ** lags + v1 * phi1 ** lags) / (v0 + v1)

    def tau_integrated(self, fs, max_lag_s=None):
        (phi0, phi1), (a0, a1) = self._scales(fs)
        v0, v1 = a0 ** 2 / (1 - phi0 ** 2), a1 ** 2 / (1 - phi1 ** 2)
        t0 = (1.0 / fs) * (1.0 / (1 - phi0) - 0.5)
        t1 = (1.0 / fs) * (1.0 / (1 - phi1) - 0.5)
        return float((v0 * t0 + v1 * t1) / (v0 + v1))

    def arma_representation(self, fs):
        """AR poles and the MA zero of the exact ARMA(2,1) representation."""
        (phi0, phi1), (a0, a1) = self._scales(fs)
        # numerator: a0^2 |1 - phi1 z|^2 + a1^2 |1 - phi0 z|^2
        c0 = a0 ** 2 * (1 + phi1 ** 2) + a1 ** 2 * (1 + phi0 ** 2)
        c1 = -(a0 ** 2 * phi1 + a1 ** 2 * phi0)
        # c0 + 2 c1 cos(w) = k |1 - theta e^{-iw}|^2 -> theta from c1/c0
        ratio = c1 / c0
        roots = np.roots([ratio, 1.0, ratio])
        theta = roots[np.argmin(np.abs(roots))]
        return {'ar_poles': np.array([phi0, phi1]), 'ma_zero': float(np.real(theta))}


@dataclass(kw_only=True)
class PowerLaw(Process):
    """Long-memory ARFIMA(0, d, 0): a scale-free negative control.

    ``S(f) ~ |2 sin(pi f / fs)|^{-2d}``, so the low-frequency spectral exponent
    is ``chi = 2d``.  The ACF decays as a power law, the integrated timescale
    diverges, and no finite characteristic tau exists.
    """

    d: float = 0.45

    def __post_init__(self):
        self.name = 'power_law'
        self.label = f'Power law (ARFIMA d={self.d:g}, $\\chi$={2*self.d:g})'
        self.tau_true = np.nan
        self.params = {'d': self.d, 'chi': 2 * self.d}

    def acf(self, lags, fs=None):
        lags = np.asarray(lags, dtype=float)
        d = self.d
        return np.exp(gammaln(1 - d) - gammaln(d)
                      + gammaln(lags + d) - gammaln(1 + lags - d))

    def acov(self, nlags):
        d = self.d
        k = np.arange(nlags + 1, dtype=float)
        var = np.exp(gammaln(1 - 2 * d) - 2 * gammaln(1 - d))
        return var * self.acf(k)

    def simulate(self, n_seconds, fs, rng):
        n = int(round(n_seconds * fs))
        return davies_harte(self.acov(n), n, rng)

    def psd(self, freqs, fs):
        freqs = np.asarray(freqs, dtype=float)
        w = np.pi * freqs / fs
        with np.errstate(divide='ignore'):
            base = np.abs(2 * np.sin(w)) ** (-2 * self.d)
        base = np.where(freqs == 0, np.inf, base)
        return (2.0 / fs) * base

    def tau_integrated(self, fs, max_lag_s=None):
        return np.inf if self.d > 0 else 0.0


@dataclass(kw_only=True)
class AR1PlusAR2(Process):
    """Aperiodic relaxation plus a *broadband* stochastic rhythm.

    The closest of these processes to a real recording.  ``AR1PlusSine`` adds
    a line no autoregressive model can represent, and ``AR2Damped`` is a
    rhythm with no separate aperiodic component; neural data is neither.  It
    is a broadband background with a broadband peak on top, both stochastic,
    which is what a spectrum with an alpha or beta bump actually looks like.

    The sum of an AR(1) and an AR(2) is ARMA(3, 2), so no pure AR(p) model
    reproduces it exactly -- but unlike an added sinusoid it can be
    approximated well, and the rhythm can be modelled rather than avoided.
    """

    tau: float = 0.05
    f_osc: float = 10.0
    tau_osc: float = 0.15
    snr: float = 1.0        # rhythm variance / aperiodic variance

    def __post_init__(self):
        self.name = 'ar1_ar2'
        self.has_oscillation = True
        self.label = (f'AR(1)+AR(2), $\\tau$={self.tau*1e3:g} ms, '
                      f'{self.f_osc:g} Hz')
        self.tau_true = self.tau
        self.params = {'tau': self.tau, 'f_osc': self.f_osc,
                       'tau_osc': self.tau_osc, 'snr': self.snr}

    def _parts(self, fs):
        phi1 = np.array([tau_to_phi(self.tau, fs)])
        r = np.exp(-1.0 / (self.tau_osc * fs))
        theta = 2 * np.pi * self.f_osc / fs
        phi2 = poles_to_ar(np.array([r * np.exp(1j * theta),
                                     r * np.exp(-1j * theta)]))
        return phi1, phi2

    def simulate(self, n_seconds, fs, rng):
        n = int(round(n_seconds * fs))
        phi1, phi2 = self._parts(fs)
        aperiodic = sim_ar_process(phi1, n, rng)
        rhythm = sim_ar_process(phi2, n, rng)
        aperiodic = aperiodic / aperiodic.std()
        rhythm = rhythm / rhythm.std()
        return aperiodic + np.sqrt(self.snr) * rhythm

    def psd(self, freqs, fs):
        phi1, phi2 = self._parts(fs)
        # Unit-variance components, so the spectra add in the same proportion
        # the signals do.
        a = ar_psd(phi1, freqs, fs, 1.0)
        o = ar_psd(phi2, freqs, fs, 1.0)
        a = a / np.trapezoid(a, freqs)
        o = o / np.trapezoid(o, freqs)
        return a + self.snr * o

    def acf(self, lags, fs):
        phi1, phi2 = self._parts(fs)
        lags = np.atleast_1d(np.asarray(lags))
        n = int(np.max(lags))
        w = 1.0 / (1.0 + self.snr)
        return (w * ar_acf(phi1, n)[lags.astype(int)]
                + (1 - w) * ar_acf(phi2, n)[lags.astype(int)])


# The higher-order AR(p) every simulation in the paper uses is ARP_MODES, the
# AR(7) of the modes figure (its coefficients are PHI_ARP there): three
# relaxations, of 200, 20 and 2 ms, and two rhythms, 10 Hz ringing for 150 ms
# and 40 Hz for 60 ms.  The relaxation knees at 0.8 and 8 Hz are an order of
# magnitude apart and both inside the band a 4 s Welch window resolves.
ARP_MODES_7 = ((0.200, 0.0), (0.150, 10.0), (0.060, 40.0),
               (0.020, 0.0), (0.002, 0.0))
# The alternative, an AR(5): a 250 ms relaxation, a 5 Hz mode whose envelope
# decays in 100 ms (half a cycle, so a damped relaxation by the paper's rule
# rather than a rhythm) and a 50 Hz rhythm ringing for 20 ms.  Under the rule
# an estimator is asked to apply -- the most important non-rhythmic mode --
# its own timescale is the 5 Hz mode's 100 ms envelope, which outranks the
# 250 ms relaxation in spectral importance.  The benchmark simulates it
# alongside the AR(7), and the figure scripts draw either (TSMODES_ARP=5).
ARP_MODES_5 = ((0.250, 0.0), (0.020, 50.0), (0.100, 5.0))
ARP_VARIANTS = {'7': ARP_MODES_7, '5': ARP_MODES_5}
ARP_MODES = ARP_MODES_7
# Every column of the benchmark figure has the same constructed timescale:
# the AR(1)'s tau, the AR(2) oscillator's envelope decay, the AR(7)'s
# dominant relaxation, and the finite sample's tau, all 200 ms, so that the
# estimates can share one axis.
BENCH_TAU = 0.200
# The AR(2) oscillator of the benchmark and of the oscillations column of
# fig_bias, as (tau, f): a 10 Hz rhythm whose envelope decays in 200 ms, two
# cycles, a peak forty times its low-frequency plateau.
AR2_RHYTHM = (BENCH_TAU, 10.0)


@dataclass(kw_only=True)
class ARpModes(Process):
    """A genuine high-order AR(p) built from a list of modes.

    The other processes fail the AR(1) assumption by not being autoregressive
    at all -- an added sinusoid, a sum of independent relaxations, a
    long-memory process.  This one *is* an AR(p), exactly, so it separates two
    things that are otherwise confounded: whether an estimator can cope with a
    process outside the model family, and whether it can cope with a process
    inside the family but of higher order than AR(1).

    ``modes`` is a list of ``(tau_seconds, freq_hz)``; a zero frequency gives a
    real pole, a non-zero one a conjugate pair.  The default is ``ARP_MODES``,
    the AR(7) of the modes figure, so every AR(p) simulated in the paper is
    the same process.  ``variant`` names an alternative in the result tables
    (``'5'`` gives ``arp_modes_5``); ``arp_process('5')`` builds one.
    """

    modes: tuple = ARP_MODES
    variant: str = None

    def __post_init__(self):
        self.name = 'arp_modes' if self.variant is None else f'arp_modes_{self.variant}'
        order = sum(1 if f == 0 else 2 for _, f in self.modes)
        self.label = f'AR({order}), {len(self.modes)} modes'
        # The target is the true model's own aperiodic timescale, under the
        # same definition an estimator would apply to a fitted model, rather
        # than the slowest relaxation by construction.  The two can differ (a
        # rhythm ringing less than a cycle is non-rhythmic by that definition);
        # for the default modes they agree on the 200 ms relaxation, and for
        # the AR(5) variant they disagree: the rule picks the 5 Hz mode's
        # 100 ms envelope over the 250 ms relaxation.  Defining
        # the target any other way would score estimators against a quantity
        # they were not asked for.
        self.tau_true = np.nan   # set below, once the poles are known
        self.params = {'modes': tuple(self.modes)}
        # No additive lines: the rhythms here are poles of the process, not
        # delta functions added to its spectrum.  `lines` is for the latter.
        self.lines = ()
        self._fs_for_truth = 500.0
        self.tau_true = float(ARModes.from_phi(
            self.phi(self._fs_for_truth), self._fs_for_truth).tau_relaxation)

    def poles(self, fs):
        out = []
        for tau, freq in self.modes:
            r = np.exp(-1.0 / (tau * fs))
            if freq == 0:
                out.append(complex(r, 0.0))
            else:
                th = 2 * np.pi * freq / fs
                out += [r * np.exp(1j * th), r * np.exp(-1j * th)]
        return np.array(out)

    def phi(self, fs):
        return poles_to_ar(self.poles(fs))

    def simulate(self, n_seconds, fs, rng):
        return sim_ar_process(self.phi(fs), int(round(n_seconds * fs)), rng)

    def psd(self, freqs, fs):
        return ar_psd(self.phi(fs), freqs, fs, 1.0)

    def acf(self, lags, fs):
        lags = np.atleast_1d(np.asarray(lags))
        return ar_acf(self.phi(fs), int(np.max(lags)))[lags.astype(int)]


def arp_process(choice='7'):
    """The constructed AR(p) by variant: ``'7'`` is the paper's AR(7) (the
    default ``ARpModes()``), ``'5'`` the alternative AR(5)."""
    choice = str(choice)
    if choice == '7':
        return ARpModes()
    return ARpModes(modes=ARP_VARIANTS[choice], variant=choice)


@dataclass(kw_only=True)
class AR1Mixture(Process):
    """A log-uniform mixture of independent AR(1) relaxations.

    Approximates a power law over the band spanned by the component
    timescales while remaining short-memory, so the integrated timescale
    exists but is dominated by the slowest components.
    """

    tau_min: float = 0.002
    tau_max: float = 2.0
    n_components: int = 12

    def __post_init__(self):
        self.name = 'ar1_mixture'
        self.label = (f'AR(1) mixture, $\\tau\\in$[{self.tau_min*1e3:g}, '
                      f'{self.tau_max*1e3:g}] ms')
        self.tau_true = np.nan
        self.params = {'tau_min': self.tau_min, 'tau_max': self.tau_max,
                       'n_components': self.n_components}
        self.taus = np.logspace(np.log10(self.tau_min), np.log10(self.tau_max),
                                self.n_components)

    def _scales(self, fs):
        phis = np.array([tau_to_phi(t, fs) for t in self.taus])
        # equal variance per octave -> equal variance per component on a log grid
        v = 1.0 / (1 - phis ** 2)
        a = np.sqrt((1.0 / self.n_components) / v)
        return phis, a

    def simulate(self, n_seconds, fs, rng):
        n = int(round(n_seconds * fs))
        phis, a = self._scales(fs)
        return sum(ai * sim_ar_process([p], n, rng) for p, ai in zip(phis, a))

    def psd(self, freqs, fs):
        phis, a = self._scales(fs)
        return sum(ai ** 2 * ar_psd_density([p], freqs, fs)
                   for p, ai in zip(phis, a))

    def acf(self, lags, fs):
        lags = np.asarray(lags, dtype=float)
        phis, a = self._scales(fs)
        v = a ** 2 / (1 - phis ** 2)
        num = sum(vi * p ** lags for vi, p in zip(v, phis))
        return num / v.sum()

    def tau_integrated(self, fs, max_lag_s=None):
        phis, a = self._scales(fs)
        v = a ** 2 / (1 - phis ** 2)
        t = (1.0 / fs) * (1.0 / (1 - phis) - 0.5)
        return float(np.sum(v * t) / np.sum(v))
