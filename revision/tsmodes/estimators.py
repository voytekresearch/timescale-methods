"""A single fitting/evaluation API shared by every simulation and dataset.

Four families of timescale estimator are implemented behind one interface:

``ar1_direct``  ordinary least squares AR(1) on the time series
``ar1_acf``     AR(1) autocorrelation form fit to the ACF
``ar1_psd``     AR(1) spectral form fit to the power spectral density
``arp_direct``  AR(p) on the time series (Burg), order by BIC
``arp_acf``     AR(p) autocorrelation form fit to the ACF, order by BIC
``arp_psd``     AR(p) spectral form fit to the PSD, order by BIC

The ACF and PSD estimators are deliberately symmetric: both fit the same AR(p)
model, both use robust (soft-L1) regression so that a narrowband oscillation is
treated as an outlier rather than as evidence about the aperiodic timescale, and
both select order by BIC.  Earlier comparisons that fit only a bare exponential
to the ACF handicap the time-domain estimator, and the resulting failures say
more about the fitting procedure than about the domain.

Every estimator consumes the same :class:`Signal` container (one PSD, one ACF,
computed once) and returns the same :class:`TimescaleFit` record, so bias,
error, goodness of fit and model order are comparable across methods.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import get_window, welch

from .modes import (poles_to_ar, ARModes, ar_acf_fft, ar_psd,
                    ar_to_real_pole_reflection, ar_to_reflection,
                    real_pole_reflection_to_ar, reflection_to_ar,
                    stabilize, LN10)

# ---------------------------------------------------------------------------
# tau <-> phi
# ---------------------------------------------------------------------------

def phi_to_tau(phi, fs):
    """``tau = -(1/fs) / log(phi)``; NaN for non-decaying phi."""
    phi = np.asarray(phi, dtype=float)
    out = np.full(phi.shape, np.nan) if phi.ndim else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        val = -1.0 / (fs * np.log(np.abs(phi)))
    good = (phi > 0) & (phi < 1)
    return np.where(good, val, out) if phi.ndim else (float(val) if good else np.nan)


def tau_to_phi(tau, fs):
    return np.exp(-1.0 / (np.asarray(tau, dtype=float) * fs))


# ---------------------------------------------------------------------------
# Shared signal container
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A time series plus the PSD and ACF every estimator shares."""

    sig: np.ndarray
    fs: float
    freqs: np.ndarray = None
    powers: np.ndarray = None
    lags: np.ndarray = None
    corrs: np.ndarray = None
    f_range: tuple = None
    nperseg: int = None
    n_segments: int = None
    name: str = ''

    @property
    def n_samples(self):
        return len(self.sig)

    @property
    def duration(self):
        return len(self.sig) / self.fs

    @property
    def fit_mask(self):
        lo, hi = self.f_range
        return (self.freqs >= lo) & (self.freqs <= hi) & (self.freqs > 0)


N_LAGS = 250          # lag budget for ACF-domain fitting

# Order grids are coarse and searched with early stopping.  BIC is smooth in p,
# so a log-spaced grid finds the same minimum as an exhaustive search at a
# fraction of the cost, and both domains use the same grid so that the selected
# orders are comparable.
ORDER_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20)
#: Stop the order search this many grid points after the best so far.
#:
#: The spectral search does not use it (None): ORDER_GRID is not uniform, it
#: steps 6, 8, 10, 13, 16, 20, so BIC is not unimodal over it and stopping
#: early quit before the gains arrived.  On mouse LFP that returned order 3
#: with 2.7% of the log spectrum unexplained where order 20 leaves 0.6%.
#:
#: The lag-domain search does use it.  Welch bins from non-overlapping
#: segments are approximately independent, which is what BIC assumes, but
#: neighbouring autocorrelation lags are strongly dependent, so the criterion
#: counts far more information than the data hold and occasionally buys a
#: large order for a process that is plainly AR(1).  The guard bounds that.
ORDER_PATIENCE_PSD = None
ORDER_PATIENCE_ACF = 3


def multitaper_psd(sig, fs, nw=4.0, n_tapers=None):
    """Multitaper spectrum of a whole record, and its implied lag window.

    Welch trades resolution for variance by cutting the record up, which a
    short record cannot afford: 296 fMRI samples leave either too few
    frequency bins to fit or too few segments to average.  Multitapering
    averages orthogonal tapers of the *whole* record instead, so it buys the
    same variance reduction without spending record length.
    """
    from scipy.signal.windows import dpss
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    # A short record cannot support a large time-bandwidth product: DPSS
    # requires more samples than tapers, and a resolution bandwidth of 2NW/n
    # wider than the band is meaningless anyway.  Windowed analyses cut the
    # same recording into pieces a few samples long, so clamp rather than
    # fail.
    nw = float(np.clip(nw, 1.25, max(1.25, n / 8.0)))
    n_tapers = int(2 * nw - 1) if n_tapers is None else int(n_tapers)
    n_tapers = int(np.clip(n_tapers, 1, max(1, n - 1)))
    tapers = dpss(n, nw, n_tapers)
    power = np.zeros(n // 2 + 1)
    for taper in tapers:
        power += np.abs(np.fft.rfft(sig * taper)) ** 2
    power *= 2.0 / (n_tapers * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # The estimate is the average of tapered periodograms, so the lag window
    # it implies is the average of the tapers' autocorrelations.
    lagwin = np.mean([np.correlate(t, t, 'full')[n - 1:] for t in tapers],
                     axis=0)
    return freqs, power, lagwin / lagwin[0], n_tapers


def prepare(sig, fs, welch_win_s=2.0, nlags_s=None, nlags=None, f_range=None,
            name='', detrend=False, exclude_bands=(), welch_overlap=0.5,
            method='welch', nw=4.0):
    """Compute the PSD and ACF used by every estimator.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, Hz.
    welch_win_s : float
        Welch segment length in seconds (capped at the signal duration).
    nlags_s : float, optional
        ACF lag range in seconds.  By default a fixed budget of ``N_LAGS``
        lags is used, which keeps the cost of ACF-domain fitting constant and
        spans many timescales at any sampling rate.  A longer range was
        tried and rejected: on the AR(7) benchmark one second of lags at
        500 Hz widened the lag-domain estimate (taper-corrected tail lags are
        noisy) without moving its median.
    f_range : tuple, optional
        Frequency range used for spectral fitting.  Defaults to
        ``(1 / welch_segment_seconds, fs / 2)``.
    detrend : False or str
        Per-segment detrending passed to Welch's method.  The signal is
        mean-centered as a whole, and removing the mean of every segment on
        top of that acts as a high-pass at the segment's resolution: it takes
        power out of the lowest bins, where a slow relaxation's knee sits.
        On the AR(7) benchmark (200 ms relaxation, 4 s segments, 100 s) the
        spectral AR(p) fit returned a median of 183 ms with per-segment
        detrending and 197 ms without, the same as Burg's method, so the
        default is no per-segment detrending.
    exclude_bands : sequence of (lo, hi)
        Frequency bands (e.g. line noise) removed before spectral fitting.
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    n = len(sig)

    if method == 'multitaper':
        freqs, powers, lagwin, n_segments = multitaper_psd(sig, fs, nw=nw)
        nperseg = n
    else:
        nperseg = int(min(n, round(welch_win_s * fs)))
        nperseg = max(nperseg, 8)
        noverlap = int(round(welch_overlap * nperseg))
        freqs, powers = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                              window='hann', detrend=detrend, scaling='density')
        n_segments = max(1, 1 + (n - nperseg) // (nperseg - noverlap))
        lagwin = None

    if f_range is None:
        f_range = (fs / nperseg, fs / 2)

    if nlags is None:
        nlags = N_LAGS if nlags_s is None else int(round(nlags_s * fs))
    nlags = int(max(4, min(n // 2, n - 2, nlags)))

    keep = np.ones(len(freqs), dtype=bool)
    for lo, hi in exclude_bands:
        keep &= ~((freqs >= lo) & (freqs <= hi))

    # The autocorrelation is derived from the same band-limited spectrum the
    # spectral estimators see, rather than from the raw signal.  Otherwise the
    # two domains are fit to different data: the raw autocorrelation carries
    # drift below the fit band and everything above it, so a lag-domain fit
    # would report a timescale set by frequencies the spectral fit excluded.
    # Cut only where content is deliberately excluded.  The lower edge of
    # f_range is usually just the Welch resolution floor, and rejecting
    # everything below it would put a brick wall in the spectrum that the lag
    # domain sees as ringing, costing poles to describe an artefact of the
    # analysis.  A low edge the caller actually asked for is a different
    # matter and is applied.
    f_lo = f_range[0] if f_range[0] > 1.5 * fs / nperseg else 0.0
    band = (freqs >= f_lo) & (freqs <= f_range[1])
    corrs = band_limited_acf(freqs, powers, nlags, band & keep, band,
                             nperseg=nperseg, window='hann', lagwin=lagwin)
    lags = np.arange(len(corrs))          # the taper correction may shorten it

    out = Signal(sig=sig, fs=fs, freqs=freqs, powers=powers, lags=lags,
                 corrs=corrs, f_range=f_range, nperseg=nperseg,
                 n_segments=n_segments, name=name)
    out.exclude_mask = keep
    out.band_mask = band
    out.raw_corrs = _acf(sig, len(corrs) - 1)
    # Adjacent multitaper bins inside the resolution bandwidth 2*NW/T are
    # correlated; the spectral fits thin to that spacing.
    out.fit_decimate = int(round(2 * nw)) if method == 'multitaper' else 1
    return out


def band_limited_acf(freqs, powers, nlags, keep, band, nperseg=None,
                     window='hann', lagwin=None):
    """Autocorrelation of a spectrum restricted to the analysis band.

    Bins inside ``band`` but excluded by ``keep`` (line noise and its
    harmonics) are filled by log-linear interpolation from their neighbours
    rather than zeroed, so that notching does not ring in the lag domain.

    Welch's estimate is the periodogram of tapered segments, so the
    autocovariance it implies is the true one multiplied by the taper's own
    autocorrelation -- 0.66 at a lag a quarter of the segment length, for a
    Hann window.  Left in, that taper masquerades as a faster decay and biases
    every timescale read off the result, so it is divided out.
    """
    powers = np.asarray(powers, dtype=float).copy()
    if np.any(band & ~keep) and np.any(band & keep):
        good = band & keep
        powers[band & ~keep] = np.exp(np.interp(
            freqs[band & ~keep], freqs[good], np.log(np.maximum(powers[good],
                                                                1e-300))))
    powers[~band] = 0.0
    acov = np.fft.irfft(powers)[: nlags + 1]
    if lagwin is not None or nperseg:
        if lagwin is None:
            taper = get_window(window, int(nperseg))
            lagwin = np.correlate(taper, taper, 'full')[int(nperseg) - 1:]
        lagwin = np.asarray(lagwin, dtype=float)
        lagwin = lagwin[: nlags + 1] / lagwin[0]
        # Below about a fifth of the taper's energy the correction amplifies
        # noise faster than it removes bias, so those lags are dropped.
        acov = acov[: max(2, int(np.sum(lagwin > 0.2)))]
        acov = acov / lagwin[: len(acov)]
    return acov / acov[0]


def spectrum_acov(signal, powers, nlags=None):
    """Autocovariance implied by a spectrum on ``signal``'s frequency grid,
    restricted to the band ``signal.corrs`` was derived from.

    :func:`prepare` derives the autocorrelation from the spectrum inside the
    analysis band, so a model fit to that spectrum is translated to the lag
    domain the same way: its spectrum on the same grid, zero outside the same
    band, inverse transformed.  The model's full-band autocorrelation would
    instead carry its extrapolation outside the band, mostly the plateau below
    the lower edge, and disagree with the data even where the fit is exact.
    A model spectrum carries no Welch taper, so none is divided out.  The
    result is left unnormalized so that additive parts of a spectrum (the mode
    components) can be scaled by their total; ``powers`` may be 2d, one
    spectrum per row.
    """
    nlags = len(signal.corrs) - 1 if nlags is None else int(nlags)
    powers = np.array(powers, dtype=float)
    powers[..., ~signal.band_mask] = 0.0
    return np.fft.irfft(powers, axis=-1)[..., : nlags + 1]


def _acf(sig, nlags):
    """Biased (FFT) sample autocorrelation, lags 0..nlags."""
    x = sig - sig.mean()
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(x, nfft)
    corrs = np.fft.irfft(spec * np.conj(spec), nfft)[: nlags + 1]
    return corrs / corrs[0]


# ---------------------------------------------------------------------------
# Fit record
# ---------------------------------------------------------------------------

@dataclass
class TimescaleFit:
    """Result of one estimator applied to one signal."""

    method: str
    tau: float = np.nan
    tau_relaxation: float = np.nan
    order: int = 1
    phi: np.ndarray = None
    offset: float = np.nan
    modes: ARModes = None
    gof: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    success: bool = True

    @property
    def summary(self):
        return None if self.modes is None else self.modes.summary()


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------

def r2_logpsd(powers, powers_fit):
    y = np.log10(powers)
    yh = np.log10(powers_fit)
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def rmse_logpsd(powers, powers_fit):
    return float(np.sqrt(np.mean((np.log10(powers) - np.log10(powers_fit)) ** 2)))


def is_divergence(powers, powers_fit):
    """Itakura-Saito divergence, the Whittle log-likelihood mismatch."""
    r = powers / powers_fit
    return float(np.mean(r - np.log(r) - 1.0))


def spectral_gof(signal, powers_fit, mask=None, refit_scale=True):
    """Spectral goodness of fit over the analysis band.

    The overall scale is refit in log-power before scoring so that estimators
    which never fit an offset (the time-domain fits) are scored on spectral
    *shape*, comparably to the PSD fits.
    """
    mask = signal.fit_mask & signal.exclude_mask if mask is None else mask
    p = signal.powers[mask]
    pf = powers_fit[mask] if len(powers_fit) == len(signal.powers) else powers_fit
    if refit_scale:
        pf = pf * np.exp(np.mean(np.log(p) - np.log(pf)))
    return {'r2_logpsd': r2_logpsd(p, pf),
            'rmse_logpsd': rmse_logpsd(p, pf),
            'is_div': is_divergence(p, pf)}


def acf_fit_gof(signal, phi, height=None, offset=None, drop_zero_lag=True):
    """Goodness of fit of an AR(p) model against the empirical ACF.

    When ``height`` and ``offset`` are not supplied they are refit by least
    squares, so a model estimated in the frequency domain is scored on the
    shape of the autocorrelation it implies.  Together with
    :func:`spectral_gof`, which refits the overall scale, this makes the two
    domains score every model on equal terms.
    """
    lags, corrs = signal.lags, signal.corrs
    if drop_zero_lag:
        lags, corrs = lags[1:], corrs[1:]
    model = ar_acf_fft(phi, int(lags[-1]))[lags.astype(int)]
    if height is None or offset is None:
        design = np.column_stack([model, np.ones_like(model)])
        coef, *_ = np.linalg.lstsq(design, corrs, rcond=None)
        pred = design @ coef
    else:
        pred = height * model + offset
    return acf_gof(corrs, pred)


def acf_gof(corrs, corrs_fit):
    ss_res = np.sum((corrs - corrs_fit) ** 2)
    ss_tot = np.sum((corrs - corrs.mean()) ** 2)
    return {'r2_acf': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            'rmse_acf': float(np.sqrt(np.mean((corrs - corrs_fit) ** 2)))}


# ---------------------------------------------------------------------------
# Time-domain AR fitting
# ---------------------------------------------------------------------------

def ar_ols(sig, order):
    """Conditional least-squares AR(p) coefficients and residual variance."""
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    n = len(x)
    if order == 0:
        return np.zeros(0), float(np.mean(x ** 2))
    xmat = np.column_stack([x[order - k - 1: n - k - 1] for k in range(order)])
    y = x[order:]
    phi, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    resid = y - xmat @ phi
    return phi, float(np.mean(resid ** 2))


def burg(sig, order):
    """Burg AR(p) coefficient estimates.

    Returned in the ``x_t = sum_k phi_k x_{t-k} + eps_t`` convention, matching
    ``timescales.autoreg.burg`` (reimplemented here so this sub-module has no
    dependency on the installed build of the parent package).
    """
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    if order == 0:
        return np.zeros(0)
    ef, eb = x[1:], x[:-1]
    phi = np.zeros(0)
    for _ in range(order):
        den = ef @ ef + eb @ eb
        k = 2.0 * (eb @ ef) / den if den > 0 else 0.0
        phi = np.concatenate([phi - k * phi[::-1], [k]]) if len(phi) else np.array([k])
        ef, eb = ef[1:] - k * eb[1:], eb[:-1] - k * ef[:-1]
    return phi


def ar_burg(sig, order):
    """Burg AR(p) coefficients and residual variance."""
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    if order == 0:
        return np.zeros(0), float(np.mean(x ** 2))
    phi = burg(x, order)
    n = len(x)
    xmat = np.column_stack([x[order - k - 1: n - k - 1] for k in range(order)])
    resid = x[order:] - xmat @ phi
    return phi, float(np.mean(resid ** 2))


def population_ar(rho, order):
    """The asymptotic (population) AR(p) fit to a known autocorrelation.

    Yule-Walker applied to the *true* ACF returns the AR(p) model minimizing the
    Whittle/Itakura-Saito divergence to the true spectrum, i.e. what any
    consistent AR(p) estimator converges to.  Comparing this to finite-sample
    fits separates model mismatch from estimation error.
    """
    from scipy.linalg import solve_toeplitz
    rho = np.asarray(rho, dtype=float)
    if len(rho) < order + 1:
        raise ValueError('Need the ACF out to lag `order`.')
    return solve_toeplitz(rho[:order], rho[1:order + 1])


def select_order_time(sig, orders, method='burg', criterion='bic'):
    """Objective AR order selection from the conditional Gaussian likelihood.

    The lagged design matrix is built once at the largest order and sliced,
    rather than rebuilt for every candidate; on long recordings that dominates
    the cost of the search.
    """
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    n = len(x)
    orders = list(orders)
    top = max(orders)
    design = np.column_stack([x[top - k - 1: n - k - 1] for k in range(top)])
    target = x[top:]

    scores, fits = [], []
    for p in orders:
        phi = burg(x, p) if method == 'burg' else np.linalg.lstsq(
            design[:, :p], target, rcond=None)[0]
        resid = target - design[:, :p] @ phi
        s2 = float(np.mean(resid ** 2))
        pen = np.log(n) if criterion == 'bic' else 2.0
        scores.append(n * np.log(max(s2, 1e-300)) + pen * (p + 1))
        fits.append((phi, s2))
    best = int(np.argmin(scores))
    return orders[best], np.asarray(scores), fits[best]


# ---------------------------------------------------------------------------
# Spectral AR fitting
# ---------------------------------------------------------------------------

def init_ar_from_psd(powers, order, eps=1e-12):
    """Yule-Walker AR coefficients from a one-sided, uniformly spaced PSD.

    The inverse real FFT of the one-sided PSD is the autocovariance sequence;
    the first ``order + 1`` lags give the Yule-Walker system.  (Same
    initialization as ``timescales.autoreg.fit.init_ar_from_psd``.)
    """
    from scipy.linalg import solve_toeplitz
    powers = np.maximum(np.asarray(powers, dtype=float), eps)
    acov = np.fft.irfft(powers)[: order + 1]
    return solve_toeplitz(acov[:-1], acov[1:])


def _ar_psd_guess(freqs, powers, fs, order):
    """Yule-Walker initialization from the empirical PSD shape."""
    try:
        phi0 = init_ar_from_psd(np.asarray(powers, dtype=float), order)
        if not np.all(np.isfinite(phi0)):
            raise ValueError
    except Exception:
        phi0 = np.zeros(order)
        phi0[0] = 0.8
    offset0 = float(np.exp(np.mean(np.log(powers))))
    return np.concatenate([phi0, [offset0]])


# Robust (soft-L1) regression protects a broadband fit from a narrowband
# rhythm, which is an outlier in the frequency domain.  It is not free: when
# there is no outlier, down-weighting the tails of an ordinary residual
# distribution biases the fit.  Robustness is therefore a choice about the
# data, made explicitly by the caller, not a default.
PSD_LOSS = 'soft_l1'
PSD_F_SCALE = 0.15     # log10-power units; Welch noise is well inside this
ACF_LOSS = 'soft_l1'
ACF_F_SCALE = 0.05     # correlation units


def loss_settings(robust, domain):
    """Loss function and soft margin for a fit, given whether to be robust.

    Robustness is offered in the frequency domain only.  A rhythm is an
    outlier there -- it occupies a handful of bins out of a thousand -- so
    down-weighting large residuals removes it and leaves the aperiodic fit
    intact.  In the lag domain it is not an outlier: an added sinusoid
    displaces *every* lag, by about the autocorrelation's own standard
    deviation.  A soft margin small enough to reject it rejects the data with
    it, and one large enough to keep the data is ordinary least squares.  We
    measured this: on an AR(1) with an equal-power 10 Hz sinusoid the lag-domain
    fit reaches R^2 = 0.84 at every margin from 0.05 to 0.4 and 0.97 at 0.8,
    where the loss has become least squares in all but name.  There is no
    setting at which it helps, so it is not offered.
    """
    if not robust or domain != 'psd':
        return 'linear', None
    return PSD_LOSS, PSD_F_SCALE
ACF_MAXFEV = 600
ACF_TOL = 1e-6      # the lag-domain objective is smooth; 1e-8 buys nothing
K_BOUND = 0.9999
#: Require every fitted AR(p) model to contain at least one relaxation mode.
#: A model whose poles are all complex has no decay time to report, which is a
#: failure of the fit as a measurement of a timescale even when it describes
#: the data well.  See tsmodes.modes.real_pole_reflection_to_ar.
REQUIRE_RELAXATION = True       # reflection coefficients; |k| < 1 <=> stationary.
                       # Bounds the longest representable timescale at
                       # -1/(fs log K_BOUND), i.e. 20 s at 500 Hz.
MAXFEV = 4000


def fit_ar_psd(freqs, powers, fs, order, guess=None, k_bound=K_BOUND,
               loss='linear', f_scale=None, maxfev=MAXFEV,
               freqs_full=None, powers_full=None,
               require_relaxation=REQUIRE_RELAXATION):
    """Fit the AR(p) spectral form to a PSD, in log10 power.

    The model is optimized over reflection coefficients rather than AR
    coefficients: every point in the box ``|k_i| < 1`` is a stationary model,
    the parameters are all O(1), and the fit cannot wander into a region where
    mode timescales are undefined.  Regression is robust (soft-L1) so that a
    narrowband oscillation is treated as an outlier rather than as evidence
    about the aperiodic timescale.

    ``freqs``/``powers`` are the points actually fit; ``powers_full`` (uniform,
    DC to Nyquist) is used only for initialization.
    """
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)
    if guess is None:
        init_p = powers if powers_full is None else powers_full
        guess = _ar_psd_guess(freqs, init_p, fs, order)
    guess = np.asarray(guess, dtype=float)
    # See fit_ar_acf: the leading parameter is a real pole when a relaxation
    # mode is required, and a reflection coefficient otherwise.
    if require_relaxation:
        a0, k0 = ar_to_real_pole_reflection(guess[:-1], k_bound=k_bound)
        to_ar = real_pole_reflection_to_ar
    else:
        refl = np.clip(ar_to_reflection(guess[:-1]), -k_bound, k_bound)
        a0, k0 = refl[0], refl[1:]
        def to_ar(a, k):
            return reflection_to_ar(np.concatenate([[a], k]))
    p0 = np.concatenate([[a0], k0, [max(guess[-1], 1e-12)]])
    lo0 = 1e-3 if require_relaxation else -k_bound
    bounds = (np.concatenate([[lo0], np.full(order - 1, -k_bound), [1e-16]]),
              np.concatenate([np.full(order, k_bound), [np.inf]]))

    k_idx = np.arange(1, order + 1)
    z_inv = np.exp(-2j * np.pi * np.outer(freqs, k_idx) / fs)

    def model(_, *params):
        phi = to_ar(params[0], np.asarray(params[1:-1]))
        return np.log10(params[-1] / np.abs(1.0 - z_inv @ phi) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            params, _ = curve_fit(model, freqs, np.log10(powers), p0=p0,
                                  bounds=bounds, maxfev=maxfev, loss=loss,
                                  f_scale=f_scale)
            success = True
        except Exception:
            params, success = p0, False
    return (to_ar(params[0], np.asarray(params[1:-1])), float(params[-1]),
            success)


def select_order_psd(freqs, powers, fs, orders, criterion='bic', **kwargs):
    """AR order selection on the PSD via a log-power residual BIC.

    ``BIC(p) = n_f log(RSS/n_f) + (p + 1) log(n_f)`` with ``n_f`` the number of
    fitted frequency bins.  Welch bins from non-overlapping segments are
    approximately independent, which is the assumption this makes.
    """
    n_f = len(freqs)
    scores, fits, tried = [], [], []
    since_best = 0
    for p in orders:
        phi, offset, ok = fit_ar_psd(freqs, powers, fs, p, **kwargs)
        pred = ar_psd(phi, freqs, fs, offset)
        rss = np.sum((np.log10(powers) - np.log10(pred)) ** 2)
        pen = np.log(n_f) if criterion == 'bic' else 2.0
        scores.append(n_f * np.log(max(rss / n_f, 1e-300)) + pen * (p + 1))
        fits.append((phi, offset, ok))
        tried.append(p)
        since_best = 0 if scores[-1] <= min(scores) else since_best + 1
        if ORDER_PATIENCE_PSD is not None and since_best >= ORDER_PATIENCE_PSD:
            break
    best = int(np.argmin(scores))
    return tried[best], np.asarray(scores), fits[best]


# ---------------------------------------------------------------------------
# ACF fitting
# ---------------------------------------------------------------------------

def exp_decay(lags, fs, tau, height, offset):
    """AR(1) autocorrelation form: ``A exp(-k/(tau fs)) + B``."""
    return height * np.exp(-lags / (tau * fs)) + offset


def init_ar_from_acf(corrs, order):
    """Yule-Walker AR coefficients from an empirical autocorrelation.

    The ACF-domain analogue of :func:`init_ar_from_psd`.
    """
    from scipy.linalg import solve_toeplitz
    corrs = np.asarray(corrs, dtype=float)
    if len(corrs) < order + 1:
        raise ValueError('Need the ACF out to lag `order`.')
    try:
        phi = solve_toeplitz(corrs[:order], corrs[1:order + 1])
        if not np.all(np.isfinite(phi)):
            raise ValueError
    except Exception:
        phi = np.zeros(order)
        phi[0] = max(min(corrs[1], 0.95), 0.05) if len(corrs) > 1 else 0.8
    return phi


def _acf_guesses(lags, corrs, fs, order):
    """Starting points for the lag-domain fit.

    Three, because the objective is badly non-convex when a rhythm is present:
    Yule-Walker, a pure exponential at the 1/e lag, and a structural guess that
    puts a real pole on the decay and conjugate pairs on the ringing.  The
    third matters most.  Without it the optimizer spends every pole on the
    rhythm and returns a model with no relaxation mode at all, which reads as
    a claim about the process when it is a claim about the search.
    """
    guesses = []
    yw = init_ar_from_acf(corrs, order)
    guesses.append(np.concatenate([yw, [corrs[0], 0.0]]))

    below = np.where(corrs < np.exp(-1.0))[0]
    lag_e = lags[below[0]] if len(below) else max(lags[-1], 1.0)
    tau0 = max(lag_e / fs, 2.0 / fs)
    phi_relax = np.clip(tau_to_phi(tau0, fs), 0.05, 0.98)
    phi0 = np.zeros(order)
    phi0[0] = phi_relax
    guesses.append(np.concatenate([phi0, [corrs[0], 0.0]]))

    if order >= 3:
        # The ringing frequency, read off the autocorrelation itself so the
        # guess stays inside the lag domain.
        n = int(1 << int(np.ceil(np.log2(max(len(corrs) * 4, 256)))))
        spec = np.abs(np.fft.rfft(corrs - corrs.mean(), n)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        keep = freqs > fs / max(lags[-1], 1.0)
        if np.any(keep):
            f0 = float(freqs[keep][np.argmax(spec[keep])])
            poles = [complex(phi_relax, 0.0)]
            theta = 2 * np.pi * f0 / fs
            for j in range((order - 1) // 2):
                r = 0.9 ** (j + 1)          # first pair sharp, later ones damped
                poles += [r * np.exp(1j * theta), r * np.exp(-1j * theta)]
            while len(poles) < order:
                poles.append(complex(0.1, 0.0))
            try:
                guesses.append(np.concatenate([poles_to_ar(np.array(poles[:order])),
                                               [corrs[0], 0.0]]))
            except ValueError:
                pass
    return guesses


def fit_ar_acf(lags, corrs, fs, order, guess=None, k_bound=K_BOUND,
               loss='linear', f_scale=None, maxfev=ACF_MAXFEV,
               drop_zero_lag=True, with_offset=True,
               require_relaxation=REQUIRE_RELAXATION):
    """Fit the AR(p) autocorrelation form to an empirical ACF.

    The direct counterpart of :func:`fit_ar_psd`: the same AR(p) model, the same
    reflection-coefficient parameterization, and the same robust loss, applied
    in the lag domain instead of the frequency domain.  The model is
    ``A rho_p(k) + B``, where ``rho_p`` is the exact AR(p) autocorrelation, ``A``
    absorbs a white-noise floor at lag zero and ``B`` a constant offset -- the
    same two nuisance parameters used by the conventional exponential ACF fit.

    Several starting points are tried and the lowest-cost solution kept, because
    the lag-domain objective is non-convex when a rhythm is present.

    Returns
    -------
    phi, height, offset, success
    """
    lags = np.asarray(lags, dtype=float)
    corrs = np.asarray(corrs, dtype=float)
    full_lags, full_corrs = lags, corrs
    if drop_zero_lag:
        lags, corrs = lags[1:], corrs[1:]
    nlags = int(lags[-1])
    idx = lags.astype(int)
    # Fix the transform length once: it keeps the cached exponential matrix hot
    # and stops a near-unit-root trial point from triggering a huge FFT.  The
    # final model is re-evaluated with the adaptive length by the caller.
    nfft = int(2 ** np.ceil(np.log2(max(8 * (nlags + 1), 4096))))

    # The autocorrelation at the first lag is at most one, so a fitted height
    # far above one and a large compensating offset describe a curve the data
    # cannot have.  Leaving these loose lets a slow exponential trade against a
    # negative offset and return a timescale set by the offset, not the decay.
    off_lo, off_hi = (-0.5, 0.5) if with_offset else (-1e-12, 1e-12)
    # With require_relaxation the first parameter is a real pole in (0, 1)
    # rather than a reflection coefficient, so the model is guaranteed to have
    # a relaxation mode to report.
    lo0 = 1e-3 if require_relaxation else -k_bound
    bounds = (np.concatenate([[lo0], np.full(order - 1, -k_bound), [0.05, off_lo]]),
              np.concatenate([np.full(order, k_bound), [1.5, off_hi]]))
    to_ar = (real_pole_reflection_to_ar if require_relaxation
             else lambda a, k: reflection_to_ar(np.concatenate([[a], k])))

    if order == 1:
        # rho(k) = phi^k exactly, so the transform is unnecessary.  This keeps
        # the AR(1) fit cheap even when the lag range is very long.
        def model(_, *params):
            return params[-2] * params[0] ** lags + params[-1]
    else:
        def model(_, *params):
            phi = to_ar(params[0], np.asarray(params[1:-2]))
            return params[-2] * ar_acf_fft(phi, nlags, nfft=nfft)[idx] + params[-1]

    if guess is not None:
        starts = [np.asarray(guess, dtype=float)]
    else:
        # Every start is tried at every order.  Dropping all but Yule-Walker
        # above order 2 was a false economy: the lag-domain objective is
        # non-convex, and the start that finds the relaxation pole is exactly
        # the one that was being discarded.
        starts = _acf_guesses(full_lags, full_corrs, fs, order)
    def _encode(g):
        if require_relaxation:
            a, k = ar_to_real_pole_reflection(g[:-2], k_bound=k_bound)
            return np.concatenate([[a], k, g[-2:]])
        return np.concatenate([np.clip(ar_to_reflection(g[:-2]),
                                       -k_bound, k_bound), g[-2:]])

    starts = [_encode(np.asarray(g, dtype=float)) for g in starts]

    best, success = None, False
    for start in starts:
        start = np.clip(start, bounds[0] + 1e-9, bounds[1] - 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                params, _ = curve_fit(model, lags, corrs, p0=start,
                                      bounds=bounds, maxfev=maxfev, loss=loss,
                                      f_scale=f_scale, ftol=ACF_TOL,
                                      xtol=ACF_TOL, gtol=ACF_TOL,
                                      x_scale='jac')
            except Exception:
                continue
        resid = corrs - model(lags, *params)
        cost = float(np.sum(f_scale ** 2 *
                            (np.sqrt(1 + (resid / f_scale) ** 2) - 1))
                     if f_scale else np.sum(resid ** 2))
        if best is None or cost < best[0]:
            best, success = (cost, params), True
    params = starts[0] if best is None else best[1]
    return (to_ar(params[0], np.asarray(params[1:-2])),
            float(params[-2]), float(params[-1]), success)


def acf_effective_n(lags, corrs):
    """Effective number of independent lags for lag-domain order selection:
    the number of lags divided by the 1/e lag.  Looser counts were tried on
    the benchmark (dividing by the square root of the 1/e lag, by a quarter
    of it, or counting every lag) and all of them over-fit even a plain
    AR(1), returning orders of 4 to 10 and timescales a quarter of the
    truth in a third of the repetitions."""
    below = np.where(np.abs(corrs) < np.exp(-1.0))[0]
    lag_e = float(lags[below[0]]) if len(below) else float(lags[-1])
    return max(6, int(round((len(lags) - 1) / max(lag_e, 1.0))))


def select_order_acf(lags, corrs, fs, orders, criterion='bic', **kwargs):
    """AR order selection in the ACF domain, mirroring :func:`select_order_psd`.

    ``BIC(p) = n log(RSS/n) + (p + 2) log(n)``, but with ``n`` the number of
    *independent* lags rather than the number of fitted ones.  Neighbouring
    autocorrelation lags are strongly dependent -- the curve is smooth on the
    scale of the decay it describes -- so counting all of them treats the data
    as far more informative than it is and buys orders the record cannot
    support: on one realization of a plain AR(1), all 250 lags gave order 8
    and a timescale of 13 ms against a truth of 50.  Dividing by the 1/e lag
    is the crudest defensible correction and is enough to stop it.
    """
    n = acf_effective_n(lags, corrs)
    scores, fits, tried = [], [], []
    previous = None
    since_best = 0
    for p in orders:
        # Warm start from the previous order: the objective is non-convex and
        # this both speeds up and stabilizes the search.
        starts = [None]
        if previous is not None and len(previous[0]) < p:
            # Pad to the new order rather than requiring p-1: ORDER_GRID skips
            # (6, 8, 10, 13, 16, 20), so an equality test silently disabled the
            # warm start over most of the grid.
            k = ar_to_reflection(previous[0])
            warm = reflection_to_ar(
                np.concatenate([k, np.zeros(p - len(k))]))
            starts.append(np.concatenate([warm, [previous[1], previous[2]]]))
        best = None
        for start in starts:
            cand = fit_ar_acf(lags, corrs, fs, p, guess=start, **kwargs)
            pred_c = cand[1] * ar_acf_fft(cand[0], int(lags[-1]))[lags.astype(int)] + cand[2]
            rss_c = float(np.sum((corrs - pred_c) ** 2))
            if best is None or rss_c < best[0]:
                best = (rss_c, cand)
        phi, height, offset, ok = best[1]
        previous = (phi, height, offset)
        pred = height * ar_acf_fft(phi, int(lags[-1]))[lags.astype(int)] + offset
        rss = float(np.sum((corrs - pred) ** 2))
        pen = np.log(n) if criterion == 'bic' else 2.0
        scores.append(n * np.log(max(rss / n, 1e-300)) + pen * (p + 2))
        fits.append((phi, height, offset, ok))
        tried.append(p)
        since_best = 0 if scores[-1] <= min(scores) else since_best + 1
        if ORDER_PATIENCE_ACF is not None and since_best >= ORDER_PATIENCE_ACF:
            break
    best = int(np.argmin(scores))
    return tried[best], np.asarray(scores), fits[best]


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

MAX_PSD_POINTS = 1200


def psd_fit_points(signal, max_points=MAX_PSD_POINTS):
    """Frequencies and powers used by the spectral fits.

    Bins are thinned to be uniform in *log* frequency, which is the spacing
    that gives each octave equal weight.  On a linear grid 80% of the bins of
    a 1-100 Hz analysis band lie above 50 Hz, so a least-squares fit in log
    power is overwhelmingly a fit to the high-frequency tail: on an AR(2)
    oscillator that pushed the best AR(1) to tau = 250 ms, chasing a steep
    tail it cannot match, where log spacing gives 76 ms and a visible fit to
    the low-frequency plateau.  Where AR(1) is the right model the two agree
    to three figures, so the change costs nothing and buys interpretability.

    Multitaper bins are additionally thinned to the estimator's own resolution
    bandwidth, because within it they are not independent.  Order selection
    counts fitted points as independent observations, so leaving them in makes
    the data look eight times larger than it is and BIC buys orders the record
    cannot support: on 296 fMRI samples it selected p = 20.
    """
    mask = signal.fit_mask & signal.exclude_mask
    freqs, powers = signal.freqs[mask], signal.powers[mask]
    decim = int(getattr(signal, 'fit_decimate', 1) or 1)
    if decim > 1:
        freqs, powers = freqs[::decim], powers[::decim]
    if len(freqs) > 2 and freqs[0] > 0:
        n_keep = min(max_points, len(freqs))
        edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_keep)
        idx = np.unique(np.clip(np.searchsorted(freqs, edges), 0,
                                len(freqs) - 1))
        freqs, powers = freqs[idx], powers[idx]
    elif len(freqs) > max_points:
        step = int(np.ceil(len(freqs) / max_points))
        freqs, powers = freqs[::step], powers[::step]
    return freqs, powers


def est_ar1_direct(signal, **kwargs):
    """Ordinary least squares AR(1) on the time series."""
    phi, s2 = ar_ols(signal.sig, 1)
    tau = phi_to_tau(float(phi[0]), signal.fs)
    modes = ARModes.from_phi(phi, signal.fs, offset=s2,
                             freqs=signal.freqs[signal.fit_mask])
    fit = TimescaleFit(method='ar1_direct', tau=tau, tau_relaxation=tau, order=1,
                       phi=phi, offset=s2, modes=modes,
                       params={'phi1': float(phi[0]), 'resid_var': s2})
    fit.gof = spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, 1.0))
    fit.gof.update(acf_fit_gof(signal, phi))
    return fit


def est_ar1_acf(signal, robust=False, **kwargs):
    """AR(1) autocorrelation form fit to the ACF."""
    kwargs.setdefault('loss', loss_settings(robust, 'acf')[0])
    kwargs.setdefault('f_scale', loss_settings(robust, 'acf')[1])
    phi, height, offset, ok = fit_ar_acf(signal.lags, signal.corrs, signal.fs,
                                         1, **kwargs)
    phi, _, _ = stabilize(phi)
    tau = phi_to_tau(float(phi[0]), signal.fs)
    mask = signal.fit_mask & signal.exclude_mask
    modes = ARModes.from_phi(phi, signal.fs, freqs=signal.freqs[mask])
    fit = TimescaleFit(method='ar1_acf', tau=tau, tau_relaxation=tau, order=1,
                       phi=phi, offset=np.nan, modes=modes, success=ok,
                       params={'phi1': float(phi[0]), 'height': height,
                               'acf_offset': offset})
    fit.gof = acf_fit_gof(signal, phi, height, offset)
    fit.gof.update(spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, 1.0)))
    return fit


def est_arp_acf(signal, orders=None, criterion='bic', order=None,
                max_lags=None, robust=False, **kwargs):
    """AR(p) autocorrelation form fit robustly to the ACF, order by BIC.

    The fit uses every lag ``prepare`` produced unless ``max_lags`` caps it.
    A fixed cap of 250 lags used to sit here, which silently discarded any
    longer lag range asked for at preparation and, at 500 Hz, fit a 200 ms
    relaxation from half a second of lags.
    """
    kwargs.setdefault('loss', loss_settings(robust, 'acf')[0])
    kwargs.setdefault('f_scale', loss_settings(robust, 'acf')[1])
    keep = len(signal.lags) if max_lags is None else min(len(signal.lags), max_lags + 1)
    lags, corrs = signal.lags[:keep], signal.corrs[:keep]
    if order is None:
        top = int(np.clip(len(lags) // 20, 2, 20))
        orders = [p for p in ORDER_GRID if p <= top] if orders is None else list(orders)
        order, scores, (phi, height, offset, ok) = select_order_acf(
            lags, corrs, signal.fs, orders, criterion=criterion, **kwargs)
    else:
        phi, height, offset, ok = fit_ar_acf(lags, corrs, signal.fs, order,
                                             **kwargs)
        scores, orders = None, [order]
    phi, _, stabilized = stabilize(phi)
    mask = signal.fit_mask & signal.exclude_mask
    modes = ARModes.from_phi(phi, signal.fs, freqs=signal.freqs[mask])
    fit = TimescaleFit(method='arp_acf', tau=modes.tau_dominant,
                       tau_relaxation=modes.tau_relaxation, order=order,
                       phi=phi, offset=np.nan, modes=modes, success=ok,
                       params={'height': height, 'acf_offset': offset,
                               'order_scores': scores,
                               'orders': np.asarray(orders),
                               'stabilized': stabilized,
                               'order_at_ceiling': bool(order == max(orders))})
    fit.gof = acf_fit_gof(signal, phi, height, offset)
    fit.gof.update(spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, 1.0)))
    return fit


def est_ar1_psd(signal, robust=False, loss=None, f_scale=None, **kwargs):
    """AR(1) spectral form fit to the PSD."""
    if loss is None:
        loss, f_scale = loss_settings(robust, 'psd')
    mask = signal.fit_mask & signal.exclude_mask
    freqs, powers = psd_fit_points(signal)
    phi, offset, ok = fit_ar_psd(freqs, powers, signal.fs, 1, loss=loss,
                                 f_scale=f_scale, powers_full=signal.powers,
                                 **kwargs)
    phi, offset, stabilized = stabilize(phi, offset)
    tau = phi_to_tau(float(phi[0]), signal.fs)
    modes = ARModes.from_phi(phi, signal.fs, offset=offset,
                             freqs=signal.freqs[mask])
    fit = TimescaleFit(method='ar1_psd', tau=tau, tau_relaxation=tau, order=1,
                       phi=phi, offset=offset, modes=modes, success=ok,
                       params={'phi1': float(phi[0]), 'offset': offset,
                               'stabilized': stabilized})
    fit.gof = spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, offset))
    fit.gof.update(acf_fit_gof(signal, phi))
    return fit


def est_arp_direct(signal, orders=None, criterion='bic', method='burg', **kwargs):
    """AR(p) on the time series with objective order selection."""
    orders = list(ORDER_GRID) if orders is None else list(orders)
    order, scores, (phi, s2) = select_order_time(signal.sig, orders,
                                                 method=method,
                                                 criterion=criterion)
    mask = signal.fit_mask & signal.exclude_mask
    modes = ARModes.from_phi(phi, signal.fs, offset=s2, freqs=signal.freqs[mask])
    fit = TimescaleFit(method='arp_direct', tau=modes.tau_dominant,
                       tau_relaxation=modes.tau_relaxation, order=order,
                       phi=phi, offset=s2, modes=modes,
                       params={'resid_var': s2, 'order_scores': scores,
                               'orders': np.asarray(orders),
                               'order_at_ceiling': bool(order == max(orders))})
    fit.gof = spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, 1.0))
    fit.gof.update(acf_fit_gof(signal, phi))
    return fit


def est_arp_psd(signal, orders=None, criterion='bic', order=None,
                robust=False, loss=None, f_scale=None, **kwargs):
    """AR(p) spectral form fit to the PSD with objective order selection."""
    if loss is None:
        loss, f_scale = loss_settings(robust, 'psd')
    mask = signal.fit_mask & signal.exclude_mask
    freqs, powers = psd_fit_points(signal)
    kw = dict(loss=loss, f_scale=f_scale, powers_full=signal.powers)
    kw.update(kwargs)
    if order is None:
        # Never fit more parameters than the spectrum can support: cap the
        # search at one AR coefficient per five fitted frequency bins.
        top = int(np.clip(len(freqs) // 5, 2, 20))
        orders = [p for p in ORDER_GRID if p <= top] if orders is None else list(orders)
        order, scores, (phi, offset, ok) = select_order_psd(
            freqs, powers, signal.fs, orders, criterion=criterion, **kw)
    else:
        phi, offset, ok = fit_ar_psd(freqs, powers, signal.fs, order, **kw)
        scores, orders = None, [order]
    phi, offset, stabilized = stabilize(phi, offset)
    modes = ARModes.from_phi(phi, signal.fs, offset=offset, freqs=freqs)
    fit = TimescaleFit(method='arp_psd', tau=modes.tau_dominant,
                       tau_relaxation=modes.tau_relaxation, order=order,
                       phi=phi, offset=offset, modes=modes, success=ok,
                       params={'offset': offset, 'order_scores': scores,
                               'orders': np.asarray(orders),
                               'stabilized': stabilized,
                               'order_at_ceiling': bool(order == max(orders))})
    fit.gof = spectral_gof(signal, ar_psd(phi, signal.freqs, signal.fs, offset))
    fit.gof.update(acf_fit_gof(signal, phi))
    return fit


ESTIMATORS = {
    'ar1_direct': est_ar1_direct,
    'ar1_acf': est_ar1_acf,
    'ar1_psd': est_ar1_psd,
    'arp_direct': est_arp_direct,
    'arp_acf': est_arp_acf,
    'arp_psd': est_arp_psd,
}

ESTIMATOR_LABELS = {
    'ar1_direct': 'AR(1) direct',
    'ar1_acf': 'AR(1) ACF fit',
    'ar1_psd': 'AR(1) PSD fit',
    'arp_direct': 'AR(p) direct',
    'arp_acf': 'AR(p) ACF fit',
    'arp_psd': 'AR(p) PSD fit',
}


def estimate(signal, method, **kwargs):
    """Run one estimator by name on a prepared :class:`Signal`."""
    if method not in ESTIMATORS:
        raise KeyError(f'Unknown estimator {method!r}. Options: {list(ESTIMATORS)}')
    return ESTIMATORS[method](signal, **kwargs)


#: Estimators that accept a ``robust`` flag (the fitted, as opposed to
#: closed-form, methods).
FITTED = ('ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd')


def estimate_all(signal, methods=None, options=None, robust=False):
    """Run several estimators on the same prepared signal.

    ``robust`` sets soft-L1 losses for every fitted estimator at once, and is
    the switch to use when the data contain a rhythm.
    """
    methods = list(ESTIMATORS) if methods is None else list(methods)
    options = {} if options is None else dict(options)
    if robust:
        for method in FITTED:
            options[method] = dict(options.get(method, {}), robust=True)
    out = {}
    for method in methods:
        try:
            out[method] = estimate(signal, method, **options.get(method, {}))
        except Exception as exc:  # keep the benchmark running
            out[method] = TimescaleFit(method=method, success=False,
                                       params={'error': repr(exc)})
    return out
