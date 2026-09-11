"""The six timescale estimators compared in this paper.

A 3 x 2 grid of fitting domain (time series, autocorrelation, spectrum) and
model order (AR(1), AR(p)).  Every estimator consumes the same prepared
:class:`~tsmodes.prep.Signal` and returns the same :class:`TimescaleFit`
record, so bias, error, goodness of fit and model order are comparable across
methods.

This is analysis code.  The estimators themselves -- :class:`timescales.ar.ARPSD`
and :class:`timescales.ar.ARACF` and the functions behind them -- are general
and live in the ``timescales`` package.  What is here is the particular set of
comparisons this paper makes, and the settings it makes them with: the order
grid, the early-stopping patience, the robust-loss margins, and the rule for
thinning spectral bins before fitting.
"""

import numpy as np

from timescales.ar import ar_burg, ar_ols, select_order_time
from timescales.ar.core import (fit_ar_acf, fit_ar_psd, select_order_acf,
                                select_order_psd)
from timescales.ar.metrics import (acf_fit_gof, acf_gof, is_divergence,  # noqa: F401
                                   r2_logpsd, rmse_logpsd, spectral_gof)
from timescales.conversions import phi_to_tau, tau_to_phi  # noqa: F401
from timescales.decomposition import ARModes
from timescales.decomposition.modes import ar_psd, stabilize

from dataclasses import dataclass, field

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
    kwargs.setdefault('maxfev', ACF_MAXFEV)
    kwargs.setdefault('require_relaxation', REQUIRE_RELAXATION)
    kwargs.setdefault('tol', ACF_TOL)
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
    kwargs.setdefault('maxfev', ACF_MAXFEV)
    kwargs.setdefault('require_relaxation', REQUIRE_RELAXATION)
    kwargs.setdefault('tol', ACF_TOL)
    keep = len(signal.lags) if max_lags is None else min(len(signal.lags), max_lags + 1)
    lags, corrs = signal.lags[:keep], signal.corrs[:keep]
    if order is None:
        top = int(np.clip(len(lags) // 20, 2, 20))
        orders = [p for p in ORDER_GRID if p <= top] if orders is None else list(orders)
        order, scores, (phi, height, offset, ok) = select_order_acf(
            lags, corrs, signal.fs, orders, criterion=criterion,
            patience=ORDER_PATIENCE_ACF, **kwargs)
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
                                 require_relaxation=REQUIRE_RELAXATION, **kwargs)
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
    kw = dict(loss=loss, f_scale=f_scale, powers_full=signal.powers,
              require_relaxation=REQUIRE_RELAXATION)
    kw.update(kwargs)
    if order is None:
        # Never fit more parameters than the spectrum can support: cap the
        # search at one AR coefficient per five fitted frequency bins.
        top = int(np.clip(len(freqs) // 5, 2, 20))
        orders = [p for p in ORDER_GRID if p <= top] if orders is None else list(orders)
        order, scores, (phi, offset, ok) = select_order_psd(
            freqs, powers, signal.fs, orders, criterion=criterion,
            patience=ORDER_PATIENCE_PSD, **kw)
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
