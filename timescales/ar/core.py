"""Fitting an AR(p) model to a power spectrum or an autocorrelation.

Both fits estimate the same object -- the AR coefficients ``phi`` -- from a
different representation of the same process, so the two are written to mirror
each other.  Both are optimized over *reflection* coefficients rather than AR
coefficients: every point in the box ``|k_i| < 1`` is a stationary model, the
parameters are all of order one, and the optimizer cannot wander into a region
where the model has no finite timescale.

These are the free functions behind :class:`~timescales.ar.ARPSD` and
:class:`~timescales.ar.ARACF`.  Use the classes unless you need the pieces.

Nothing here carries a default chosen for a particular recording or analysis:
every numerical setting is an argument.
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit

from timescales.conversions import tau_to_phi
from timescales.decomposition.modes import (LN10, ar_acf_fft, ar_psd,
                                            ar_to_real_pole_reflection,
                                            ar_to_reflection, poles_to_ar,
                                            real_pole_reflection_to_ar,
                                            reflection_to_ar)

#: Largest reflection-coefficient magnitude the optimizer may reach.  It bounds
#: the slowest representable relaxation at ``-1 / (fs log K_BOUND)``: 20 s at
#: 500 Hz.  Raise it to represent slower decays, at the cost of conditioning.
K_BOUND = 0.9999


def init_ar_from_psd(powers, order, eps=1e-12):
    """Yule-Walker AR coefficients from a one-sided, uniformly spaced PSD."""
    from scipy.linalg import solve_toeplitz
    powers = np.asarray(powers, dtype=float)
    acov = np.fft.irfft(np.maximum(powers, eps))
    if len(acov) < order + 1:
        raise ValueError('Need at least `order` + 1 autocovariance lags.')
    return solve_toeplitz(acov[:order], acov[1:order + 1])


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


def reflection_to_ar_jac(k):
    """AR coefficients and ``d phi / d k`` for the Levinson step-up recursion.

    Forward-mode differentiation of :func:`reflection_to_ar`.  Each step is
    ``phi <- [phi - k_m reverse(phi), k_m]``, in which ``phi`` depends only on
    the earlier reflection coefficients, so the derivative recursion mirrors
    the value recursion exactly and costs the same ``O(p^2)``.

    Returns
    -------
    phi : 1d array, (p,)
    dphi : 2d array, (p, p)
        ``dphi[i, j] = d phi_i / d k_j``.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    p = len(k)
    phi = np.zeros(0)
    dphi = np.zeros((0, p))
    for m in range(p):
        km = k[m]
        if len(phi) == 0:
            phi = np.array([km])
            dphi = np.zeros((1, p))
            dphi[0, m] = 1.0
            continue
        rev, drev = phi[::-1], dphi[::-1]
        new_dphi = np.vstack([dphi - km * drev, np.zeros((1, p))])
        new_dphi[:-1, m] -= rev
        new_dphi[-1, m] = 1.0
        phi = np.concatenate([phi - km * rev, [km]])
        dphi = new_dphi
    return phi, dphi


def real_pole_reflection_to_ar_jac(a, k):
    """AR coefficients and their Jacobian for the constrained parameterization.

    ``A(z^-1) = (1 - a z^-1) A_rest(z^-1)`` with ``A_rest`` built from the
    reflection coefficients ``k``.  Returns the derivative with respect to
    ``[a, k_1, ..., k_{p-1}]``.
    """
    a = float(a)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if k.size == 0:
        return np.array([a]), np.ones((1, 1))
    rest, drest = reflection_to_ar_jac(k)
    head = np.array([1.0, -a])
    rpoly = np.concatenate(([1.0], -rest))
    poly = np.convolve(head, rpoly)
    dpoly = np.empty((len(poly), 1 + len(k)))
    dpoly[:, 0] = np.convolve([0.0, -1.0], rpoly)
    for j in range(len(k)):
        dpoly[:, j + 1] = np.convolve(head, np.concatenate(([0.0], -drest[:, j])))
    return -poly[1:], -dpoly[1:]


def _logpsd_jac(z_inv, denom, phi, dphi_dparam, offset):
    """``d log10 S / d params`` for the AR(p) spectral model.

    With ``D(f) = 1 - sum_k phi_k z^-k`` and ``S = offset / |D|^2``,

        ``d log10 S / d phi_j = (2 / ln 10) Re(z^-j conj(D)) / |D|^2``

    and ``d log10 S / d offset = 1 / (offset ln 10)``.  The chain rule through
    the reflection parameterization is the matrix product with ``dphi``.
    """
    weight = 2.0 / (LN10 * np.abs(denom) ** 2)
    dlog_dphi = weight[:, None] * np.real(z_inv * np.conj(denom)[:, None])
    jac = np.empty((len(denom), dphi_dparam.shape[1] + 1))
    jac[:, :-1] = dlog_dphi @ dphi_dparam
    jac[:, -1] = 1.0 / (offset * LN10)
    return jac


def fit_ar_psd(freqs, powers, fs, order, guess=None, k_bound=K_BOUND,
               loss='linear', f_scale=None, maxfev=4000,
               freqs_full=None, powers_full=None,
               require_relaxation=False):
    """Fit the AR(p) spectral form to a PSD, in log10 power.

    The model is optimized over reflection coefficients rather than AR
    coefficients: every point in the box ``|k_i| < 1`` is a stationary model,
    the parameters are all O(1), and the fit cannot wander into a region where
    mode timescales are undefined.  Regression is robust (soft-L1) so that a
    narrowband oscillation is treated as an outlier rather than as evidence
    about the slow decay.

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
        to_ar_jac = real_pole_reflection_to_ar_jac
    else:
        refl = np.clip(ar_to_reflection(guess[:-1]), -k_bound, k_bound)
        a0, k0 = refl[0], refl[1:]
        def to_ar(a, k):
            return reflection_to_ar(np.concatenate([[a], k]))
        def to_ar_jac(a, k):
            return reflection_to_ar_jac(np.concatenate([[a], k]))
    p0 = np.concatenate([[a0], k0, [max(guess[-1], 1e-12)]])
    lo0 = 1e-3 if require_relaxation else -k_bound
    bounds = (np.concatenate([[lo0], np.full(order - 1, -k_bound), [1e-16]]),
              np.concatenate([np.full(order, k_bound), [np.inf]]))

    k_idx = np.arange(1, order + 1)
    z_inv = np.exp(-2j * np.pi * np.outer(freqs, k_idx) / fs)

    def model(_, *params):
        phi = to_ar(params[0], np.asarray(params[1:-1]))
        return np.log10(params[-1] / np.abs(1.0 - z_inv @ phi) ** 2)

    def jac(_, *params):
        # The residual is smooth in the reflection coefficients, so the
        # derivative is available in closed form.  Supplying it replaces the
        # p + 1 extra model evaluations a finite-difference Jacobian needs at
        # every iteration, which is most of the cost of a high-order fit, and
        # it is exact where a difference quotient is not.
        phi, dphi = to_ar_jac(params[0], np.asarray(params[1:-1]))
        denom = 1.0 - z_inv @ phi
        return _logpsd_jac(z_inv, denom, phi, dphi, params[-1])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            params, _ = curve_fit(model, freqs, np.log10(powers), p0=p0,
                                  bounds=bounds, maxfev=maxfev, loss=loss,
                                  f_scale=f_scale, jac=jac)
            success = True
        except Exception:
            params, success = p0, False
    return (to_ar(params[0], np.asarray(params[1:-1])), float(params[-1]),
            success)


def select_order_psd(freqs, powers, fs, orders, criterion='bic', patience=None,
                     **kwargs):
    """AR order selection on the PSD via a log-power residual BIC.

    ``BIC(p) = n_f log(RSS/n_f) + (p + 1) log(n_f)`` with ``n_f`` the number of
    fitted frequency bins.  Welch bins from non-overlapping segments are
    approximately independent, which is the assumption this makes.
    """
    n_f = len(freqs)
    scores, fits, tried = [], [], []
    since_best = 0
    previous = None
    for p in orders:
        # Warm start from the previous order.  A fit at order p is close to the
        # fit at order p - 1 with the extra reflection coefficients near zero,
        # so padding the previous solution starts the optimizer near its
        # answer.  Without it every order is solved from a fresh Yule-Walker
        # guess and the high orders, where the objective is worst conditioned,
        # take hundreds of iterations to get back to what the previous order
        # already knew.
        start = None
        if previous is not None and len(previous[0]) < p:
            k_prev = ar_to_reflection(previous[0])
            warm = reflection_to_ar(np.concatenate([k_prev, np.zeros(p - len(k_prev))]))
            start = np.concatenate([warm, [previous[1]]])
        phi, offset, ok = fit_ar_psd(freqs, powers, fs, p, guess=start, **kwargs)
        previous = (phi, offset)
        pred = ar_psd(phi, freqs, fs, offset)
        rss = np.sum((np.log10(powers) - np.log10(pred)) ** 2)
        pen = np.log(n_f) if criterion == 'bic' else 2.0
        scores.append(n_f * np.log(max(rss / n_f, 1e-300)) + pen * (p + 1))
        fits.append((phi, offset, ok))
        tried.append(p)
        since_best = 0 if scores[-1] <= min(scores) else since_best + 1
        if patience is not None and since_best >= patience:
            break
    best = int(np.argmin(scores))
    return tried[best], np.asarray(scores), fits[best]


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
               loss='linear', f_scale=None, maxfev=600, tol=1e-6,
               drop_zero_lag=True, with_offset=True,
               require_relaxation=False):
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
                                      f_scale=f_scale, ftol=tol,
                                      xtol=tol, gtol=tol,
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


def select_order_acf(lags, corrs, fs, orders, criterion='bic', patience=None,
                     n_eff=None, **kwargs):
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
    n = acf_effective_n(lags, corrs) if n_eff is None else int(n_eff)
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
        if patience is not None and since_best >= patience:
            break
    best = int(np.argmin(scores))
    return tried[best], np.asarray(scores), fits[best]
