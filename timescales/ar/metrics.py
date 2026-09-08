"""Goodness of fit in the spectral and lag domains.

A model fit in one domain is scored in both, so that "the spectral fit
describes the autocorrelation" is a measured statement rather than an
assumption.  Spectral scores are computed on log power with the overall
scale refit, so estimators that never fit an offset are scored on shape.
"""

import numpy as np

from timescales.decomposition.modes import ar_acf_fft

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
