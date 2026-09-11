"""Time-domain AR(p) estimation and order selection.

Conditional least squares and Burg's method on the samples themselves, the
population (Yule-Walker) fit to a known autocorrelation, and BIC order
selection from the conditional Gaussian likelihood.
"""

import numpy as np

from .spectral import burg

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
