"""Analytic finite-sample behaviour of AR(1) timescale estimates.

For a mean-corrected AR(1) fit by least squares on ``n`` samples the classical
small-sample results are

    E[phi_hat] - phi  ~=  -(1 + 3 phi) / n          (Marriott-Pope, Kendall)
    Var[phi_hat]      ~=  (1 - phi^2) / n

Propagating these through ``tau = -1 / (fs log phi)`` with
``dtau/dphi = tau^2 fs / phi`` gives, for ``phi -> 1`` (i.e. ``tau fs >> 1``)
and recording duration ``T = n / fs``, two expressions that depend on the data
only through the dimensionless ratio ``T / tau``:

    bias(tau) / tau  ~=  -2 tau / T   =  -2 / (T/tau)
    sd(tau)   / tau  ~=  sqrt(2 tau / T)  =  sqrt(2 / (T/tau))

The bias expression keeps the second-order delta term: ``tau(phi)`` is strongly
convex near ``phi = 1``, and the Jensen contribution ``0.5 tau'' Var[phi_hat]``
cancels half of the ``-4 tau / T`` first-order term.  Both expressions depend on
the data only through the dimensionless ratio ``T / tau``.

So a 10% standard error needs ``T/tau ~ 200`` and a bias below 10% needs
``T/tau ~ 20``: the required recording duration scales with the timescale being
measured, not with the sampling rate.
"""

import numpy as np


def phi_of_tau(tau, fs):
    return np.exp(-1.0 / (np.asarray(tau, dtype=float) * fs))


def tau_of_phi(phi, fs):
    """Timescale of an AR(1) coefficient, scalar in and scalar out.

    ``np.where`` returns a 0-d array for scalar input, and a list of those
    lands in pandas as an ``object`` column.  That is invisible until
    something numeric touches it: the finite-sampling figure crashed inside
    ``fill_between`` on the freshly computed table while working perfectly on
    the same table after a CSV round trip had quietly cast it to float.
    """
    phi_arr = np.asarray(phi, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = -1.0 / (fs * np.log(phi_arr))
    out = np.where((phi_arr > 0) & (phi_arr < 1), out, np.nan)
    return float(out) if np.ndim(phi) == 0 else out


def ar1_phi_bias(phi, n):
    """Leading-order bias of the least-squares AR(1) coefficient."""
    return -(1.0 + 3.0 * np.asarray(phi, dtype=float)) / n


def ar1_phi_sd(phi, n):
    """Leading-order standard deviation of the least-squares AR(1) coefficient."""
    phi = np.asarray(phi, dtype=float)
    return np.sqrt(np.maximum(1.0 - phi ** 2, 0.0) / n)


def ar1_phi_ci(phi, n, alpha=0.05, bias_correct=True):
    """Normal-approximation confidence interval for phi_hat around true phi."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    centre = np.asarray(phi, dtype=float) + (ar1_phi_bias(phi, n) if bias_correct else 0.0)
    sd = ar1_phi_sd(phi, n)
    return centre - z * sd, centre + z * sd, centre, sd


def ar1_tau_moments(tau, duration, fs, second_order=True):
    """Delta-method mean and sd of tau_hat for an AR(1) recorded for ``duration``.

    ``tau(phi)`` is convex, so the second-order (Jensen) term is not negligible
    at short durations; it is included by default.
    """
    tau = np.asarray(tau, dtype=float)
    n = np.asarray(duration, dtype=float) * fs
    phi = phi_of_tau(tau, fs)
    d1 = tau ** 2 * fs / phi
    bias = d1 * ar1_phi_bias(phi, n)
    var = ar1_phi_sd(phi, n) ** 2
    if second_order:
        d2 = (fs * tau ** 2 / phi ** 2) * (2 * tau * fs - 1.0)
        bias = bias + 0.5 * d2 * var
    sd = d1 * np.sqrt(var)
    return bias, sd


def ar1_tau_ci(tau, duration, fs, alpha=0.05):
    """Confidence interval for tau_hat, obtained by mapping the phi interval."""
    n = np.asarray(duration, dtype=float) * fs
    phi = phi_of_tau(tau, fs)
    lo, hi, centre, _ = ar1_phi_ci(phi, n, alpha=alpha)
    return tau_of_phi(np.clip(lo, 1e-12, 1 - 1e-12), fs), \
        tau_of_phi(np.clip(hi, 1e-12, 1 - 1e-12), fs), \
        tau_of_phi(np.clip(centre, 1e-12, 1 - 1e-12), fs)


def sample_tau_finite(tau, duration, fs, n_draws, rng):
    """Draw tau_hat from the analytic finite-sample sampling distribution.

    ``phi_hat ~ N(phi + bias, sd^2)`` with the moments above, mapped through
    ``tau(phi)``.  Draws with ``phi_hat`` outside ``(0, 1)`` map to NaN: an
    estimate that is not a decaying AR(1).
    """
    n = duration * fs
    phi = phi_of_tau(tau, fs)
    draws = rng.normal(phi + ar1_phi_bias(phi, n), ar1_phi_sd(phi, n), size=n_draws)
    return tau_of_phi(draws, fs)


def sample_phi_finite(phi, duration, fs, n_draws, rng):
    """Draw phi_hat from the analytic finite-sample sampling distribution."""
    n = duration * fs
    return rng.normal(phi + ar1_phi_bias(phi, n), ar1_phi_sd(phi, n), size=n_draws)


def relative_sd_asymptotic(duration_over_tau):
    """``sd(tau_hat) / tau ~= sqrt(2 / (T/tau))``."""
    return np.sqrt(2.0 / np.asarray(duration_over_tau, dtype=float))


def relative_bias_asymptotic(duration_over_tau):
    """``bias(tau_hat) / tau ~= -2 / (T/tau)`` (second-order delta method)."""
    return -2.0 / np.asarray(duration_over_tau, dtype=float)


def duration_for_relative_sd(tau, rel_sd):
    """Recording duration needed for a target relative standard error."""
    return 2.0 * np.asarray(tau, dtype=float) / np.asarray(rel_sd, dtype=float) ** 2


def duration_for_relative_bias(tau, rel_bias):
    """Recording duration needed for a target relative bias magnitude."""
    return 2.0 * np.asarray(tau, dtype=float) / np.abs(np.asarray(rel_bias, dtype=float))


def bias_corrected_phi(phi_hat, n):
    """Marriott-Pope bias correction applied to an estimate."""
    return np.asarray(phi_hat, dtype=float) + (1.0 + 3.0 * np.asarray(phi_hat, dtype=float)) / n
