"""Tests for the analytic finite-sample results."""

import pandas as pd
import numpy as np
import pytest

from tsmodes.estimators import ar_ols
from tsmodes.finite import (ar1_phi_bias, ar1_phi_ci, ar1_phi_sd,
                            ar1_tau_ci, ar1_tau_moments, bias_corrected_phi,
                            duration_for_relative_bias,
                            duration_for_relative_sd, phi_of_tau,
                            relative_bias_asymptotic, relative_sd_asymptotic,
                            sample_tau_finite, tau_of_phi)
from tsmodes.sims import sim_ar_process

FS = 500.0
TAU = 0.05


def monte_carlo_tau(duration, n_reps, seed=0):
    rng = np.random.default_rng(seed)
    phi = phi_of_tau(TAU, FS)
    n = int(duration * FS)
    out = np.empty(n_reps)
    for i in range(n_reps):
        sig = sim_ar_process([phi], n, rng, burn_in=2000)
        out[i] = tau_of_phi(ar_ols(sig, 1)[0][0], FS)
    return out


@pytest.mark.parametrize('duration', [1.0, 5.0, 50.0])
def test_analytic_sd_matches_monte_carlo(duration):
    est = monte_carlo_tau(duration, 800, seed=int(duration))
    _, sd = ar1_tau_moments(TAU, duration, FS)
    assert np.nanstd(est) == pytest.approx(sd, rel=0.2)


@pytest.mark.parametrize('duration', [1.0, 5.0, 50.0])
def test_analytic_bias_matches_monte_carlo(duration):
    est = monte_carlo_tau(duration, 2000, seed=100 + int(duration))
    bias, _ = ar1_tau_moments(TAU, duration, FS)
    observed = np.nanmean(est) - TAU
    assert observed == pytest.approx(bias, abs=0.35 * abs(bias) + 1e-4)


def test_relative_moments_depend_only_on_duration_over_tau():
    """The same T/tau gives the same relative bias and spread at any tau/fs."""
    for tau, fs in ((0.02, 1000.0), (0.5, 250.0)):
        for ratio in (50.0, 500.0):
            bias, sd = ar1_tau_moments(tau, ratio * tau, fs)
            assert bias / tau == pytest.approx(relative_bias_asymptotic(ratio), rel=0.1)
            assert sd / tau == pytest.approx(relative_sd_asymptotic(ratio), rel=0.1)


def test_duration_solvers_invert_the_asymptotics():
    assert relative_sd_asymptotic(
        duration_for_relative_sd(TAU, 0.1) / TAU) == pytest.approx(0.1)
    assert relative_bias_asymptotic(
        duration_for_relative_bias(TAU, 0.1) / TAU) == pytest.approx(-0.1)


def test_ten_percent_standard_error_needs_two_hundred_taus():
    assert duration_for_relative_sd(TAU, 0.1) / TAU == pytest.approx(200.0)


def test_sampling_distribution_matches_its_own_moments():
    rng = np.random.default_rng(0)
    draws = sample_tau_finite(TAU, 5.0, FS, 200000, rng)
    bias, sd = ar1_tau_moments(TAU, 5.0, FS)
    assert np.nanmean(draws) - TAU == pytest.approx(bias, rel=0.15)
    assert np.nanstd(draws) == pytest.approx(sd, rel=0.15)


def test_confidence_interval_widens_as_duration_shrinks():
    widths = []
    for duration in (1.0, 10.0, 100.0):
        lo, hi, _ = ar1_tau_ci(TAU, duration, FS)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_confidence_interval_covers_the_truth_at_the_nominal_rate():
    duration = 20.0
    est = monte_carlo_tau(duration, 1500, seed=7)
    lo, hi, _ = ar1_tau_ci(TAU, duration, FS, alpha=0.05)
    coverage = np.mean((est >= lo) & (est <= hi))
    assert coverage == pytest.approx(0.95, abs=0.05)


def test_bias_correction_reduces_bias():
    duration = 2.0
    rng = np.random.default_rng(3)
    phi = phi_of_tau(TAU, FS)
    n = int(duration * FS)
    raw, corrected = [], []
    for _ in range(1500):
        sig = sim_ar_process([phi], n, rng, burn_in=2000)
        p = ar_ols(sig, 1)[0][0]
        raw.append(p)
        corrected.append(bias_corrected_phi(p, n))
    assert abs(np.mean(corrected) - phi) < abs(np.mean(raw) - phi)


def test_phi_moments_match_monte_carlo():
    duration = 2.0
    rng = np.random.default_rng(4)
    phi = phi_of_tau(TAU, FS)
    n = int(duration * FS)
    est = np.array([ar_ols(sim_ar_process([phi], n, rng, burn_in=2000), 1)[0][0]
                    for _ in range(2000)])
    assert np.mean(est) - phi == pytest.approx(ar1_phi_bias(phi, n), rel=0.25)
    assert np.std(est) == pytest.approx(ar1_phi_sd(phi, n), rel=0.15)


def test_phi_ci_is_centred_on_the_bias_corrected_value():
    lo, hi, centre, sd = ar1_phi_ci(0.95, 1000)
    assert centre < 0.95
    assert lo < centre < hi
    assert hi - lo == pytest.approx(2 * 1.959963985 * sd, rel=1e-6)


def test_scalar_input_gives_scalar_output():
    """A 0-d array in a pandas column becomes an object column silently.

    The finite-sampling figure crashed inside ``fill_between`` on a freshly
    computed table while working on the same table after a CSV round trip,
    because ``np.where`` had returned 0-d arrays that pandas stored as
    objects and the round trip cast to float.
    """
    from tsmodes.finite import tau_of_phi, ar1_tau_ci
    assert isinstance(tau_of_phi(0.9, FS), float)
    assert np.ndim(tau_of_phi(0.9, FS)) == 0
    for value in ar1_tau_ci(0.05, 1.0, FS):
        assert isinstance(value, float), 'confidence bound is not a float'

    frame = pd.DataFrame({'lo': [ar1_tau_ci(0.05, d, FS)[0]
                                 for d in (0.5, 1.0, 2.0)]})
    assert frame['lo'].dtype == np.float64, 'object column would crash plotting'

    # arrays still come back as arrays
    out = tau_of_phi(np.array([0.9, 0.8]), FS)
    assert isinstance(out, np.ndarray) and out.shape == (2,)
