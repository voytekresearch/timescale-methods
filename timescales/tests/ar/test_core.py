"""The free functions behind the AR estimators, and the utilities feeding them."""

import numpy as np
import pytest

from timescales.ar.core import (acf_effective_n, exp_decay, fit_ar_acf,
                                fit_ar_psd, init_ar_from_acf, init_ar_from_psd,
                                select_order_acf, select_order_psd)
from timescales.ar.metrics import (acf_gof, is_divergence, r2_logpsd,
                                   rmse_logpsd)
from timescales.ar.time import ar_burg, ar_ols, population_ar, select_order_time
from timescales.conversions import tau_to_phi
from timescales.decomposition.modes import ar_acf, ar_psd
from timescales.sim import AR1, AR2Damped, ARpModes, sim_ar_process
from timescales.utils import compute_acf, compute_psd

FS = 500.0
FREQS = np.linspace(1.0, 250.0, 400)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_init_from_psd_recovers_an_ar1():
    phi_true = np.array([0.9])
    freqs = np.fft.rfftfreq(2048, d=1 / FS)
    powers = ar_psd(phi_true, freqs, FS, 1.0)
    np.testing.assert_allclose(init_ar_from_psd(powers, 1), phi_true, atol=1e-3)


def test_init_from_acf_recovers_an_ar1():
    corrs = ar_acf(np.array([0.9]), 100)
    np.testing.assert_allclose(init_ar_from_acf(corrs, 1), [0.9], atol=1e-6)


def test_init_from_acf_needs_enough_lags():
    with pytest.raises(ValueError):
        init_ar_from_acf(np.array([1.0, 0.5]), 5)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def test_fit_ar_psd_recovers_coefficients():
    phi_true = np.array([1.4, -0.6])
    powers = ar_psd(phi_true, FREQS, FS, 2.0)
    phi, offset, ok = fit_ar_psd(FREQS, powers, FS, 2)
    assert ok
    np.testing.assert_allclose(phi, phi_true, atol=1e-3)
    assert offset == pytest.approx(2.0, rel=1e-2)


def test_fit_ar_acf_recovers_coefficients():
    phi_true = np.array([1.4, -0.6])
    lags = np.arange(201, dtype=float)
    corrs = ar_acf(phi_true, 200)
    phi, height, offset, ok = fit_ar_acf(lags, corrs, FS, 2)
    assert ok
    np.testing.assert_allclose(phi, phi_true, atol=5e-3)


def test_relaxation_constraint_always_yields_a_real_pole():
    """What the constraint promises: a decay time exists, whatever the data."""
    rng = np.random.default_rng(0)
    sig = AR2Damped(tau=0.05, f_osc=40.0).simulate(60.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(2 * FS), f_range=(1, 250))
    for order in (3, 5, 7):
        phi, _, _ = fit_ar_psd(freqs, powers, FS, order, require_relaxation=True)
        poles = np.roots(np.r_[1.0, -phi])
        assert any(abs(p.imag) < 1e-8 and p.real > 0 for p in poles)


def test_robust_loss_reduces_the_pull_of_a_narrowband_peak():
    """A rhythm is an outlier in the frequency domain; soft-L1 down-weights it."""
    rng = np.random.default_rng(1)
    sig = AR1(tau=0.05).simulate(200.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(1, 250))
    spiked = powers * (1 + 40 * np.exp(-0.5 * ((freqs - 40.0) / 1.0) ** 2))
    plain, _, _ = fit_ar_psd(freqs, spiked, FS, 1)
    robust, _, _ = fit_ar_psd(freqs, spiked, FS, 1, loss='soft_l1', f_scale=0.15)
    clean, _, _ = fit_ar_psd(freqs, powers, FS, 1)
    assert abs(robust[0] - clean[0]) < abs(plain[0] - clean[0])


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------

def test_select_order_psd_returns_scores_for_every_order_tried():
    powers = ar_psd(np.array([0.9]), FREQS, FS, 1.0)
    order, scores, fit = select_order_psd(FREQS, powers, FS, [1, 2, 3])
    assert order in (1, 2, 3)
    assert len(scores) == 3 and len(fit) == 3


def test_select_order_psd_patience_stops_early():
    powers = ar_psd(np.array([0.9]), FREQS, FS, 1.0)
    _, all_scores, _ = select_order_psd(FREQS, powers, FS, range(1, 12))
    _, few_scores, _ = select_order_psd(FREQS, powers, FS, range(1, 12), patience=1)
    assert len(few_scores) <= len(all_scores)


def test_acf_effective_n_shrinks_with_slower_decay():
    """Dependent lags carry less information, which is what n_eff encodes."""
    lags = np.arange(251, dtype=float)
    fast = acf_effective_n(lags, ar_acf(np.array([tau_to_phi(0.005, FS)]), 250))
    slow = acf_effective_n(lags, ar_acf(np.array([tau_to_phi(0.2, FS)]), 250))
    assert fast > slow
    assert slow >= 6


def test_select_order_acf_honours_an_explicit_n_eff():
    lags = np.arange(251, dtype=float)
    corrs = ar_acf(np.array([1.4, -0.6]), 250)
    strict, _, _ = select_order_acf(lags, corrs, FS, range(1, 8), n_eff=6)
    loose, _, _ = select_order_acf(lags, corrs, FS, range(1, 8), n_eff=250)
    assert strict <= loose


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('estimator', [ar_ols, ar_burg])
def test_time_domain_estimators_recover_an_ar1(estimator):
    rng = np.random.default_rng(0)
    phi_true = tau_to_phi(0.05, FS)
    sig = sim_ar_process(np.array([phi_true]), 100000, rng)
    phi, resid_var = estimator(sig, 1)
    assert phi[0] == pytest.approx(phi_true, rel=0.02)
    assert resid_var > 0


def test_population_ar_is_exact_on_a_known_acf():
    phi_true = np.array([1.4, -0.6])
    np.testing.assert_allclose(population_ar(ar_acf(phi_true, 50), 2),
                               phi_true, atol=1e-8)


def test_select_order_time_prefers_the_true_order():
    rng = np.random.default_rng(2)
    sig = sim_ar_process(np.array([1.4, -0.6]), 40000, rng)
    order, scores, _ = select_order_time(sig, range(1, 8))
    assert order == 2
    assert len(scores) == 7


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_are_perfect_on_an_exact_fit():
    powers = ar_psd(np.array([0.9]), FREQS, FS, 1.0)
    assert r2_logpsd(powers, powers) == pytest.approx(1.0)
    assert rmse_logpsd(powers, powers) == pytest.approx(0.0)
    assert is_divergence(powers, powers) == pytest.approx(0.0)
    corrs = ar_acf(np.array([0.9]), 100)
    assert acf_gof(corrs, corrs)['r2_acf'] == pytest.approx(1.0)


def test_metrics_degrade_with_a_worse_fit():
    powers = ar_psd(np.array([0.9]), FREQS, FS, 1.0)
    worse = ar_psd(np.array([0.5]), FREQS, FS, 1.0)
    assert r2_logpsd(powers, worse) < 1.0
    assert rmse_logpsd(powers, worse) > 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def test_compute_psd_defaults_to_a_periodogram_and_honours_f_range():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=4096)
    freqs, powers = compute_psd(sig, FS)
    assert freqs[0] == 0.0 and freqs[-1] == pytest.approx(FS / 2)
    freqs, powers = compute_psd(sig, FS, nperseg=512, f_range=(1, 100))
    assert freqs[0] >= 1.0 and freqs[-1] <= 100.0
    assert freqs.shape == powers.shape


def test_compute_acf_is_normalized_and_bounded():
    rng = np.random.default_rng(0)
    sig = sim_ar_process(np.array([0.9]), 20000, rng)
    lags, corrs = compute_acf(sig, 100)
    assert lags[0] == 0 and len(lags) == 101
    assert corrs[0] == pytest.approx(1.0)
    assert np.all(np.abs(corrs) <= 1.0 + 1e-12)


def test_compute_acf_refuses_more_lags_than_samples():
    with pytest.raises(ValueError):
        compute_acf(np.zeros(10), 20)


def test_exp_decay_matches_the_ar1_acf():
    lags = np.arange(50, dtype=float)
    tau = 0.05
    np.testing.assert_allclose(exp_decay(lags, FS, tau, 1.0, 0.0),
                               tau_to_phi(tau, FS) ** lags, rtol=1e-10)


# ---------------------------------------------------------------------------
# Analytic Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('order', [1, 2, 5, 12, 25])
def test_reflection_jacobian_matches_finite_differences(order):
    from timescales.ar.core import reflection_to_ar_jac
    from timescales.decomposition.modes import reflection_to_ar
    rng = np.random.default_rng(order)
    k = rng.uniform(-0.8, 0.8, order)
    phi, dphi = reflection_to_ar_jac(k)
    np.testing.assert_allclose(phi, reflection_to_ar(k), rtol=1e-12)
    step = 1e-6
    for j in range(order):
        up, dn = k.copy(), k.copy()
        up[j] += step
        dn[j] -= step
        numeric = (reflection_to_ar(up) - reflection_to_ar(dn)) / (2 * step)
        np.testing.assert_allclose(dphi[:, j], numeric, atol=1e-6)


@pytest.mark.parametrize('order', [1, 2, 5, 12, 25])
def test_real_pole_jacobian_matches_finite_differences(order):
    from timescales.ar.core import real_pole_reflection_to_ar_jac
    from timescales.decomposition.modes import real_pole_reflection_to_ar
    rng = np.random.default_rng(order)
    a, k = 0.7, rng.uniform(-0.8, 0.8, order - 1)
    phi, dphi = real_pole_reflection_to_ar_jac(a, k)
    np.testing.assert_allclose(phi, real_pole_reflection_to_ar(a, k), rtol=1e-12)
    step = 1e-6
    numeric = (real_pole_reflection_to_ar(a + step, k)
               - real_pole_reflection_to_ar(a - step, k)) / (2 * step)
    np.testing.assert_allclose(dphi[:, 0], numeric, atol=1e-6)
    for j in range(order - 1):
        up, dn = k.copy(), k.copy()
        up[j] += step
        dn[j] -= step
        numeric = (real_pole_reflection_to_ar(a, up)
                   - real_pole_reflection_to_ar(a, dn)) / (2 * step)
        np.testing.assert_allclose(dphi[:, j + 1], numeric, atol=1e-6)


@pytest.mark.parametrize('order', [1, 3, 8, 20])
def test_logpsd_jacobian_matches_finite_differences(order):
    """The derivative the optimizer is given must be the derivative it has."""
    from timescales.ar.core import _logpsd_jac, real_pole_reflection_to_ar_jac
    from timescales.decomposition.modes import real_pole_reflection_to_ar
    rng = np.random.default_rng(order)
    a, k, offset = 0.9, rng.uniform(-0.5, 0.5, order - 1), 2.5
    z = np.exp(-2j * np.pi * np.outer(FREQS, np.arange(1, order + 1)) / FS)

    def model(a, k, offset):
        phi = real_pole_reflection_to_ar(a, k)
        return np.log10(offset / np.abs(1.0 - z @ phi) ** 2)

    phi, dphi = real_pole_reflection_to_ar_jac(a, k)
    jac = _logpsd_jac(z, 1.0 - z @ phi, phi, dphi, offset)
    assert jac.shape == (len(FREQS), order + 1)
    step = 1e-7
    numeric = np.empty_like(jac)
    numeric[:, 0] = (model(a + step, k, offset) - model(a - step, k, offset)) / (2 * step)
    for j in range(order - 1):
        up, dn = k.copy(), k.copy()
        up[j] += step
        dn[j] -= step
        numeric[:, j + 1] = (model(a, up, offset) - model(a, dn, offset)) / (2 * step)
    numeric[:, -1] = (model(a, k, offset + step) - model(a, k, offset - step)) / (2 * step)
    np.testing.assert_allclose(jac, numeric, atol=1e-5, rtol=1e-4)


def test_analytic_jacobian_reaches_the_same_fit_as_a_numerical_one():
    """Speed must not come from a different answer."""
    from scipy.optimize import curve_fit
    from timescales.ar.core import _ar_psd_guess
    from timescales.decomposition.modes import (ar_to_reflection,
                                                reflection_to_ar)
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.06, 40.0)))
    order = len(proc.phi(FS))          # one real pole and two conjugate pairs
    powers = ar_psd(proc.phi(FS), FREQS, FS, 1.0)

    analytic = fit_ar_psd(FREQS, powers, FS, order)

    # the same optimization with scipy's finite-difference Jacobian
    guess = _ar_psd_guess(FREQS, powers, FS, order)
    refl = np.clip(ar_to_reflection(guess[:-1]), -0.9999, 0.9999)
    p0 = np.concatenate([refl, [max(guess[-1], 1e-12)]])
    bounds = (np.concatenate([np.full(order, -0.9999), [1e-16]]),
              np.concatenate([np.full(order, 0.9999), [np.inf]]))
    z = np.exp(-2j * np.pi * np.outer(FREQS, np.arange(1, order + 1)) / FS)

    def model(_, *params):
        phi = reflection_to_ar(np.asarray(params[:-1]))
        return np.log10(params[-1] / np.abs(1.0 - z @ phi) ** 2)

    params, _ = curve_fit(model, FREQS, np.log10(powers), p0=p0, bounds=bounds,
                          maxfev=4000)
    numeric_phi = reflection_to_ar(np.asarray(params[:-1]))

    # both recover the process, which is the property that matters
    np.testing.assert_allclose(analytic[0], proc.phi(FS), atol=1e-3)
    np.testing.assert_allclose(numeric_phi, proc.phi(FS), atol=1e-3)
    np.testing.assert_allclose(analytic[0], numeric_phi, atol=1e-3)


def test_order_search_warm_starts_do_not_change_the_answer():
    """The search reuses the previous order's solution; the fit is unchanged."""
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0)))
    powers = ar_psd(proc.phi(FS), FREQS, FS, 1.0)
    order, _, (phi, offset, _) = select_order_psd(FREQS, powers, FS, range(1, 8))
    direct = fit_ar_psd(FREQS, powers, FS, order)
    np.testing.assert_allclose(ar_psd(phi, FREQS, FS, offset),
                               ar_psd(direct[0], FREQS, FS, direct[1]), rtol=1e-3)
