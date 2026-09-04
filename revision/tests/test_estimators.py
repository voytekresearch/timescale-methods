"""Tests for the shared estimator API."""

import numpy as np
import pytest

from tsmodes.estimators import (ESTIMATORS, acf_fit_gof, ar_burg, ar_ols,
                                estimate,
                                estimate_all, fit_ar_psd, phi_to_tau,
                                population_ar, prepare, select_order_psd,
                                select_order_time, tau_to_phi)
from tsmodes.modes import ARModes, ar_acf, ar_psd
from tsmodes.sims import (AR1, AR1PlusSine, AR2Damped, SumAR1,
                          sim_ar_process)


FS = 500.0


@pytest.fixture(scope='module')
def ar1_signal():
    rng = np.random.default_rng(0)
    proc = AR1(tau=0.05)
    sig = proc.simulate(200.0, FS, rng)
    return proc, prepare(sig, FS)


def test_tau_phi_roundtrip():
    for tau in (0.005, 0.05, 0.5):
        assert phi_to_tau(tau_to_phi(tau, FS), FS) == pytest.approx(tau, rel=1e-12)


def test_phi_to_tau_rejects_non_decaying():
    assert np.isnan(phi_to_tau(-0.5, FS))
    assert np.isnan(phi_to_tau(1.2, FS))


def test_all_estimators_recover_ar1_tau(ar1_signal):
    proc, signal = ar1_signal
    fits = estimate_all(signal)
    assert set(fits) == set(ESTIMATORS)
    for name, fit in fits.items():
        assert fit.success, name
        assert fit.tau == pytest.approx(proc.tau, rel=0.15), name


def test_order_selection_picks_one_for_ar1(ar1_signal):
    _, signal = ar1_signal
    assert select_order_time(signal.sig, range(1, 11))[0] == 1
    mask = signal.fit_mask & signal.exclude_mask
    order, _, _ = select_order_psd(signal.freqs[mask], signal.powers[mask], FS,
                                   range(1, 11), powers_full=signal.powers)
    assert order == 1


def test_order_selection_picks_two_for_damped_oscillator():
    rng = np.random.default_rng(1)
    proc = AR2Damped(tau=0.05, f_osc=10.0)
    signal = prepare(proc.simulate(200.0, FS, rng), FS)
    assert select_order_time(signal.sig, range(1, 11))[0] == 2


def test_arp_recovers_damped_oscillator_tau_and_frequency():
    rng = np.random.default_rng(2)
    proc = AR2Damped(tau=0.05, f_osc=10.0)
    signal = prepare(proc.simulate(400.0, FS, rng), FS)
    for method in ('arp_direct', 'arp_psd'):
        fit = estimate(signal, method)
        mode = fit.modes.dominant_mode()
        assert mode.kind == 'oscillatory', method
        assert mode.tau == pytest.approx(proc.tau, rel=0.2), method
        assert mode.freq == pytest.approx(proc.f_osc, rel=0.05), method


def test_ar_burg_and_ols_agree_on_long_signals():
    rng = np.random.default_rng(3)
    phi = np.array([0.6, 0.2])
    sig = sim_ar_process(phi, 200000, rng)
    assert np.allclose(ar_burg(sig, 2)[0], ar_ols(sig, 2)[0], atol=1e-3)
    assert np.allclose(ar_ols(sig, 2)[0], phi, atol=2e-2)


def test_population_ar_recovers_exact_ar_coefficients():
    """Yule-Walker on the true ACF of an AR(p) returns that AR(p)."""
    for phi in (np.array([0.9]), np.array([1.4, -0.6]),
                np.array([0.5, 0.2, -0.1, 0.05])):
        rho = ar_acf(phi, 200)
        assert np.allclose(population_ar(rho, len(phi)), phi, atol=1e-8)


def test_population_ar_of_higher_order_pads_with_zeros():
    phi = np.array([1.4, -0.6])
    rho = ar_acf(phi, 200)
    est = population_ar(rho, 5)
    assert np.allclose(est[:2], phi, atol=1e-8)
    assert np.allclose(est[2:], 0.0, atol=1e-8)


def test_spectral_fit_recovers_known_ar_spectrum():
    """Fitting the AR(p) form to its own noiseless spectrum returns phi."""
    fs = 500.0
    phi = np.array([1.2, -0.5, 0.1])
    freqs = np.linspace(1.0, fs / 2, 400)
    powers = ar_psd(phi, freqs, fs, 2.0)
    phi_hat, offset, ok = fit_ar_psd(freqs, powers, fs, 3,
                                     guess=np.concatenate([phi * 0.5, [1.0]]))
    assert ok
    assert np.allclose(phi_hat, phi, atol=1e-4)
    assert offset == pytest.approx(2.0, rel=1e-3)


def test_prepare_shares_one_psd_and_acf():
    rng = np.random.default_rng(4)
    sig = AR1(0.05).simulate(20.0, FS, rng)
    signal = prepare(sig, FS)
    assert signal.corrs[0] == pytest.approx(1.0)
    assert signal.powers.shape == signal.freqs.shape
    assert signal.duration == pytest.approx(20.0)
    assert signal.fit_mask.sum() > 10


def test_estimate_all_survives_a_failing_estimator():
    rng = np.random.default_rng(5)
    signal = prepare(AR1(0.05).simulate(5.0, FS, rng), FS)
    fits = estimate_all(signal, methods=['ar1_psd'],
                        options={'ar1_psd': {'orders': 'nonsense'}})
    assert 'ar1_psd' in fits


def test_sum_of_ar1_poles_are_the_latent_poles_with_an_ma_zero():
    """The exact ARMA(2,1) form keeps both latent poles but adds a zero."""
    proc = SumAR1(tau0=0.01, tau1=0.2, weight=0.5)
    rep = proc.arma_representation(FS)
    assert np.allclose(np.sort(rep['ar_poles']),
                       np.sort([np.exp(-1 / (0.01 * FS)), np.exp(-1 / (0.2 * FS))]))
    assert abs(rep['ma_zero']) < 1.0
    assert abs(rep['ma_zero']) > 1e-3

    # A pure AR(2) population fit is therefore *not* at the latent poles.
    rho = proc.acf(np.arange(4001), FS)
    phi2 = population_ar(rho, 2)
    fitted = np.sort(np.real(ARModes.from_phi(phi2, FS).poles))
    assert not np.allclose(fitted, np.sort(rep['ar_poles']), atol=1e-3)


def test_modes_attached_to_every_ar_fit(ar1_signal):
    _, signal = ar1_signal
    for method in ('ar1_direct', 'ar1_psd', 'arp_direct', 'arp_psd'):
        fit = estimate(signal, method)
        assert fit.modes is not None
        assert len(fit.modes.modes) >= 1
        assert fit.modes.summary()[0][0] > 0


# ---------------------------------------------------------------------------
# ACF-domain AR(p) fitting
# ---------------------------------------------------------------------------

def test_acf_fit_recovers_ar1_coefficient():
    """The AR(1) ACF form recovers phi as well as the time-domain fit."""
    rng = np.random.default_rng(10)
    proc = AR1(tau=0.05)
    signal = prepare(proc.simulate(300.0, FS, rng), FS)
    from tsmodes.estimators import fit_ar_acf
    phi, height, offset, ok = fit_ar_acf(signal.lags, signal.corrs, FS, 1)
    assert ok
    assert phi_to_tau(float(phi[0]), FS) == pytest.approx(0.05, rel=0.15)
    assert height == pytest.approx(1.0, abs=0.15)


def test_acf_fit_recovers_a_damped_oscillator():
    """Where a bare exponential fails, the AR(2) ACF form does not."""
    rng = np.random.default_rng(11)
    proc = AR2Damped(tau=0.05, f_osc=10.0)
    signal = prepare(proc.simulate(300.0, FS, rng), FS)
    from tsmodes.estimators import fit_ar_acf
    phi, _, _, ok = fit_ar_acf(signal.lags, signal.corrs, FS, 2)
    assert ok
    mode = ARModes.from_phi(phi, FS).modes[0]
    assert mode.kind == 'oscillatory'
    assert mode.tau == pytest.approx(0.05, rel=0.2)
    assert mode.freq == pytest.approx(10.0, rel=0.05)


def test_acf_fit_returns_a_stationary_model():
    """Reflection-coefficient parameterization cannot leave the stable region."""
    rng = np.random.default_rng(12)
    from tsmodes.estimators import fit_ar_acf
    for proc in (AR1(0.05), AR2Damped(0.05, 10.0), SumAR1(0.01, 0.2, 0.5)):
        signal = prepare(proc.simulate(120.0, FS, rng), FS)
        for order in (1, 3, 6):
            phi, _, _, _ = fit_ar_acf(signal.lags, signal.corrs, FS, order)
            assert ARModes.from_phi(phi, FS).is_stationary, (proc.name, order)


def test_psd_fit_returns_a_stationary_model():
    rng = np.random.default_rng(13)
    for proc in (AR1(0.05), AR2Damped(0.05, 10.0), SumAR1(0.01, 0.2, 0.5)):
        signal = prepare(proc.simulate(120.0, FS, rng), FS)
        mask = signal.fit_mask & signal.exclude_mask
        for order in (1, 3, 8):
            phi, _, _ = fit_ar_psd(signal.freqs[mask], signal.powers[mask], FS,
                                   order, powers_full=signal.powers)
            assert ARModes.from_phi(phi, FS).is_stationary, (proc.name, order)


def test_acf_and_psd_estimators_agree_when_the_model_is_right():
    """AR(1) data: the two domains must not disagree; that is the whole point."""
    rng = np.random.default_rng(14)
    signal = prepare(AR1(tau=0.05).simulate(300.0, FS, rng), FS)
    acf = estimate(signal, 'ar1_acf').tau
    psd = estimate(signal, 'ar1_psd').tau
    assert acf == pytest.approx(psd, rel=0.15)
    assert acf == pytest.approx(0.05, rel=0.2)


def test_acf_order_selection_finds_two_for_a_damped_oscillator():
    rng = np.random.default_rng(15)
    signal = prepare(AR2Damped(0.05, 10.0).simulate(300.0, FS, rng), FS)
    fit = estimate(signal, 'arp_acf')
    assert fit.order <= 4
    mode = fit.modes.dominant_mode()
    assert mode.kind == 'oscillatory'
    assert mode.freq == pytest.approx(10.0, rel=0.1)


def test_reflection_parameterization_makes_stabilization_unnecessary():
    """No spectral fit should need the minimum-phase repair any more."""
    rng = np.random.default_rng(16)
    for proc in (AR1(0.05), AR2Damped(0.05, 10.0), SumAR1(0.01, 0.2, 0.5)):
        signal = prepare(proc.simulate(120.0, FS, rng), FS)
        for method in ('ar1_psd', 'arp_psd', 'arp_acf'):
            fit = estimate(signal, method)
            assert not fit.params.get('stabilized', False), (proc.name, method)


# ---------------------------------------------------------------------------
# The two domains must describe the same data
# ---------------------------------------------------------------------------

def test_welch_acf_recovers_the_true_autocorrelation():
    """The prepared ACF matches the analytic one, taper bias removed."""
    rng = np.random.default_rng(4)
    tau, n = 0.05, int(FS * 600)
    phi = np.exp(-1.0 / (tau * FS))
    signal = prepare(sim_ar_process(np.array([phi]), n, rng), FS,
                     welch_win_s=4.0)
    lags = signal.lags / FS
    assert np.max(np.abs(signal.corrs - np.exp(-lags / tau))) < 0.05


def test_out_of_band_drift_does_not_reach_the_lag_domain():
    """Content the spectral fit excludes must not move the ACF estimate.

    Both estimators are supposed to describe the analysis band.  If the ACF
    were taken from the raw signal, slow drift below the band would dominate
    it and the two domains would silently answer different questions.
    """
    rng = np.random.default_rng(5)
    tau, n = 0.05, int(FS * 400)
    phi = np.exp(-1.0 / (tau * FS))
    x = sim_ar_process(np.array([phi]), n, rng)
    drift = 6.0 * np.sin(2 * np.pi * 0.05 * np.arange(n) / FS)

    clean = prepare(x, FS, welch_win_s=4.0, f_range=(1.0, 200.0))
    drifted = prepare(x + drift, FS, welch_win_s=4.0, f_range=(1.0, 200.0))

    tau_clean = estimate(clean, 'ar1_acf').tau
    tau_drifted = estimate(drifted, 'ar1_acf').tau
    assert abs(tau_drifted - tau_clean) / tau_clean < 0.15
    # ... and the raw autocorrelation is the thing that would have failed.
    assert drifted.raw_corrs[-1] > 0.5 > clean.raw_corrs[-1]


def test_line_noise_is_interpolated_not_notched():
    """Excluded bins are filled in, so notching does not ring in the lags."""
    rng = np.random.default_rng(6)
    n = int(FS * 200)
    phi = np.array([np.exp(-1.0 / (0.05 * FS))])
    x = sim_ar_process(phi, n, rng)
    line = 3.0 * np.sin(2 * np.pi * 60.0 * np.arange(n) / FS)
    notched = prepare(x + line, FS, welch_win_s=4.0,
                      exclude_bands=[(58.0, 62.0)])
    plain = prepare(x, FS, welch_win_s=4.0)
    assert np.max(np.abs(notched.corrs - plain.corrs)) < 0.08


# ---------------------------------------------------------------------------
# Robustness is a choice, not a default
# ---------------------------------------------------------------------------

def test_robust_is_off_unless_asked_for():
    from tsmodes.estimators import loss_settings
    assert loss_settings(False, 'psd') == ('linear', None)
    assert loss_settings(True, 'psd')[0] == 'soft_l1'


def test_robustness_is_never_applied_in_the_lag_domain():
    """A rhythm displaces every lag, so it cannot be treated as an outlier."""
    from tsmodes.estimators import loss_settings
    assert loss_settings(False, 'acf') == ('linear', None)
    assert loss_settings(True, 'acf') == ('linear', None)


def test_the_lag_domain_fit_does_not_lose_the_aperiodic_component():
    """An AR(1) with an added sinusoid must still yield its decay.

    The component is real and the fit must find it.  It need not appear as a
    real pole: the optimizer typically returns a conjugate pair at a few Hz
    that rings for a fifth of a cycle, which is a relaxation in everything but
    the algebraic classification.  What must not happen is that every pole is
    spent on the rhythm.
    """
    rng = np.random.default_rng(0)
    signal = prepare(AR1PlusSine(0.05, 10.0, 1.0).simulate(200.0, FS, rng), FS,
                     welch_win_s=4.0)
    fit = estimate(signal, 'arp_acf')
    slow = [m for m in fit.modes.modes if m.n_cycles < 1.0]
    assert slow, 'every mode is a rhythm; the aperiodic component was lost'
    tau = max(m.tau for m in slow)
    assert 0.02 < tau < 0.3

    rhythm = [m for m in fit.modes.modes if m.n_cycles >= 1.0]
    assert rhythm and abs(max(rhythm, key=lambda m: m.n_cycles).freq - 10.0) < 0.5


def test_higher_order_lag_fits_do_not_get_worse():
    """Nested models cannot fit worse; if they do, the search is failing."""
    rng = np.random.default_rng(0)
    signal = prepare(AR1PlusSine(0.05, 10.0, 1.0).simulate(200.0, FS, rng), FS,
                     welch_win_s=4.0)
    best = -np.inf
    for order in (2, 5, 8):
        fit = estimate(signal, 'arp_acf', order=order)
        r2 = acf_fit_gof(signal, fit.phi, fit.params['height'],
                         fit.params['acf_offset'])['r2_acf']
        assert r2 > best - 0.02, f'order {order} fits worse than a lower order'
        best = max(best, r2)


def test_robust_fitting_protects_the_aperiodic_estimate_from_a_rhythm():
    rng = np.random.default_rng(7)
    signal = prepare(AR1PlusSine(0.05, 10.0, 2.0).simulate(300.0, FS, rng), FS,
                     welch_win_s=4.0)
    plain = estimate(signal, 'ar1_psd', robust=False).tau
    robust = estimate(signal, 'ar1_psd', robust=True).tau
    assert abs(robust - 0.05) < abs(plain - 0.05)


def test_estimate_all_can_switch_robustness_on_in_one_place():
    rng = np.random.default_rng(8)
    signal = prepare(AR1PlusSine(0.05, 10.0, 2.0).simulate(200.0, FS, rng), FS,
                     welch_win_s=4.0)
    fits = estimate_all(signal, methods=['ar1_psd'], robust=True)
    assert abs(fits['ar1_psd'].tau - 0.05) / 0.05 < 0.25
