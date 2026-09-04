"""Tests for the AR(p) pole/mode decomposition."""

import itertools

import numpy as np
import pytest

from tsmodes.modes import (ARModes, ar_acf, ar_impulse_response, ar_psd,
                           ar_to_poles, group_poles, mode_importance,
                           pair_to_ar2, pole_log_factor, pole_residues,
                           poles_to_ar, psd_from_poles, subset_error,
                           variance_decomposition, LN10)


def random_stable_ar(order, rng, max_radius=0.97):
    """Draw AR coefficients with all poles strictly inside the unit circle."""
    poles = []
    while len(poles) < order:
        if order - len(poles) >= 2 and rng.random() < 0.6:
            r = rng.uniform(0.2, max_radius)
            theta = rng.uniform(0.05, np.pi - 0.05)
            poles += [r * np.exp(1j * theta), r * np.exp(-1j * theta)]
        else:
            poles.append(rng.uniform(-max_radius, max_radius) + 0j)
    return poles_to_ar(np.array(poles))


ORDERS = [1, 2, 3, 4, 5, 8, 12]


@pytest.mark.parametrize('order', ORDERS)
def test_pole_coefficient_roundtrip(order):
    rng = np.random.default_rng(order)
    phi = random_stable_ar(order, rng)
    assert np.allclose(poles_to_ar(ar_to_poles(phi)), phi, atol=1e-8)


@pytest.mark.parametrize('order', ORDERS)
def test_psd_exactly_reconstructed_from_pole_factorization(order):
    """The AR(p) PSD equals the product of its first-order pole factors."""
    rng = np.random.default_rng(100 + order)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 977)
    for _ in range(5):
        phi = random_stable_ar(order, rng)
        offset = float(rng.uniform(0.1, 10))
        direct = ar_psd(phi, freqs, fs, offset)
        factored = psd_from_poles(ar_to_poles(phi), freqs, fs, offset)
        assert np.allclose(direct, factored, rtol=1e-10, atol=0)


@pytest.mark.parametrize('order', ORDERS)
def test_psd_reconstructed_from_grouped_modes(order):
    """Grouping conjugate pairs into damped oscillators is also exact."""
    rng = np.random.default_rng(200 + order)
    fs = 250.0
    freqs = np.linspace(0, fs / 2, 513)
    phi = random_stable_ar(order, rng)
    model = ARModes.from_phi(phi, fs, offset=3.0, freqs=freqs)
    full = model.psd(freqs)
    from_modes = model.psd_from_modes(freqs, refit_scale=False)
    assert np.allclose(full, from_modes, rtol=1e-10, atol=0)

    # And the product of the per-mode factor spectra reproduces the same shape.
    prod = np.ones_like(freqs)
    for mode in model.modes:
        prod = prod * mode.psd(freqs)
    assert np.allclose(full, 3.0 * prod, rtol=1e-10, atol=0)


def test_conjugate_pair_is_a_real_ar2():
    fs = 500.0
    r, theta = 0.96, 2 * np.pi * 12 / fs
    pair = (r * np.exp(1j * theta), r * np.exp(-1j * theta))
    phi2 = pair_to_ar2(pair)
    assert np.allclose(phi2, [2 * r * np.cos(theta), -r ** 2])
    assert np.allclose(poles_to_ar(np.array(pair)), phi2)


def test_mode_tau_and_frequency_match_construction():
    fs = 500.0
    tau, f_osc = 0.08, 12.0
    r = np.exp(-1.0 / (tau * fs))
    theta = 2 * np.pi * f_osc / fs
    phi = poles_to_ar(np.array([r * np.exp(1j * theta), r * np.exp(-1j * theta)]))
    model = ARModes.from_phi(phi, fs)
    assert len(model.modes) == 1
    mode = model.modes[0]
    assert mode.kind == 'oscillatory'
    assert mode.tau == pytest.approx(tau, rel=1e-10)
    assert mode.freq == pytest.approx(f_osc, rel=1e-10)


def test_real_pole_tau_matches_ar1():
    fs = 1000.0
    tau = 0.02
    phi = np.array([np.exp(-1.0 / (tau * fs))])
    model = ARModes.from_phi(phi, fs)
    assert model.modes[0].kind == 'relaxation'
    assert model.modes[0].tau == pytest.approx(tau, rel=1e-12)
    assert model.modes[0].freq == 0.0


def test_negative_real_pole_is_a_nyquist_mode():
    fs = 400.0
    model = ARModes.from_phi(np.array([-0.7]), fs)
    assert model.modes[0].kind == 'nyquist'
    assert model.modes[0].freq == pytest.approx(fs / 2)


@pytest.mark.parametrize('order', [2, 3, 5, 8])
def test_impulse_response_is_the_sum_of_pole_modes(order):
    """psi_n = sum_k c_k lam_k^n: poles are additive in the time domain."""
    rng = np.random.default_rng(300 + order)
    phi = random_stable_ar(order, rng)
    poles = ar_to_poles(phi)
    if len(np.unique(np.round(poles, 10))) < order:
        pytest.skip('repeated roots')
    c = pole_residues(poles)
    n = 200
    psi_direct = ar_impulse_response(phi, n)
    psi_modes = np.real(np.sum(c[:, None] * poles[:, None] ** np.arange(n)[None, :], axis=0))
    assert np.allclose(psi_direct, psi_modes, atol=1e-8)


@pytest.mark.parametrize('order', [1, 2, 3, 5])
def test_theoretical_acf_matches_impulse_response_convolution(order):
    rng = np.random.default_rng(400 + order)
    phi = random_stable_ar(order, rng, max_radius=0.9)
    nlags = 60
    psi = ar_impulse_response(phi, 6000)
    acov = np.array([np.sum(psi[:len(psi) - k] * psi[k:]) for k in range(nlags + 1)])
    assert np.allclose(ar_acf(phi, nlags), acov / acov[0], atol=1e-6)


@pytest.mark.parametrize('order', [1, 2, 3, 5])
def test_theoretical_acf_matches_psd_integral(order):
    """Wiener-Khinchin consistency between the mode PSD and the mode ACF."""
    rng = np.random.default_rng(500 + order)
    fs = 200.0
    phi = random_stable_ar(order, rng, max_radius=0.9)
    n = 200000
    freqs = np.fft.rfftfreq(n, 1 / fs)
    psd = ar_psd(phi, freqs, fs, 1.0)
    acov = np.fft.irfft(psd)[: 41]
    assert np.allclose(ar_acf(phi, 40), acov / acov[0], atol=1e-6)


def test_importance_matches_brute_force_deletion():
    """Closed-form importance == delete the mode, refit scale, measure RMSE."""
    rng = np.random.default_rng(7)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 401)
    phi = random_stable_ar(7, rng)
    model = ARModes.from_phi(phi, fs, offset=2.0, freqs=freqs)
    full = np.log10(model.psd(freqs))
    for i, mode in enumerate(model.modes):
        keep = [j for j in range(len(model.modes)) if j != i]
        reduced = np.log10(model.psd_from_modes(freqs, indices=keep, refit_scale=True))
        brute = np.sqrt(np.mean((full - reduced) ** 2))
        assert brute == pytest.approx(mode.importance, rel=1e-8)


def test_importance_is_matches_brute_force_itakura_saito():
    rng = np.random.default_rng(8)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 401)
    phi = random_stable_ar(6, rng)
    model = ARModes.from_phi(phi, fs, offset=1.5, freqs=freqs)
    full = model.psd(freqs)
    for i, mode in enumerate(model.modes):
        keep = [j for j in range(len(model.modes)) if j != i]
        poles = [lam for j in keep for lam in model.modes[j].poles]
        base = psd_from_poles(poles, freqs, fs, 1.0) if poles else np.ones_like(freqs)
        scale = np.mean(full / base)          # IS-optimal scale
        red = scale * base
        ratio = full / red
        brute = float(np.mean(ratio - np.log(ratio) - 1.0))
        assert brute == pytest.approx(mode.importance_is, rel=1e-6, abs=1e-12)


def test_subset_error_matches_brute_force_and_exhaustive_search():
    rng = np.random.default_rng(9)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 257)
    model = ARModes.from_phi(random_stable_ar(6, rng), fs, offset=1.0, freqs=freqs)
    n = len(model.modes)
    full = np.log10(model.psd(freqs))
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            reduced = np.log10(model.psd_from_modes(freqs, indices=combo,
                                                    refit_scale=True))
            brute = np.sqrt(np.mean((full - reduced) ** 2))
            assert brute == pytest.approx(
                subset_error(model.modes, combo, freqs, fs), rel=1e-8, abs=1e-12)

    curve = model.dominant_subset()['curve']
    expect = [min(subset_error(model.modes, c, freqs, fs)
                  for c in itertools.combinations(range(n), size))
              for size in range(n + 1)]
    assert np.allclose(curve, expect)


def test_dominant_subset_returns_smallest_set_within_tolerance():
    fs = 500.0
    # One dominant slow relaxation plus a negligible fast pole.
    phi = poles_to_ar(np.array([0.99 + 0j, 0.02 + 0j]))
    model = ARModes.from_phi(phi, fs)
    dom = model.dominant_subset(tol=0.05)
    assert dom['n_modes'] == 1
    assert model.modes[dom['indices'][0]].tau == pytest.approx(
        -1 / (fs * np.log(0.99)), rel=1e-10)


def test_full_model_has_zero_reconstruction_error():
    rng = np.random.default_rng(11)
    fs = 500.0
    model = ARModes.from_phi(random_stable_ar(9, rng), fs)
    dom = model.dominant_subset(tol=0.0)
    assert dom['error'] == pytest.approx(0.0, abs=1e-10)
    assert dom['r2_curve'][-1] == pytest.approx(1.0)
    assert dom['rel_curve'][0] == pytest.approx(1.0)


def test_relative_subset_error_is_a_log_spectral_r_squared():
    rng = np.random.default_rng(21)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 401)
    model = ARModes.from_phi(random_stable_ar(6, rng), fs, offset=2.0, freqs=freqs)
    dom = model.dominant_subset(tol=0.1)
    full = np.log10(model.psd(freqs))
    keep = dom['indices']
    reduced = np.log10(model.psd_from_modes(freqs, indices=keep, refit_scale=True))
    r2 = 1 - np.sum((full - reduced) ** 2) / np.sum((full - full.mean()) ** 2)
    assert r2 == pytest.approx(dom['r2_curve'][dom['n_modes']], rel=1e-8)
    assert r2 >= 0.99


def test_variance_decomposition_totals_match_theory():
    rng = np.random.default_rng(12)
    for order in (2, 3, 5):
        phi = random_stable_ar(order, rng, max_radius=0.9)
        dec = variance_decomposition(ar_to_poles(phi))
        expected = ar_acf(phi, 1, normalize=False)[0]
        assert dec['total_variance'] == pytest.approx(expected, rel=1e-6)


def test_cross_mode_variance_is_non_negligible():
    """Per-mode variance shares are not well defined: cross terms are large."""
    fs = 500.0
    phi = poles_to_ar(np.array([0.99 + 0j, 0.9 + 0j]))
    dec = variance_decomposition(ar_to_poles(phi))
    assert abs(dec['cross_fraction']) > 0.05


def test_group_poles_pairs_conjugates():
    poles = np.array([0.5 + 0j, 0.8 + 0.3j, 0.8 - 0.3j, -0.4 + 0j])
    groups = group_poles(poles)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1, 2]


def test_modes_are_ranked_by_importance():
    rng = np.random.default_rng(13)
    model = ARModes.from_phi(random_stable_ar(8, rng), 500.0)
    imp = [m.importance for m in model.modes]
    assert imp == sorted(imp, reverse=True)


def test_summary_is_tau_frequency_importance():
    model = ARModes.from_phi(np.array([0.9]), 500.0)
    (tau, freq, importance), = model.summary()
    assert tau == pytest.approx(-1 / (500.0 * np.log(0.9)))
    assert freq == 0.0
    assert importance > 0


def test_mode_log_factors_sum_to_full_log_spectrum():
    rng = np.random.default_rng(14)
    fs = 500.0
    freqs = np.linspace(0, fs / 2, 301)
    model = ARModes.from_phi(random_stable_ar(6, rng), fs, offset=4.0, freqs=freqs)
    g = model.mode_log_factors(freqs)
    assert np.allclose(np.log(model.psd(freqs)), np.log(4.0) - g.sum(axis=0),
                       atol=1e-10)


def test_non_stationary_model_flagged():
    model = ARModes.from_phi(np.array([1.2]), 500.0)
    assert not model.is_stationary
    assert not np.isfinite(model.modes[0].tau)


@pytest.mark.parametrize('order', [1, 2, 3, 5, 8])
def test_acf_mode_components_sum_to_the_acf(order):
    """Modes are additive in the ACF even though their spectral factors multiply."""
    from tsmodes.modes import acf_mode_components
    rng = np.random.default_rng(600 + order)
    phi = random_stable_ar(order, rng, max_radius=0.95)
    poles = ar_to_poles(phi)
    if len(np.unique(np.round(poles, 8))) < order:
        pytest.skip('repeated roots')
    nlags = 120
    comps = acf_mode_components(phi, nlags)
    assert np.allclose(comps.sum(axis=0), ar_acf(phi, nlags), atol=1e-8)
    assert np.isreal(comps).all()


def test_acf_components_of_an_ar1_is_a_single_exponential():
    from tsmodes.modes import acf_mode_components
    comps = acf_mode_components(np.array([0.9]), 30)
    assert comps.shape[0] == 1
    assert np.allclose(comps[0], 0.9 ** np.arange(31), atol=1e-10)


def test_stabilize_preserves_the_spectrum_and_makes_poles_stationary():
    from tsmodes.modes import stabilize
    fs = 500.0
    freqs = np.linspace(0.5, fs / 2, 501)
    # A model with one pole outside the unit circle.
    phi = poles_to_ar(np.array([1.4 + 0j, 0.8 * np.exp(0.4j), 0.8 * np.exp(-0.4j)]))
    assert np.any(np.abs(ar_to_poles(phi)) >= 1.0)
    phi_s, offset_s, changed = stabilize(phi, offset=2.0)
    assert changed
    assert np.all(np.abs(ar_to_poles(phi_s)) < 1.0)
    assert np.allclose(ar_psd(phi, freqs, fs, 2.0),
                       ar_psd(phi_s, freqs, fs, offset_s), rtol=1e-9)


def test_stabilize_is_a_no_op_for_stationary_models():
    from tsmodes.modes import stabilize
    rng = np.random.default_rng(31)
    phi = random_stable_ar(5, rng)
    phi_s, offset_s, changed = stabilize(phi, offset=3.0)
    assert not changed
    assert np.allclose(phi_s, phi)
    assert offset_s == 3.0


def test_stabilized_model_has_a_finite_acf():
    from tsmodes.modes import stabilize
    phi = poles_to_ar(np.array([1.05 + 0j, 0.5 + 0j]))
    phi_s, _, _ = stabilize(phi)
    acf = ar_acf(phi_s, 500)
    assert np.all(np.isfinite(acf))
    assert abs(acf[-1]) <= 1.0


@pytest.mark.parametrize('order', [1, 2, 3, 5, 8])
def test_fft_acf_matches_the_exact_recursion(order):
    """The FFT route to the AR(p) ACF is used for fitting; it must be exact."""
    from tsmodes.modes import ar_acf_fft
    rng = np.random.default_rng(700 + order)
    phi = random_stable_ar(order, rng, max_radius=0.95)
    nlags = 200
    assert np.allclose(ar_acf_fft(phi, nlags), ar_acf(phi, nlags), atol=1e-9)


def test_fft_acf_is_accurate_for_a_slow_pole():
    from tsmodes.modes import ar_acf_fft
    phi = np.array([0.995])
    assert np.allclose(ar_acf_fft(phi, 500), ar_acf(phi, 500), atol=1e-6)


@pytest.mark.parametrize('order', [1, 2, 3, 5, 8, 12])
def test_reflection_coefficient_roundtrip(order):
    from tsmodes.modes import ar_to_reflection, reflection_to_ar
    rng = np.random.default_rng(800 + order)
    phi = random_stable_ar(order, rng, max_radius=0.95)
    k = ar_to_reflection(phi)
    assert np.all(np.abs(k) < 1.0)
    assert np.allclose(reflection_to_ar(k), phi, atol=1e-8)


def test_reflection_coefficients_inside_unit_interval_imply_stationarity():
    from tsmodes.modes import reflection_to_ar
    rng = np.random.default_rng(42)
    for _ in range(50):
        k = rng.uniform(-0.98, 0.98, size=rng.integers(1, 10))
        phi = reflection_to_ar(k)
        assert np.all(np.abs(ar_to_poles(phi)) < 1.0)


@pytest.mark.parametrize('order', [1, 2, 3, 5, 8])
def test_impulse_mode_components_sum_to_the_impulse_response(order):
    from tsmodes.modes import impulse_mode_components
    rng = np.random.default_rng(900 + order)
    phi = random_stable_ar(order, rng, max_radius=0.95)
    poles = ar_to_poles(phi)
    if len(np.unique(np.round(poles, 8))) < order:
        pytest.skip('repeated roots')
    n = 300
    comps = impulse_mode_components(phi, n)
    assert np.allclose(comps.sum(axis=0), ar_impulse_response(phi, n), atol=1e-8)


def test_impulse_components_of_an_ar1_is_a_single_geometric_decay():
    from tsmodes.modes import impulse_mode_components
    comps = impulse_mode_components(np.array([0.9]), 40)
    assert comps.shape[0] == 1
    assert np.allclose(comps[0], 0.9 ** np.arange(40), atol=1e-10)


# ---------------------------------------------------------------------------
# Is an "oscillatory" mode actually a rhythm?
# ---------------------------------------------------------------------------

def _model(spec, fs=500.0):
    poles = []
    for tau, freq in spec:
        r = np.exp(-1.0 / (tau * fs))
        if freq == 0:
            poles.append(r + 0j)
        else:
            th = 2 * np.pi * freq / fs
            poles += [r * np.exp(1j * th), r * np.exp(-1j * th)]
    return ARModes.from_phi(poles_to_ar(np.array(poles)), fs)


def test_n_cycles_counts_periods_before_the_envelope_decays():
    model = _model([(0.3, 10.0), (0.05, 10.0), (0.1, 0.0)])
    osc = {round(m.freq): m for m in model.modes if m.kind == 'oscillatory'}
    assert osc[10].n_cycles == pytest.approx(10.0 * osc[10].tau)
    relax = [m for m in model.modes if m.kind == 'relaxation'][0]
    assert relax.n_cycles == 0.0


def test_a_conjugate_pair_need_not_be_a_rhythm():
    """The constructed AR(8) of the modes figure has no oscillation in it.

    Every conjugate pair rings for under one cycle, and because modes
    multiply in the spectrum rather than adding, the product of five such
    factors is monotone: there is no peak anywhere for a marker to point at.
    """
    model = _model([(0.100, 0.0), (0.050, 10.0), (0.010, 40.0),
                    (0.002, 0.0), (0.003, 180.0)])
    osc = [m for m in model.modes if m.kind == 'oscillatory']
    assert len(osc) == 3
    assert all(m.n_cycles < 1.0 for m in osc)

    freqs = np.linspace(0.1, 250.0, 4000)
    assert np.all(np.diff(model.psd(freqs)) < 0), 'spectrum is not monotone'


def test_the_same_pair_alone_does_make_a_peak():
    """Visibility is a property of the whole model, not of one mode."""
    fs, tau, f0 = 500.0, 0.05, 10.0
    freqs = np.linspace(0.1, 250.0, 4000)
    alone = _model([(tau, f0)], fs)
    i = int(np.argmax(alone.psd(freqs)))
    assert 0 < i < len(freqs) - 1
    assert abs(freqs[i] - f0) < 1.5
