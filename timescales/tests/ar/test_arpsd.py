"""ARPSD: fitting an AR(p) model to a power spectrum.

The theoretical spectrum of a known AR(p) is the strongest available test:
the answer is known exactly, so the fit can be checked against it rather than
against another estimate.  Fits to simulated data follow, where the sampling
noise is real but the truth is still known.
"""

import numpy as np
import pytest

from timescales.ar import ARPSD
from timescales.ar.core import K_BOUND
from timescales.conversions import phi_to_tau, tau_to_phi
from timescales.decomposition.modes import ar_psd
from timescales.sim import AR1, AR2Damped, ARpModes, sim_ar_process
from timescales.utils import compute_psd

FS = 500.0
FREQS = np.linspace(1.0, 250.0, 500)


def _theory(phi, offset=1.0, freqs=FREQS):
    return freqs, ar_psd(np.atleast_1d(phi), freqs, FS, offset)


# ---------------------------------------------------------------------------
# Exact recovery from a theoretical spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tau', [0.005, 0.02, 0.1, 0.5])
def test_ar1_recovers_tau_from_its_own_spectrum(tau):
    freqs, powers = _theory(tau_to_phi(tau, FS))
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    assert ar.success_
    assert ar.tau == pytest.approx(tau, rel=1e-3)
    assert ar.rsq > 0.999


@pytest.mark.parametrize('order', [2, 3, 5, 7])
def test_arp_recovers_coefficients_from_its_own_spectrum(order):
    rng = np.random.default_rng(order)
    # A stationary model built from poles, so the truth is known exactly.
    poles = [0.9] + [0.6 * np.exp(1j * a) for a in rng.uniform(0.3, 2.5, (order - 1) // 2)]
    poles = poles + [np.conj(p) for p in poles[1:]]
    while len(poles) < order:
        poles.append(0.3)
    phi = np.real(-np.poly(np.array(poles))[1:])
    freqs, powers = _theory(phi)
    ar = ARPSD(order=order, fs=FS).fit(freqs, powers)
    assert ar.rsq > 0.999
    np.testing.assert_allclose(np.sort(np.abs(np.roots(np.r_[1.0, -ar.phi]))),
                               np.sort(np.abs(np.array(poles))), atol=2e-2)


def test_ar2_oscillator_recovers_envelope_and_frequency():
    proc = AR2Damped(tau=0.1, f_osc=20.0)
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=2, fs=FS).fit(freqs, powers)
    modes = ar.decompose()
    assert len(modes.modes) == 1
    mode = modes.modes[0]
    assert mode.kind == 'oscillatory'
    assert mode.tau == pytest.approx(0.1, rel=0.02)
    assert mode.freq == pytest.approx(20.0, rel=0.02)


def test_arp_finds_every_constructed_mode():
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.02, 0.0)))
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=4, fs=FS).fit(freqs, powers)
    taus = np.sort(ar.decompose().taus)
    np.testing.assert_allclose(taus, np.sort([0.2, 0.15, 0.02]), rtol=0.05)


# ---------------------------------------------------------------------------
# Recovery from simulated data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tau', [0.01, 0.05, 0.2])
def test_ar1_recovers_tau_from_simulation(tau):
    rng = np.random.default_rng(0)
    sig = AR1(tau=tau).simulate(200.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(0.5, 250))
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    assert ar.tau == pytest.approx(tau, rel=0.15)
    assert ar.is_stationary


def test_order_selection_prefers_one_for_a_noisy_ar1_spectrum():
    """BIC should buy no extra order for an AR(1) with ordinary noise on it.

    The criterion is undefined on an exactly-fit curve, where the residual is
    already zero at order one and every score is numerical noise, so the test
    adds the scatter a real estimate would have.
    """
    rng = np.random.default_rng(4)
    freqs, powers = _theory(tau_to_phi(0.05, FS))
    powers = powers * 10 ** rng.normal(0.0, 0.05, len(powers))
    ar = ARPSD(order=range(1, 9), fs=FS).fit(freqs, powers)
    assert ar.order_ == 1
    assert ar.tau == pytest.approx(0.05, rel=0.05)


def test_order_selection_is_undefined_on_an_exact_fit():
    """Documented edge case: with a zero residual the scores are meaningless.

    The fit is still right; only the choice among orders is arbitrary.
    """
    freqs, powers = _theory(tau_to_phi(0.05, FS))
    ar = ARPSD(order=range(1, 9), fs=FS).fit(freqs, powers)
    assert ar.rsq > 0.999
    assert ar.tau == pytest.approx(0.05, rel=0.01)


def test_order_selection_on_simulated_ar1_stays_low_and_keeps_tau():
    """With sampling noise BIC may buy an order or two; tau must survive it.

    Welch bins are not independent, which is the assumption BIC makes, so a
    dense linear grid lets the criterion over-count evidence.  Thinning to
    log-spaced bins is the usual remedy and is left to the caller.
    """
    rng = np.random.default_rng(1)
    sig = AR1(tau=0.05).simulate(200.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(0.5, 250))
    keep = np.unique(np.searchsorted(
        freqs, np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 120)))
    keep = keep[keep < len(freqs)]
    ar = ARPSD(order=range(1, 9), fs=FS).fit(freqs[keep], powers[keep])
    assert ar.order_ <= 3
    assert ar.tau == pytest.approx(0.05, rel=0.2)


def test_order_selection_buys_order_for_an_oscillator():
    rng = np.random.default_rng(2)
    sig = AR2Damped(tau=0.1, f_osc=20.0).simulate(200.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(0.5, 250))
    ar = ARPSD(order=range(1, 9), fs=FS).fit(freqs, powers)
    assert ar.order_ >= 2
    osc = [m for m in ar.decompose().modes if m.kind == 'oscillatory']
    assert osc and osc[0].freq == pytest.approx(20.0, rel=0.1)


def test_higher_order_never_fits_worse_than_ar1():
    rng = np.random.default_rng(3)
    sig = ARpModes(modes=((0.2, 0.0), (0.15, 10.0))).simulate(200.0, FS, rng)
    freqs, powers = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(0.5, 250))
    low = ARPSD(order=1, fs=FS).fit(freqs, powers)
    high = ARPSD(order=6, fs=FS).fit(freqs, powers)
    assert high.rsq > low.rsq


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_defaults_to_ar1():
    assert ARPSD(fs=FS).order == 1


def test_fs_is_required():
    with pytest.raises(ValueError):
        ARPSD(order=1)


def test_robust_loss_requires_a_margin():
    with pytest.raises(ValueError):
        ARPSD(order=1, fs=FS, loss='soft_l1')
    ARPSD(order=1, fs=FS, loss='soft_l1', f_scale=0.1)


def test_reports_before_fit_are_refused():
    ar = ARPSD(order=1, fs=FS)
    with pytest.raises(RuntimeError):
        _ = ar.tau
    with pytest.raises(RuntimeError):
        ar.predict()
    assert 'unfitted' in repr(ar)


def test_rejects_mismatched_and_nonpositive_input():
    freqs, powers = _theory(0.9)
    with pytest.raises(ValueError):
        ARPSD(order=1, fs=FS).fit(freqs[:-1], powers)
    with pytest.raises(ValueError):
        ARPSD(order=1, fs=FS).fit(freqs, powers * -1)


def test_settings_are_not_hidden_module_state():
    """Every numerical setting is reachable and changeable per instance."""
    ar = ARPSD(order=1, fs=FS, k_bound=0.99, maxfev=10, criterion='aic')
    assert (ar.k_bound, ar.maxfev, ar.criterion) == (0.99, 10, 'aic')


def test_k_bound_limits_the_slowest_timescale():
    """The bound on reflection coefficients is what bounds tau, as documented."""
    slowest = -1.0 / (FS * np.log(K_BOUND))
    freqs, powers = _theory(tau_to_phi(10 * slowest, FS))
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    assert ar.tau <= slowest * 1.01


def test_predict_matches_the_analytic_spectrum():
    freqs, powers = _theory(0.95)
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    np.testing.assert_allclose(ar.predict(),
                               ar_psd(ar.phi, freqs, FS, ar.offset), rtol=1e-12)


def test_predict_accepts_new_frequencies():
    freqs, powers = _theory(0.95)
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    other = np.linspace(2.0, 100.0, 37)
    assert ar.predict(other).shape == other.shape


def test_fit_is_stationary_and_decomposable():
    freqs, powers = _theory([1.6, -0.8])
    ar = ARPSD(order=2, fs=FS).fit(freqs, powers)
    assert ar.is_stationary
    modes = ar.decompose()
    assert np.all(np.isfinite(modes.taus))
    assert ar.taus.shape == ar.freqs_hz.shape


def test_simulate_round_trips_through_the_fit():
    freqs, powers = _theory(tau_to_phi(0.05, FS))
    ar = ARPSD(order=1, fs=FS).fit(freqs, powers)
    sig = ar.simulate(100.0, rng=np.random.default_rng(0))
    assert len(sig) == int(100.0 * FS)
    f2, p2 = compute_psd(sig, FS, nperseg=int(4 * FS), f_range=(0.5, 250))
    assert ARPSD(order=1, fs=FS).fit(f2, p2).tau == pytest.approx(0.05, rel=0.2)


def test_plot_draws_without_error():
    import matplotlib
    matplotlib.use('Agg')
    freqs, powers = _theory([1.6, -0.8])
    ar = ARPSD(order=2, fs=FS).fit(freqs, powers)
    assert ar.plot(plot_modes=True) is not None


def test_orders_are_recorded_for_a_selected_fit():
    freqs, powers = _theory(0.95)
    ar = ARPSD(order=range(1, 5), fs=FS).fit(freqs, powers)
    assert ar.order_ in range(1, 5)
    assert len(ar.order_scores_) == len(ar.orders_)


def test_a_sequence_of_one_order_behaves_like_that_order():
    freqs, powers = _theory(0.95)
    assert ARPSD(order=[3], fs=FS).fit(freqs, powers).order_ == 3


def test_empty_order_grid_is_refused():
    freqs, powers = _theory(0.95)
    with pytest.raises(ValueError):
        ARPSD(order=[], fs=FS).fit(freqs, powers)


def test_relaxation_constraint_costs_one_order():
    """An AR(2) forced to hold a real pole is two real poles, so it cannot
    represent an oscillation; the constrained fit needs order three."""
    proc = AR2Damped(tau=0.1, f_osc=20.0)
    freqs, powers = _theory(proc.phi(FS))
    two = ARPSD(order=2, fs=FS, require_relaxation=True).fit(freqs, powers)
    assert all(m.kind != 'oscillatory' for m in two.decompose().modes)
    three = ARPSD(order=3, fs=FS, require_relaxation=True).fit(freqs, powers)
    osc = [m for m in three.decompose().modes if m.kind == 'oscillatory']
    assert osc and osc[0].freq == pytest.approx(20.0, rel=0.05)
    assert np.isfinite(three.tau)


def test_unconstrained_is_the_default():
    assert ARPSD(fs=FS).require_relaxation is False


def test_pure_oscillator_has_no_aperiodic_timescale_but_a_dominant_one():
    """A model whose every mode is a rhythm has an envelope, not a timescale.

    Reporting the envelope as `tau` would misname it, so `tau` is NaN and the
    leading mode is available separately.
    """
    proc = AR2Damped(tau=0.1, f_osc=20.0)
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=2, fs=FS).fit(freqs, powers)
    assert np.isnan(ar.tau)
    assert ar.tau_dominant == pytest.approx(0.1, rel=0.05)
    assert 'all modes rhythmic' in repr(ar)


def test_a_relaxation_plus_a_rhythm_reports_the_relaxation():
    proc = ARpModes(modes=((0.2, 0.0), (0.05, 30.0)))
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=3, fs=FS).fit(freqs, powers)
    assert ar.tau == pytest.approx(0.2, rel=0.1)
    assert np.isfinite(ar.tau_dominant)


def test_plotted_mode_curves_are_the_factor_spectra():
    """The spectrum is a product of factors, so the plot must draw factors.

    The partial-fraction components are negative over most of the band for an
    oscillatory mode.  Drawing those on a log axis was wrong, and this pins
    the correct decomposition.
    """
    import matplotlib
    matplotlib.use('Agg')
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.06, 40.0)))
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=5, fs=FS).fit(freqs, powers)
    modes = ar.decompose()

    # partial fractions: additive, and negative for the oscillatory modes
    comps = modes.psd_components(freqs)
    np.testing.assert_allclose(comps.sum(axis=0), ar.predict(), rtol=1e-6)
    osc = [i for i, m in enumerate(modes.modes) if m.kind == 'oscillatory']
    assert osc, 'expected an oscillatory mode'
    assert any((comps[i] < 0).mean() > 0.5 for i in osc)

    # factors: multiplicative, strictly positive, and what the plot draws
    product = np.ones_like(freqs)
    for mode in modes.modes:
        factor = mode.psd(freqs)
        assert np.all(factor > 0)
        product = product * factor
    scale = np.exp(np.mean(np.log(ar.predict()) - np.log(product)))
    np.testing.assert_allclose(scale * product, ar.predict(), rtol=1e-8)

    ax = ar.plot(plot_modes=True)
    curves = ax.get_lines()[2:]
    assert len(curves) == len(modes.modes)
    for curve, mode in zip(curves, modes.modes):
        y = curve.get_ydata()
        assert np.all(y > 0)
        expected = mode.psd(freqs)
        expected = expected / np.exp(np.mean(np.log(expected)))
        expected = expected * np.exp(np.mean(np.log(ar.predict())))
        np.testing.assert_allclose(y, expected, rtol=1e-10)
        # scaling is a constant, so the shape is exactly the factor's
        np.testing.assert_allclose(y / y[0], mode.psd(freqs) / mode.psd(freqs)[0],
                                   rtol=1e-10)


def test_order_ceiling_is_flagged_and_warned():
    """A search that ends at its largest order has not seen the curve turn."""
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.06, 40.0)))
    freqs, powers = _theory(proc.phi(FS))
    with pytest.warns(RuntimeWarning, match='largest offered'):
        ar = ARPSD(order=range(1, 3), fs=FS).fit(freqs, powers)
    assert ar.order_at_ceiling_ and ar.order_ == 2

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')                 # no warning when it turns
        ar = ARPSD(order=range(1, 12), fs=FS).fit(freqs, powers)
    assert not ar.order_at_ceiling_


def test_fixed_order_never_reports_a_ceiling():
    freqs, powers = _theory(0.9)
    ar = ARPSD(order=5, fs=FS).fit(freqs, powers)
    assert ar.order_at_ceiling_ is False


def test_error_by_band_exposes_what_overall_rsq_hides():
    """Overall R^2 on a linear grid is dominated by the top decade."""
    proc = ARpModes(modes=((0.5, 0.0), (0.2, 3.0)))
    freqs = np.linspace(1.0, 250.0, 500)
    powers = ar_psd(proc.phi(FS), freqs, FS, 1.0)
    ar = ARPSD(order=4, fs=FS).fit(freqs, powers)
    edges, error = ar.error_by_band(4)
    assert edges.shape == (5,) and error.shape == (4,)
    assert edges[0] == pytest.approx(freqs[0]) and edges[-1] == pytest.approx(freqs[-1])
    assert np.all(np.isfinite(error)) and np.all(error >= 0)
    # an exact fit is exact in every band
    assert np.nanmax(error) < 1e-6


def test_error_by_band_is_large_where_the_model_is_wrong():
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.06, 40.0)))
    freqs, powers = _theory(proc.phi(FS))
    under = ARPSD(order=1, fs=FS).fit(freqs, powers)
    assert np.nanmax(under.error_by_band(4)[1]) > 0.1


def test_plot_modes_accepts_a_count():
    import matplotlib
    matplotlib.use('Agg')
    proc = ARpModes(modes=((0.2, 0.0), (0.15, 10.0), (0.06, 40.0), (0.02, 0.0)))
    freqs, powers = _theory(proc.phi(FS))
    ar = ARPSD(order=6, fs=FS).fit(freqs, powers)
    assert len(ar.plot(plot_modes=2).get_lines()) == 2 + 2
    assert len(ar.plot(plot_modes=True).get_lines()) == 2 + len(ar.decompose().modes)
