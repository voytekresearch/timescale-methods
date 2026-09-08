"""ARACF: fitting an AR(p) model to an autocorrelation.

Mirrors the ARPSD suite: exact recovery from a theoretical autocorrelation
first, then recovery from simulated data, then the interface itself.
"""

import numpy as np
import pytest

from timescales.ar import ARACF, ARPSD
from timescales.conversions import tau_to_phi
from timescales.decomposition.modes import ar_acf, ar_psd
from timescales.sim import AR1, AR2Damped, ARpModes
from timescales.utils import compute_acf, compute_psd

FS = 500.0
NLAGS = 250
LAGS = np.arange(NLAGS + 1, dtype=float)


def _theory(phi, nlags=NLAGS):
    return np.arange(nlags + 1, dtype=float), ar_acf(np.atleast_1d(phi), nlags)


# ---------------------------------------------------------------------------
# Exact recovery from a theoretical autocorrelation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tau', [0.005, 0.02, 0.1, 0.5])
def test_ar1_recovers_tau_from_its_own_acf(tau):
    lags, corrs = _theory(tau_to_phi(tau, FS))
    ar = ARACF(order=1, fs=FS).fit(lags, corrs)
    assert ar.success_
    assert ar.tau == pytest.approx(tau, rel=1e-2)
    assert ar.rsq > 0.999


def test_ar2_oscillator_recovers_envelope_and_frequency():
    proc = AR2Damped(tau=0.1, f_osc=20.0)
    lags, corrs = _theory(proc.phi(FS))
    ar = ARACF(order=2, fs=FS).fit(lags, corrs)
    assert ar.rsq > 0.99
    osc = [m for m in ar.decompose().modes if m.kind == 'oscillatory']
    assert osc
    assert osc[0].freq == pytest.approx(20.0, rel=0.05)
    assert osc[0].tau == pytest.approx(0.1, rel=0.1)


def test_arp_recovers_a_relaxation_plus_a_rhythm():
    proc = ARpModes(modes=((0.15, 0.0), (0.08, 12.0)))
    lags, corrs = _theory(proc.phi(FS))
    ar = ARACF(order=3, fs=FS).fit(lags, corrs)
    assert ar.rsq > 0.99
    modes = ar.decompose()
    osc = [m for m in modes.modes if m.kind == 'oscillatory']
    assert osc and osc[0].freq == pytest.approx(12.0, rel=0.1)


def test_the_two_domains_agree_on_the_same_process():
    """The PSD and ACF fits estimate the same phi, so they must agree."""
    proc = AR1(tau=0.08)
    lags, corrs = _theory(proc.phi(FS))
    freqs = np.linspace(1.0, 250.0, 500)
    powers = ar_psd(proc.phi(FS), freqs, FS, 1.0)
    acf_tau = ARACF(order=1, fs=FS).fit(lags, corrs).tau
    psd_tau = ARPSD(order=1, fs=FS).fit(freqs, powers).tau
    assert acf_tau == pytest.approx(psd_tau, rel=0.02)


# ---------------------------------------------------------------------------
# Recovery from simulated data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tau', [0.01, 0.05, 0.2])
def test_ar1_recovers_tau_from_simulation(tau):
    rng = np.random.default_rng(0)
    sig = AR1(tau=tau).simulate(200.0, FS, rng)
    lags, corrs = compute_acf(sig, NLAGS)
    ar = ARACF(order=1, fs=FS).fit(lags, corrs)
    assert ar.tau == pytest.approx(tau, rel=0.2)
    assert ar.is_stationary


def test_order_selection_buys_order_for_an_oscillator():
    rng = np.random.default_rng(2)
    sig = AR2Damped(tau=0.1, f_osc=20.0).simulate(200.0, FS, rng)
    lags, corrs = compute_acf(sig, NLAGS)
    ar = ARACF(order=range(1, 9), fs=FS).fit(lags, corrs)
    assert ar.order_ >= 2
    osc = [m for m in ar.decompose().modes if m.kind == 'oscillatory']
    assert osc and osc[0].freq == pytest.approx(20.0, rel=0.15)


def test_higher_order_never_fits_worse_than_ar1():
    rng = np.random.default_rng(3)
    sig = AR2Damped(tau=0.1, f_osc=20.0).simulate(200.0, FS, rng)
    lags, corrs = compute_acf(sig, NLAGS)
    low = ARACF(order=1, fs=FS).fit(lags, corrs)
    high = ARACF(order=4, fs=FS).fit(lags, corrs)
    assert high.rsq > low.rsq


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_defaults_to_ar1_and_unconstrained():
    ar = ARACF(fs=FS)
    assert ar.order == 1
    assert ar.require_relaxation is False


def test_fs_is_required():
    with pytest.raises(ValueError):
        ARACF(order=1)


def test_reports_before_fit_are_refused():
    ar = ARACF(order=1, fs=FS)
    with pytest.raises(RuntimeError):
        _ = ar.tau
    assert 'unfitted' in repr(ar)


def test_rejects_mismatched_and_short_input():
    lags, corrs = _theory(0.95)
    with pytest.raises(ValueError):
        ARACF(order=1, fs=FS).fit(lags[:-1], corrs)
    with pytest.raises(ValueError):
        ARACF(order=1, fs=FS).fit(lags[:2], corrs[:2])


def test_offset_can_be_switched_off():
    lags, corrs = _theory(tau_to_phi(0.05, FS))
    ar = ARACF(order=1, fs=FS, with_offset=False).fit(lags, corrs)
    assert abs(ar.acf_offset) < 1e-6
    assert ar.tau == pytest.approx(0.05, rel=0.05)


def test_offset_absorbs_a_constant_pedestal():
    """A shifted autocorrelation is fit by the offset, not by a slower decay."""
    lags, corrs = _theory(tau_to_phi(0.05, FS))
    ar = ARACF(order=1, fs=FS).fit(lags, corrs * 0.8 + 0.1)
    assert ar.tau == pytest.approx(0.05, rel=0.1)
    assert ar.acf_offset == pytest.approx(0.1, abs=0.05)


def test_settings_are_not_hidden_module_state():
    ar = ARACF(order=1, fs=FS, k_bound=0.99, maxfev=10, tol=1e-3, n_eff=17)
    assert (ar.k_bound, ar.maxfev, ar.tol, ar.n_eff) == (0.99, 10, 1e-3, 17)


def test_predict_matches_the_fitted_curve():
    lags, corrs = _theory(0.95)
    ar = ARACF(order=1, fs=FS).fit(lags, corrs)
    pred = ar.predict()
    assert pred.shape == corrs.shape
    assert np.corrcoef(pred, corrs)[0, 1] > 0.99


def test_effective_sample_size_controls_the_order_bought():
    """`n_eff` is the documented guard on BIC over dependent lags."""
    rng = np.random.default_rng(5)
    sig = AR2Damped(tau=0.1, f_osc=20.0).simulate(200.0, FS, rng)
    lags, corrs = compute_acf(sig, NLAGS)
    strict = ARACF(order=range(1, 12), fs=FS, n_eff=6).fit(lags, corrs)
    loose = ARACF(order=range(1, 12), fs=FS, n_eff=len(lags)).fit(lags, corrs)
    assert strict.order_ <= loose.order_


def test_patience_stops_the_order_search_early():
    rng = np.random.default_rng(6)
    sig = AR1(tau=0.05).simulate(100.0, FS, rng)
    lags, corrs = compute_acf(sig, NLAGS)
    stopped = ARACF(order=range(1, 20), fs=FS, patience=1).fit(lags, corrs)
    assert len(stopped.order_scores_) < 19


def test_plot_draws_without_error():
    import matplotlib
    matplotlib.use('Agg')
    proc = AR2Damped(tau=0.1, f_osc=20.0)
    lags, corrs = _theory(proc.phi(FS))
    ar = ARACF(order=3, fs=FS).fit(lags, corrs)
    assert ar.plot(plot_modes=True) is not None


def test_simulate_round_trips_through_the_fit():
    lags, corrs = _theory(tau_to_phi(0.05, FS))
    ar = ARACF(order=1, fs=FS).fit(lags, corrs)
    sig = ar.simulate(200.0, rng=np.random.default_rng(0))
    lags2, corrs2 = compute_acf(sig, NLAGS)
    assert ARACF(order=1, fs=FS).fit(lags2, corrs2).tau == pytest.approx(0.05, rel=0.2)
