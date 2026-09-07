"""Tests that each generative process matches its closed-form spectrum/ACF."""

import numpy as np
import pytest
from scipy.signal import welch

from tsmodes.modes import ar_acf, ar_to_poles
from tsmodes.sims import (AR1, AR1Mixture, AR1PlusSine, AR2Damped,
                          SumAR1, ar_psd_density, sim_ar_process,
                          tau_to_phi)

FS = 500.0
PROCESSES = [AR1(tau=0.05), AR1PlusSine(tau=0.05, f_osc=10.0, snr=1.0), AR2Damped(tau=0.05, f_osc=10.0),
             SumAR1(tau0=0.01, tau1=0.2, weight=0.5), AR1Mixture()]


def _welch(sig, fs, win_s=2.0):
    nperseg = int(win_s * fs)
    freqs, powers = welch(sig, fs, nperseg=nperseg, noverlap=nperseg // 2)
    w = np.hanning(nperseg)
    enbw = fs * np.sum(w ** 2) / np.sum(w) ** 2
    return freqs, powers, enbw


@pytest.mark.parametrize('proc', PROCESSES, ids=[p.name for p in PROCESSES])
def test_empirical_psd_matches_theory(proc):
    rng = np.random.default_rng(0)
    sig = proc.simulate(300.0, FS, rng)
    freqs, powers, enbw = _welch(sig, FS)
    theory = proc.psd_with_lines(freqs, FS, enbw)
    mask = (freqs > 1.0) & (freqs < 200.0)
    err = np.median(np.abs(np.log10(powers[mask] / theory[mask])))
    assert err < 0.05, f'{proc.name}: median |log10 ratio| = {err:.3f}'


@pytest.mark.parametrize('proc', PROCESSES[:4], ids=[p.name for p in PROCESSES[:4]])
def test_empirical_acf_matches_theory(proc):
    rng = np.random.default_rng(1)
    sig = proc.simulate(600.0, FS, rng)
    x = sig - sig.mean()
    nlags = int(FS * 0.4)
    emp = np.correlate(x, x, 'full')[len(x) - 1: len(x) + nlags]
    emp = emp / emp[0]
    theory = proc.acf(np.arange(nlags + 1), FS)
    assert np.max(np.abs(emp - theory)) < 0.05, proc.name


def test_ar1_variance_and_integrated_timescale():
    proc = AR1(tau=0.05)
    phi = tau_to_phi(0.05, FS)
    assert proc.tau_integrated(FS) == pytest.approx(
        (1 / FS) * (1 / (1 - phi) - 0.5), rel=1e-12)
    # The trapezoid convention returns the continuous-time tau, not tau + 1/(2 fs).
    assert proc.tau_integrated(FS) == pytest.approx(0.05, rel=1e-3)


def test_integrated_timescale_of_mixture_is_variance_weighted():
    proc = SumAR1(tau0=0.01, tau1=0.2, weight=0.5)
    lags = np.arange(0, int(FS * 20))
    numeric = np.trapezoid(proc.acf(lags, FS), dx=1 / FS)
    assert proc.tau_integrated(FS) == pytest.approx(numeric, rel=1e-3)


def test_ar2_damped_poles_match_requested_tau_and_frequency():
    proc = AR2Damped(tau=0.08, f_osc=12.0)
    poles = ar_to_poles(proc.phi(FS))
    r = np.abs(poles[0])
    assert -1 / (FS * np.log(r)) == pytest.approx(0.08, rel=1e-10)
    assert FS * abs(np.angle(poles[0])) / (2 * np.pi) == pytest.approx(12.0, rel=1e-10)


def test_sim_ar_process_matches_theoretical_variance():
    rng = np.random.default_rng(3)
    phi = np.array([0.8, -0.3])
    sig = sim_ar_process(phi, 400000, rng)
    assert sig.var() == pytest.approx(ar_acf(phi, 1, normalize=False)[0], rel=0.05)


def test_ar_psd_density_integrates_to_the_variance():
    phi = np.array([0.7])
    freqs = np.linspace(0, FS / 2, 200001)
    total = np.trapezoid(ar_psd_density(phi, freqs, FS), freqs)
    assert total == pytest.approx(ar_acf(phi, 1, normalize=False)[0], rel=1e-3)


def test_ar1_plus_sine_line_power_matches_amplitude():
    proc = AR1PlusSine(tau=0.05, f_osc=10.0, snr=2.0)
    var_ar = 1 / (1 - tau_to_phi(0.05, FS) ** 2)
    assert proc.line_power(FS) == pytest.approx(2.0 * var_ar, rel=1e-12)


def test_processes_cannot_be_built_positionally():
    """Positional construction silently used the defaults; it must now fail.

    ``Process`` declares its own fields first, so ``AR1PlusSine(0.05, 4, 4)``
    bound tau, frequency and amplitude to ``name``, ``label`` and
    ``tau_true``, then ``__post_init__`` overwrote all three.  The call
    returned a default process and nothing complained.
    """
    with pytest.raises(TypeError):
        AR1PlusSine(0.05, 4.0, 4.0)
    with pytest.raises(TypeError):
        AR1(0.05)
    strong = AR1PlusSine(tau=0.05, f_osc=4.0, snr=4.0)
    assert strong.params == {'tau': 0.05, 'f_osc': 4.0, 'snr': 4.0}
