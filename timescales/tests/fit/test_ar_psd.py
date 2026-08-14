"""Test ARPSD model."""
import numpy as np
from timescales.sim.ar import sim_ar_spectrum
from timescales.fit import ARPSD

def test_ar_psd_model():

    # Simulate AR(1) PSD
    freqs = np.linspace(0, 500, 1000)
    fs = 1000
    powers = sim_ar_spectrum(freqs, fs, phi=np.array([0.5]))

    # 1d: Fit AR(1) PSD
    arpsd = ARPSD(1, fs)
    arpsd.fit(freqs, powers)
    arpsd.plot()

    assert arpsd.params[0].round(1) == 0.5
    assert arpsd.is_stationary
    sig = arpsd.simulate(1, 1000)
    assert len(sig) == 1000

    # 2d
    arpsd = ARPSD(1, fs)
    arpsd.fit(freqs, np.vstack((powers, powers)))
    arpsd.plot()

    assert arpsd.params[0, 0].round(1) == 0.5
    assert arpsd.params[1, 0].round(1) == 0.5
    is_stationary = arpsd.is_stationary
    assert is_stationary[0] and is_stationary[1]
    for i in range(2):
        sig = arpsd.simulate(1, 1000, index=i)
        assert len(sig) == 1000

    # A stationary AR(2) may have coefficients outside (-1, 1)
    phi = np.array([1.5, -0.75])
    powers = sim_ar_spectrum(freqs, fs, phi=phi)
    arpsd = ARPSD(2, fs)
    arpsd.fit(freqs, np.vstack((powers, powers)))

    assert arpsd.guess.shape == (2, 3)
    assert arpsd.bounds.shape == (2, 2, 3)
    assert np.allclose(arpsd.params[:, :-1], phi)
    assert arpsd.is_stationary.all()
