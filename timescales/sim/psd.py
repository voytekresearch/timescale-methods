"""Power spectral density simulations."""


def sim_lorentzian(freqs, knee_freq, exponent=2., offset=0., constant=0.):
    """Simulate a Lorentzian with a timescale.

    Parameters
    ----------
    freqs : 1d array
        Frequency definition.
    knee_freq : float
        Frequency corresponding to the desired timescale.
    exponent : float, optional, deafult: 2.
        Slope of the post-knee log-log spectra.
    offset : float, optional, default: 0.
        Y-intercept.
    constant : float, optional, default: 0.
        Constant scaling factor that tapers high-frequency log-log power.

    Returns
    -------
    powers : 1d array
        Power spectral density.
    """
    f_e = freqs**exponent
    fk_e = knee_freq**exponent

    powers = (
        (10**offset + (constant * fk_e) + (constant * f_e)) / (fk_e + f_e)
    )

    return powers
