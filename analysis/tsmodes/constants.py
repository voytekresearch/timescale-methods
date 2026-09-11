"""Process constants specific to this paper's figures and benchmark.

The generic machinery lives in :mod:`timescales.sim.processes`; what is here
is the particular AR(7), AR(5) and AR(2) the paper simulates, and the single
constructed timescale every column of the benchmark figure shares.
"""

from timescales.sim import ARpModes

# The sampling rate of every simulation in the paper, and the constructed
# timescale of the AR(1) simulations (fig_bias, the oscillation sweep and the
# finite-sampling stage).  They live here, not in each script, because a
# figure and the stage that produced its table have to agree on them and DVC
# cannot check that two constants in two files still match.
SIM_FS = 500.0
AR1_TAU = 0.05

# magnitude apart and both inside the band a 4 s Welch window resolves.
ARP_MODES_7 = ((0.200, 0.0), (0.150, 10.0), (0.060, 40.0),
               (0.020, 0.0), (0.002, 0.0))
# The alternative, an AR(5): a 250 ms relaxation, a 5 Hz mode whose envelope
# decays in 100 ms (half a cycle, so a damped relaxation by the paper's rule
# rather than a rhythm) and a 50 Hz rhythm ringing for 20 ms.  Under the rule
# an estimator is asked to apply -- the most important non-rhythmic mode --
# its own timescale is the 5 Hz mode's 100 ms envelope, which outranks the
# 250 ms relaxation in spectral importance.  The benchmark simulates it
# alongside the AR(7), and the figure scripts draw either (TSMODES_ARP=5).
ARP_MODES_5 = ((0.250, 0.0), (0.020, 50.0), (0.100, 5.0))
ARP_VARIANTS = {'7': ARP_MODES_7, '5': ARP_MODES_5}
ARP_MODES = ARP_MODES_7
# Every column of the benchmark figure has the same constructed timescale:
# the AR(1)'s tau, the AR(2) oscillator's envelope decay, the AR(7)'s
# dominant relaxation, and the finite sample's tau, all 200 ms, so that the
# estimates can share one axis.
BENCH_TAU = 0.200
# The AR(2) oscillator of the benchmark and of the oscillations column of
# fig_bias, as (tau, f): a 10 Hz rhythm whose envelope decays in 200 ms, two
# cycles, a peak forty times its low-frequency plateau.
AR2_RHYTHM = (BENCH_TAU, 10.0)


def arp_process(choice='7'):
    """The constructed AR(p) by variant: ``'7'`` is the paper's AR(7),
    ``'5'`` the alternative AR(5)."""
    choice = str(choice)
    return ARpModes(modes=ARP_VARIANTS[choice],
                    variant=None if choice == '7' else choice)
