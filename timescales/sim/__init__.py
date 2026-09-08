"""Initialize sim sub-module."""

from .spikes import sim_spikes_synaptic, sim_spikes_prob, sim_poisson, sample_spikes, bin_spikes
from .acf import sim_acf_cos, sim_exp_decay, sim_damped_cos
from .psd import sim_lorentzian
from .ar import sim_ar, sim_ar_spectrum
from .ou import sim_ou
from .branching import sim_branching
from .processes import (Process, AR1, AR1PlusSine, AR2Damped, SumAR1,
                        AR1PlusAR2, ARpModes, AR1Mixture, sim_ar_process,
                        ar_psd_density, integrated_timescale_from_acf)


def __getattr__(name):
    """Re-export neurodsp's synaptic kernel without importing neurodsp here.

    The spike, Ornstein-Uhlenbeck and branching simulations need neurodsp;
    fitting and decomposing a model does not.  Importing it lazily keeps
    ``import timescales`` to numpy and scipy.  Install with
    ``pip install timescales[all]`` to use these simulations.
    """
    if name == 'sim_synaptic_kernel':
        from neurodsp.sim import sim_synaptic_kernel
        return sim_synaptic_kernel
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
