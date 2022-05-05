"""Simulation, transformation, and fitting pipeline framework."""


import operator as op
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.signal import resample

from timescales.fit import PSD, ACF
from timescales.fit.utils import progress_bar

from neurodsp.utils.norm import normalize_sig


class Pipe:
    """Simulations, PSD/ACF, and fitting pipelines.

    Parameters
    ----------
    n_seconds : float
        Length of signal, in seconds.
    fs : float
        Sampling rate, in Hertz.
    seeds : 1d array of int
        Random seeds for reproducible simulations.
    """

    def __init__(self, n_seconds, fs, seeds=None):

        self.n_seconds = n_seconds
        self.fs = fs

        self.seeds = seeds
        self.seed = None

        self.sig = None

        self.model = None
        self.models = []
        self.pipe = None

        self.results = None
        self.result = None


    def run(self, n_jobs=-1, progress=None):
        """Run analysis pipeline.

        Parameters
        ----------
        n_jobs : int
            Number of jobs to run in parralel.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        self.results = [] if  self.results is None else self.results

        with Pool(processes=n_jobs) as pool:
            mapping = pool.imap(self._run, self.seeds)
            results = list(progress_bar(mapping, progress, len(self.seeds)))

        self.models = [r[0] for r in results]
        self.results = np.array([r[1] for r in results])


    def _run(self, seed):
        """Proxy function to allow parallelziation.

        Parameters
        ----------
        seed : int
            Random seed to set.
        """
        np.random.seed(seed)

        # Clear
        self.model = None
        self.result = None

        for node in self.pipe:
            getattr(self, node['step'])(*node['args'], **node['kwargs'])

        return self.model, self.result


    def add_step(self, step, *args, **kwargs):
        """Add a step to the pipeline.

        Parameters
        ----------
        step : {'simulate', 'sample', 'transform', fit'}
            Method to run.
        *args
            Positional arugments for the specified method.
        **kwargs
            Keyword arguemnts for the specified method.
        """
        if self.pipe is None:
            self.pipe = []

        self.pipe.append({'step': step, 'args': args, 'kwargs': kwargs})


    def fit(self, return_attrs, **fit_kwargs):
        """Fit timescale of simulation.

        Parameters
        ----------
        return_attrs : str or list of str or {'knee_freq', 'tau', 'rsq'}
            Model attributes to specifically store. These are attributes
            of PSD or ACF objects set upon fitting.
        **fit_kwargs
            Keyword arguments passed to the fit method of the
            PSD or ACF objects.

        Notes
        -----
        Assumes method based on transform method call.
        """

        self.model.fit(**fit_kwargs)

        if isinstance(return_attrs, str):
            return_attrs = [return_attrs]

        self.result = [getattr(self.model, r, None) for r in return_attrs]


    def transform(self, method, **compute_kwargs):
        """Fit timescale of simulation.

        Parameters
        ----------
        method : {'PSD', 'ACF'}
            Fitting method.
        **compute_kwargs
            Additional arguments to pass to compute_spectrum() or
            compute_acf() methods in the PSD or ACF objects.
        """

        if method == 'PSD':
            self.model = PSD()
            self.model.compute_spectrum(self.sig, self.fs,
                                        **compute_kwargs)
        elif method == 'ACF':
            self.model = ACF()
            self.model.compute_acf(self.sig, self.fs,
                                   **compute_kwargs)


    def simulate(self, sim_func, *sim_args, operator='add', rescale=None,
                 mean=None, variance=None, **sim_kwargs):
        """Simulate aperiodic signal.

        Parameters
        ----------
        sim_func : func
            Simulation function.
        operator : {'add', 'mul', 'sub', 'div'} or {'+', '*', '-', '/'}
            Operator to combine signals.
        rescale : tuple of (float, float), optional, default: None
            Minimum and maximum y-values of simulation.
        mean : float, optional, default: None
            Mean to normalize to.
        variance : float, opational, default: None
            Variance to normalize to.
        *sim_args
            Additonal simulation positional arugments.
        **sim_kwargs
            Additional simulation keyword arguments.

        Notes
        -----
        Either rescale or (mean, std) should be passed for each simulation.
        """

        # Handle normalization/rescaling options
        if rescale is not None and (mean or variance):
            raise ValueError('If rescale is defined, (mean, variance) must be None.')
        elif rescale is None and mean is None and variance is None:
            # No rescaling or normalizing
            transform = lambda x : x
        elif rescale is not None:
            # Rescale
            transform = lambda x : Pipe.rescale(x, rescale)
        else:
            # Normalize
            transform = lambda x : Pipe.normalize(x, mean, variance)

        # How to combine signals
        if operator in ['add', '+']:
            operator = op.add
        elif operator in ['mul', '*']:
            operator = op.mul
        elif operator in ['sub', '-']:
            operator = op.sub
        else:
            operator = op.truediv

        # Set seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Simulate
        if self.sig is None:
            self.sig = transform(sim_func(self.n_seconds, self.fs, *sim_args, **sim_kwargs))
        else:
            self.sig = operator(
                self.sig, transform(sim_func(self.n_seconds, self.fs, *sim_args, **sim_kwargs))
            )


    def sample(self, n_resample=None):
        """Sample binary array from probabilties.

        Parameters
        ----------
        n_resample : int, optional, default: None
            Resample signal array before sampling.

        Notes
        -----
        Assumes the sig attribute is the target probability array.
        """
        if n_resample:
            # Upsampling can decrease computation time
            self.sig = resample(self.sig, n_resample)

        self.sig = self.sig > np.random.rand(len(self.sig))


    def bin(self, bin_size):
        """Bin signal.

        Parameters
        ----------
        bin_size : int
            Number of samples per bin.
        """
        self.sig = self.sig.reshape(-1, bin_size).sum(axis=1)


    @staticmethod
    def rescale(sig, norm_range=(0, 1)):
        """Normalize signal from lower to upper."""
        if sig.ndim == 2:
            for ind in range(len(sig)):
                sig[ind] = ACF.normalize(sig[ind])
        else:
            sig = np.interp(sig, (sig.min(), sig.max()), norm_range)
        return sig


    @staticmethod
    def normalize(sig, mean=None, variance=None):
        """Normalize signal with given mean and variance."""
        return normalize_sig(sig, mean, variance)
