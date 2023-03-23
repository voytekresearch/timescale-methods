import os
import time
from contextlib import redirect_stdout, redirect_stderr
import logging, sys
logging.disable(sys.maxsize)

from functools import partial
from tqdm.notebook import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import numpy as np

from timescales.fit import PSD
from timescales.sim import sim_branching
from timescales.conversions import convert_knee

import abcTau
from abcTau.preprocessing import extract_stats
import mrestimator as mre

from abcTau.generative_models import oneTauOU
from abcTau.distance_functions import linear_distance

class MyModel(abcTau.Model):

    def __init__(self):
        pass

    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    def generate_data(self, theta):

        syn_data, numBinData = oneTauOU(theta, self.deltaT, self.binSize, self.T, self.numTrials,
                                        self.data_mean, self.data_var)

        syn_sumStat = abcTau.summary_stats.comp_sumStat(
            syn_data, self.summStat_metric, self.ifNorm, self.deltaT, self.binSize, self.T, numBinData, self.maxTimeLag
        )

        return syn_sumStat

    # Computes the summary statistics
    def summary_stats(self, data):
        sum_stat = data
        return sum_stat

    # Choose the method for computing distance (from basic_functions)
    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0:
            d = 10**4
        else:
            d = linear_distance(data, synth_data)
        return d



def compute_taus(knee):
    fs = 1000
    datasave_path = 'example_abc_results/'
    inter_save_direc = 'example_abc_results/'
    inter_filename = 'abc_intermediate_results'
    filename = 'sim_00'
    filenameSave = filename
    summStat_metric = 'comp_ac_fft'
    ifNorm = True
    deltaT = 1
    binSize = 1
    disp = None
    maxTimeLag = 100
    generativeModel = 'oneTauOU'
    distFunc = 'linear_distance'
    t_min = 1.0
    t_max = 100.0
    priorDist = [stats.uniform(loc=t_min, scale=t_max - t_min)]
    epsilon_0 = 1
    min_samples = 50
    steps = 30
    minAccRate = 0.1
    parallel = False
    n_procs = 1

    # Simulate
    n_seconds = 1
    y = sim_branching(n_seconds, fs, convert_knee(knee), 1000, mean=0, variance=1)
    y = y.reshape(1, -1)

    # Fit MR
    with redirect_stderr(open(os.devnull, 'w')):
        start = time.time()
        rk = mre.coefficients(y, dtunit='step')
        m0 = mre.fit(rk, fitfunc=mre.f_exponential_offset)
        end = time.time()
        time_mr = end-start

    # Fit aABC
    with redirect_stdout(None):

        start = time.time()

        data_sumStat, data_mean, data_var, T, numTrials =  extract_stats(
            y, deltaT, binSize, summStat_metric, ifNorm, maxTimeLag
        )

        kwargs = dict(data_mean=data_mean, data_var=data_var, T=T, numTrials=numTrials,
                    generativeModel=generativeModel, distFunc=distFunc, summStat_metric=summStat_metric,
                    ifNorm=ifNorm, deltaT=deltaT, binSize=binSize, maxTimeLag=maxTimeLag)

        abc_results, final_step = abcTau.fit.fit_withABC(
            MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,
            datasave_path,filenameSave, epsilon_0, min_samples,
            steps, minAccRate, parallel, n_procs, disp, **kwargs
        )
        end = time.time()
        time_abc = end-start

    # Fit AR spectrum
    start = time.time()
    psd = PSD()
    psd.compute_spectrum(y[0], fs, ar_order=5, f_range=(1, 100))
    psd.fit()
    end = time.time()
    time_ar = end-start

    # Collect results
    knees_mr = convert_knee(m0.tau / fs)
    knees_abc = convert_knee(abc_results[final_step-1][0].mean() / fs)
    knees_ar = psd.knee_freq

    return knees_ar, knees_mr, knees_abc, time_ar, time_mr, time_abc
