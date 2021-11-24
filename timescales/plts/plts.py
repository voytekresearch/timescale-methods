"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_connected_scatter(taus_a, taus_b, ax, title, paired=True, alpha_scatter=.5,
                           alpha_line=.1,  ylim=None, ylabel=None, ticklabels=None):

    xs_a = np.random.uniform(1, 1.2, size=len(taus_a))
    ax.scatter(xs_a, taus_a, alpha=alpha_scatter)


    xs_b = np.random.uniform(1.8, 2, size=len(taus_b))
    ax.scatter(xs_b, taus_b, alpha=alpha_scatter)

    vp = ax.violinplot([taus_a, taus_b], showextrema=False)
    vp['bodies'][1].set_color('C1')
    vp['bodies'][1].set_facecolor('C1')
    vp['bodies'][1].set_edgecolor('C1')

    b = vp['bodies'][0]
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf,
                                              np.mean(b.get_paths()[0].vertices[:, 0]))

    b = vp['bodies'][1]
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],
                                              np.mean(b.get_paths()[0].vertices[:, 0]), np.inf)

    if paired:
        for i, (t_psd, t_acf) in enumerate(zip(taus_a, taus_b)):
            ax.plot([xs_a[i], xs_b[i]], [t_psd, t_acf], color='k', alpha=alpha_line)

    ax.set_title(title)
    ax.set_ylabel('Tau')
    ax.set_xticks([1, 2])

    ticklabels = ['PSD', 'ACF'] if ticklabels is None else ticklabels
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(0, 3)

    if ylabel is None:
        ax.set_ylabel('Tau')
    else:
        ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)
