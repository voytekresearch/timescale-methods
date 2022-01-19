"""Plotting functions."""

import numpy as np


def plot_connected_scatter(param_a, param_b, ax, title, paired=True, alpha_scatter=.5,
                           alpha_line=.1,  ylim=None, ylabel=None, xticklabels=None):
    """Plot connected violin scattter plots.
    Parameters
    ----------
    param_a : 1d array
        First parameter to plot.
    param_b : 1d array
        Second parameter to plot.
    ax : AxesSubplot
        Subplot to plot onto.
    title : str
        Title of plot.
    paired : bool, optional, default: True
        Plot lines connected pairs of points from param_a and param_b.
    alpha_scatter : float, optional, default: .5
        Transparency of scatter points.
    alpha_line : float, optional, default: .1
        Transparency of scatter paired lines.
    ylim : tuple of (float, float), optional, default: None
        Y-axis limits.
    ylabel : str, optional, default: None
        Y-axis label.
    xticklabels: list, optional, default: None
        X-axis labels.
    """
    xs_a = np.random.uniform(1, 1.2, size=len(param_a))
    ax.scatter(xs_a, param_a, alpha=alpha_scatter)


    xs_b = np.random.uniform(1.8, 2, size=len(param_b))
    ax.scatter(xs_b, param_b, alpha=alpha_scatter)

    vp = ax.violinplot([param_a, param_b], showextrema=False)
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
        for i, (t_psd, t_acf) in enumerate(zip(param_a, param_b)):
            ax.plot([xs_a[i], xs_b[i]], [t_psd, t_acf], color='k', alpha=alpha_line)

    ax.set_title(title)
    ax.set_ylabel('Tau')
    ax.set_xticks([1, 2])

    xticklabels = ['PSD', 'ACF'] if xticklabels is None else xticklabels
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, 3)

    if ylabel is None:
        ax.set_ylabel('Tau')
    else:
        ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax
