"""Plotting functions."""

import numpy as np


def plot_connected_scatter(param_a, param_b, ax, title=None, paired=True,
                           scatter_a_loc=None, scatter_b_loc=None, violin_locs=None,
                           colors=None, alpha_scatter=.5, alpha_line=.1,
                           xticks=None, xlim=None, ylim=None, ylabel=None, xticklabels=None):
    """Plot connected violin scattter plots.
    Parameters
    ----------
    param_a : 1d or 2d array
        First parameter to plot.
    param_b : 1d or 2d array
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


    if title is None:
        title = ''

    # Scatter points
    if scatter_a_loc is None:
        xs_a = np.random.uniform(1, 1.2, size=len(param_a))
    else:
        xs_a = np.random.uniform(*scatter_a_loc, size=len(param_a))

    if scatter_a_loc is None:
        xs_b = np.random.uniform(1.8, 2, size=len(param_b))
    else:
        xs_b = np.random.uniform(*scatter_b_loc, size=len(param_b))

    if colors is None:
        c0 = 'C0'
        c1 = 'C1'
    else:
        c0 = colors[0]
        c1 = colors[1]

    ax.scatter(xs_a, param_a, alpha=alpha_scatter, color=c0)
    ax.scatter(xs_b, param_b, alpha=alpha_scatter, color=c1)

    # Violinplots
    if violin_locs is None:
        vp = ax.violinplot([param_a, param_b], showextrema=False)
    else:
        vp = ax.violinplot([param_a, param_b], positions=violin_locs, showextrema=False)


    vp['bodies'][0].set_color(c0)
    vp['bodies'][0].set_facecolor(c0)
    vp['bodies'][0].set_edgecolor(c0)

    vp['bodies'][1].set_color(c1)
    vp['bodies'][1].set_facecolor(c1)
    vp['bodies'][1].set_edgecolor(c1)

    b = vp['bodies'][0]
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf,
                                              np.mean(b.get_paths()[0].vertices[:, 0]))

    b = vp['bodies'][1]
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],
                                              np.mean(b.get_paths()[0].vertices[:, 0]), np.inf)

    if paired:
        for i, (t_psd, t_acf) in enumerate(zip(param_a, param_b)):
            ax.plot([xs_a[i], xs_b[i]], [t_psd, t_acf], color='k', alpha=alpha_line)

    # Axis settings
    ax.set_title(title)
    ax.set_ylabel('Tau')

    if xticks is None:
        ax.set_xticks([1, 2])
    else:
        ax.set_xticks(xticks)

    xticklabels = ['PSD', 'ACF'] if xticklabels is None else xticklabels
    ax.set_xticklabels(xticklabels, fontsize=24)

    if ylabel is None:
        ax.set_ylabel('Tau')
    else:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)
