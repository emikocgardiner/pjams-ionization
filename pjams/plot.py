"""Plotting module.

Provides convenience methods for generating standard plots and components using `matplotlib`.

"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


FIGSIZE = 6
FONTSIZE = 13
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2
COLORS = np.array(['#ff3333', '#ff8c1a', '#bfbd2e',  '#5cd699', '#51a6fb', '#a65eed'])


mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["lines.solid_capstyle"] = 'round'


def figax_single(height=None, **kwargs):
    mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["lines.solid_capstyle"] = 'round'
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
    mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
    mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

    if height is None:
        height = FIGSIZE * GOLDEN_RATIO
    figsize_single = [FIGSIZE, height]
    adjust_single = dict(left=0.15, bottom=0.15, right=0.95, top=0.95)

    kwargs.setdefault('figsize', figsize_single)
    for kk, vv in adjust_single.items():
        kwargs.setdefault(kk, vv)

    return figax(**kwargs)


def figax_double(height=None, **kwargs):
    mpl.style.use('default')   # avoid dark backgrounds from dark theme vscode
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["lines.solid_capstyle"] = 'round'
    plt.rcParams["font.size"] = FONTSIZE
    plt.rcParams["legend.fontsize"] = FONTSIZE*0.8
    mpl.rcParams['xtick.labelsize'] = FONTSIZE*0.8
    mpl.rcParams['ytick.labelsize'] = FONTSIZE*0.8

    if height is None:
        height = 2 * FIGSIZE * GOLDEN_RATIO

    figsize_double = [2*FIGSIZE, height]
    adjust_double = dict(left=0.10, bottom=0.10, right=0.98, top=0.95)

    kwargs.setdefault('figsize', figsize_double)
    for kk, vv in adjust_double.items():
        kwargs.setdefault(kk, vv)

    return figax(**kwargs)


def figax(figsize=[7, 5], ncols=1, nrows=1, sharex=False, sharey=False, squeeze=True,
          scale=None, xscale='log', xlabel='', xlim=None, yscale='log', ylabel='', ylim=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          widths=None, heights=None, grid=True, **kwargs):
    """Create matplotlib figure and axes instances.

    Convenience function to create fig/axes using `plt.subplots`, and quickly modify standard
    parameters.

    Parameters
    ----------
    figsize : (2,) list, optional
        Figure size in inches.
    ncols : int, optional
        Number of columns of axes.
    nrows : int, optional
        Number of rows of axes.
    sharex : bool, optional
        Share xaxes configuration between axes.
    sharey : bool, optional
        Share yaxes configuration between axes.
    squeeze : bool, optional
        Remove dimensions of length (1,) in the `axes` object.
    scale : [type], optional
        Axes scaling to be applied to all x/y axes.  One of ['log', 'lin'].
    xscale : str, optional
        Axes scaling for xaxes ['log', 'lin'].
    xlabel : str, optional
        Label for xaxes.
    xlim : [type], optional
        Limits for xaxes.
    yscale : str, optional
        Axes scaling for yaxes ['log', 'lin'].
    ylabel : str, optional
        Label for yaxes.
    ylim : [type], optional
        Limits for yaxes.
    left : [type], optional
        Left edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    bottom : [type], optional
        Bottom edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    right : [type], optional
        Right edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    top : [type], optional
        Top edge of axes space, set using `plt.subplots_adjust()`, as a fraction of figure.
    hspace : [type], optional
        Height space between axes if multiple rows are being used.
    wspace : [type], optional
        Width space between axes if multiple columns are being used.
    widths : [type], optional
    heights : [type], optional
    grid : bool, optional
        Add grid lines to axes.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        New matplotlib figure instance containing axes.
    axes : [ndarray] `matplotlib.axes.Axes`
        New matplotlib axes, either a single instance or an ndarray of axes.

    """

    if scale is not None:
        xscale = scale
        yscale = scale

    scales = [xscale, yscale]
    for ii in range(2):
        if scales[ii].startswith('lin'):
            scales[ii] = 'linear'

    xscale, yscale = scales

    if (widths is not None) or (heights is not None):
        gridspec_kw = dict()
        if widths is not None:
            gridspec_kw['width_ratios'] = widths
        if heights is not None:
            gridspec_kw['height_ratios'] = heights
        kwargs['gridspec_kw'] = gridspec_kw

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=ncols, nrows=nrows,
                             sharex=sharex, sharey=sharey, **kwargs)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)

    if ylim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(ylim) == (2,):
            ylim = np.array(ylim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols,)

    ylim = np.broadcast_to(ylim, shape)

    if xlim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(xlim) == (2,):
            xlim = np.array(xlim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols)

    xlim = np.broadcast_to(xlim, shape)
    _, xscale, xlabel = np.broadcast_arrays(axes, xscale, xlabel)
    _, yscale, ylabel = np.broadcast_arrays(axes, yscale, ylabel)

    for idx, ax in np.ndenumerate(axes):
        ax.set(xscale=xscale[idx], xlabel=xlabel[idx], yscale=yscale[idx], ylabel=ylabel[idx])
        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])
        if ylim[idx] is not None:
            ax.set_ylim(ylim[idx])

        if grid is True:
            ax.set_axisbelow(True)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)
            # ax.grid(True, which='major', axis='both', c='0.6', zorder=2, alpha=0.4)
            # ax.grid(True, which='minor', axis='both', c='0.8', zorder=2, alpha=0.4)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes