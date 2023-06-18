"""Plotting module.

Provides convenience methods for generating standard plots and components using `matplotlib`.

"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as DisplayImage


FIGSIZE = 6
FONTSIZE = 13
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2
COLORS = np.array(['#ff3333', '#ff8c1a', '#bfbd2e',  '#5cd699', '#51a6fb', '#a65eed'])
ALPHA = 0.6

MYFONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 65)
SMALLFONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 55)


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



#########################################################
#### PIL Images
#########################################################

def pil_image(imgfiles, debug=True, head=0):
    """ 
    Build a PIL image using the imgs in imgfiles
    First row should include y-axes and bottom row should include x-axes.
    """
    nrows = len(imgfiles)
    ncols = len(imgfiles[0])
    if debug: print(f"{nrows=}, {ncols=}")

    # get side lengths
    corner_img = Image.open(imgfiles[nrows-1, 0])
    middle_img = Image.open(imgfiles[0,1])
    side = middle_img.width
    left = corner_img.width - middle_img.width
    bott = corner_img.height - middle_img.height
    if debug: print(f"{corner_img.size=}, {middle_img.size=}")
    if debug: print(f"{side=}, {left=}, {bott=}, {head=}")

    # make allimage
    allimage=Image.new("RGBA", (ncols*side+left, nrows*side+bott+head))

    # paste mini images
    yy = head
    for rr in range(nrows):
        xx = 0 
        for cc in range(ncols):
            if debug: print(f"{rr=}, {cc=}, {xx=}, {yy=}")
            img = Image.open(imgfiles[rr,cc])
            allimage.paste(img, ((xx, yy)))
            if cc==0: xx += left+side
            else: xx += side
        yy += side
    
    details = {
        'nrows':nrows, 'ncols':ncols, 'side':side, 'left':left, 'bott':bott, 'head':head
    }
    return allimage, details



#########################################################
##### Ionization Fraction PIL Image Functions
#########################################################

def pil_ionfrac_header(allimage, side, leftax, header): 
   # add header
    draw = ImageDraw.Draw(allimage)
    myFont = MYFONT 
    smallFont = SMALLFONT

    mname = "Mass-Weighted"
    # mw, mh = draw.textsize(mname)
    vname = "Volume-Weighted"
    # vw, vh = draw.textsize(mname)
    ename = "Emission-Weighted"
    # ew, eh = draw.textsize(mname)

    t1 = leftax+(1.5*side)
    t2 = t1+3*side
    t3 = t2+3*side

    draw.text((t1,0), mname, font=myFont, fill=(0,0,0), anchor='mt')
    draw.text((t2,0), vname, font=myFont, fill=(0,0,0), anchor='mt')
    draw.text((t3,0), ename, font=myFont, fill=(0,0,0), anchor='mt')

    draw.text((leftax+60,header-60), "1 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+40+side,header-60), "10 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+30+2*side,header-60), "100 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+60+3*side,header-60), "1 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+40+4*side,header-60), "10 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+30+5*side,header-60), "100 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+60+6*side,header-60), "1 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+40+7*side,header-60), "10 km/s", font=smallFont, fill=(26,38,134))
    draw.text((leftax+30+8*side,header-60), "100 km/s", font=smallFont, fill=(26,38,134))
    return allimage


def pil_ionfrac_section_breaks(allimage, side, leftax, header):
    # add section breaks
    line1 = [(leftax+3*side,0), (leftax+3*side, header)]
    line2 = [(leftax+6*side,0), (leftax+6*side, header)]
    draw = ImageDraw.Draw(allimage)
    draw.line(line1,fill=(0,0,0),width=5)
    draw.line(line2,fill=(0,0,0),width=5)
    return allimage


def ionfrac_pil_image(imgfiles, debug=False):
    allimage, details = pil_image(imgfiles, head=150, debug=debug)
    side = details['side']
    left = details['left']
    head = details['head']
    allimage = pil_ionfrac_header(allimage, side, left, head)
    allimage = pil_ionfrac_section_breaks(allimage, side, left, head)
    # allimage = add_mass_labels(allimage, side, left, head, labels, debug=debug)
    return allimage