import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import logging

from plotting import crossplot as xp

logger = logging.getLogger(__name__)


def ex_rpt(x, c, **kw):
    return kw.pop('level', 7.)+np.log(min(x)) - np.log(x) + c

def plot_rpt(x, rpt, rpt_keywords, sizes, colors, contours, fig=None, ax=None, **kwargs):
    """
    Plot any RPT (rock physics template) that can be described by a function rpt(x), which can be
    parameterized by a constant. E.G. rpt(x, const=contours[i])

    :param x:
        np.array
        x values used to draw the rockphysics template, preferably less than about 10 items long for creating
        nice plots
    :param rpt:
        function
        Rock physics template function of x
        Should take a second argument which is used to parameterize the function
        e.g.
        def rpt(x, c, **rpt_keywords):
            return c*x + rpt_keywords.pop('zero_crossing', 0)
    :param rpt_keywords:
        dict
        Dictionary with keywords passed on to the rpt function
    :param sizes:
        float or np.array
        determines the size of the markers
        if np.array it must be same size as x
    :param colors
        str or np.array
        determines the colors of the markers
        in np.array it must be same size as x
    :param contours:
        list
        list of constants used to parametrize the rpt function
    """
    #
    # some initial setups
    lw = kwargs.pop('lw', 0.5)
    tc = kwargs.pop('c', 'k')
    edgecolor = kwargs.pop('edgecolor', 'none')
    if sizes is None:
        sizes = 90
    if colors is None:
        colors = 'red'

    #
    # set up plotting environment
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
        ax = fig.subplots()

    # start drawing the RPT
    for const in contours:
        # First draw lines of the RPT
        ax.plot(
            x,
            rpt(x, const, **rpt_keywords),
            lw=lw,
            c=tc,
            label='_nolegend_',
            **kwargs
        )
        # Next draw the points
        ax.scatter(
            x,
            rpt(x, const, **rpt_keywords),
            c=colors,
            s=sizes,
            edgecolor=edgecolor,
            label='_nolegend_',
            **kwargs
        )


if __name__ == '__main__':
    x = np.linspace(2000, 6000, 6)
    contours = [0, 1, 2]
    colors = np.linspace(100, 200, 6)
    sizes = np.ones(6)*100
    plot_rpt(x, ex_rpt, {'level': 0}, sizes, colors, contours)

    plt.show()