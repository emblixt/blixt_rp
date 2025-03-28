import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
import logging
from copy import deepcopy
import bruges.rockphysics.anisotropy as bra

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.core.well_new as cw
from blixt_rp.core.core import StratUnit, Interval, Intervals
from blixt_utils.plotting.log_plotter import LogPlotter, LogColumn, Line, seismic_color_map, SeismicTraces
import blixt_utils.utils as uu
import blixt_utils.misc.wavelets as bumw
from blixt_utils.utils import print_info

logger = logging.getLogger(__name__)


def find_nearest(data, value):
    return np.nanargmin(np.abs(data - value))


def plot_logs(well: cw.Well,
              log_columns: list,
              wis: Intervals | None = None,
              wi_names: list | None = None,
              buffer: float | None = None,
              width: int | None = None,
              height: int | None = None,
              scales: list | None = None,
              wavelet=None, **kwargs):
    """
    Attempts to plot logs side by side, or together, in different "columns", utilizing the interactive plotting
    possibilities in bokeh
    :param well:
        well object, new format from well_new.py
    :param log_columns:
        list
        List  of lists that contain the names of the logs to plot in each column
        E.G.
            [['rhob', 'neu'], ['vp', 'vs']]
    :param wis:
        Intervals
        Container of working intervals,
    :param wi_names:
        optional list of strings
        Name of working intervals to use in this plot
        set to None to plot whole well
    :param buffer:
        float
        distance in meters
        Log is plotted from top of working interval - buffer to base of working interval + buffer
        Default is 50 m
    :param width:
    :param height:
    :param scales:
        list of strings determining the x scale of each column
        ['log', 'linear'
    :param wavelet:
        dict
        dictionary with three keys:
            'wavelet': contains the wavelet amplitude
            'time': contains the time data [s]
            'header': a dictionary with info about the wavelet
        see blixt_utils.io.io.read_petrel_wavelet() for example
    :kwargs
        keyword arguments
        backus_length: The Backus averaging length in m
        suffix: string. Added to the default title
    :return:
    """
    # Check that the requested logs exists in the well
    flat_list = [ x for xs in log_columns for x in xs ]
    for _log in flat_list:
        if _log not in well.get_log_names:
            print_info('Log {} does not exist in {}', 'error', raiser='ioerror')

    if scales is None:
        scales = ['linear'] * len(log_columns)
    if buffer is None:
        buffer = 50.
    if width is None:
        width = 800.
    if height is None:
        height = 1000.

    time_step = kwargs.pop('time_step', 0.001)
    c_f = kwargs.pop('center_frequency', 30.)
    duration = kwargs.pop('duration', 0.512)
    scaling = kwargs.pop('scaling', 60.0)
    suffix = kwargs.pop('suffix', '')

    if isinstance(wi_names, str):
        wi_names = list(wi_names)

    # Gather data that should be plotted in each of the columns
    plot_columns = []
    for _i, _c in enumerate(log_columns):
        plot_columns.append(
            LogColumn(
                'test',
                lines=[well.get_log_curve(_l).get_line() for _l in _c],
                scale=scales[_i])
        )
    # for i, _c in enumerate(plot_columns):
    #     print(i)
    #     for _l in _c.lines:
    #         print(_l.x_range)
    plotter = LogPlotter(width=800, height=1000, columns = plot_columns)
    return plotter



