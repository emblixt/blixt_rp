import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pint
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
import logging
from copy import deepcopy
import bruges.rockphysics.anisotropy as bra

from blixt_rp.plotting.plot_logs import chi_rotation

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.core.well_new as cw
from blixt_rp.core.core import StratUnit, Interval, Intervals
from blixt_rp.core.log_curve_new import LogCurve, _to_depth
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
              rel_widths: list | None = None):
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
    :param rel_widths:
        list of floats determining the relative width of each column
        [1., 2., 1.
    :return:
    """
    # Check that the requested logs exists in the well
    flat_list = [ x for xs in log_columns for x in xs ]
    for _log in flat_list:
        if _log not in well.get_log_names:
            print_info('Log {} does not exist in {}', 'error', raiser='ioerror')

    if scales is None:
        scales = ['linear'] * len(log_columns)
    if rel_widths is None:
        rel_widths = [1.] * len(log_columns)
    if buffer is None:
        buffer = 50.
    if width is None:
        width = 800.
    if height is None:
        height = 1000.

    if isinstance(wi_names, str):
        wi_names = list(wi_names)

    # Gather data that should be plotted in each of the columns
    plot_columns = []
    for _i, _c in enumerate(log_columns):
        plot_columns.append(
            LogColumn(
                'test',
                rel_width=rel_widths[_i],
                lines=[well.get_log_curve(_l).get_line() for _l in _c],
                scale=scales[_i])
        )
    # for i, _c in enumerate(plot_columns):
    #     print(i)
    #     for _l in _c.lines:
    #         print(_l.x_range)
    plotter = LogPlotter(width=800, height=1000, columns = plot_columns)
    return plotter

def get_wiggles_in_depth(
        vp: LogCurve,
        vs: LogCurve,
        rho: LogCurve,
        time_depth_twt: pint.Quantity,
        dt: pint.Quantity | None,
        avo_angles: float | np.ndarray | None = None,
        chi_angles: float | np.ndarray | None = None,
        wavelet=None,
        **kwargs):
    """
    Converts the three input logs to time
    Then calculates the reflectivity
    Then convolves with the wavelet
    And finally converts back to depth

    :param vp:
        LogCurve of Vp data
        Must be in MD domain
    :param vs:
        LogCurve of Vs data
        Must be in MD domain
    :param rho:
        LogCurve of Rho data
        Must be in MD domain
    :param time_depth_twt:
        array like pint Quantity of length N
        The (irregular) twt data for the depth-time relation
    :param dt:
        The sample rate of the resampled log data when converted to TWT
    :param avo_angles:
        float or array like container with incidence angles, of length M [deg]
    :param chi_angles:
        float or array like container with chi angles, of length M [deg]
        If both avo_angles and chi_angels are specified, avo_angles will be used
    :param wavelet:
        dict
        dictionary with three keys:
            'wavelet': contains the wavelet amplitude
            'time': contains the time data [s]
            'header': a dictionary with info about the wavelet
        see blixt_utils.io.io.read_petrel_wavelet() for example
    :return:
       (M, N) size np.ndarray.
       M seismic traces, each of length N
    """
    from blixt_rp.rp.rp_core import reflectivity
    # Get kwargs:
    c_f = kwargs.pop('center_frequency', 30.)  # Hz
    duration = kwargs.pop('duration', 0.512)  # seconds


    # Check input data:
    _length = len(vp)
    for lc in [vp, vs, rho]:
        if len(lc) != _length:
            err_txt = 'Each log curve must have same length: {} != {}'.format(
                _length, len(lc))
            print_info(err_txt, 'error', logger, 'IOError')

    if chi_angles is None and avo_angles is None:
        angles = np.linspace(0, 35, 100)
        eei = False
    elif chi_angles is not None and avo_angles is not None:
        angles = avo_angles
        eei = False
    elif avo_angles is not None:
        angles = avo_angles
        eei = False
    else:
        angles = chi_angles
        eei = True
    m = len(angles) if isinstance(angles, np.ndarray) else 1

    if wavelet is None:
        wavelet = bumw.ricker(duration, dt.to('second').magnitude, c_f)

    # Convert the input logs to TWT domain, with a regular sampling rate
    vp_t = vp.to_twt(time_depth_twt, dt)
    vs_t = vs.to_twt(time_depth_twt, dt)
    rho_t = rho.to_twt(time_depth_twt, dt)

    # calculate reflectivity as a function of incidence or chi angle
    refl_func = reflectivity(vp_t.values, None,
                        vs_t.values, None,
                        rho_t.values, None,
                        eei=eei,
                        along_wiggle=True)

    # Convolve the reflectivity with the wavelet
    amp_t = np.zeros((m, len(vp_t)))
    for _i, _x in enumerate(angles):
        amp_t[_i, :] = bumw.convolve_with_refl(
            wavelet['wavelet'],
            refl_func(_x),
            verbose=False)

    # Convert back to depth domain
    amp = _to_depth(
        time_depth_twt,
        vp_t.depth.depth,
        amp_t)

    return amp
