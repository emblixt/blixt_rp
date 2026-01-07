import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pint
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
import logging
import inspect
from copy import deepcopy
import bruges.rockphysics.anisotropy as bra
from .. import ureg, Q_
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

from bokeh.models import ColumnDataSource, Span, Select, Button, Slider, CheckboxGroup, Div, CustomJS
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, TextInput
from bokeh.events import RangesUpdate
from bokeh.plotting import show, row, column, figure
from bokeh.io import output_file

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.core.well_new as cw
import blixt_rp.rp.rp_core as rp
from blixt_rp.core.core import StratUnit, Interval, Intervals, LogTable, Template, Cutoffs
from blixt_rp.core.log_curve_new import LogCurve, _to_depth, replace_data
from blixt_rp.plotting.log_plotter import (LogPlotter, LogColumn, Line,  add_strat_table, add_color_amp)
import blixt_utils.utils as uu
import blixt_utils.misc.wavelets as bumw
from blixt_utils.utils import nan_corrcoef, print_info
from blixt_rp.core.seismic import SeismicTraces

logger = logging.getLogger(__name__)

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    CrosshairTool(),
    ResetTool(),
    SaveTool()
]

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
    # TODO Move the following lines to a new class (LineSource) in log_plotter.py
    # To be able to use a cutoff on the well log data, we need to harmonize the logs to have the same
    # length and sample rate
    well.harmonize_logs()

    # Check that the requested logs exists in the well
    # flat_list = [ x for xs in log_columns for x in xs ]
    # for _log in flat_list[:]:
    for _col in log_columns:
        # Iterate over a copy of the list by slicing!
        for _log in _col[:]:
            if _log not in well.get_log_names:
                print_info('Log {} does not exist in {}'.format(_log, well.name), 'warning', logger)
                # flat_list.remove(_log)
                _col.remove(_log)

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
                'column_{}'.format(str(_i)),
                rel_width=rel_widths[_i],
                # TODO First create one (ONE) single source from the well, and then re-use this source in all the
                # TODO different columns, and with a CDSView too
                lines=[well.get_log_curve(_l).get_line() for _l in _c],
                scale=scales[_i])
        )
    # for i, _c in enumerate(plot_columns):
    #     print(i)
    #     for _l in _c.lines:
    #         print(_l.x_range)
    plotter = LogPlotter(width=800, height=1000, columns = plot_columns)
    return plotter

def plot_chi_rotation(well: cw.Well,
                      brine_log_table: LogTable,
                      hc_log_table: LogTable,
                      fluid_log: LogCurve | None = None,
                      litho_log: LogCurve | None = None,
                      wis: Intervals | None = None,
                      width: int | None = None,
                      height: int | None = None,
                      rel_widths: list | None = None):
    """
    Returns the bokeh elements that can be used to build an interactive plotter
    that draws the eei at different chi angles
    Elastic logs Vp, Vs and Rho are needed for both the brine case, and a hc case,
    are needed to evaluate the optimal chi angle

    :param well:
    :param brine_log_table:
        LogTable
        Must contain the log types: 'P velocity', 'S velocity', 'Density'
    :param hc_log_table:
        LogTable
        Must contain the log types: 'P velocity', 'S velocity', 'Density'
    :param fluid_log:
        LogCurve
        A log curve object containing a log that should represent the fluid response (e.g. Soil).
        Used to correlate with the EEI at all Chi angles
    :param litho_log:
        LogCurve
        A log curve object containing a log that should represent the lithology response (e.g. Vsh).
        Used to correlate with the EEI at all Chi angles
    :param wis:
    :param width:
    :param height:
    :param rel_widths:
    :return:
    """
    # TODO 1. check that the logs have the same length and step
    # TODO 2. The center of the Chi line must update after changing zoom or recalculating the EEI
    # TODO 3. Create a Class for this
    # TODO 4. Plot the uncertainty in log AI vs. log GI plot
    # Well must have LogCurve with TWT (or OWT) data, so first check that this exists
    twt = None
    log_types = well.get_log_types
    if 'Two-way time' in log_types:
        twt = well.get_logs_of_type('Two-way time')[0]  # Take the first occurrence
    elif 'One-way time' in log_types:
        twt = well.get_logs_of_type('One-way time')[0].get_twt_from_owt()
    else:
        err_txt = 'No TWT data exists in well {}'.format(well.name)
        print_info(err_txt, 'error', logger, 'IOError')


    # Create a combined log table
    _dict = {'P velocity': [], 'S velocity': [], 'Density': []}
    for _key in list(_dict.keys()):
        for _lt in [brine_log_table, hc_log_table]:
            _dict[_key].append(_lt[_key])
    log_table = LogTable(_dict)

    # Check that the required logs exists in the well
    for _log in log_table.log_names:
        if _log not in well.get_log_names:
            err_txt = 'Log {} does not exist in well {}'.format(_log, well.name)
            print_info(err_txt, 'error', logger, 'IOError')

    if width is None:
        width = 800
    if height is None:
        height = 800

    # Create title
    title = Div(text=well.name, width=int(width), height=10)

    # Create controller widgets
    chi_slider = Slider(title='Chi angle [deg]', start=-90, end=90, step=3, value=23)
    backus_slider = Slider(title='Backus window length [m]', start=0, end=18, step=1, value=5)
    freq_slider = Slider(title='Wavelet central freq. [Hz]', start=10, end=40, step=5, value=20)

    # Create initial data:
    vp_orig = well.get_log_curve(brine_log_table['P velocity'])
    step =  vp_orig.step().magnitude
    depth = vp_orig.depth.values
    vp_brine = vp_orig
    vs_brine = well.get_log_curve(brine_log_table['S velocity'])
    rho_brine = well.get_log_curve(brine_log_table['Density'])
    vp_hc = well.get_log_curve(hc_log_table['P velocity'])
    vs_hc = well.get_log_curve(hc_log_table['S velocity'])
    rho_hc = well.get_log_curve(hc_log_table['Density'])

    chi_angles = np.arange(-90, 91, 1)
    time_depth_twt = twt.take_sampling_from(vp_orig)
    dt = Q_(1, 'millisecond')

    # Find min / max values of the EEI's
    def find_eei_extremes_and_correlations(mask=None):
        _min = 1E6; _max = -1E6; _max_diff = -1E6; _chi_at_max = None
        _cc_litho_b = []; _cc_litho_h = []; _cc_fluid_b = []; _cc_fluid_h = []  # containers for CC
        _f_log = None; _l_log = None
        if mask is None:
            _vp_b = vp_brine.values
            _vs_b = vs_brine.values
            _rho_b = rho_brine.values
            _vp_h = vp_hc.values
            _vs_h = vs_hc.values
            _rho_h = rho_hc.values
            if fluid_log is not None:
                _f_log = fluid_log.values
            if litho_log is not None:
                _l_log = litho_log.values
        else:
            _vp_b = vp_brine.values[mask]
            _vs_b = vs_brine.values[mask]
            _rho_b = rho_brine.values[mask]
            _vp_h = vp_hc.values[mask]
            _vs_h = vs_hc.values[mask]
            _rho_h = rho_hc.values[mask]
            if fluid_log is not None:
                _f_log = fluid_log.values[mask]
                # print('XXX fluid: ', np.nanmin(_f_log), np.nanmax(_f_log))
            if litho_log is not None:
                _l_log = litho_log.values[mask]
                # print('XXX litho: ', np.nanmin(_l_log), np.nanmax(_l_log))

        for _chi in chi_angles:
            _eei_b, _eei_h = calc_eei(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _chi)
            _diff = np.abs(np.nanmedian(_eei_b) - np.nanmedian(_eei_h))
            # if np.nanmin(_eei_b) < _min:
            #     _min = np.nanmin(_eei_b)
            # if np.nanmin(_eei_h) < _min:
            #     _min = np.nanmin(_eei_h)
            # if np.nanmax(_eei_b) > _max:
            #     _max = np.nanmax(_eei_b)
            # if np.nanmax(_eei_h) > _max:
            #     _max = np.nanmax(_eei_h)
            if np.nanstd(_eei_b) > _max:
                 _max = np.nanstd(_eei_b)
            if _diff > _max_diff:
                _max_diff = _diff
                _chi_at_max = _chi
            if fluid_log is not None:
                _cc_fluid_b.append(nan_corrcoef(_eei_b, _f_log)[0, 1])
                _cc_fluid_h.append(nan_corrcoef(_eei_h, _f_log)[0, 1])
            if litho_log is not None:
                _cc_litho_b.append(nan_corrcoef(_eei_b, _l_log)[0, 1])
                _cc_litho_h.append(nan_corrcoef(_eei_h, _l_log)[0, 1])
        # return _min, _max, _chi_at_max, _cc_litho_b, _cc_litho_h, _cc_fluid_b, _cc_fluid_h
        return (np.nanmean(_eei_b) - 0.5 * _max, np.nanmean(_eei_b) + 0.5 * _max,
                _chi_at_max, _cc_litho_b, _cc_litho_h, _cc_fluid_b, _cc_fluid_h)

    eei_min, eei_max, chi_at_max_diff, cc_litho_b, cc_litho_h, cc_fluid_b, cc_fluid_h = (
        find_eei_extremes_and_correlations())
    cc_fluid_cds = ColumnDataSource(dict(x=chi_angles, cc_b=cc_fluid_b, cc_h=cc_fluid_h))
    cc_litho_cds = ColumnDataSource(dict(x=chi_angles, cc_b=cc_litho_b, cc_h=cc_litho_h))

    def create_lines_from_line_cds(_line_cds):
        _line_brine = Line(x='eei_brine', y='md', cds=_line_cds, style=Template(**{
            'name': 'EEI brine, chi: {}, ba: {}'.format(chi_slider.value, backus_slider.value),
            'line_color': 'blue', 'min': 1.2 * eei_min, 'max': 0.8 * eei_max
        }))
        _line_hc = Line(x='eei_hc', y='md', cds=_line_cds, style=Template(**{
            'name': 'EEI hc, chi: {}, ba: {}'.format(chi_slider.value, backus_slider.value),
            'line_color': 'red', 'min': 1.2 * eei_min, 'max': 0.8 * eei_max
        }))
        _line_diff = Line(x='eei_diff', y='md', cds=_line_cds, style=Template(**{
            'name': 'DIFF., chi: {}, ba: {}'.format(chi_slider.value, backus_slider.value),
            'line_color': 'black', 'min': -0.1, 'max': 0.3
        }))
        return _line_brine, _line_hc, _line_diff

    line_cds = ColumnDataSource(
        calc_all(
            vp_brine.values, vs_brine.values, rho_brine.values,
            vp_hc.values, vs_hc.values, rho_hc.values,
            depth, backus_slider.value, step, chi_slider.value
        )
    )
    line_brine, line_hc, line_diff = create_lines_from_line_cds(line_cds)

    # Calculate synthetic EEI seismic
    def get_amp_cdss(_ba, _freq):
        if _ba > 0:
            _vp_b_ba, _vs_b_ba, _rho_b_ba, _vp_h_ba, _vs_h_ba, _rho_h_ba = calc_backus_avg(
                vp_brine.values, vs_brine.values, rho_brine.values,
                vp_hc.values, vs_hc.values, rho_hc.values,
                _ba, step)
        else:
            _vp_b_ba, _vs_b_ba, _rho_b_ba, _vp_h_ba, _vs_h_ba, _rho_h_ba = (
                vp_brine.values, vs_brine.values, rho_brine.values, vp_hc.values, vs_hc.values, rho_hc.values)

        # Create LogCurves that contain the backus averaged elastic logs
        _vp_brine_ba = replace_data(vp_brine, Q_(_vp_b_ba, vp_brine.units), vp_brine.depth, False,
                                    'backus {}m'.format(_ba), suffix='ba')
        _vs_brine_ba = replace_data(vs_brine, Q_(_vs_b_ba, vs_brine.units), vs_brine.depth, False,
                                    'backus {}m'.format(_ba), suffix='ba')
        _rho_brine_ba = replace_data(rho_brine, Q_(_rho_b_ba, rho_brine.units), rho_brine.depth, False,
                                     'backus {}m'.format(_ba), suffix='ba')
        _vp_hc_ba = replace_data(vp_hc, Q_(_vp_h_ba, vp_hc.units), vp_hc.depth, False,
                                 'backus {}m'.format(_ba), suffix='ba')
        _vs_hc_ba = replace_data(vs_hc, Q_(_vs_h_ba, vs_hc.units), vs_hc.depth, False,
                                 'backus {}m'.format(_ba), suffix='ba')
        _rho_hc_ba = replace_data(rho_hc, Q_(_rho_h_ba, rho_hc.units), rho_hc.depth, False,
                                  'backus {}m'.format(_ba), suffix='ba')

        _amp_brine = get_wiggles_in_depth(
            _vp_brine_ba,
            _vs_brine_ba,
            _rho_brine_ba,
            time_depth_twt.data,
            dt,
            chi_angles=chi_angles,
            center_frequency=_freq
        )
        # _amp_brine_cds = ColumnDataSource({'value': [_amp_brine.T]})
        _amp_hc = get_wiggles_in_depth(
            _vp_hc_ba,
            _vs_hc_ba,
            _rho_hc_ba,
            time_depth_twt.data,
            dt,
            chi_angles=chi_angles,
            center_frequency = _freq
        )
        return _amp_brine, _amp_hc

    amp_brine, amp_hc = get_amp_cdss(backus_slider.value, freq_slider.value)

    amp_hc_cds = ColumnDataSource({'value': [amp_hc.T]})
    amp_diff_cds = ColumnDataSource({'value': [(amp_brine - amp_hc).T]})  # TESTING

    def create_traces(_amp_brine_cds, _amp_hc_cds):
        _seismic_traces_brine = SeismicTraces(
            x=chi_angles, y=vp_brine.depth.values,
            traces=None, cds=_amp_brine_cds,
            trace_type='chi')
        _seismic_traces_hc = SeismicTraces(
            x=chi_angles, y=vp_brine.depth.values,
            traces=None, cds=_amp_hc_cds,
            trace_type='chi')
        return _seismic_traces_brine, _seismic_traces_hc

    # seismic_traces_brine, seismic_traces_hc = create_traces(amp_brine_cds, amp_hc_cds)
    seismic_traces_diff, seismic_traces_hc = create_traces(amp_diff_cds, amp_hc_cds)

    def spans(_chi):
        _span = Span(
            location=_chi, dimension='height',
            line_width=0.5,
            line_dash='dashed',
            line_color='black'
        )
        return _span

    _spans = spans(chi_slider.value)

    # Create plots
    xplot = figure(width=int(width -200), height=int(height-200), tools=tools)  # Intercept vs. Gradient crossplot
    xplot.toolbar.logo = None
    hplot = figure(width=int(width - 200), height=200, tools=[PanTool(), WheelZoomTool()])  # Histogram
    hplot.toolbar.logo = None

    if fluid_log:
        t_f = 'Cross correlation with {}: {}'.format(fluid_log.log_type, fluid_log.name)
    else:
        t_f = 'No fluid log to correlate with'
    if litho_log:
        t_l = 'Cross correlation with {}: {}'.format(litho_log.log_type, litho_log.name)
    else:
        t_l = 'No lithology log to correlate with'
    cc_fluid_fig = figure(title=t_f, width=int(width - 200), height=150, tools=[PanTool(), WheelZoomTool()])   # cross correlation
    cc_fluid_fig.toolbar.logo = None
    cc_fluid_fig.add_layout(_spans)
    cc_litho_fig = figure(title=t_l, width=int(width - 200), height=150, tools=[PanTool(), WheelZoomTool()])   # cross correlation
    cc_litho_fig.toolbar.logo = None
    cc_litho_fig.add_layout(_spans)

    plotter = LogPlotter(width=width, height=height)
    c1 = LogColumn('EEI', lines=[line_brine, line_hc], rel_width=1)
    c2 = LogColumn('WIS', lines=[], rel_width=0.3)
    compare_lines = []
    if litho_log is not None:
        compare_lines.append(litho_log.get_line())
    if fluid_log is not None:
        compare_lines.append(fluid_log.get_line())
    if len(compare_lines) == 0:
        c3 = LogColumn('DIFF', lines=[line_diff], rel_width=0.7)
    else:
        c3 = LogColumn('COMPARE', lines=compare_lines, rel_width=0.7)

    # c3 = LogColumn('EEI_BRINE', seismic_traces=seismic_traces_brine, rel_width=1)
    c4 = LogColumn('EEI_HC', seismic_traces=seismic_traces_hc, rel_width=1)
    c5 = LogColumn('EEI_DIFF', seismic_traces=seismic_traces_diff, rel_width=1)
    plotter.columns = [c1, c2, c3, c4, c5]
    grid = plotter.figure()
    # grid.children[2][0].add_layout(_spans)
    grid.children[3][0].add_layout(_spans)
    grid.children[4][0].add_layout(_spans)
    # Fake some data to add a legend
    grid.children[3][0].scatter([0], [0], fill_color=None, line_color=None, size=0,
                                legend_label='HC, ba: {}, Hz: {}'.format(backus_slider.value, freq_slider.value))
    grid.children[4][0].scatter([0], [0], fill_color=None, line_color=None, size=0,
                                legend_label='Brine - HC, ba: {}, Hz: {}'.format(backus_slider.value, freq_slider.value))
    def calc_xplot_cds_dict(_ba, _chi, _mask):
        return calc_all(
                vp_brine.values[_mask], vs_brine.values[_mask], rho_brine.values[_mask],
                vp_hc.values[_mask], vs_hc.values[_mask], rho_hc.values[_mask],
                depth[_mask], _ba, step, _chi
            )
    mask = np.array(np.ones(len(depth)), bool)
    xplot_cds = ColumnDataSource(calc_xplot_cds_dict(
        backus_slider.value, chi_slider.value, mask
    ))

    # xplot.scatter(
    #     x='int_brine', y='grad_brine', source=xplot_cds, color='blue', marker='circle',
    #     legend_label='Brine, ba: {}'.format(backus_slider.value), alpha=0.7
    # )
    # xplot.scatter(
    #     x='int_hc', y='grad_hc', source=xplot_cds, color='red', marker='circle',
    #     legend_label='HC, ba: {}'.format(backus_slider.value), alpha=0.7
    # )
    # y_range = np.max([np.abs(np.nanmin(xplot_cds.data['grad_hc'])), np.abs(np.nanmax(xplot_cds.data['grad_hc']))])
    # x_range = np.max([np.abs(np.nanmin(xplot_cds.data['int_hc'])), np.abs(np.nanmax(xplot_cds.data['int_hc']))])
    xplot.scatter(
        x='log_ai_b', y='log_gi_b', source=xplot_cds, color='blue', marker='circle',
        legend_label='Brine, ba: {}'.format(backus_slider.value), alpha=0.7
    )
    xplot.scatter(
        x='log_ai_h', y='log_gi_h', source=xplot_cds, color='red', marker='circle',
        legend_label='HC, ba: {}'.format(backus_slider.value), alpha=0.7
    )

    # x_r = xplot.x_range
    # y_r = xplot.~
    y_range = 0.5 * (np.nanmax(xplot_cds.data['log_gi_h']) - np.nanmin(xplot_cds.data['log_gi_h']))
    x_center = float(np.nanmedian(np.append(xplot_cds.data['log_ai_h'], xplot_cds.data['log_ai_b'])))
    y_center = float(np.nanmedian(np.append(xplot_cds.data['log_gi_h'], xplot_cds.data['log_gi_b'])))
    # y_range = np.max([np.abs(np.nanmin(xplot_cds.data['log_gi_h'])), np.abs(np.nanmax(xplot_cds.data['log_gi_h']))])
    # x_range = np.max([np.abs(np.nanmin(xplot_cds.data['log_ai_h'])), np.abs(np.nanmax(xplot_cds.data['log_ai_h']))])
    x_start = 0.8 * np.nanmin(xplot_cds.data['log_ai_h'])
    x_end = 1.2 * np.nanmax(xplot_cds.data['log_ai_h'])
    y_start = 0.8 * np.nanmin(xplot_cds.data['log_gi_h'])
    y_end = 1.2 * np.nanmax(xplot_cds.data['log_gi_h'])
    bins = np.linspace(1.2 * eei_min, 0.8 * eei_max, 100)
    # bins = 40
    _x, _y = return_chi_line(y_range, chi_slider.value, x_center=x_center, y_center=y_center)
    chi_line_cds = ColumnDataSource(dict(x=_x, y=_y))
    xplot.line(x='x', y='y', source=chi_line_cds)

    def draw_histogram(_eei_brine, _eei_hc):
        hplot.legend.items = []
        hplot.renderers = []
        _x = _eei_brine[~np.isnan(_eei_brine)]
        hist, edges = np.histogram(_x, density=True, bins=bins)
        hplot.quad(top=hist, bottom=0, fill_color='blue', alpha=0.7, line_color='white',
                   left=edges[:-1], right=edges[1:], legend_label='Brine, ba: {}'.format(backus_slider.value))
        _x = _eei_hc[~np.isnan(_eei_hc)]
        hist, edges = np.histogram(_x, density=True, bins=bins)
        hplot.quad(top=hist, bottom=0, fill_color='red', alpha=0.7, line_color='white',
                   left=edges[:-1], right=edges[1:], legend_label='HC, ba: {}'.format(backus_slider.value))

    draw_histogram(xplot_cds.data['eei_brine'], xplot_cds.data['eei_hc'])

    def draw_cc():
        if fluid_log is not None:
            cc_fluid_fig.line(x='x', y='cc_b', source=cc_fluid_cds, color='blue', legend_label='CC brine')
            cc_fluid_fig.line(x='x', y='cc_h', source=cc_fluid_cds, color='red', legend_label='CC hc')
            cc_fluid_fig.x_range.start = -90; cc_fluid_fig.x_range.end = 90
            cc_fluid_fig.y_range.start = -0.7; cc_fluid_fig.y_range.end = 0.7
        if litho_log is not None:
            cc_litho_fig.line(x='x', y='cc_b', source=cc_litho_cds, color='blue', legend_label='CC brine')
            cc_litho_fig.line(x='x', y='cc_h', source=cc_litho_cds, color='red', legend_label='CC hc')
            cc_litho_fig.x_range.start = -90; cc_litho_fig.x_range.end = 90
            cc_litho_fig.y_range.start = -0.7; cc_litho_fig.y_range.end = 0.7
        cc_litho_fig.xaxis.axis_label = 'Chi angle [deg]'

    draw_cc()

    data_table = add_strat_table(grid, stratigraphy=wis.get_intervals_dict(well.name), width=width, column_index=1)

    grid.children[0][0].legend.click_policy = 'hide'
    # xplot.xaxis.axis_label = 'Intercept'
    # xplot.yaxis.axis_label = 'Gradient'
    xplot.xaxis.axis_label = 'log AI'
    xplot.yaxis.axis_label = 'log GI'
    xplot.legend.title = 'Chi: {}'.format(chi_slider.value)
    hplot.legend.title = 'Chi: {}'.format(chi_slider.value)
    # xplot.x_range = x_r  # this did not help, the range kept updating
    # xplot.y_range = y_r
    xplot.y_range.start = y_start
    xplot.y_range.end = y_end
    xplot.x_range.start = x_start
    xplot.x_range.end = x_end
    hplot.xaxis.axis_label = 'EEI'
    xplot.legend.click_policy = 'hide'
    hplot.legend.click_policy = 'hide'

    # Call back functions
    def chi_callback(attr, old, new):
        line_cds.data = calc_all(
            vp_brine.values, vs_brine.values, rho_brine.values,
            vp_hc.values, vs_hc.values, rho_hc.values,
            depth, backus_slider.value, step, new
        )

        _mask = mask_based_on_depth(vp_orig.depth.values,
                                   grid.children[0][0].y_range.end, grid.children[0][0].y_range.start)

        _xplot_dict = calc_xplot_cds_dict(backus_slider.value, new, _mask)
        xplot_cds.data = _xplot_dict

        draw_histogram(_xplot_dict['eei_brine'], _xplot_dict['eei_hc'])

        # _y_range = 0.5 * (np.nanmax(xplot_cds.data['log_gi_h']) - np.nanmin(xplot_cds.data['log_gi_h']))
        _x_center = float(np.nanmedian(np.append(xplot_cds.data['log_ai_h'], xplot_cds.data['log_ai_b'])))
        _y_center = float(np.nanmedian(np.append(xplot_cds.data['log_gi_h'], xplot_cds.data['log_gi_b'])))
        _x, _y = return_chi_line(
            y_range, new, x_center=_x_center, y_center=_y_center)
        chi_line_cds.data = dict(x=_x, y=_y)
        xplot.x_range.start = np.nanmin(xplot_cds.data['log_ai_h'])
        xplot.x_range.end = np.nanmax(xplot_cds.data['log_ai_h'])
        xplot.y_range.start = np.nanmin(xplot_cds.data['log_gi_h'])
        xplot.y_range.end = np.nanmax(xplot_cds.data['log_gi_h'])

        _eei_min, _eei_max, _chi_at_max_diff, _cc_litho_b, _cc_litho_h, _cc_fluid_b, _cc_fluid_h = (
            find_eei_extremes_and_correlations(_mask))
        cc_fluid_cds.data = dict(x=chi_angles, cc_b=_cc_fluid_b, cc_h=_cc_fluid_h)
        cc_litho_cds.data = dict(x=chi_angles, cc_b=_cc_litho_b, cc_h=_cc_litho_h)

    def backus_callback(attr, old, new):
        _amp_brine, _amp_hc = get_amp_cdss(new, freq_slider.value)
        amp_hc_cds.data['value'] = [_amp_hc.T]
        amp_diff_cds.data['value'] = [(_amp_brine - _amp_hc).T]

        line_cds.data = calc_all(
            vp_brine.values, vs_brine.values, rho_brine.values,
            vp_hc.values, vs_hc.values, rho_hc.values,
            depth, new, step, chi_slider.value
        )

    def freq_callback(attr, old, new):
        _amp_brine, _amp_hc = get_amp_cdss(chi_slider.value, new)
        amp_hc_cds.data['value'] = [_amp_hc.T]
        amp_diff_cds.data['value'] = [(_amp_brine - _amp_hc).T]

    # Call backs
    legend_callback = CustomJS(args=dict(chi=chi_slider, ba=backus_slider, freq=freq_slider, spans=_spans,
                                         c1_legend=grid.children[0][0].legend[0],
                                         c3_legend=grid.children[2][0].legend[0],
                                         c4_legend=grid.children[3][0].legend[0],
                                         c5_legend=grid.children[4][0].legend[0],
                                         hplot_legend=hplot.legend[0],
                                         xplot_legend=xplot.legend[0]), code=legend_code)

    chi_slider.on_change('value_throttled', chi_callback)
    backus_slider.on_change('value_throttled', backus_callback)
    freq_slider.on_change('value_throttled', freq_callback)
    chi_slider.js_on_change('value', legend_callback)
    backus_slider.js_on_change('value', legend_callback)
    freq_slider.js_on_change('value', legend_callback)

    # # Test callback from change in Zoom:
    # grid.children[0][0].js_on_event(RangesUpdate,
    #                                 CustomJS(args=dict(p=grid.children[0][0]),
    #                                          code='console.log("Range update " + p.y_range.start)'))

    return title, grid,chi_slider, backus_slider, freq_slider, data_table, xplot, hplot, cc_fluid_fig, cc_litho_fig

def compare_synth_with_seismic(
        well: cw.Well,
        ref_log_table: LogTable,
        vp_vs_rho_table: LogTable,
        seismic_traces: SeismicTraces,
        wis: Intervals | None = None,
        width: int | None = None,
        height: int | None = None):
    """
    Plots the synthetic seismic based on the elastic logs in vp_vs_rho_table along with the real seismic (seismic_traces)
    for sake of comparison
    :param well:
        Well object
    :param ref_log_table:
        LogTable with some well logs useful for comparison / reference
    :param vp_vs_rho_table:
        LogTable with Vp, Vs and Rho data that are used to calculate the synthetics
    :param seismic_traces:
        SeismicTraces object with real seismic data
    :param wis:
        Intervals
        If provided the working intervals in wis will be plotted
    :param width:
    :param height:
    :return:
    """
    if width is None:
        width = 800.
    if height is None:
        height = 800.

    # Well must have LogCurve with TWT (or OWT) data, so first check that this exists
    twt = None
    log_types = well.get_log_types
    if 'Two-way time' in log_types:
        twt = well.get_logs_of_type('Two-way time')[0]  # Take the first occurrence
    elif 'One-way time' in log_types:
        twt = well.get_logs_of_type('One-way time')[0].get_twt_from_owt()
    else:
        err_txt = 'No TWT data exists in well {}'.format(well.name)
        print_info(err_txt, 'error', logger, 'IOError')

    # Check that the required logs exists in the well
    for _log in (ref_log_table.log_names + vp_vs_rho_table.log_names):
        if _log not in well.get_log_names:
            err_txt = 'Log {} does not exist in well {}'.format(_log, well.name)
            print_info(err_txt, 'error', logger, 'IOError')

    # Create controller widgets
    backus_slider = Slider(title='Backus window length [m]', start=0, end=20, step=2, value=0, width=int(width/3.) - 5)
    freq_slider = Slider(title='Wavelet central freq. [Hz]', start=10, end=40, step=5, value=20, width=int(width/3.) - 5)
    shift_slider = Slider(title='Shift in MD [m]', start=-40, end=40, step=2, value=0,  width=int(width/3.) - 5)

    # Create initial data:
    vp_orig = well.get_log_curve(vp_vs_rho_table['P velocity'])
    step =  vp_orig.step().magnitude
    depth = vp_orig.depth.values
    vs_orig = well.get_log_curve(vp_vs_rho_table['S velocity'])
    rho_orig = well.get_log_curve(vp_vs_rho_table['Density'])
    ai_orig = vp_orig.values * rho_orig.values
    ai_min = np.nanmin(ai_orig)
    ai_max = np.nanmax(ai_orig)
    time_depth_twt = twt.take_sampling_from(vp_orig)
    dt = Q_(1, 'millisecond')
    avo_angles = np.arange(seismic_traces.x[0], seismic_traces.x[-1], 1)

    def create_ai_line(_line_cds):
        return Line(
            x='ai_ba', y='md', cds=_line_cds, style=Template(**{
                'name': 'AI ba: {}m'.format(backus_slider.value),
                'line_color': 'blue', 'min': ai_min, 'max': ai_max
            })
        )
    def calc_ai(_vp, _vs, _rho, _depth, _ba, _step):
        if _ba == 0.:
            _vp_ba = vp_orig.values
            _vs_ba = vs_orig.values
            _rho_ba = rho_orig.values
        else:
            _vp_ba, _vs_ba, _rho_ba, _, _, _ = calc_backus_avg(
                _vp, _vs, _rho,
                _vp, _vs, _rho,
                _ba, _step)
        return dict(
            ai_ba=_vp_ba * _rho_ba,
            vp=_vp_ba,
            vs=_vs_ba,
            rho=_rho_ba,
            md=_depth
        )

    line_cds = ColumnDataSource(
        calc_ai(
            vp_orig.values, vs_orig.values, rho_orig.values,
            depth, backus_slider.value, step
        )
    )

    line_ai = create_ai_line(line_cds)

    # Calculate synthetic seismic
    def get_amp_cds(_ba, _freq):
        if _ba == 0.:
            _vp_b_ba = vp_orig.values
            _vs_b_ba = vs_orig.values
            _rho_b_ba = rho_orig.values
        else:
            _vp_b_ba, _vs_b_ba, _rho_b_ba, _, _, _ = calc_backus_avg(
                vp_orig.values, vs_orig.values, rho_orig.values,
                vp_orig.values, vs_orig.values, rho_orig.values,
                _ba, step)

        # Create LogCurves that contain the backus averaged elastic logs
        _vp_ba = replace_data(vp_orig, Q_(_vp_b_ba, vp_orig.units), vp_orig.depth, False,
                                    'backus {}m'.format(_ba), suffix='ba')
        _vs_ba = replace_data(vs_orig, Q_(_vs_b_ba, vs_orig.units), vs_orig.depth, False,
                                    'backus {}m'.format(_ba), suffix='ba')
        _rho_ba = replace_data(rho_orig, Q_(_rho_b_ba, rho_orig.units), rho_orig.depth, False,
                                     'backus {}m'.format(_ba), suffix='ba')

        _amp = get_wiggles_in_depth(_vp_ba, _vs_ba, _rho_ba,
                                    time_depth_twt.data, dt, avo_angles=avo_angles,
                                    center_frequency=_freq)
        return _amp

    amp = get_amp_cds(backus_slider.value, freq_slider.value)
    amp_cds = ColumnDataSource({'value': [amp.T]})

    def create_synth_traces(_amp_cds, _ba, _freq):
        _seismic_traces = SeismicTraces(
            x=avo_angles, y=depth,
            traces=None, cds=_amp_cds,
            trace_type='avo',
            title='Synth. ba: {}m, f: {}Hz'.format(_ba, _freq)
        )
        return _seismic_traces

    synth_traces = create_synth_traces(amp_cds, backus_slider.value, freq_slider.value)

    # This is necessary to get the cds of the real seismic, so that we can shift it later
    def create_real_traces(_seismic_traces, _cds):
        _seismic_traces = SeismicTraces(
            x=_seismic_traces.x, y=_seismic_traces.y,
            traces=None, cds=_cds,
            trace_type=_seismic_traces.trace_type,
            title=_seismic_traces._title
        )
        return _seismic_traces

    orig_seismic_values = deepcopy(seismic_traces.cds.data['value'][0])
    data_cds = seismic_traces.cds
    real_traces = create_real_traces(seismic_traces, data_cds)


    # create a grid plot
    plotter = LogPlotter(width=width, height=height)
    c1 = LogColumn('REF', lines=[well.get_log_curve(_x).get_line() for _x in ref_log_table.log_names], rel_width=1)
    if wis is not None:
        c2 = LogColumn('WIS', lines=[], rel_width=0.3)
    else:
        c2 = LogColumn('WIS', lines=[], rel_width=0)
    c3 = LogColumn('AI', lines=[line_ai], rel_width=1)
    c4 = LogColumn('SYNTH', seismic_traces=synth_traces, rel_width=1)
    # c5 = LogColumn('SEISMIC', seismic_traces=seismic_traces, rel_width=1)
    c5 = LogColumn('SEISMIC', seismic_traces=real_traces, rel_width=1)
    plotter.columns = [c1, c2, c3, c4, c5]
    grid = plotter.figure()
    # Fake some data to add a legend to the real seismic
    grid.children[4][0].scatter([0], [0], fill_color=None, line_color=None, size=0,
                                legend_label='Shift = {}m MD'.format(shift_slider.value))
    grid.children[4][0].legend.label_text_font_size = '10px'
    for ax in [grid.children[_i][0] for _i in [3, 4]]:
        ax.x_range.start = avo_angles[0]
        ax.x_range.end = avo_angles[-1]

    # Add possibility to modify colors on the real seismic
    color_amp_factor = add_color_amp(grid.children[4][0])

    if wis is not None:
        data_table = add_strat_table(grid, stratigraphy=wis.get_intervals_dict(well.name), column_index=1)
    else:
        data_table = Div(text='', width=10, height=10)

    # call back functions
    def backus_callback(attr, old, new):
        _amp = get_amp_cds(new, freq_slider.value)
        amp_cds.data['value'] = [_amp.T]
        # print(grid.children[2][0].xaxis[0].axis_label)

        line_cds.data = calc_ai(
            vp_orig.values, vs_orig.values, rho_orig.values,
            depth, new, step
        )

    def freq_callback(attr, old, new):
        _amp = get_amp_cds(backus_slider.value, new)
        amp_cds.data['value'] = [_amp.T]

    def shift_callback(attr, old, new):
        _shifted_data = np.roll(orig_seismic_values, int(new/step), 0)
        data_cds.data['value'] = [_shifted_data]

    legend_callback = CustomJS(args=dict(ba=backus_slider, freq=freq_slider, shift=shift_slider,
                                         c3_axis=grid.children[2][0].xaxis[0],
                                         c3_legend=grid.children[2][0].legend[0],
                                         c4_axis=grid.children[3][0].xaxis[0],
                                         c5_legend=grid.children[4][0].legend[0]),
                               code="""
                                const _ba = ba.value;
                                const _freq = freq.value;
                                const _shift = shift.value;
                                c3_axis.axis_label = 'AI ba:' + _ba + 'm';
                                c3_legend.items[0].label.value = 'AI ba:' + _ba + 'm';
                                //c4_legend.items[0].label.value = 'Synth. ba: ' + _ba + 'm, f: ' + _freq + 'Hz: avo';
                                c4_axis.axis_label = 'Synth. ba: ' + _ba + 'm, f: ' + _freq + 'Hz: avo';
                                c5_legend.items[0].label.value = 'Shift = ' + _shift + 'm MD'
                                console.log('Test: ', _ba); 
                               """)
    backus_slider.on_change('value_throttled', backus_callback)
    freq_slider.on_change('value_throttled', freq_callback)
    shift_slider.on_change('value_throttled', shift_callback)
    backus_slider.js_on_change('value', legend_callback)
    freq_slider.js_on_change('value', legend_callback)
    shift_slider.js_on_change('value', legend_callback)

    return grid, backus_slider, freq_slider, shift_slider, data_table, color_amp_factor


def interactive_edits(well: cw.Well,
                      log_columns: list,
                      wis: Intervals | None = None):
    # This script is called  by the main.py script under blixt_projects/bokeh_testing
    # And can be invoked by calling:
    # C:\Users\emb\Documents\PycharmProjects\blixt_projects>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show bokeh_testing
    """
    Allow the user to apply some predefined smoothing and spike removal methods on the given well.
    Backus averaging is not one of smoothing methods, as it should be applied elsewhere (an example of how it is implemented
    can be found in the test_interactive_edit() test in test_plot_logs.py).
    NB The smoothing done here should be applied on logs other that tne Vp, Vs, and Rho logs, and preferably after
    Backus averaging has been applied on those. This is because it is wise to have a similar "smoothness" on both the
    elastic logs as on other logs (e.g Vsh and Porosity) so that we don't introduce false information

    The smoothing is done on all logs in the plot, but in the case of multiple logs in a Column, only the first
    smoothened result is shown

    All smoothened logs are saved in the output las file.

    Remember to press "Apply" before saving

    :param well:
        well object, new format from well_new.py
    :param log_columns:
        list
        List  of lists that contain the names of the logs to plot in each column
        E.G.
            [['rhob', 'neu'], ['vp', 'vs']]
    :param wis:
        Intervals
    :return:
    """

    select_editors = CheckboxGroup(
        labels=['Fill gaps', 'Remove spikes', 'Smooth'],
        active=[2])
    post_fix = TextInput(value='edit', title='Log suffix')
    despike_clip_sel = Select(title='Clip level', value='moderate', options= ['low', 'moderate', 'high'],
                              description=' Lower clip level and longer window length clips away more data')
    despike_window_len = Slider(title='Despike window length [m]', start=1, end=20, step=1, value=2)
    # backus_window_len = Slider(title='Backus window length [m]', start=2, end=18, step=1, value=5)
    smooth_method_sel = Select(title='Smoothing method', value='convolution', options= ['convolution', 'median'])
    smooth_window_sel = Select(title='Smoothing window type', value='hanning',
                               options=['flat', 'hanning', 'hamming', 'bartlett', 'blackman'])
    smooth_window_len = Slider(title='Smoothing window length [m]', start=5, end=100, step=5, value=11)
    run_smoothing = Button(label='Apply edits', button_type='success')
    save_result = Button(label='Save to .las file', button_type='success')

    # Create a column plot with the requested logs
    plotter = plot_logs(well, log_columns)

    # add the smoothened data to each column
    line_cdss = []
    for _i, _column in enumerate(plotter.columns):
        # print(_column)
        # Instead of plotting the smoothened version of each log, we only show the first
        # Basically the same as: _log_name = log_columns[_i][0]
        for _j, _log_name in enumerate(log_columns[_i][0:1]):
            _this_smooth_res = well.get_log_curve(_log_name).smooth(
                window_len=smooth_window_len.value,
                method=smooth_method_sel.value,
                window=smooth_window_sel.value
            )
            # Because each log in this column is added with same name ('smooth') to the ColumnDataSource
            # only one will be plotted
            _this_cds = ColumnDataSource(dict(smooth=_this_smooth_res.values, md=_this_smooth_res.depth.values))
            _this_style = _this_smooth_res.style
            _this_style.line_width = 3.
            _this_style.line_color = 'red'
            _this_style.name = _this_style.name + '_smooth'
            line_cdss.append(_this_cds)
            _column.add_line(Line(x='smooth', y='md', source=_this_cds, style=_this_style))

    def callback():
        _step = None
        for _i, _column in enumerate(plotter.columns):
            # and to each log within each column
            # _log_name = log_columns[_i][0]
            for _j, _log_name in enumerate(log_columns[_i]):
                _this_smooth_res = well.get_log_curve(_log_name).copy(post_fix.value)
                if _step is None:
                    _step = _this_smooth_res.step().magnitude
                _this_step = _this_smooth_res.step().magnitude
                if _this_step != _step:
                    raise IOError('Current depth step ({}) is different from last ({})'.format(_this_step, _step))
                if select_editors.active.count(0) > 0:  # Fill gaps
                    print('Fill gaps')
                    _this_smooth_res = _this_smooth_res.fill_gaps()
                if select_editors.active.count(1) > 0:  # Remove spikes
                    print('Remove spikes at {} level'.format(despike_clip_sel.value))
                    level = _this_smooth_res.std
                    if despike_clip_sel.value == 'low':
                        level = 0.1 * level
                    elif despike_clip_sel.value == 'moderate':
                        level = 0.5 * level
                    elif despike_clip_sel.value == 'high':
                        level = 1.0 * level
                    # print('XXX1', level, despike_window_len.value)
                    _this_smooth_res = _this_smooth_res.despike(max_clip=level, window_len=despike_window_len.value)
                if select_editors.active.count(2) > 0:  # smooth data
                    # _this_smooth_res = well.get_log_curve(_log_name).smooth(
                    _this_smooth_res = _this_smooth_res.smooth(
                        window_len=smooth_window_len.value,
                        method=smooth_method_sel.value,
                        window=smooth_window_sel.value
                    )
                line_cdss[_i].data['smooth'] = _this_smooth_res.values
                well.add_log(_this_smooth_res, if_log_exists='overwrite')
        # l_names = well.get_log_names
        # print(l_names)
        # print([well.get_log_curve(_name).units for _name in l_names])
        # print(well.get_log_curve(l_names[-1]).header)

    def save():
        file_name = save_as_las()
        well.write_las(file_name, overwrite=True)

    run_smoothing.on_click(callback)
    save_result.on_click(save)
    # "Realize" the figure
    grid = plotter.figure(title=well.name)
    if wis is not None:
        data_table = add_strat_table(grid, stratigraphy=wis.get_intervals_dict(well.name), column_index=None)
    else:
        data_table = Div(text='', width=10, height=10)

    # show(grid)
    return (grid, select_editors, despike_clip_sel, despike_window_len,
            smooth_method_sel, smooth_window_sel, smooth_window_len, post_fix, run_smoothing, save_result, data_table)


def plot_trends(x_name: str, wells: list, log_table: LogTable, wis:Intervals, wi_name: str,  cutoffs: Cutoffs,
                results_folder: str | None = None, verbose: bool =True, suffix: str | None = None,
                de_trend_loc: float | None = None, de_trend_scale: float | None = None,
                skip: list | None = None, **kwargs):
    """
    Plots the depth trends (TVD) and performs an optional de-trending for each individual log within the given working interval, for all wells

    :param x_name:
        name of the log which represents the independent variable (typically TVD_ML (below mudline) for depth trends)
    :param wells:
        list of well_new objects
    :param log_table:
        Logtable object with the logs we will trend analyze
    :param wis:
    :param wi_name:
        Name of the working interval we are calculate trend data on
    :param cutoffs:
       Cutoffs Object with CutoffRule's
       E.G. {'depth': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
    :param results_folder:
        str
        full pathname of folder where results plots should be saved.
        If verbose is False, no plots are saved
    :param verbose:
        bool
        If True plots are generated, and least_square optimisation shows progress
    :param suffix:
        str
        String used in title and in saved figure names, to distinguish different cases
    :param de_trend_loc:
        Typically the depth of the target reservoir when analysing depth trends.
        The data is then shifted to this depth
    :param de_trend_scale:
        The uncertainty of the location above
    :param skip:
        List of Log types we don't calculate a trend for.
        Default values ['Discrete flag']
    :param kwargs:

    :return:
    """
    from blixt_utils.misc.curve_fitting import (residuals, linear_function, depth_trend, exp_function,
                                                calculate_depth_trend)
    from blixt_utils.utils import mask_string
    from blixt_utils.plotting import crossplot as xp

    if skip is None:
        skip = ['Discrete flag']
    down_weight_outliers = kwargs.pop('down_weight_outliers', False)
    # target_function = exp_function
    # x0 = [1000., -1000., -0.001]
    target_function = linear_function
    x0 = [1., 1.]
    # TODO
    # We need a x0 for each log type

    res = {
        'success': False,
        'message': 'No fit done',
        'x': x0
    }

    if suffix is None:
        suffix = ''

    x_trends = {}
    data = {}
    x = {}
    data_detrended = {}
    x_detrended = {}
    de_trended_x = None
    de_trended_data = None

    # Start looping over the different log types
    print_info('START ANALYZING DEPTH TRENDS:', '', None, verbose, False)
    for log_type in log_table.log_types:
        if log_type in skip:
            continue
        if log_table.multi_log:
            raise NotImplementedError('This only supports log_tables with one log per log_type, should be extended')
        log_name = log_table[log_type]
        print_info('Working on log type: {}, on log: {}'.format(log_type, log_name), ' -', None, verbose, False)
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = None, None

        # Start looping over wells and collect the data in one container
        data_container = np.zeros(0)  # empty container
        x_container = np.zeros(0)
        legend_items = []
        x_min = 1E6
        x_max = -1E6
        for well in wells:
            print_info(' - in well: {}'.format(well.name), ' -', None, verbose, False)
            if x_name not in well.get_log_names:
                warn_txt = '{} log is missing in well {}'.format(x_name, well.name)
                print_info(warn_txt, 'warning', logger)
                continue
            if log_name not in well.get_log_names:
                warn_txt = '{} log is missing in well {}'.format(log_name, well.name)
                print_info(warn_txt, 'warning', logger)
                continue
            print_info('Well: {}'.format(well.name), '  *', None, verbose, False)

            # Take into account the log_table when using the cutoffs
            #TODO  Note that this won't work for multilog log_tables
            _cutoffs = cutoffs.use_log_table(log_table=log_table)

            # Take into account the requested working interval
            _cutoffs.append(wis.get_cutoff_rule(wi_name, well.name))

            # Create a dictionary of log data for the well where all logs
            well_dict = well.dict(cutoffs=_cutoffs, use_cutoffs=True)

            xdata = well_dict[x_name].magnitude
            ydata = well_dict[log_name].magnitude

            if down_weight_outliers:
                # Remove outliers here, 2 std out?
                _f = 1.5
                outlier_mask = (np.nanmedian(xdata) - _f * np.nanstd(xdata) < xdata) & \
                               (xdata < np.nanmedian(xdata) + _f * np.nanstd(xdata))
                xdata = xdata[outlier_mask]
                ydata = ydata[outlier_mask]

            # Index of NaN's
            nans = np.isnan(ydata)
            if len(ydata[~nans]) < 5:
                warn_txt = 'To few data points in {}, in well {}, after masking'.format(log_name, well.name)
                print_info(warn_txt, 'warning', logger)
                continue

            # Gather the data
            data_container = np.append(data_container, ydata[~nans])
            x_container = np.append(x_container, xdata[~nans])

            # Prepare for plotting
            legend_items.append(well.name)
            if np.min(xdata) < x_min:
                x_min = np.min(xdata)
            if np.max(xdata) > x_max:
                x_max = np.max(xdata)

            if verbose:
                xp.plot(
                    xdata,
                    ydata,
                    cdata=well.style.fill_color,
                    mdata=well.style.marker,
                    xtempl=well.get_log_curve(x_name).style.dict()[x_name],
                    ytempl=well.get_log_curve(log_name).style.dict()[log_name],
                    edge_color=False,
                    ax=ax
                )

        # Try fit the function to the data
        print_info('Finished adding all wells for log type {}'.format(log_type), ' -', None, verbose, False)
        print_info('Trying to fit {} to data'.format(target_function.__name__), ' - ', None, verbose, False)
        try:
            res = calculate_depth_trend(data_container, x_container, target_function, x0, loss='soft_l1',
                                        down_weight_outliers=False,
                                        verbose=False)[0]  # There is only one interval in this case

        except ValueError as error:
            warn_txt = 'Depth trend could not calculated for {} for all wells'.format(log_type)
            print_info(warn_txt, 'warning', logger)
            print(error)
            pass
        info_txt = '  * Success: {}\n  * {}\n  * x0: {}\n  * x: {}'.format(
            res['success'], res['message'], x0, res['x']
        )
        print_info(info_txt, '', None, verbose, False)
        x_trends[log_name] = res['x']
        data[log_name] = data_container
        x[log_name] = x_container

        new_x = np.linspace(x_min, x_max)

        # Try de-trend the data
        if de_trend_loc is not None:
            de_trended_x = np.random.normal(
                loc=de_trend_loc,
                scale=de_trend_scale,
                size=len(data_container))

            def detrend(_z, _y, new_z):
                return _y + target_function(new_z, *res['x']) - target_function(_z, *res['x'])

            de_trended_data = detrend(x_container, data_container, de_trended_x)
            data_detrended[log_name] = de_trended_data
            x_detrended[log_name] = de_trended_x

        if verbose:
            ax.plot(new_x, target_function(new_x, *res['x']))
            legend_items.append('{}, {}'.format(log_name, target_function.__name__))
            # legend_items.append('{} = {:.3}xTVD + {:.3}'.format(log_name, res.x[0], res.x[1]))

            if de_trend_loc is not None:
                ax.scatter(de_trended_x, de_trended_data, c='gray', alpha=0.2,
                           edgecolors='none')
                legend_items.append('De-trended data')

            ax.set_title('{}: {}. {} {} {}'.format(log_type, log_name, str(cutoffs), wi_name, suffix))
            buffer = 0.05 * (x_max - x_min)
            # ax.set_xlim(x_max + buffer, x_min - buffer)
            ax.set_xlim(x_min - buffer, x_max + buffer)
            this_legend = ax.legend(
                legend_items,
                prop=FontProperties(size='smaller'),
                scatterpoints=1,
                markerscale=2,
                loc=1
            )


        if verbose:
            if results_folder:
                _outfile = os.path.join(results_folder, '{} trend {} {} {} in {}.png'.format(
                    x_name.upper(), log_type, log_name, suffix, wi_name))
                print('Saving to: {}'.format(_outfile))
                fig.savefig(_outfile)
            else:
                plt.show()

    return x_trends, data, x, data_detrended, x_detrended


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

def return_chi_line(_y_half_range, _chi, x_center=0., y_center=0.):
    """
    Returns the end points of a line that goes through the
    origo from -y_half_range to +y_half_range

    :param _y_half_range:
    :param _chi:
        float
        Chi angle in deg
    :return:
    """
    _xl = [-1.1 * _y_half_range * np.tan(np.pi * _chi / 180.) + x_center,
           1.1 * _y_half_range * np.tan(np.pi * _chi / 180.) + x_center]
    _yl = [1.1 * _y_half_range + y_center, -1.1 * _y_half_range + y_center]
    return _xl, _yl

def mask_based_on_depth(_depth, _top, _base):
    return (_depth >= _top) & (_depth <= _base)

def calc_intercept_gradient(_vp, _vs, _rho):
    # The calculated intercept and gradient are always one item shorter than the input, so we
    # add a zero at the end
    _intercept = rp.intercept(_vp, None, _rho, None, along_wiggle=True)
    _gradient = rp.gradient(_vp, None, _vs, None, _rho, None, along_wiggle=True)
    return np.append(_intercept, np.zeros(1)), np.append(_gradient, np.zeros(1))

def calc_backus_avg(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _ba, _step):
    if _ba > 0:
        _ba_b = bra.backus(_vp_b, _vs_b, _rho_b, _ba, _step)
        _ba_h = bra.backus(_vp_h, _vs_h, _rho_h, _ba, _step)
        return _ba_b[0], _ba_b[1], _ba_b[2], _ba_h[0], _ba_h[1], _ba_h[2]
    else:
        return _vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h

def calc_eei(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _chi):
    _eei_b = rp.eei(_vp_b, _vs_b, _rho_b)(_chi)
    _eei_h = rp.eei(_vp_h, _vs_h, _rho_h)(_chi)
    return _eei_b, _eei_h

def calc_all(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _depth, _ba, _step, _chi):
    _vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h = calc_backus_avg(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _ba, _step)
    _eei_b, _eei_h = calc_eei(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, _chi)
    _ai_b, _ai_h = calc_eei(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, 0.)  # acoustic impedance
    _gi_b, _gi_h = calc_eei(_vp_b, _vs_b, _rho_b, _vp_h, _vs_h, _rho_h, 90.)  # gradient impedance
    _i_b, _g_b = calc_intercept_gradient(_vp_b, _vs_b, _rho_b)
    _i_h, _g_h = calc_intercept_gradient(_vp_h, _vs_h, _rho_h)
    return dict(
        eei_brine=_eei_b,
        eei_hc=_eei_h,
        eei_diff=(_eei_b - _eei_h) / _eei_h,
        int_brine=_i_b,
        grad_brine=_g_b,
        int_hc=_i_h,
        grad_hc=_g_h,
        log_ai_b=np.log10(_ai_b),
        log_gi_b=np.log10(_gi_b),
        log_ai_h=np.log10(_ai_h),
        log_gi_h=np.log10(_gi_h),
        md=_depth
    )

legend_code = """
    const _chi = chi.value;
    const _ba = ba.value;
    const _freq = freq.value;
    spans.location = _chi;
    c1_legend.items[0].label.value = 'EEI brine, chi:' + _chi + ' ba:' + _ba;
    c1_legend.items[1].label.value = 'EEI hc, chi:' + _chi + ' ba:' + _ba;
    \\ c3_legend.items[0].label.value = 'DIFF., chi:' + _chi + ' ba:' + _ba;
    c4_legend.items[0].label.value = 'HC, ba:' + _ba + ' Hz:' + _freq;
    c5_legend.items[0].label.value = 'Brine - HC, ba:' + _ba + ' Hz:' + _freq;
    hplot_legend.title = 'Chi:' + _chi;
    hplot_legend.items[0].label.value = 'Brine, ba:' + _ba;
    hplot_legend.items[1].label.value = 'HC, ba:' + _ba;
    xplot_legend.title = 'Chi:' + _chi;
    xplot_legend.items[0].label.value = 'Brine, ba:' + _ba;
    xplot_legend.items[1].label.value = 'HC, ba:' + _ba;
    console.log('Chi angle: ', _chi, xplot_legend.title); 
    """


def save_as_las():
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    file_name = asksaveasfilename(filetypes=(("las file", "*.las"),("All Files", "*.*")))
    return file_name
