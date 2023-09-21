"""
Calculates the TOC using the functions from Passey et al. 1990 "A practical model for organic richness ...",
which are implemented in rp_core.py
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from blixt_rp.rp_utils.version import info
from blixt_utils.plotting.helpers import axis_plot, annotate_plot, header_plot, deltalogr_plot
from blixt_rp.core.log_curve import LogCurve
from blixt_utils.misc.templates import default_templates

text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}

my_global = None


def test(a=my_global):
    global my_global
    mutable_object = {}
    if my_global is None:
        my_global = 0

    def on_press(event):
        global my_global
        my_global += 1
        print(my_global, 'you pressed', event.button, event.xdata, event.ydata)
        _key = event.button
        mutable_object['key'] = _key

    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x**2

    ax.plot(x, y)

    cid = fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()


def get_values(array1, array2, depth, md):
    """
    Returns the values of array1 and array2 at the given depth md
    :param array1:
        numpy.ndarray
    :param array2:
        numpy.ndarray
    :param depth:
        numpy.ndarray
        Containing the MD of the well
    """
    this_i = np.argmin((depth - md)**2)
    return array1[this_i], array2[this_i]


def calc_toc(
        r, ac, md, r0=None, ac0=None, start_r=None, lom=None, true_toc=None, mask=None, mask_desc=None, templates=None,
        axes=None, header_axes=None,
        intervals=None, interval_names=None,
        down_weight_intervals=None,
        discrete_intervals=None,
        ylim=None,
        reference_logs=None,
        verbose=False
):
    """
    Calculates the TOC using the functions from Passey et al. 1990 "A practical model for organic richness ...",
    which are implemented in rp_core.py
    By clicking on the plot, you select to use r and ac at the picked depth as the baseline r0 and ac0
    The "trend" TOC is calculated using a linear trend in r and ac as baseline, so that the manual picking is not
    necessary
    :param r:
        LogCurve object
        LogCurve containing the resistivity
    :param ac:
        LogCurve object
        LogCurve containing the sonic
    :param md:
        LogCurve object
        LogCurve containing the MD of the well
    :param r0:
        float
        Picked baseline value of the resistivity
    :param ac0:
        float
        Picked baseline value of the sonic
    :param start_r:
        float
        Lower limit in resistivity range in the delta log r plot, Ohm m
    :param lom:
        float
        Unitless number somewhere between 6 and 11 which describes the level of organic metamorphism units
        (Hood et al. 1975)
    :param true_toc:
        dict
        Dictionary with keys 'toc' and 'depth' which contains true values of TOC at given measured depths
    :param mask:
        Boolean numpy array of same length as the LogCurves r, s and d
        A False value indicates that the data is masked out
    :param mask_desc:
        str
        Description of what cutoffs the mask is based on.
    :param axes:
        dict
        Dictionary with the the following keys; 'r_ac_ax', 'dlr_ax', 'toc_ax' which each have an Axes object that the
        data will be plotted in.
        If reference_logs is not None, then this dict must contain an additional axes called 'ref_ax'.
        If axes are provided, the "verbose" keyword will be ignored
    :param header_axes:
        dict
        Dictionary with the the following keys; 'r_ac_ax', 'dlr_ax', 'toc_ax' which each have an Axes object that
        will hold the headers of the plots.
        If reference_logs is not None, then this dict must contain an additional axes called 'ref_ax'.
    :param intervals:
        list of lists with top and base (in MD) of N intervals, e.g.
        [[interval1_top, interval1_base], [interval2_top, interval2_base], ...]
    :param interval_names:
        list of N names to annotate the intervals
    :param down_weight_intervals:
        list
        List of lists, where each inner list contains the start MD, stop MD and weight, for each interval we like to
        downweight. With weight = 0 the interval is simply ignored when calculating the background trend
    :param discrete_intervals:
        list
        list of MD values of where we allow the trend function to have discontinuities
    :param reference_logs:
        LogCurve object or list of LogCurve objects
        LogCurve(s) containing other data from this well to display in one extra sub plot
        This axis is called 'ref_ax' and if input axes are provided, it must be found there
    :param ylim:
        list of min max value of the y axis

    Returns:
        A two item list of LogCurve objects
        First is the automated TOC using the trends as baselines, and the second is the TOC calulated using the
        input r0 and ac0 as baseline

    """
    from blixt_rp.rp.rp_core import delta_log_r, toc_from_delta_log_r
    if r0 is None:
        r0 = 2.1
    if ac0 is None:
        ac0 = 90.
    if start_r is None:
        start_r = 0.013
    if lom is None:
        lom = 9.
    if mask is None:
        mask = np.array(np.ones(len(r.data)), dtype=bool)  # All True values -> all data is included
    if (mask is not None) and (mask_desc is None):
        mask_desc = 'UNKNOWN'
    if templates is None:
        templates = default_templates

    ax_names = ['r_ac_ax', 'dlr_ax', 'toc_ax']
    if reference_logs is not None:
        ax_names += ['ref_ax']
    if axes is not None:
        verbose = False
        preset_axes = True
        for n in ax_names:
            if n not in list(axes.keys()):
                raise IOError('The {} axis is missing among the provided axes')
            if n not in list(header_axes.keys()):
                raise IOError('The {} axis is missing among the provided header axes')
    else:
        preset_axes = False

    depth = md.data
    if ylim is None:
        md_min = np.min(depth[mask])
        md_max = np.max(depth[mask])
    else:
        md_min = ylim[0]
        md_max = ylim[1]
    fig = None
    if verbose:
        if reference_logs is None:
            fig = plt.figure(figsize=(12, 10))
        else:
            fig = plt.figure(figsize=(16, 10))
        title_txt = 'TOC estimation in well {}'.format(r.well)
        if mask_desc is not None:
            title_txt += ', using mask: {}'.format(mask_desc)
        fig.suptitle(title_txt)
        if intervals is not None:
            ax_names = ['md_ax'] + ax_names
            width_ratios = [0.2, 1, 1, 1]
        else:
            width_ratios = [1, 1, 1]
        if reference_logs is not None:
            width_ratios += [1]

        n_cols = len(ax_names)
        n_rows = 2
        height_ratios = [1, 5]
        spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols,
                                height_ratios=height_ratios, width_ratios=width_ratios,
                                hspace=0., wspace=0.,
                                left=0.05, bottom=0.03, right=0.98, top=0.96)
        header_axes = {}
        axes = {}
        for i in range(len(ax_names)):
            header_axes[ax_names[i]] = fig.add_subplot(spec[0, i])
            axes[ax_names[i]] = fig.add_subplot(spec[1, i])

        if intervals is not None:
            header_plot(header_axes['md_ax'], None, None, None, title='MD [m]')
            annotate_plot(axes['md_ax'], depth[mask], intervals=intervals, interval_names=interval_names,
                          ylim=[md_min, md_max])

    if down_weight_intervals is not None:
        if isinstance(down_weight_intervals, list):
            if len(down_weight_intervals[0]) != 3:
                raise IOError('down_weight_intervals must contain lists with 3 items')
        else:
            raise IOError('down_weight_intervals must be a list')
        # replace the depths [m MD] with the indexes for the for this depth
        for _i, intrvl in enumerate(down_weight_intervals):
            down_weight_intervals[_i][0] = np.argmin(np.sqrt((depth-intrvl[0])**2))
            down_weight_intervals[_i][1] = np.argmin(np.sqrt((depth-intrvl[1])**2))

    if discrete_intervals is not None:
        if not isinstance(discrete_intervals, list):
            raise IOError('discrete_intervals must be a list')
        # replace the depths [m MD] with the indexes for the for this depth
        for _i, intrvl in enumerate(discrete_intervals):
            discrete_intervals[_i] = np.argmin(np.sqrt((depth-intrvl)**2))
            discrete_intervals[_i] = np.argmin(np.sqrt((depth-intrvl)**2))

    # find the linear trends matching the data. These are used as baseline in the DeltaLogR calculation
    fit_parameters_rdep = r.calc_depth_trend(
        md.data,
        mask=mask,
        down_weight_outliers=True,
        down_weight_intervals=down_weight_intervals,
        discrete_intervals=discrete_intervals,
        verbose=False
    )
    fitted_rdep = r.apply_trend_function(
        md.data,
        fit_parameters_rdep,
        discrete_intervals=discrete_intervals,
        verbose=False
    )
    fit_parameters_ac = ac.calc_depth_trend(
        md.data,
        mask=mask,
        down_weight_outliers=True,
        down_weight_intervals=down_weight_intervals,
        discrete_intervals=discrete_intervals,
        verbose=False
    )
    fitted_ac = ac.apply_trend_function(
        md.data,
        fit_parameters_ac,
        discrete_intervals=discrete_intervals,
        verbose=False
    )

    dlr_trend = delta_log_r(r.data, ac.data, fitted_rdep, fitted_ac)
    toc_trend = toc_from_delta_log_r(dlr_trend, lom)

    def re_plot(_sr, _r0, _ac0):
        _dlr_picked = delta_log_r(r.data, ac.data, _r0, _ac0)
        _toc_picked = toc_from_delta_log_r(_dlr_picked, lom)
        if verbose or preset_axes:
            axes['dlr_ax'].clear()
            header_axes['dlr_ax'].clear()
            axes['toc_ax'].clear()
            header_axes['toc_ax'].clear()

            log_types = ['Sonic', 'Resistivity']
            limits = [[200., 0.], [_sr, 10000 * _sr]]
            styles = [{'lw': templates[x]['line width'],
                       'color': templates[x]['line color'],
                       'ls': templates[x]['line style']} for x in log_types]
            legends = ['{} [{}]'.format(x, templates[x]['unit']) for x in log_types]
            # xlims = deltalogr_plot(axes['dlr_ax'], depth[mask],
            #                        [ac.data[mask], r.data[mask]],
            #                        limits, styles, yticks=False, ylim=[md_min, md_max])
            _x1 = ac.data; _x1[~mask] = np.nan
            _x2 = r.data; _x2[~mask] = np.nan
            xlims = deltalogr_plot(axes['dlr_ax'], depth,
                                   [_x1, _x2],
                                   limits, styles, yticks=False, ylim=[md_min, md_max])
            header_plot(header_axes['dlr_ax'], xlims, legends, styles)
            header_axes['dlr_ax'].text(1.5, 1.0, '$start r$={:.4f} $\\Omega m$'.format(
                _sr), ha='center', va='bottom', **text_style)

            legends = ['TOC [%], picked baseline', 'TOC [%], trend baseline']
            limits = [[templates['TOC']['min'], templates['TOC']['max']]] * 2
            styles = [
                {'lw': templates['TOC']['line width'],
                 'color': templates['TOC']['line color'],
                 'ls': templates['TOC']['line style']},
                {'lw': templates['TOC']['line width'],
                 'color': 'b',
                 'ls': templates['TOC']['line style']},
            ]
            # xlims = axis_plot(axes['toc_ax'], depth[mask],
            #                   [_toc_picked.value[mask], toc_trend.value[mask]],
            #                   limits, styles, yticks=False, ylim=[md_min, md_max])
            _x1 = _toc_picked.value; _x1[~mask] = np.nan
            _x2 = toc_trend.value; _x2[~mask] = np.nan
            xlims = axis_plot(axes['toc_ax'], depth,
                          [_x1, _x2],
                          limits, styles, yticks=False, ylim=[md_min, md_max])
            if true_toc is not None:
                axes['toc_ax'].scatter(true_toc['toc'], true_toc['depth'],
                                       c='red', marker=templates['TOC']['marker'])
                xlims.append(xlims[0])
                legends.append('TOC from cuttings/core [%]')
                styles.append({'lw': 0, 'color': 'red', 'marker': templates['TOC']['marker']})
            header_plot(header_axes['toc_ax'], xlims, legends, styles)
            header_axes['toc_ax'].text(1.5, 1.0, '$r_0$={:.1f}, $ac_0$={:.1f}, LOM={:.1f}'.format(
                _r0, _ac0, lom), ha='center', va='bottom', **text_style)
        return _dlr_picked, _toc_picked

    if verbose or preset_axes:
        log_types = ['Resistivity', 'Resistivity', 'Sonic', 'Sonic']
        lognames = ['RDEP', 'RDEP trend', 'AC', 'AC trend']
        limits = [[0., 10.] for x in log_types[:2]]
        limits += [[200., 0.] for x in log_types[2:]]
        cls = ['r', 'r', 'g', 'g']
        lws = [1, 1, 1, 1]
        lss = ['-', '--', '-', '--']
        styles = [{'lw': lws[i], 'color': cls[i], 'ls': lss[i]} for i in range(4)]
        legends = ['{} [{}]'.format(x, templates['Resistivity']['unit']) for x in lognames[:2]]
        legends += ['{} [{}]'.format(x, templates['Sonic']['unit']) for x in lognames[2:]]

        # logs_to_plot = [r.data[mask],
        #                 fitted_rdep[mask],
        #                 ac.data[mask],
        #                 fitted_ac[mask]
        _x1 = r.data; _x1[~mask] = np.nan
        _x2 = fitted_rdep; _x2[~mask] = np.nan
        _x3 = ac.data; _x3[~mask] = np.nan
        _x4 = fitted_ac; _x4[~mask] = np.nan
        logs_to_plot = [_x1, _x2, _x3, _x4 ]

        # xlims = axis_plot(axes['r_ac_ax'], depth[mask], logs_to_plot,
        xlims = axis_plot(axes['r_ac_ax'], depth, logs_to_plot,
                          limits, styles, yticks=not((intervals is not None) or preset_axes), ylim=[md_min, md_max])
        # header_plot(header_axes['r_ac_ax'], limits, legends, styles)
        header_plot(header_axes['r_ac_ax'], xlims, legends, styles)

        if reference_logs is not None:
            if isinstance(reference_logs, LogCurve):
                reference_logs = [reference_logs]
            logs_to_plot = [xx.data for xx in reference_logs]
            styles = [{'lw': templates[xx.log_type]['line width'],
                       'color': templates[xx.log_type]['line color'],
                       'ls': templates[xx.log_type]['line style']}
                      for xx in reference_logs]
            limits = [[templates[xx.log_type]['min'], templates[xx.log_type]['max']] for xx in reference_logs]
            legends = ['{} [{}]'.format(xx.name, templates[xx.log_type]['unit']) for xx in reference_logs]
            xlims = axis_plot(axes['ref_ax'], depth, logs_to_plot,
                              limits, styles, yticks=not((intervals is not None) or preset_axes), ylim=[md_min, md_max])
            header_plot(header_axes['ref_ax'], xlims, legends, styles)

    dlr_picked, toc_picked = re_plot(start_r, r0, ac0)

    def on_press(event):
        # print('you pressed', event.button, event.xdata, event.ydata)
        _r00, _ac00 = get_values(r.data, ac.data, md.data, event.ydata)
        print('At MD {:.2f}m, RDEP={:.2f}, AC={:.2f}'.format(event.ydata, _r00, _ac00))
        _, _, = re_plot(10**(0.02*_ac00 + np.log10(_r00)) / 10000., _r00, _ac00)
        plt.show()

    if verbose:
        cid = fig.canvas.mpl_connect('button_press_event', on_press)
        plt.show()

    # Replace negative values with 0. TOC should not be negative
    toc_trend.value[toc_trend.value < 0] = 0.
    toc_picked.value[toc_picked.value < 0] = 0.

    out_toc_trend = LogCurve(
        name=toc_trend.name.lower() + '_trend',
        data=toc_trend.value,
        header={
            'unit': toc_trend.unit,
            'log_type': 'TOC',
            'desc': '{}  DLR calculated using a linear trend as baseline. LOM = {:.1f}'.format(
                toc_trend.desc, lom)
        }
    )
    out_toc_picked = LogCurve(
        name=toc_picked.name.lower() + '_picked',
        data=toc_picked.value,
        header={
            'unit': toc_picked.unit,
            'log_type': 'TOC',
            'desc': '{}  DLR calculated using a picked baseline, with r0={:.2f}, ac0={:.2f}. LOM = {:.1f}'.format(
                toc_picked.desc, r0, ac0, lom)
        }
    )
    out_dlr_trend = LogCurve(
        name=dlr_trend.name.lower() + '_trend',
        data=dlr_trend.value,
        header={
            'unit': dlr_trend.unit,
            'log_type': 'Delta log R',
            'desc': '{}  DLR calculated using a linear trend as baseline. LOM = {:.1f}'.format(
                dlr_trend.desc, lom)
        }
    )
    out_dlr_picked = LogCurve(
        name=dlr_picked.name.lower() + '_picked',
        data=dlr_picked.value,
        header={
            'unit': dlr_picked.unit,
            'log_type': 'Delta log R',
            'desc': '{}  DLR calculated using a picked baseline, with r0={:.2f}, ac0={:.2f}. LOM = {:.1f}'.format(
                dlr_picked.desc, r0, ac0, lom)
        }
    )
    return out_toc_trend, out_toc_picked, out_dlr_trend, out_dlr_picked

