"""
Functionality designed to plot and calculate the elastic rock properties around a given depth (top) with the goal to
show how the rock properties changes across this depth and how that can affect the seismic signature
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from blixt_utils.plotting.helpers import axis_plot, annotate_plot, header_plot, wiggle_plot, set_up_column_plot
from blixt_utils.misc.templates import default_templates
from blixt_rp.core.well import Project, LogCurve
import blixt_rp.rp.rp_core as rp
from blixt_rp.core.models import Model, Layer, plot_wiggles
import blixt_utils.misc.wavelets as bumw
import blixt_rp.plotting.plot_logs as rpp

h_above = 50.
h_below = 50.
results = None
soft_below = None


def calc_ai_around_top(
        vp, rho, md, twt, top,
        thickness_above=None, thickness_below=None,
        ref_logs=None, ref_traces=None,
        ref_incident_angles=None,
        buffer=None,
        mask=None, mask_desc=None,
        templates=None, axes=None, header_axes=None,
        intervals=None, interval_names=None,
        wavelet=None, interactive=True,
        **kwargs
):
    """
    :param vp:
        LogCurve object
        P velocity
    :param rho:
        LogCurve object
        Density
    :param md:
        LogCurve object
        Measured depth, meter
    :param twt:
        LogCurve object
        Two way travel time, seconds
    :param top:
        float
        MD value around (+/- h_above, h_below) which we plot the data
    :param thickness_above:
        float
        thickness in meters that defines the layer to use in the calculation of the statistics
    :param thickness_below:
        float
        thickness in meters that defines the layer to use in the calculation of the statistics
    :param ref_logs:
        LogCurve object or list of LogCurve objects
        LogCurve(s) containing other data from this well to display in one extra sub plot
        This axis is called 'ref_ax' and if input axes are provided, it must be found there
    :param ref_traces:
        LogCurve object  or list of LogCurve objects
        LogCurve(s) containing a seismic trace along the well path, so it should be in the depth domain
        This axis is called 'ref_trace_ax' and if input axes are provided, it must be found there
    :param    ref_incident_angles:
        list
        List of incident angles in degrees for the different seismic reference traces

    :param axes:
        not implemented
    wavelet:
        dict
        dictionary with three keys:
            'wavelet': contains the wavelet amplitude
            'time': contains the time data [s]
            'header': a dictionary with info about the wavelet
        see blixt_utils.io.io.read_petrel_wavelet() for example
    """
    global h_above, h_below, results, soft_below

    if thickness_above is not None:
        h_above = thickness_above
    if thickness_below is not None:
        h_below = thickness_below
    if buffer is None:
        buffer = 30.
    if mask is None:
        mask = np.array(np.ones(len(vp.data)), dtype=bool)  # All True values -> all data is included
    if (mask is not None) and (mask_desc is None):
        mask_desc = 'UNKNOWN'
    if templates is None:
        templates = default_templates
    if ref_logs is not None:
        if isinstance(ref_logs, LogCurve):
            ref_logs = [ref_logs]
    if ref_traces is not None:
        if isinstance(ref_traces, LogCurve):
            ref_traces = [ref_traces]
    if (ref_incident_angles is not None) and (ref_traces is not None):
        if len(ref_incident_angles) != len(ref_traces):
            raise IOError('Length of inc. angles ({}) and ref_traces ({}) are not the same'.format(
                len(ref_incident_angles), len(ref_traces)))

    time_step = kwargs.pop('time_step', 0.001)
    c_f = kwargs.pop('center_frequency', 25.)
    duration = kwargs.pop('duration', 0.128)

    if wavelet is None:
        wavelet = bumw.ricker(duration, time_step, c_f)

    ax_names = ['md_ax', 'ai_ax', 'synth_ax', 'twt_ax']
    width_ratios = [0.2, 1, 1, 0.2]
    if ref_logs is not None:
        ax_names.insert(1, 'ref_ax')
        width_ratios.insert(1, 1)
    if ref_traces is not None:
        ax_names.insert(-2, 'ref_trace_ax')
        width_ratios.insert(-2, 1)
    # TODO
    # Check that axes never are used as input
    # if axes is not None:
    #     for n in ax_names:
    #         if n not in list(axes.keys()):
    #             raise IOError('The {} axis is missing among the provided axes')
    #         if n not in list(header_axes.keys()):
    #             raise IOError('The {} axis is missing among the provided header axes')

    depth = md.data
    # fig = None
    # if ref_logs is None:
    #     fig = plt.figure(figsize=(12, 10))
    # else:
    #     fig = plt.figure(figsize=(16, 10))

    # n_cols = len(ax_names)
    # n_rows = 2
    # height_ratios = [1, 5]
    # spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols,
    #                         height_ratios=height_ratios, width_ratios=width_ratios,
    #                         hspace=0., wspace=0.,
    #                         left=0.05, bottom=0.03, right=0.98, top=0.96)
    # header_axes = {}
    # axes = {}
    # for i in range(len(ax_names)):
    #     header_axes[ax_names[i]] = fig.add_subplot(spec[0, i])
    #     axes[ax_names[i]] = fig.add_subplot(spec[1, i])
    fig, header_axes, axes = set_up_column_plot(ax_names=ax_names, width_ratios=width_ratios)

    title_txt = 'Elastics estimation in well {}'.format(md.well)
    if mask_desc is not None:
        title_txt += ', using mask: {}'.format(mask_desc)
    fig.suptitle(title_txt)

    log_types = ['Impedance', 'P velocity', 'Density']
    lws = {ltype: _x for ltype, _x in zip(log_types, [2, 0.5, 0.5])}
    limits = [[templates[xx]['min'], templates[xx]['max']] for xx in log_types]
    styles = [{'lw': lws[xx],
               'color': templates[xx]['line color'],
               'ls': templates[xx]['line style']}
              for xx in log_types]
    legends = ['{} [{}]'.format(xx, templates[xx]['unit']) for xx in log_types]

    _x1 = vp.data; _x1[~mask] = np.nan
    _x2 = rho.data; _x2[~mask] = np.nan

    # A uniformly sampled array of time steps, from A to B
    t = np.arange(np.nanmin(twt.data), np.nanmax(twt.data), wavelet['header']['Sample rate'])

    def re_plot(_h_above, _h_below, _ref_logs, _ref_traces):
        print('Replotting using :', _h_above, _h_below)

        for ax_name in ax_names:
            axes[ax_name].clear()
            header_axes[ax_name].clear()

        md_min = top - _h_above - buffer
        md_max = top + _h_below + buffer

        header_plot(header_axes['md_ax'], None, None, None, title='MD [m]')
        annotate_plot(axes['md_ax'], depth[mask], intervals=intervals, interval_names=interval_names,
                      ylim=[md_min, md_max])

        above_mask = np.ma.masked_inside(md.data[mask], top - _h_above, top).mask
        below_mask = np.ma.masked_inside(md.data[mask], top, top + _h_below).mask
        avg_vp_above = np.nanmedian(_x1[mask][above_mask])
        avg_vp_below = np.nanmedian(_x1[mask][below_mask])
        avg_rho_above = np.nanmedian(_x2[mask][above_mask])
        avg_rho_below = np.nanmedian(_x2[mask][below_mask])

        soft_below_mask = np.ma.masked_less(
            _x1[mask][below_mask] * _x2[mask][below_mask],
            avg_vp_above * avg_rho_above).mask
        softness_below = avg_vp_above * avg_rho_above / \
                         (_x1[mask][below_mask][soft_below_mask] * _x2[mask][below_mask][soft_below_mask])

        positions = ['above', 'below']
        _results = {'top': top, 'h_above': _h_above, 'h_below': _h_below,
                   'vp_above': avg_vp_above, 'vp_below': avg_vp_below,
                   'rho_above': avg_rho_above, 'rho_below': avg_rho_below}
        _soft_below = {'softness_below': softness_below}
        # _averages = {'above': {'vp': avg_vp_above, 'rho': avg_rho_above},
        #              'below': {'vp': avg_vp_below, 'rho': avg_rho_below}}
        if _ref_logs is not None:
            for _log in _ref_logs:
                for pos in positions:
                    # _averages['above'][_log.name] = np.nanmedian(_log.data[mask][above_mask])
                    # _averages['below'][_log.name] = np.nanmedian(_log.data[mask][below_mask])
                    if pos == 'above':
                        this_mask = above_mask
                    else:
                        this_mask = below_mask
                        _soft_below['{}_soft_below'.format(_log.name)] = _log.data[mask][this_mask][soft_below_mask]
                    _results['{}_{}'.format(_log.name, pos)] = np.nanmedian(_log.data[mask][this_mask])

        if _ref_traces is not None:
            for _trace in _ref_traces:
                for pos in positions:
                    if pos == 'above':
                        this_mask = above_mask
                    else:
                        this_mask = below_mask
                    _results['{}_max_{}'.format(_trace.name, pos)] = np.max(_trace.data[mask][this_mask])
                    _results['{}_min_{}'.format(_trace.name, pos)] = np.min(_trace.data[mask][this_mask])

        xlims = axis_plot(axes['ai_ax'], depth, [_x1 * _x2, _x1, _x2],
                          limits, styles, yticks=False, ylim=[md_min, md_max])

        axes['ai_ax'].plot(
            [avg_vp_above * avg_rho_above] * 2, [top - _h_above, top], c=styles[0]['color'], ls='--')
        axes['ai_ax'].plot([avg_vp_below * avg_rho_below] * 2, [top, top + _h_below], c=styles[0]['color'], ls='--')

        for ax_name in ax_names:
            axes[ax_name].axhline(top - _h_above, c='k', ls='--')
            axes[ax_name].axhline(top, c='k', ls='-')
            axes[ax_name].axhline(top + _h_below, c='k', ls='--')

        header_plot(header_axes['ai_ax'], xlims, legends, styles)

        # get the twt that corresponds to md_min and md_max
        twt_min = twt.data[np.nanargmin((md.data - md_min)**2)]
        if np.isnan(twt_min):
            twt_min = np.nanmin(twt.data[mask])
        twt_mid = twt.data[np.nanargmin((md.data - top)**2)]
        twt_max = twt.data[np.nanargmin((md.data - md_max)**2)]
        if np.isnan(twt_max):
            twt_max = np.nanmax(twt.data[mask])

        # Build a model based on the average elastic properties
        build_model = True
        above_layer = Layer(thickness=twt_mid - twt_min + 2 * time_step,
                            vp=avg_vp_above, vs=avg_vp_above, rho=avg_rho_above)
        if np.isnan(above_layer.thickness) or (above_layer.thickness <= 0.):
            build_model = False
        below_layer = Layer(thickness=twt_max - twt_mid, vp=avg_vp_below, vs=avg_vp_below, rho=avg_rho_below)
        if np.isnan(below_layer.thickness) or (below_layer.thickness <= 0.):
            build_model = False

        if build_model:
            m = Model(depth_to_top=twt_min, layers=[above_layer, below_layer])
            # print(' - Well {}, dt_above {}, dt_below {}, timestep {}'.format(
            #    twt.well, above_layer.thickness, below_layer.thickness, time_step
            # ))
            _twt, _layer_i, _vp, _vs, _rho, _z = m.realize_model(time_step)
            _reff = rp.reflectivity(_vp, None, _vs, None, _rho, None, along_wiggle=True)
            m_wiggle = bumw.convolve_with_refl(wavelet['wavelet'], _reff(0.), verbose=False)
            t_to_d = [np.nanargmin((twt.data - _t)**2) for _t in _twt]
            #  wiggle_plot(axes['synth_ax'], _twt, m_wiggle, ylim=[twt_min, twt_max], yticks=False, ls='--')
            wiggle_plot(axes['synth_ax'], md.data[t_to_d], m_wiggle, ylim=[md_min, md_max], yticks=False, ls='--')

        vp_t = np.interp(x=t, xp=twt.data, fp=vp.data)
        rho_t = np.interp(x=t, xp=twt.data, fp=rho.data)
        # reflectivity as a function of incident angle, but this one is only valid at theta=0 as we don't
        # include vs in the calculation
        reff = rp.reflectivity(vp_t, None, vp_t, None, rho_t, None, along_wiggle=True)
        wiggle = bumw.convolve_with_refl(wavelet['wavelet'], reff(0.), verbose=False)
        t_to_d = [np.nanargmin((twt.data - _t)**2) for _t in t]

        # wiggle_plot(axes['synth_ax'], t, wiggle, ylim=[twt_min, twt_max], yticks=False)
        wiggle_plot(axes['synth_ax'], md.data[t_to_d], wiggle, ylim=[md_min, md_max], yticks=False)

        # for _md, _ls in zip([top - _h_above, top, top + _h_below], ['--', '-', '--']):
        #    this_twt = twt.data[np.argmin((md.data - _md)**2)]
        #    axes['synth_ax'].axhline(this_twt, c='k', ls=_ls)

        if build_model:
            header_plot(header_axes['synth_ax'],
                        [[np.min(wiggle), np.max(wiggle)], [np.min(m_wiggle), np.max(m_wiggle)]],
                        ['Synthetics', 'Synth. from avg.'], [{'color': 'k', 'ls': '-'},
                                                             {'color': 'k', 'ls': '--'}], title='Synthetics')
        else:
            header_plot(header_axes['synth_ax'],
                        [[np.min(wiggle), np.max(wiggle)]],
                        ['Synthetics'], [{'color': 'k', 'ls': '-'}], title='Synthetics')

        header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
        annotate_plot(axes['twt_ax'], twt.data, ylim=[twt_min, twt_max])

        if _ref_traces is not None:
            # for trace in _ref_traces:
            #     wiggle_plot(axes['ref_trace_ax'], depth, trace.data, ylim=[md_min, md_max], yticks=False)
            _, _ = rpp.plot_wiggles(
                None, depth[mask], None,
                scaling=10./np.nanmax(_ref_traces[0].data[mask]),
                ax=axes['ref_trace_ax'],
                incident_angles=ref_incident_angles,
                extract_at=None,
                input_wiggles=[_x.data[mask] for _x in _ref_traces],
                yticks=False,
                ylim=[md_min, md_max]
            )
            header_plot(header_axes['ref_trace_ax'], None, None, None, title='Seismic traces')

        if _ref_logs is not None:
            logs_to_plot = [xx.data for xx in _ref_logs]
            _styles = [{'lw': templates[xx.log_type]['line width'],
                       'color': templates[xx.log_type]['line color'],
                       'ls': templates[xx.log_type]['line style']}
                      for xx in _ref_logs]
            _limits = [[templates[xx.log_type]['min'], templates[xx.log_type]['max']] for xx in _ref_logs]
            _legends = ['{} [{}]'.format(xx.name, templates[xx.log_type]['unit']) for xx in _ref_logs]
            xlims = axis_plot(axes['ref_ax'], depth, logs_to_plot,
                              _limits, _styles, yticks=False, ylim=[md_min, md_max])
            header_plot(header_axes['ref_ax'], xlims, _legends, _styles)

        # for position in ['above', 'below']:
        #     print('Average values {} top:'.format(position))
        #     print('  ', ', '.join(['{}: {}'.format(key, value) for key, value in _averages[position].items()]))
        #     print('Seismic amplitudes {} top:'.format(position))
        #     print('  ', ', '.join(['{}: {}'.format(key, value) for key, value in _trace_amplitudes[position].items()]))
        # print('')
        return _h_above, _h_below, _results, _soft_below

    h_above, h_below, results, soft_below = re_plot(h_above, h_below, ref_logs, ref_traces)

    def on_press(event):
        global h_above, h_below, results
        # print('you pressed', event.button, event.xdata, event.ydata)
        if event.ydata > top:
            h_above, h_below, results, soft_below = re_plot(h_above, event.ydata - top, ref_logs, ref_traces)
        if event.ydata < top:
            h_above, h_below, results, soft_below = re_plot(top - event.ydata, h_below, ref_logs, ref_traces)
        # print('h_above:', h_above, ',  h_below', h_below)
        plt.show()

    if interactive:
        cid = fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()
    return h_above, h_below, results, soft_below


def test_calc_elastics():
    dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\blixt_rp\\rp_utils', '')
    project_table = os.path.join(dir_path, 'excels', 'project_table.xlsx')
    wp = Project(project_table=project_table)
    wells = wp.load_all_wells(unit_convert_using_template=True)
    templates = wp.load_all_templates()
    wis = wp.load_all_wis()
    this_well = wells[list(wells.keys())[0]]
    this_well.add_twt(wp.project_table, verbose=False)
    depth = this_well.get_logs_of_name('depth')[0]
    vp = this_well.get_logs_of_type('P velocity')[0]
    rho = this_well.get_logs_of_type('Density')[0]
    twt = this_well.get_logs_of_type('Two-way time')[0]
    poro = this_well.get_logs_of_type('Porosity')[0]
    sw = this_well.get_logs_of_type('Saturation')[0]

    intervals = []
    interval_names = []
    for key in list(wis[this_well.well].keys()):
        interval_names.append(key)
        intervals.append(wis[this_well.well][key])

    calc_ai_around_top(vp, rho, depth, twt,
                       top=1826.,
                       h_above=50., h_below=50.,
                       intervals=intervals, interval_names=interval_names,
                       templates=templates,
                       ref_logs=[sw, poro])


if __name__ == '__main__':
    test_calc_elastics()
