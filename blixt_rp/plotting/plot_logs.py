import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
import logging
from copy import deepcopy

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot, \
    deltalogr_plot, chi_rotation_plot, set_up_column_plot
import blixt_rp.rp.rp_core as rp
from blixt_utils.plotting import crossplot as xp
# from bruges.filters import ricker
import blixt_utils.misc.wavelets as bumw
import bruges.rockphysics.anisotropy as bra

logger = logging.getLogger(__name__)


def find_nearest(data, value):
    return np.nanargmin(np.abs(data - value))


def plot_logs(well, log_table, wis, wi_names, templates, buffer=None, block_name=None, savefig=None,
              intervals=None, interval_names=None, interval_colors=None, wavelet=None, **kwargs):
    """
    Attempts to draw a plot similar to the "CPI plots", for one working interval with some buffer.
    :param well:
        well object
    :param log_table:
        dict
        Dictionary of log type: log name key: value pairs which decides which log to use when selecting which
        velocity / sonic and density logs. Other logs (e.g. 'Resistivity') are selected based on their presence
        E.G.
            log_table = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob'}
    :param wis:
        dict
        Dictionary of working intervals, keys are working interval names, and the values are a two
        items list with top & bottom
    :param wi_names:
        list of strings
        Name of working intervals to use in this plot
        set to None to plot whole well
    :param templates:
        dict
        Dictionary of different templates as returned from Project().load_all_templates()
    :param buffer:
        float
        distance in meters
        Log is plotted from top of working interval - buffer to base of working interval + buffer
        Default is 50 m
    :param savefig:
        str
        Full path name to file (.png or .pdf) to which the plot is exported
        if None, the plot is displayed instead
    :param intervals:
        list of lists with top and base of N intervals that we want to highlight in the plot, e.g.
        [[interval1_top, interval1_base], [interval2_top, interval2_base], ...]
    :param interval_names:
        list of N names to annotate the intervals
    :param interval_colors:
        list of N colors to color each interval
        if equal to string 'cyclic', two hardcoded colors are used to cyclically color each interval
    wavelet:
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
        source_rock_study:
            dict
            Well names are keys, and each item is a 2 item list of resistivity and sonic baseline values used in the
            calculation of DeltaLogR and TOC. e.g.:
            {'34_6_2S': [3., 104.], ...}
        ref_toc:
            dict
            dictionary which contain core sampled TOC and LOM (and MD) values for multiple wells. eg.:
            { '34_6_2S': {'md': np.array, 'toc': np.array, 'lom': np.array}, ... }
        start_r:
            dict
            Well names are keys, with resistivity values as items. This value is the lower plotting range of the
            resistivity in the Delta log R plot
            {'34_6_2S': 0.015, ...}
    :return:
    """
    log_table = small_log_table(log_table)
    if buffer is None:
        buffer = 50.
    if block_name is None:
        block_name = cw.def_lb_name

    text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'lightgray', 'alpha': 0.5}}

    time_step = kwargs.pop('time_step', 0.001)
    c_f = kwargs.pop('center_frequency', 30.)
    duration = kwargs.pop('duration', 0.512)
    scaling = kwargs.pop('scaling', 30.0)
    suffix = kwargs.pop('suffix', '')
    wiggle_fill_style = kwargs.pop('wiggle_fill_style', 'default')
    backus_length = kwargs.pop('backus_length', 5.0)
    source_rock_study = kwargs.pop('source_rock_study', None)
    ref_toc = kwargs.pop('ref_toc', None)
    start_r = kwargs.pop('start_r', 0.015)
    override_lom = kwargs.pop('override_lom', None)

    if wiggle_fill_style == 'opposite':
        fill_pos_style = 'pos-blue'
        fill_neg_style = 'neg-red'
    else:
        fill_pos_style = 'default'
        fill_neg_style = 'default'

    if isinstance(wi_names, str):
        wi_names = list(wi_names)

    # Set up plot window
    ax_names = None
    width_ratios = None
    if source_rock_study:
        ax_names = ['gr_ax', 'md_ax', 'twt_ax', 'res_ax', 'dlr_ax', 'toc_ax', 'cpi_ax', 'ai_ax', 'synt_ax']
        width_ratios = [1, 0.2, 0.2, 1, 1, 1, 1, 1, 1.8]
    else:
        ax_names = ['gr_ax', 'md_ax', 'twt_ax', 'res_ax', 'rho_ax', 'cpi_ax', 'ai_ax', 'synt_ax']
        width_ratios = [1, 0.2, 0.2, 1, 1, 1, 1, 2]
    fig, header_axes, axes = set_up_column_plot(ax_names=ax_names, width_ratios=width_ratios)

    if wi_names is None:
        fig.suptitle('Well {} {}'.format(well.well, suffix))
    else:
        fig.suptitle('{} interval in well {} {}'.format(', '.join(wi_names), well.well, suffix))

    #
    # Start plotting data
    #
    tb = well.block[block_name]  # this log block
    depth = tb.logs['depth'].data
    if wi_names is not None:
        _md_min = 1E6
        _md_max = -1E6
        for wi_name in wi_names:
            wi_name = wi_name.upper()
            if wis[well.well][wi_name][0] < _md_min:
                _md_min = wis[well.well][wi_name][0]
            if wis[well.well][wi_name][1] > _md_max:
                _md_max = wis[well.well][wi_name][1]
        # mask = np.ma.masked_inside(depth, wis[well.well][wi_names][0]-buffer, wis[well.well][wi_names][1]+buffer).mask
        mask = np.ma.masked_inside(depth, _md_min-buffer, _md_max+buffer).mask
    else:
        mask = np.ma.masked_inside(depth, np.nanmin(depth), np.nanmax(depth)).mask
        _md_min = None
        _md_max = None
    md_min = np.min(depth[mask])
    md_max = np.max(depth[mask])

    if 'Two-way time' in tb.log_types():
        # Extract the first TWT log
        twt = tb.get_logs_of_type('Two-way time')[0].data
        twt_min = np.nanmin(twt[mask])
        twt_max = np.nanmax(twt[mask])
        info_txt = 'Using real twt data'
    else:
        twt = None
        twt_min = None
        twt_max = None
        info_txt = 'Did not find TWT log, calculating from Sonic or Vp'
    print('INFO: {}'.format(info_txt))
    logger.info(info_txt)

    #
    # Gamma ray and Caliper
    try_these_log_types = ['Gamma ray', 'Caliper', 'Bit size', 'Inclination']
    log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

    # print(md_min, md_max)
    xlims = axis_plot(axes['gr_ax'], depth[mask],
              [tb.logs[lognames[xx]].data[mask] for xx in log_types],
              limits, styles, ylim=[md_min, md_max])
    header_plot(header_axes['gr_ax'], xlims, legends, styles)

    #for ax in [axes[x] for x in ax_names if x not in ['twt_ax', 'synt_ax']]:
    if wi_names is not None:
        for ax in [axes[x] for x in ax_names if x not in ['twt_ax']]:
            ax.axhline(y=_md_min, color='k', ls='--')
            ax.axhline(y=_md_max, color='k', ls='--')
    if intervals is not None:
        for interval in intervals:
            for ax in [axes[x] for x in ax_names if x not in ['twt_ax']]:
                ax.axhline(y=interval[0], color='k', lw=0.5)

    #
    # MD
    header_plot(header_axes['md_ax'], None, None, None, title='MD [m]')
    annotate_plot(axes['md_ax'], depth[mask], intervals=intervals, interval_names=interval_names, ylim=[md_min, md_max])

    #
    # TWT
    if twt is None:
        # Get the time-depth relation (two way time as a function of md) by calculating it from vp or sonic
        _x = None
        if 'P velocity' in list(log_table.keys()):
            _x = log_table['P velocity']
        elif 'Sonic' in list(log_table.keys()):
            _x = log_table['Sonic']
        if _x is not None:
            twt = well.time_to_depth(_x, templates=templates)
            if twt is None:
                header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]\nLacking info')
                annotate_plot(axes['twt_ax'], None)
            else:
                twt_min = np.min(twt[mask])
                twt_max = np.max(twt[mask])
                header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
                annotate_plot(axes['twt_ax'], twt[mask], ylim=[twt_min, twt_max])
        else:
            header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]\nLacking info')
            annotate_plot(axes['twt_ax'], None)
    else:
        header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
        annotate_plot(axes['twt_ax'], twt[mask], ylim=[twt_min, twt_max])

    if twt is not None and wi_names is not None:
        tops_twt = [twt[find_nearest(depth, y)] for y in [_md_min, _md_max]]
        for ax in [axes[x] for x in ['twt_ax', 'synt_ax']]:
            ax.axhline(y=tops_twt[0], color='k', ls='--')
            ax.axhline(y=tops_twt[1], color='k', ls='--')
    if twt is not None and intervals is not None:
        for interval in intervals:
            tops_twt = twt[find_nearest(depth, interval[0])]
            for ax in [axes[x] for x in ['twt_ax', 'synt_ax']]:
                ax.axhline(y=tops_twt, color='k', lw=0.5)

    #
    # Resistivity
    log_types = ['Resistivity']
    lognames = {ltype: [x.name for x in well.get_logs_of_type(ltype)] for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    cls = ['r', 'b', 'k', 'g', 'c']  # should not plot more than 5 lines in this plot!
    lws = [2, 1, 1, 1, 1]
    lss = ['-', '--', ':', '.-', '-']
    styles = [{'lw': lws[i], 'color': cls[i], 'ls': lss[i]} for i in range(len(lognames['Resistivity']))]
    legends = ['{} [{}]'.format(x, templates['Resistivity']['unit']) for x in lognames['Resistivity']]

    xlims = axis_log_plot(axes['res_ax'], depth[mask], [tb.logs[x].data[mask] for x in lognames['Resistivity']],
                  limits, styles, yticks=False, ylim=[md_min, md_max])
    header_plot(header_axes['res_ax'], xlims*len(legends), legends, styles)

    if source_rock_study is not None:
        from blixt_rp.rp.rp_core import delta_log_r, toc_from_delta_log_r
        # Try DeltaLogR
        log_types = ['Sonic', 'Resistivity']
        for necessary_log_type in log_types:
            if necessary_log_type not in well.block[block_name].log_types():
                raise IOError('{} log is missing in well {}'.format(necessary_log_type, well.well))
            if necessary_log_type not in list(log_table.keys()):
                raise IOError('{} log is missing in log table'.format(necessary_log_type))

        # Use the logs provided by the log_table
        lognames = {x: log_table[x] for x in log_types}

        if start_r is not None:
            sr = start_r[well.well]
        else:
            sr = 0.015
        limits = [[200., 0.], [sr, 10000 * sr]]
        styles = [{'lw': templates[x]['line width'],
                   'color': templates[x]['line color'],
                   'ls': templates[x]['line style']} for x in log_types]
        legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

        xlims = deltalogr_plot(axes['dlr_ax'], depth[mask],
                               [tb.logs[lognames[xx]].data[mask] for xx in log_types],
                               limits, styles, yticks=False, ylim=[md_min, md_max])
        header_plot(header_axes['dlr_ax'], xlims, legends, styles)

        # Calculate delta log r and TOC according to Passey et al. 1990
        if well.well in list(source_rock_study.keys()):
            if (ref_toc is not None) and (well.well in list(ref_toc.keys())):
                avg_lom = np.median(ref_toc[well.well]['lom'])
            else:
                avg_lom = 9.5

            if (override_lom is not None) and (well.well in list(override_lom.keys())):
                old_lom = avg_lom
                avg_lom = override_lom[well.well]
            else:
                old_lom = None

            legends = ['Calculated TOC [%]']
            limits = [[templates['TOC']['min'], templates['TOC']['max']]]
            styles = [{'lw': templates['TOC']['line width'],
                       'color': templates['TOC']['line color'],
                       'ls': templates['TOC']['line style']}]
            dlr = delta_log_r(
                well.block[block_name].logs[log_table['Resistivity']].data,
                well.block[block_name].logs[log_table['Sonic']].data,
                *source_rock_study[well.well])
            toc = toc_from_delta_log_r(dlr, avg_lom)
            xlims = axis_plot(axes['toc_ax'], depth[mask],
                              [toc.value[mask]],
                              limits, styles, yticks=False, ylim=[md_min, md_max])
            if (ref_toc is not None) and (well.well in list(ref_toc.keys())):  # plot TOC data from cuttings and cores
                axes['toc_ax'].scatter(ref_toc[well.well]['toc'], ref_toc[well.well]['md'],
                                       c='red', marker=templates['TOC']['marker'])
                # modify header plot to show observed data too
                xlims.append(xlims[0])
                legends.append('TOC from cuttings/core [%]')
                styles.append({'lw': 0, 'color': 'red', 'marker': templates['TOC']['marker']})

            header_plot(header_axes['toc_ax'], xlims, legends, styles)
            if old_lom is None:
                header_axes['toc_ax'].text(1.5, 1.0,
                                           '$r_0$={:.1f}, $ac_0$={:.1f}, LOM={:.1f}'.format(
                                               *source_rock_study[well.well], avg_lom),
                                           ha='center', va='bottom', **text_style)
            else:
                header_axes['toc_ax'].text(1.5, 1.0,
                                           '$r_0$={:.1f}, $ac_0$={:.1f}, LOM={:.1f}->{:.1f}'.format(
                                               *source_rock_study[well.well], old_lom, avg_lom),
                                           ha='center', va='bottom', **text_style)

    else:
        #
        # Rho
         try_these_log_types = ['Density', 'Neutron density']
         log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
         lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
         # Replace the density with the one selected by log_table
         lognames['Density'] = log_table['Density']
         limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
         styles = [{'lw': templates[x]['line width'],
                    'color': templates[x]['line color'],
                    'ls': templates[x]['line style']} for x in log_types]
         legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

         xlims = axis_plot(axes['rho_ax'], depth[mask],
                           [tb.logs[lognames[xx]].data[mask] for xx in log_types],
                           limits, styles, yticks=False, ylim=[md_min, md_max])
         header_plot(header_axes['rho_ax'], xlims, legends, styles)

    #
    # CPI
    # try_these_log_types = ['Saturation', 'Porosity', 'Volume']
    try_these_log_types = ['Saturation', 'Porosity']
    log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

    if len(log_types) == 0:
        header_plot(header_axes['cpi_ax'], None, None, None, title='CPI is lacking')
        axis_plot(axes['cpi_ax'], None, None, None, None, ylim=[md_min, md_max])
    else:
        xlims = axis_plot(axes['cpi_ax'], depth[mask],
                  [tb.logs[lognames[xx]].data[mask] for xx in log_types],
                  limits, styles, yticks=False, ylim=[md_min, md_max])
        header_plot(header_axes['cpi_ax'], xlims, legends, styles)

    #
    # AI
    tt = ''
    ba = None
    if 'Density' in list(log_table.keys()):
        tt += log_table['Density']
        if 'P velocity' in list(log_table.keys()):
            try:
                _vp = tb.logs[log_table['P velocity']].data
                _vs = tb.logs[log_table['S velocity']].data
                _rho = tb.logs[log_table['Density']].data
                if backus_length is not None:
                    ba = bra.backus(_vp, _vs, _rho, backus_length, tb.step)
            except:
                ba = None
            ai = tb.logs[log_table['Density']].data * tb.logs[log_table['P velocity']].data
            tt += '*{}'.format(log_table['P velocity'])
        elif 'Sonic' in list(log_table.keys()):
            ai = tb.logs[log_table['Density']].data / tb.logs[log_table['Sonic']].data
            tt += '/{}'.format(log_table['Sonic'])
        else:
            ai = None
    else:
        ai = None
    if ai is not None:
        #styles = [{'lw': 1, 'color': 'k', 'ls': '--'}]
        log_types = ['Impedance']
        lognames = {'Impedance': 'AI'}
        limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
        styles = [{'lw': templates[x]['line width'],
                   'color': templates[x]['line color'],
                   'ls': templates[x]['line style']} for x in log_types]
        legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]
        # TODO
        # Now we blindly assumes AI is in m/s g/cm3 Make this more robust
        # print(tb.logs[log_table['P velocity']].header['unit'])
        xlims = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]], limits, styles,
                  yticks=False, ylim=[md_min, md_max])
        #header_plot(header_axes['ai_ax'], xlims, ['AI ({})'.format(tt)], styles)
        if ba is not None: # Backus averaged data exists
            styles_ba = deepcopy(styles)
            styles_ba[0]['lw'] = styles[0]['lw'] * 3
            ai = ba[2] * ba[0]
            _ = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]], limits, styles_ba,
                          yticks=False, ylim=[md_min, md_max])
            # modify the settings to accommodate the backus averaged data
            limits.append(limits[0])
            styles.append(styles_ba[0])
            legends.append('AI backus {:.1f}m'.format(backus_length))
            xlims.append(xlims[0])
        header_plot(header_axes['ai_ax'], xlims, legends, styles)
    else:
        header_plot(header_axes['ai_ax'], None, None, None, title='AI is lacking')
        axis_plot(axes['ai_ax'], None, None, None, None, ylim=[md_min, md_max])


    #
    # Wiggles
    t = np.arange(np.nanmin(twt), np.nanmax(twt), time_step)  # A uniformly sampled array of time steps, from A to B
    if 'P velocity' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['P velocity']].data)
    elif 'Sonic' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=1./tb.logs[log_table['Sonic']].data)
    else:
        vp_t = None
    if 'S velocity' in list(log_table.keys()):
        vs_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['S velocity']].data)
    elif 'Shear sonic' in list(log_table.keys()):
        vs_t = np.interp(x=t, xp=twt, fp=1./tb.logs[log_table['Shear sonic']].data)
    else:
        vs_t = None
    if 'Density' in list(log_table.keys()):
        rho_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['Density']].data)
    else:
        rho_t = None

    if (vp_t is not None) and (vs_t is not None) and (rho_t is not None):
        reff = rp.reflectivity(vp_t, None, vs_t, None, rho_t, None, along_wiggle=True)
    else:
        reff = None

    if reff is not None:
        if wavelet is None:
            wavelet = bumw.ricker(duration, time_step, c_f)

        # Translate the mask to the time variable
        t_mask = np.ma.masked_inside(t[:-1], np.nanmin(twt[mask]), np.nanmax(twt[mask])).mask

        header_plot(header_axes['synt_ax'], None, None, None,
                    title='Incidence angle\n{} f={:.0f} Hz, l={:.3f} s'.format(wavelet['header']['Name'], c_f, duration))
        for inc_a in range(0, 35, 5):
            # wig = np.convolve(w, np.nan_to_num(reff(inc_a)), mode='same')
            wig = bumw.convolve_with_refl(wavelet['wavelet'], reff(inc_a), verbose=False)
            # wiggle_plot(axes['synt_ax'], t[:-1][t_mask], wig[t_mask], inc_a, scaling=scaling,
            wiggle_plot(axes['synt_ax'], t[:-1][t_mask], wig[:-1][t_mask], inc_a, scaling=scaling,
                        fill_pos_style=fill_pos_style, fill_neg_style=fill_neg_style, ylim=[twt_min, twt_max],
                        yticks=False)
    else:
        header_plot(header_axes['synt_ax'], None, None, None, title='Refl. coeff. lacking')
        wiggle_plot(axes['synt_ax'], None, None, None)

    if savefig is not None:
        fig.savefig(savefig)
    else:
        plt.show()


def overview_plot(wells, log_table, wis, wi_name, templates, log_types=None, block_name=None, savefig=None):
    """
    Overview plot designed to show data coverage in given working interval together with sea water depth
    Wells with no TVD data are plotted as dashed lines, with TVD the well is drawn with a solid line
    :param wells:
    :param log_table:
        dict
        Dictionary of log type: log name key: value pairs which decides which log, under each log type, to plot
        E.G.
            log_table = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob',
               'Porosity': 'phie',
               'Volume': 'vcl'}
    :param wis:
    :param wi_name:
    :param templates:
    :param block_name:
    :param savefig
        str
        full pathname of file to save the plot to
    :return:
    """
    if block_name is None:
        block_name = cw.def_lb_name
    if log_types is None:
        log_types = list(log_table.keys())
    _savefig = False
    if savefig is not None:
        _savefig = True

    sea_style = {'lw': 10, 'color': 'b', 'alpha': 0.5}
    kb_style = {'lw': 10, 'color': 'k', 'alpha': 0.5}
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('{} interval'.format(wi_name), pad=10.)

    wi_name = wi_name.upper()

    # create fake x axes and plotting ranges
    x = np.arange(len(wells))
    pw = 0.7 * len(wells) / (len(wells) + 1)  # approximate plotting width for each well

    # create common TVD depth axis for all wells
    y_max = -1E6

    wnames = []
    c_styles = {}  # style of line that defines the center of each well.
    for i, well in enumerate(wells.values()):
        wnames.append(well.well)
        c_styles[well.well] = {'color': 'k', 'ls': '--', 'lw': 1}
        # extract the relevant log block
        tb = well.block[block_name]

        # Try finding the depth interval of the desired working interval 'wi_name'
        # TODO missing working intervals are tricky to capture, sometimes they cause a TypeError, sometimes not
        # So I need to handle both cases
        try:
            well.calc_mask({}, name='XXX', wis=wis, wi_name=wi_name)
            mask = tb.masks['XXX'].data
            int_exists = True
        except TypeError:
            print('{} not present in well {}. Continue'.format(wi_name, well.well))
            mask = None
            int_exists = False
        if int_exists:
            if mask.all():
                # All values in mask is True, meaning that all data in well is masked out
                print('{} not present in well {}. Continue'.format(wi_name, well.well))
                mask = None
                int_exists = False

        if int_exists:
            if 'tvd' in well.log_names():
                c_styles[well.well] = {'color': 'k', 'ls': '-', 'lw': 1}
                depth_key = 'tvd'
            else:
                depth_key = 'depth'
            if wis[well.well][wi_name][1] > y_max:
                y_max = wis[well.well][wi_name][1]
            #if np.nanmax(tb.logs[depth_key].data[mask]) > y_max:
            #    y_max = np.nanmax(tb.logs[depth_key].data[mask])

            # plot the desired logs
            missing_logs_txt = ''
            for ltype in log_types:
                lname = log_table[ltype]
                if lname not in well.log_names():  # skip this log
                    missing_logs_txt += '{}\n'.format(lname)
                    continue
                if np.isnan(tb.logs[lname].data[mask]).all(): # all nan's
                    missing_logs_txt += '{}\n'.format(lname)
                    continue
                x = uu.norm(tb.logs[lname].data[mask], method='median')
                styles = {'lw': templates[ltype]['line width'],
                          'color': templates[ltype]['line color'], 'ls': templates[ltype]['line style']}
                ax.plot(i + x*pw, tb.logs[depth_key].data[mask], **styles)
            if len(missing_logs_txt) > 1:
                ax.text(i, min(tb.logs[depth_key].data[mask]), 'Missing logs: \n'+missing_logs_txt[:-1],
                    bbox = {'boxstyle': 'round', 'facecolor': 'orangered', 'alpha': 0.5},
                    verticalalignment = 'bottom',
                    horizontalalignment = 'center')
            else:
                ax.text(i, min(tb.logs[depth_key].data[mask]), 'No missing logs',
                        bbox={'boxstyle': 'round', 'facecolor': 'lightgreen', 'alpha': 0.5},
                        verticalalignment='bottom',
                        horizontalalignment='center')

        water_depth = well.get_from_well_info('water depth', templates=templates)
        kb = well.get_from_well_info('kb', templates=templates)
        ax.plot([i, i],
                [0.,  # this is not exact, because the linewidth makes the lines look longer than what they are
                np.abs(water_depth) + kb], label='_nolegend_',
                **sea_style)
        ax.plot([i, i], [0., templates[well.well]['kb']], label='_nolegend_', **kb_style)
        ax.axvline(i, label='_nolegend_', **c_styles[well.well])

    ax.set_ylim(y_max, 0)
    ax.set_ylabel('TVD [m]')
    ax.get_xaxis().set_ticks(range(len(wnames)))
    ax.get_xaxis().set_ticklabels(wnames)
    ax.legend(['{}: {}'.format(xx, log_table[xx]) for xx in list(log_table.keys())])
    fig.tight_layout()

    if _savefig:
        fig.savefig(savefig)
    else:
        plt.show()


def plot_depth_trends(wells, log_table, wis, wi_name, templates, cutoffs,
                      block_name=None, results_folder=None, verbose=True, suffix=None, **kwargs):
    """
    Plots the depth trends (TVD) for each individual log within the given working interval, for all wells

    :param wells:
    :param log_table:
    :param wis:
    :param wi_name:
    :param templates:
    :param cutoffs:
        dict
       E.G. {'depth': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
    :param block_name:
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
    :param kwargs:
    :return:
    """
    from scipy.optimize import least_squares
    from blixt_utils.misc.curve_fitting import residuals, linear_function
    from blixt_utils.utils import mask_string

    if suffix is None:
        suffix = ''
    buffer = 0.
    y_log_name = kwargs.pop('y_log_name', 'tvd')
    if block_name is None:
        block_name = cw.def_lb_name

    depth_trends = {}
    # Start looping over the different log types
    for log_type in log_table:
        log_name = log_table[log_type]
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = None, None

        # Start looping over wells and plot the data in TVD domain
        data_container = np.zeros(0)  # empty container
        tvd_container = np.zeros(0)
        legend_items = []
        tvd_min = 1E6
        tvd_max = -1E6
        for well in wells:
            if log_name not in wells[well].log_names():
                warn_txt = '{} log is missing in well {}'.format(log_name, well)
                logger .warning(warn_txt)
                print(warn_txt)
                continue
            wells[well].calc_mask(cutoffs, 'my_mask', log_table=log_table, wis=wis, wi_name=wi_name)
            mask = wells[well].block[block_name].masks['my_mask'].data
            legend_items.append(well)
            xdata = wells[well].block[block_name].logs[log_name].data[mask]
            if len(xdata) < 5:
                warn_txt = 'To few data points in {}, in well {}, after masking'.format(log_name, well)
                logger .warning(warn_txt)
                print(warn_txt)
                continue
            data_container = np.append(data_container, xdata)
            ydata = wells[well].block[block_name].logs['tvd'].data[mask]
            if len(ydata) < 5:
                warn_txt = 'To few data points in {}, in well {}, after masking'.format(log_name, well)
                logger .warning(warn_txt)
                print(warn_txt)
                continue
            tvd_container = np.append(tvd_container, ydata)
            if np.min(ydata) < tvd_min:
                tvd_min = np.min(ydata)
            if np.max(ydata) > tvd_max:
                tvd_max = np.max(ydata)
            if verbose:
                xp.plot(
                    xdata,
                    ydata,
                    cdata=templates[well]['color'],
                    mdata=templates[well]['symbol'],
                    xtempl=templates[log_type],
                    ytempl=templates['TVD'],
                    edge_color=False,
                    ax=ax
                )
        # Calculate depth trend
        verbosity_level = 0
        if verbose:
            verbosity_level = 2
        res = least_squares(residuals, [1., 1.], args=(tvd_container, data_container),
                            kwargs={'target_function': linear_function}, verbose=verbosity_level)

        depth_trends[log_name] = res.x
        new_tvd = np.linspace(tvd_min, tvd_max)
        if verbose:
            ax.plot(linear_function(new_tvd, *res.x), new_tvd)
            legend_items.append('{} = {:.3}xTVD + {:.3}'.format(log_name, res.x[0], res.x[1]))

            ax.set_title('{}: {}. {} {}'.format(log_type, log_name, mask_string(cutoffs, wi_name), suffix))
            ax.set_ylim(tvd_max, tvd_min)
            this_legend = ax.legend(
                legend_items,
                prop=FontProperties(size='smaller'),
                scatterpoints=1,
                markerscale=2,
                loc=1
            )

            if results_folder:
                if suffix != '' and suffix[0] != '_':
                    suffix = '_{}'.format(suffix)
                fig.savefig(os.path.join(results_folder, 'Depth trend {} in {}{}.png'.format(
                    log_name, wi_name, suffix
                )))
            else:
                plt.show()
    return depth_trends


def chi_rotation(well, log_tables, wis, wi_name, templates, buffer=None, chi_angles=None,
                 common_limit=None, ref_logtable=None,
                 plot_tops=None, annotate_tops=None,
                 block_name=None, results_folder=None, verbose=True, suffix=None, **kwargs):
    """
    For the given set of brine / oil / gas elastic logs, it plots extended elastic impedance at different chi angles
    in the selected working interval
    Args:
        well:
            well object
        log_tables:
            list
            List of (three) logtables which determines which elastic logs to use in the calculation of  the
            extended elastic impedance. E.G.
                [ {'P velocity': 'vp_brine', 'S velocity': 'vs_brine', ...},
                  {'P velocity': 'vp_oil', 'S velocity': 'vs_oil', ...}, ... ]
            The order is assumed to be brine, oil, gas, and they will be plotted in blue, green, red
        wis:
        wi_name:
        templates:
        buffer:
        chi_angles:
            list
            List of floats specifying the Chi angles (deg) to plot
            Defaults to [0., 5., 10., 15., 20., 30., 40., 50.]
        common_limit:
            list, tuple
            Common min, max values for the EEI at all Chi angles
        ref_logtable:
            dict
            A log table that is used as reference to make it easier for the user to orient themself
            on where we are looking.
            Preferably a gamma ray log, e.g. {'Gamma ray': 'gr'}
        plot_tops:
            list
            List of MD [m] values to plot horizontal lines across all plots
        annotate_tops:
            list
            List of strings, with same length as plot_tops, which are used to annotate the tops
        block_name:
        results_folder:
        verbose:
        suffix:
        **kwargs:

    Returns:

    """
    if buffer is None:
        buffer = 50.
    if chi_angles is None:
        chi_angles = [0., 5., 10., 15., 20., 30., 40., 50.]
    if block_name is None:
        block_name = cw.def_lb_name
    if suffix is None:
        suffix = ''
    if common_limit is not None:
        if len(common_limit) != 2:
            raise IOError('Common limits must have length 2, a min and max value')

    required_log_types = ['P velocity', 'S velocity', 'Density']
    # Check that the given log tables contain the required log types
    for log_table in log_tables:
        for log_type in required_log_types:
            if log_type not in log_table:
                raise IOError('Log type {} not included in log table'.format(log_type))

    tb = well.block[block_name]  # this log block
    depth = tb.logs['depth'].data
    mask = np.ma.masked_inside(depth, wis[well.well][wi_name][0] - buffer, wis[well.well][wi_name][1] + buffer).mask

    # Calculate the EEI for the different elastic logs specified in the log tables
    # First calculate common average values used in the normalization
    vp0 = np.nanmean(tb.logs[log_tables[0]['P velocity']].data[mask])
    vs0 = np.nanmean(tb.logs[log_tables[0]['S velocity']].data[mask])
    rho0 = np.nanmean(tb.logs[log_tables[0]['Density']].data[mask])
    eeis = []
    for log_table in log_tables:
        vp = tb.logs[log_table['P velocity']].data[mask]
        vs = tb.logs[log_table['S velocity']].data[mask]
        rho = tb.logs[log_table['Density']].data[mask]
        eeis.append(rp.eei(vp, vs, rho, vp0=vp0, vs0=vs0, rho0=rho0))

    # Calculate all EEI's and find the common limits for all log types at each chi angle
    all_chis = np.zeros((len(depth[mask]), len(chi_angles), len(log_tables)))
    limits = []
    common_limits = []
    for i, chi in enumerate(chi_angles):
        x_min = 1.E6
        x_max = -1.E6
        this_min, this_max = None, None
        for j in range(len(log_tables)):
            eei_at_this_chi = eeis[j](chi)
            all_chis[:, i, j] = eei_at_this_chi
            this_min = np.nanmin(eei_at_this_chi)
            this_max = np.nanmax(eei_at_this_chi)
            if this_max > x_max:
                x_max = this_max
            if this_min < x_min:
                x_min = this_min
        these_limits = [this_min, this_max]
        common_limits.append([these_limits] * len(log_tables))

    if common_limit is None:
        limits = common_limits
    else:
        for i, chi in enumerate(chi_angles):
            limits.append([common_limit] * len(log_tables))

    if ref_logtable is not None:
        ref_log_type = list(ref_logtable.keys())[0]
        ref_log_name = ref_logtable[ref_log_type]
        ref_data = tb.logs[ref_log_name].data[mask]
        ref_template = templates[ref_log_type]
    else:
        ref_data = None
        ref_template = None

    fig, axes, header_axes = chi_rotation_plot(all_chis, depth[mask], chi_angles, limits,
                                               reference_log=ref_data,
                                               reference_template=ref_template
                                               )
    if plot_tops is not None:
        for ax in axes:
            for this_top in plot_tops:
                ax.axhline(y=this_top, color='k', ls='--')
        if annotate_tops is not None:
            for i, this_name in enumerate(annotate_tops):
                axes[0].text(axes[0].get_xlim()[0], plot_tops[i], this_name, ha='left', va='bottom')

    axes[0].set_ylabel('Measured Depth [m]')
    fig.suptitle('EEI in {} interval in well {}, {}'.format(wi_name, well.well, suffix))

    if results_folder:
        if suffix != '' and suffix[0] != '_':
            suffix = '_{}'.format(suffix)
        fig.savefig(os.path.join(results_folder, 'EEI Chi {:.0f}-{:.0f} in {} in {}{}.png'.format(
            chi_angles[0], chi_angles[-1], wi_name, well.well, suffix
        )))
    else:
        plt.show()


def plot_wiggles(reflectivity, twt, wavelet, incident_angles=None, extract_at=None, extract_at_styles=None,
                 input_wiggles=None,
                 yticks=True, ax=None, title=None, **kwargs):
    """
    Plot seismic traces at different incident angles for the given reflectivity function and wavelet.
    UNLESS input_wiggles is used, in which case those are used and the reflectivity function and wavelet are
    ignored

    :param    reflectivity:
        A function of incident_angle [deg] which returns an array of reflectivities
    :param    twt:
        array
        Two way time in seconds
        OR
        depth (TWT, MD or TVD or ...) when input_wiggles are used, which can be in any depth domain

    :param    wavelet:
        dict
        dictionary with three keys:
            'wavelet': contains the wavelet amplitude
            'time': contains the time data [s]
            'header': a dictionary with info about the wavelet
        see blixt_utils.io.io.read_petrel_wavelet() for example

    :param    incident_angles:
        list
        List of incident angles in degrees that are used  in the reflectivity function
    :param    extract_at:
        float or list of floats
        TWT value(s) at which amplitudes are extracted as a function of incident angle
    :param extract_at_styles:
        dict or list of dicts
        Each dictionary are used to control the linestyle of the axhlines that indicate the extract_at depths
        E.G. [{'c': 'b', 'lw':2}, ...]
    :param input_wiggles:
        list, same length as incident_angles
        List of arrays containing the wiggles at the different incident angles
        Contradicts the normal behavior of plot_wiggles, and uses the input wiggles directly instead of
        calculating them from reflectivity and wavelet
    :param    :param yticks:
        bool
        if False the yticklabels are not shown
    :param    ax:
        matplotlib.pyplot Axes object
    :param    title:
        str
        Title of the plot
    :param    kwargs:
        Keyword arguments passed on to wiggle_plot()

    Returns:
        ava_curves:
            np.ndarray
            size len(incident_angles), len(extract_at)
            Array with AVA (amplitude versus angle) curves, one for each "extract_at" float.
            None if extract_at is None
        extract_at_indices
            list of indices of where the AVA curves have been extracted

    """
    scaling = kwargs.pop('scaling', 80)
    fill_pos_style = kwargs.pop('fill_pos_style', 'pos-blue')
    fill_neg_style = kwargs.pop('fill_neg_style', 'neg-red')
    if input_wiggles is None and wavelet is None:
        _dt = (np.nanmax(twt) - np.nanmin(twt)) / (len(twt) -1)
        wavelet = bumw.ricker(0.096, _dt, 25)
    if incident_angles is None:
        incident_angles = [10., 15., 20., 25., 30., 35., 40.]
    if extract_at is not None:
        if isinstance(extract_at, float):
            extract_at = [extract_at]
        elif not isinstance(extract_at, list):
            raise IOError('extract_at must be a float, or list of floats')
    if extract_at_styles is not None:
        if isinstance(extract_at_styles, dict):
            extract_at_styles = [extract_at_styles]
        if len(extract_at_styles) != len(extract_at):
            warn_txt = 'Length of extract_at ({}) must be the same as extract_at_styles ({])'.format(
                len(extract_at), len(extract_at_styles)
            )
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            extract_at_styles = None
    if input_wiggles is not None:
        if len(input_wiggles) != len(incident_angles):
            raise IOError('The number of input wiggles ({}) must the same as number of incident angles ({})'.format(
                len(input_wiggles), len(incident_angles)
            ))
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = ''

    # Find indexes where the AVA curves should be extracted
    if extract_at is not None:
        ava_curves = np.zeros((len(incident_angles), len(extract_at)))
        extract_at_indices = []
        for this_twt in extract_at:
            extract_at_indices.append(np.argmin((twt - this_twt)**2))
    else:
        ava_curves = None
        extract_at_indices = None

    for i, inc_angle in enumerate(incident_angles):
        if input_wiggles is None:
            wiggle = bumw.convolve_with_refl(wavelet['wavelet'], reflectivity(inc_angle))
        else:
            wiggle = input_wiggles[i]

        if extract_at is not None:
            for j, t in enumerate(extract_at_indices):
                ava_curves[i, j] = wiggle[t]

        wiggle_plot(ax, twt, wiggle, inc_angle, scaling=scaling, fill_pos_style=fill_pos_style,
                    fill_neg_style=fill_neg_style, **kwargs)

    ax.grid(which='major', alpha=0.5)
    if extract_at is not None:
        for i, _y in enumerate(extract_at):
            if extract_at_styles is not None:
                ax.axhline(_y, 0, 1, **extract_at_styles[i])
            else:
                ax.axhline(_y, 0, 1, c='k', ls='--')
    if not yticks:
        ax.get_yaxis().set_ticklabels([])
    else:
        ax.set_ylabel('TWT [s]')
    ax.set_xlabel('Incident angle [deg]')

    return ava_curves, extract_at_indices
