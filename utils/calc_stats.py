from datetime import datetime
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

import utils.io as uio
from plotting import crossplot as xp
from utils.utils import nan_corrcoef

logger = logging.getLogger(__name__)
def_msk_name = 'Mask'  # default mask name
def_lb_name = 'LogBlock'  # default LogBlock name


def calc_stats2(
        wells,
        logname_dict,
        tops,
        intervals,
        cutoffs,
        templates=None,
        rokdoc_output=None,
        working_dir=None,
        suffix=None
):
    """
    Loop of over a set of wells, and a well tops dictionary and calculate the statistics over all wells within
    specified intervals.

    :param wells:
        dict
        Dictionary of wells containing core.well.Well objects
        eg.
            wp = Project(...)
            wells = wp.load_all_wells
    :param logname_dict:
        dict
        Dictionary of log type: log name key: value pairs to create statistics on
        The Vp, Vs, Rho and Phi logs are necessary for output to RokDoc compatible Sums & Average excel file
        E.G.
            logname_dict = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob',
               'Porosity': 'phie',
               'Volume': 'vcl'}
    :param tops:
        dict
        E.G.
        tops_file = 'C:/Users/marten/Google Drive/Blixt Geo AS/Projects/Well log plotter/Tops/strat_litho_wellbore.xlsx'
        tops = read_npd_tops(tops_file)

    :param intervals:
        list of dicts
        E.G.
            [
                {'name': 'Hekkingen sands',
                 'tops': ['HEKKINGEN FM', 'BASE HEKKINGEN']},
                {'name': 'Kolmule sands',
                 'tops': ['KOLMULE FM', 'BASE KOLMULE'
            ]
        The 'name' of the interval is used in the saved RokDoc compatible sums and averages file
        to name the averages
    :param cutoffs:
        dict
        Dictionary with log types as keys, and list with mask definition as value
        E.G.
            {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    :param templates:
        dict
        templates dictionary as returned from utils.io.project_templates()
    :param rokdoc_output:
        str
        full path name of file of which should contain the averages (RokDoc format)
        This requires that the log files include P- and S velocity, density, porosity and volume (Vsh)
    :param working_dir:
        str
        name of folder where results should be saved
    :param suffix:
        str
        Suffix added to output plots (png) to ease separating output from eachother
    :return:
    """
    # some initial setups
    suffix, cutoffs_str = fix_strings(suffix, cutoffs)
    log_types = list(logname_dict.keys())
    logs = list(logname_dict.values())
    ncols = len(logs)
    well_names = list(wells.keys())

    # Test if necessary log types for creating a RokDoc compatible output are present
    rokdoc_output = test_types(log_types, rokdoc_output)

    #    # open a figure for each log that is being analyzed
    figs_and_axes = [plt.subplots(figsize=(8, 6)) for xx in range(len(logs))]
    interval_axis = []
    interval_ticks = ['', ]

    # Start looping of all intervals
    for j, interval in enumerate(intervals):
        print('Interval: {}'.format(interval['name']))
        interval_axis.append(j)
        interval_ticks.append(interval['name'])

        # create container for results
        results, results_per_well, depth_from_top = create_containers(logs)

        # collect data
        collect_data_for_this_interval(wells, logs, tops, interval, results, results_per_well, depth_from_top, cutoffs)

        # create plot of logs vs depth and fill the interval plots
        plot_logs_vs_depth(logs, wells, interval, ncols, well_names, results_per_well, depth_from_top,
                           working_dir, cutoffs_str, suffix)

        # plot averages and standard deviations for each well
        plot_averages(j, logs, wells, results, results_per_well, figs_and_axes)

        # create histogram plot
        plot_histograms(logs, results, interval, ncols, well_names, results_per_well, working_dir, cutoffs_str, suffix)

        # Write result to RokDoc compatible excel Sums And Average xls file:
        save_rokdoc_output(rokdoc_output, results, interval, logname_dict, cutoffs_str, suffix)

    # arrange the interval plots
    well_names.append('All')
    for i, fax in enumerate(figs_and_axes):
        fax[1].legend(well_names, prop={'size': 10})
        fax[1].grid(True)
        if templates is not None:
            fax[1].set_xlim(templates[log_types[i]]['min'], templates[log_types[i]]['max'])
        fax[1].set_title(cutoffs_str)
        fax[1].set_yticklabels(interval_ticks)
        fax[0].tight_layout()
        if working_dir is not None:
            fax[0].savefig(os.path.join(
                working_dir,
                '{}_{}_intervals{}.png'.format(logs[i],
                                               cutoffs_str.replace('>', '_gt_').replace('<', '_lt_').replace('=',
                                                                                                             '_eq_'),
                                               suffix)))
            plt.close('all')


def fix_strings(_suffix, _cutoffs):
    """
    :param _suffix:
        str
        Suffix added to output plots (png) to ease separating output from eachother
    :param _cutoffs:
        dict
        Dictionary with log types as keys, and list with mask definition as value
        E.G.
            {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    :return:
        str
        Strings used in the analysis
    """
    if _suffix is None:
        _suffix = ''
    elif (len(_suffix) > 0) and (_suffix[0] != '_'):
        _suffix = '_' + _suffix

    _cutoffs_str = ''
    for key in _cutoffs:
        _cutoffs_str += '{}{}{:.2f}, '.format(key, _cutoffs[key][0], _cutoffs[key][1])
    _cutoffs_str = _cutoffs_str.rstrip(', ')

    return _suffix, _cutoffs_str


def test_types(_log_types, _rokdoc_output):
    # Test if necessary log types for creating a RokDoc compatible output are present
    for necessary_type in ['P velocity', 'S velocity', 'Density', 'Porosity', 'Volume']:
        if necessary_type not in _log_types:
            # set the output to None
            _rokdoc_output = None
            warn_txt = 'Logs of type {}, are lacking for storing output as a RokDoc Sums and Averages file'.format(
                necessary_type)
            logger.warning(warn_txt)
            print('WARNING: {}'.format(warn_txt))
    return _rokdoc_output


def create_containers(_logs):
    _results = {}
    _results_per_well = {}
    for key in _logs:
        _results[key.lower()] = np.empty(0)
    _depth_from_top = {}
    return _results, _results_per_well, _depth_from_top


def test_top_presence(tops, interval, this_well_name, results_per_well, depth_from_top):
    results_per_well[this_well_name] = {}
    _cont = False
    try:
        logger.info('Well: {}'.format(this_well_name))
        logger.info(' Top: {}: {:.2f} [m] MD'.format(interval['tops'][0],
                                                     tops[this_well_name][interval['tops'][0].upper()]))
        logger.info(' Base: {}: {:.2f} [m] MD'.format(interval['tops'][1],
                                                      tops[this_well_name][interval['tops'][1].upper()]))
    except:
        logger.info(
            'Tops {} & {} not present in {}'.format(interval['tops'][0], interval['tops'][1], this_well_name))
        depth_from_top[this_well_name] = np.empty(0)
        for key in log_types:
            results_per_well[this_well_name][key.lower()] = np.empty(0)
        _cont = True

    return _cont


def collect_data_for_this_interval(
        wells, logs, tops, interval, results, results_per_well, depth_from_top, cutoffs,
    ):
    # start looping over the well objects
    for this_well_name, well in wells.items():
        print(' Well: {}'.format(this_well_name))
        cont = test_top_presence(tops, interval, this_well_name, results_per_well, depth_from_top)
        if cont:
            continue

        # Create the mask
        well.calc_mask(
            cutoffs,
            name=def_msk_name,
            tops=tops,
            use_tops=interval['tops'],
            log_type_input=True
        )

        # Calculate mask
        mask = well.log_blocks[def_lb_name].masks[def_msk_name].data

        this_depth = well.log_blocks[def_lb_name].logs['depth'].data[mask]
        test_depth(this_depth, interval, this_well_name)

        # calculate the depth from the top for each well
        depth_from_top[this_well_name] = this_depth - tops[this_well_name][interval['tops'][0].upper()]

        for key in logs:
            this_data = well.log_blocks[def_lb_name].logs[key].data[mask]
            results[key] = np.append(results[key], this_data)
            results_per_well[this_well_name][key] = this_data


def test_depth(_this_depth, _interval, _this_well_name):
    if len(_this_depth) > 0:
        _this_string = '   Interval {}, in well {}, has the depth range: {:.1f} - {:.1f}'.format(
            _interval['name'],
            _this_well_name,
            np.nanmin(_this_depth),
            np.nanmax(_this_depth)
        )
    else:
        _this_string = '   Interval {}, is lacking in well {}'.format(
            _interval['name'],
            _this_well_name
        )
    logger.info(_this_string)
    print('   '.format(_this_string))


def plot_logs_vs_depth(logs, wells, interval, ncols, well_names, results_per_well, depth_from_top,
                       working_dir, cutoffs_str, suffix):
    # create plot of logs vs depth
    fig, axs = plt.subplots(
        nrows=1,
        ncols=ncols,
        sharey=True,
        figsize=(ncols * 6, 8))
    for i, key in enumerate(logs):
        key = key.lower()
        for k, well in enumerate(wells.values()):
            this_well_name = uio.fix_well_name(well.well)
            axs[i].plot(
                results_per_well[this_well_name][key],
                depth_from_top[this_well_name],
                c=xp.cnames[k])

        axs[i].set_xlabel(key)
        axs[i].set_ylim(axs[i].get_ylim()[1], axs[i].get_ylim()[0])

    axs[0].set_ylabel('Depth from {} [m]'.format(interval['tops'][0]))
    axs[-1].legend(well_names, prop={'size': 10})
    fig.suptitle('{}, {}'.format(interval['name'], cutoffs_str))
    fig.tight_layout()
    if working_dir is not None:
        fig.savefig(os.path.join(working_dir, '{}_logs_depth{}.png'.format(interval['name'], suffix)))


def plot_averages(j, logs, wells, results, results_per_well, figs_and_axes):
    for i, key in enumerate(logs):
        for k, well in enumerate(wells.values()):
            this_well_name = uio.fix_well_name(well.well)

            mn = np.nanmean(results_per_well[this_well_name][key])
            std = np.nanstd(results_per_well[this_well_name][key])
            figs_and_axes[i][1].errorbar(
                mn, j, xerr=std, fmt='none', capsize=10,
                capthick=1, elinewidth=1, ecolor=xp.cnames[k])
            figs_and_axes[i][1].scatter(
                mn, j,
                marker=xp.msymbols[k],
                c=xp.cnames[k],
                s=90)

        mn = np.nanmean(results[key])
        std = np.nanstd(results[key])
        figs_and_axes[i][1].errorbar(
            mn, j, xerr=std, fmt='none', capsize=20,
            capthick=3, elinewidth=3, ecolor='pink')
        figs_and_axes[i][1].scatter(mn, j, marker='*', s=360, c='pink')
        figs_and_axes[i][1].set_xlabel(key)


def plot_histograms(logs, results, interval, ncols, well_names, results_per_well,
                    working_dir, cutoffs_str, suffix):
    fig, axs = plt.subplots(nrows=ncols, ncols=1, figsize=(9, 8 * ncols))
    for i, key in enumerate(logs):
        key = key.lower()
        n, bins, patches = axs[i].hist(
            [results_per_well[wid][key] for wid in well_names],
            10,
            histtype='bar',
            stacked=True,
            label=well_names,
            color=[xp.cnames[k] for k in range(len(well_names))]
        )
        axs[i].set_ylabel('N')
        axs[i].set_xlabel(key)
        ylim = axs[i].get_ylim()
        mn = np.nanmean(results[key])
        std = np.nanstd(results[key])
        axs[i].plot([mn, mn], ylim, c='black', lw=2)
        axs[i].plot([mn + std, mn + std], ylim, 'b--')
        axs[i].plot([mn - std, mn - std], ylim, 'b--')
    axs[-1].legend(prop={'size': 10})
    axs[0].set_title('{}, {}'.format(interval['name'], cutoffs_str))
    fig.tight_layout()
    if working_dir is not None:
        fig.savefig(os.path.join(working_dir, '{}_logs_hist{}.png'.format(interval['name'], suffix)))

def save_rokdoc_output(rokdoc_output, results, interval, logname_dict, cutoffs_str, suffix):
    # Write result to RokDoc compatible excel Sums And Average xls file:
    if rokdoc_output is not None:
        uio.write_sums_and_averages(rokdoc_output,
                                    [
                                        '{}{}'.format(interval['name'], suffix),
                                        'NONE',
                                        'MD',
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        np.nanmean(results[logname_dict['P velocity']]),
                                        np.nanmean(results[logname_dict['S velocity']]),
                                        np.nanmean(results[logname_dict['Density']]),
                                        np.nanmedian(results[logname_dict['P velocity']]),
                                        np.nanmedian(results[logname_dict['S velocity']]),
                                        np.nanmedian(results[logname_dict['Density']]),
                                        np.nanmean(results[logname_dict['P velocity']]),
                                        # this should be a mode value, but uncertain what it means
                                        np.nanmean(results[logname_dict['S velocity']]),
                                        # this should be a mode value, but uncertain what it means
                                        np.nanmean(results[logname_dict['Density']]),
                                        # this should be a mode value, but uncertain what it means
                                        'NONE',
                                        np.nanmean(results[logname_dict['Porosity']]),
                                        np.nanstd(results[logname_dict['Porosity']]),
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        np.nanstd(results[logname_dict['P velocity']]),
                                        np.nanstd(results[logname_dict['S velocity']]),
                                        np.nanstd(results[logname_dict['Density']]),
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        nan_corrcoef(results[logname_dict['P velocity']],
                                                     results[logname_dict['S velocity']])[0, 1],
                                        nan_corrcoef(results[logname_dict['P velocity']],
                                                     results[logname_dict['Density']])[0, 1],
                                        nan_corrcoef(results[logname_dict['S velocity']],
                                                     results[logname_dict['Density']])[0, 1],
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        -999.25,
                                        np.nanmean(results[logname_dict['Volume']]),
                                        np.nanstd(results[logname_dict['Volume']]),
                                        'None',
                                        0.0,
                                        cutoffs_str,
                                        datetime.now()
                                    ]
                                    )
