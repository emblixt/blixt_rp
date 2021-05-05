import os
import matplotlib.pyplot as plt
import numpy as np
import logging

import blixt_rp.rp_utils.definitions as ud
import blixt_utils.io.io as uio
from blixt_utils.plotting import crossplot as xp
from blixt_utils.misc.templates import handle_template

logger = logging.getLogger(__name__)


def fix_strings(_suffix, _cutoffs, _log_table):
    """
    :param _suffix:
        str
        Suffix added to output plots (png) to ease separating output from eachother

    :param _cutoffs:
        dict
        Dictionary with log types as keys, and list with mask definition as value
        E.G.
            {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}

    :param _log_table:
        dict
        Dictionary with log types as keys, and log name as value
        E.G.
            {'Volume': 'vsh', 'Porosity': 'phit'}
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

    _log_table_str = ''
    for key in _log_table:
        _log_table_str += '{}: {}, '.format(key, _log_table[key])
    _log_table_str = _log_table_str.rstrip(', ')

    return _suffix, _cutoffs_str, _log_table_str


def test_types(_log_types):
    # Test if necessary log types for creating a RokDoc compatible output are present
    for necessary_type in ['P velocity', 'S velocity', 'Density', 'Porosity', 'Volume']:
        if necessary_type not in _log_types:
            # set the output to None
            warn_txt = 'Logs of type {}, are lacking for storing output as a RokDoc Sums and Averages file'.format(
                necessary_type)
            logger.warning(warn_txt)
            print('WARNING: {}'.format(warn_txt))
            return False
    return True


def create_containers(_logs):
    _results = {}
    _results_per_well = {}
    for key in _logs:
        _results[key.lower()] = np.empty(0)
    _depth_from_top = {}
    return _results, _results_per_well, _depth_from_top


def plot_logs_vs_depth(logs, log_types, wells, wi_name, results_per_well, depth_from_top,
                       cutoffs_str, suffix, templates, axs, first_row):
    # create plot of logs vs depth

    for i, key in enumerate(logs):
        for k, well in enumerate(wells.values()):
            this_well_name = uio.fix_well_name(well.well)
            try:
                axs[i].plot(
                    results_per_well[this_well_name][key],
                    depth_from_top[this_well_name],
                    c=templates[this_well_name]['color'])
            except ValueError:
                warn_txt = 'Log {}, is lacking in working interval {} in well {}'.format(
                    key, wi_name, this_well_name)
                logger.warning(warn_txt)
                print('WARNING: {}'.format(warn_txt))
                continue

        if log_types[i] in list(templates.keys()):
            _,  lmts, _, _, _, _ = handle_template(templates[log_types[i]])
            axs[i].set_xlim(*lmts)
        axs[i].set_xlabel(key)
        axs[i].set_ylim(axs[i].get_ylim()[::-1])

    axs[0].set_ylabel('Depth from {}, {} [m]'.format(wi_name, suffix))
    if first_row:
        axs[-1].legend([w.well for w in list(wells.values())], prop={'size': 10})


def plot_averages(j, logs, log_types, wells, results, results_per_well, cutoffs_str, suffix, templates, avg_plots_axes):
    for i, key in enumerate(logs):
        for k, well in enumerate(wells.values()):
            this_well_name = uio.fix_well_name(well.well)

            mn = np.nanmean(results_per_well[this_well_name][key])
            std = np.nanstd(results_per_well[this_well_name][key])
            avg_plots_axes[i][1].errorbar(
                mn, j, xerr=std, fmt='none', capsize=10,
                capthick=1, elinewidth=1, ecolor=templates[this_well_name]['color'])
            avg_plots_axes[i][1].scatter(
                mn, j,
                marker=templates[this_well_name]['symbol'],
                c=templates[this_well_name]['color'],
                s=90)

        mn = np.nanmean(results[key])
        std = np.nanstd(results[key])
        avg_plots_axes[i][1].errorbar(
            mn, j, xerr=std, fmt='none', capsize=20,
            capthick=3, elinewidth=3, ecolor='pink')
        avg_plots_axes[i][1].scatter(mn, j, marker='*', s=360, c='pink')
        avg_plots_axes[i][1].grid(True)
        well_labels = [well.well for well in list(wells.values())]
        well_labels.append('All')
        avg_plots_axes[i][1].legend(well_labels, prop={'size': 10})

        if log_types[i] in list(templates.keys()):
            xlabel,  lmts, _, _, _, _ = handle_template(templates[log_types[i]])
            avg_plots_axes[i][1].set_xlim(*lmts)
            avg_plots_axes[i][1].set_xlabel(xlabel)
        else:
            avg_plots_axes[i][1].set_xlabel(key)


class SetUpCalculation:
    """
    Set up the logistics around calculating the statistics and storing the results

    """
    def __init__(self,
                 wells,
                 wi_names,
                 cutoffs,
                 log_table,
                 suffix,
                 wis,
                 templates=None,
                 rokdoc_output=None,
                 working_dir=None,
                 block_name=ud.def_lb_name,
                 ):
        """
        Loop of over a set of wells, and a well tops dictionary and calculate the statistics over all wells within
        specified working intervals.

        :param wells:
            dict
            Dictionary of wells containing core.well.Well objects
            eg.
                wp = Project(...)
                wells = wp.load_all_wells
            Statistics are calculated for the given logs across all wells, for each specified working intervals

        :param wi_names:
            str or list
            of working interval names
            E.G.
                'Hekkingen sands'
                or
                [ 'Hekkingen sands',  'Kolmule sands', ... ]
            The number of working intervals determines how the calculations are carried out

        :param cutoffs:
            dict or list of dicts
            Dictionary with log types as keys, and list with mask definition as value
            E.G.
                {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
            If this is a single dictionary, it will be used in all above given working intervals.
            If it is a list, it must have the same length as wi_names

        :param log_table:
            dict or list of dicts
            Dictionary of {log type: log name} {key: value} pairs to create statistics on
            The Vp, Vs, Rho and Phi logs are necessary for output to RokDoc compatible Sums & Average excel file
            E.G.
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
            If this is a single dictionary, it will be used in all above given working intervals.
            If it is a list, it must have the same length as wi_names

        :param suffix:
            str or list of strings
            A combination of the working interval and the suffix is used to tag each line in the output excel sheet
            If this is a single dictionary, it will be used in all above given working intervals.
            If it is a list, it must have the same length as wi_names

        :param wis:
            dict
            working intervals, as defined in the "Working intervals" sheet of the project table, and
            loaded through:
            wp = Project()
            wis = wp.load_all_wis()

        :param templates:
            dict
            templates dictionary as returned from rp_utils.io.project_templates()
        :param rokdoc_output:
            str
            full path name of file of which should contain the averages (RokDoc format)
            This requires that the log files include P- and S velocity, density, porosity and volume (Vsh)
        :param working_dir:
            str
            name of folder where results should be saved
        :param block_name:
            str
            Name of the log block from where the logs are picked

        :return:
        """
        # Check input consistency
        if isinstance(wi_names, str):
            # We are studying a single working interval
            if isinstance(log_table, list):
                raise ValueError('Only one log_table is allowed for the single working interval studied: {}'.format(
                    wi_names
                ))
            else:
                self.log_table = [log_table]
            if isinstance(cutoffs, list):
                raise ValueError('Only set of cutoffs is allowed for the single working interval studied: {}'.format(
                    wi_names
                ))
            else:
                self.cutoffs = [cutoffs]
            if isinstance(suffix, list):
                raise ValueError('Only one suffix is allowed for the single working interval studied: {}'.format(
                    wi_names
                ))
            else:
                self.suffix = [suffix]
            self.wi_names = [wi_names]
        elif isinstance(wi_names, list):
            self.wi_names = wi_names
            # we are studying multiple,working intervals
            if isinstance(log_table, dict):
                self.log_table = [log_table] * len(wi_names)
            elif isinstance(log_table, list):
                if len(wi_names) != len(log_table):
                    raise ValueError(
                        'The number of log_tables, {}, must be same as number of working intervals, {}'.format(
                            len(log_table), len(wi_names)))
            if isinstance(cutoffs, dict):
                self.cutoffs = [cutoffs] * len(wi_names)
            elif isinstance(cutoffs, list):
                if len(wi_names) != len(cutoffs):
                    raise ValueError(
                        'The number of cutoffs, {}, must be same as number of working intervals, {}'.format(
                            len(cutoffs), len(wi_names)))
            if isinstance(suffix, str):
                self.suffix = [suffix] * len(wi_names)
            elif isinstance(suffix, list):
                if len(wi_names) != len(suffix):
                    raise ValueError(
                        'The number of suffixes, {}, must be same as number of working intervals, {}'.format(
                            len(suffix), len(wi_names)))

        self.n_intervals = len(wi_names)
        self.n_logs = max([len(_x) for _x in self.log_table])
        self.n_well = len(wells)

        if templates is None:
            templates = {}
            for k, well in enumerate(wells):
                templates[uio.fix_well_name(well.well)] = {'color': xp.cnames[k], 'symbol': xp.msymbols[k]}

        if working_dir is not None:
            if not os.path.isdir(working_dir):
                warn_txt = 'The specified folder, where results should be saved, does not exist: {}'.format(working_dir)
                logger.warning(warn_txt)
                print(warn_txt)
                working_dir = None

        # Set up plots
        #self.fig_hist, self.axes_hist = plt.subplots(nrows=self.n_intervals,
        #                                   ncols=self.n_logs,
        #                                   figsize=(9 * self.n_logs, 8 * self.n_intervals))
        self.fig_logs, self.axes_logs = plt.subplots(nrows=self.n_intervals,
                                           ncols=self.n_logs,
                                           figsize=(2 * self.n_logs, 3 * self.n_intervals))
        self.avg_plots_axes = [plt.subplots(figsize=(9, 8)) for kk in  range(self.n_logs)]

        self.wells = wells
        self.wis = wis
        self.templates = templates
        self.rokdoc_output = rokdoc_output
        self.working_dir = working_dir
        self.block_name = block_name


class CalculateStats:
    def __init__(self, calc_setup):
        from blixt_rp.rp_utils.calc_stats import collect_data_for_this_interval, save_rokdoc_output
        """
        :param calc_setup:
            SetUpCalculation
            SetUpCalculation object

        """
        # Start looping over all intervals
        wi_labels = []
        for j, wi_name in enumerate(calc_setup.wi_names):
            this_label = '{}, {}'.format(wi_name, calc_setup.suffix[j])
            wi_labels.append(this_label)
            this_suffix, cutoffs_str, log_table_str = fix_strings(
                calc_setup.suffix[j],
                calc_setup.cutoffs[j],
                calc_setup.log_table[j]
            )
            print('Interval: {}. {}'.format(this_label, cutoffs_str))

            logs = [n.lower() for n in list(calc_setup.log_table[j].values())]
            log_types = [n for n in list(calc_setup.log_table[j].keys())]

            # create container for results
            results, results_per_well, depth_from_top = create_containers(logs)

            # collect data
            collect_data_for_this_interval(
                calc_setup.wells, logs, calc_setup.wis, wi_name, results, results_per_well,
                depth_from_top, calc_setup.cutoffs[j], calc_setup.log_table[j], block_name=calc_setup.block_name)

            # create plot of logs vs depth and fill the interval plots
            plot_logs_vs_depth(logs, log_types, calc_setup.wells, wi_name, results_per_well, depth_from_top,
                               cutoffs_str, calc_setup.suffix[j], templates=calc_setup.templates,
                               axs=calc_setup.axes_logs[j][:], first_row=j == 0)

            # Create plots of average values within each working interval, with one plot per log
            plot_averages(j, logs, log_types, calc_setup.wells, results, results_per_well,
                               cutoffs_str, calc_setup.suffix[j], calc_setup.templates,
                               calc_setup.avg_plots_axes)

            if calc_setup.rokdoc_output is not None:  # Try saving output excel sheeta
                if test_types(log_types):  # Test if necessary log types are present
                    save_rokdoc_output(calc_setup.rokdoc_output, results, wi_name, calc_setup.log_table[j],
                                       cutoffs_str, this_suffix)

        # Add y tick labels of the working intervals
        for i in range(len(calc_setup.avg_plots_axes)):
            calc_setup.avg_plots_axes[i][1].set_yticks(range(len(calc_setup.wi_names)))
            calc_setup.avg_plots_axes[i][1].set_yticklabels(wi_labels)

        calc_setup.fig_logs.tight_layout()
        if calc_setup.working_dir is not None:
            calc_setup.fig_logs.savefig(os.path.join(calc_setup.working_dir, 'logs_vs_depth.png'), dpi=300)
            for i, key in enumerate([n for n in list(calc_setup.log_table[0].keys())]):
                calc_setup.avg_plots_axes[i][0].savefig(os.path.join(calc_setup.working_dir, '{}.png'.format(key)),
                                                        dpi=300)


def test_calc_stat():
    from blixt_rp.core.well import Project

    wp = Project(
        name='MyProject',
        project_table='excels/project_table.xlsx')

    # For below lines to work, the wells: WELL_A, WELL_B and WELL_C should be selected in the project table
    wells = wp.load_all_wells()

    # Load templates
    templates = wp.load_all_templates()

    # Load working intervals
    wis = wp.load_all_wis()
    # and determine which intervals to calculate the statistics in
    wi_sands = ['Sand H', 'Sand F', 'Sand E', 'Sand D', 'Sand C']
    wi_shales = ['Shale G', 'Shale C']

    # Cut offs that are used to classify the data (e.g. sands or shales).
    cutoffs_sands = {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    cutoffs_shales = {'Volume': ['>', 0.5], 'Porosity': ['<', 0.1]}

    # Define which logs to use
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie', 'Volume': 'vcl'}

    # Determine which excel sheet that should contain the results
    #rd_file = os.path.join(wp.working_dir, 'results_folder/test_statistics.xlsx')
    rd_file = None

    stats_set_up = SetUpCalculation(wells, wi_sands, cutoffs_sands, log_table, 'TEST', wis, templates, rd_file,
                                    working_dir=os.path.join(wp.working_dir, 'results_folder'))

    CalculateStats(stats_set_up)



if __name__ == '__main__':
    test_calc_stat()
