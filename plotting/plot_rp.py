import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import logging

from plotting import crossplot as xp

logger = logging.getLogger(__name__)

def plot_rp(wells, logname_dict, wis, wi_name, cutoffs, templates=None,
            plot_type=None, fig=None, ax=None, block_name='Logs', savefig=None, **kwargs):

    """
    Plots some standard rock physics crossplots for the given wells

    :param wells:
        dict
        dictionary with well names as keys, and core.well.Well object as values
        As returned from core.well.Project.load_all_wells()
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
    :param wis:
        dict
        working intervals, as defined in the "Working intervals" sheet of the project table, and
        loaded through:
        wp = Project()
        wis = utils.io.project_working_intervals(wp.project_table)
    :param wi_name:
        str
        name of working interval to plot
    :param cutoffs:
        dict
        Dictionary with log types as keys, and list with mask definition as value
        E.G.
            {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    :param templates:
        dict
        templates dictionary as returned from utils.io.project_templates()
    :param plot_type:
        str
        plot_type = 'AI-VpVs': AI versus Vp/Vs plot
    :param fig:

    :param ax:

    :param block_name:
        str
        Name of the log block from where the logs are picked
    :param savefig
        str
        full pathname of file to save the plot to
    :param **kwargs:
        keyword arguments passed on to crossplot.plot()
    :return:
    """
    #
    # some initial setups
    if plot_type is None:
        plot_type = 'AI-VpVs'
    log_types = list(logname_dict.keys())
    logs = list(logname_dict.values())

    #
    # set up plotting environment
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
        ax = fig.subplots()

    # load
    well_names = []
    desc = ''
    for wname, _well in wells.items():
        print('Plotting well {}'.format(wname))
        # create a deep copy of the well so that the original is not altered
        well = deepcopy(_well)
        these_wis = wis[wname]
        if wi_name.upper() not in [_x.upper() for _x in list(these_wis.keys())]:
            warn_txt = 'Working interval {} does not exist in well {}'.format(wi_name, wname)
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            continue  # this working interval does not exist in current well
        # test if necessary logs are in the well
        skip = False
        for _log in logs:
            if _log not in list(well.block[block_name].logs.keys()):
                warn_txt = '{} log is lacking in {}'.format(_log, wname)
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                skip = True
        if skip:
            continue
        # create mask based on working interval
        well.calc_mask({}, name='XXX', wis=wis, wi_name=wi_name)
        # apply mask (which also deletes the mask itself)
        well.apply_mask('XXX')
        # create mask based on cutoffs
        well.calc_mask(cutoffs, name='cmask', log_type_input=True)
        mask = well.block[block_name].masks['cmask'].data
        desc = well.block[block_name].masks['cmask'].header.desc

        # collect data for plot
        x_data = well.block[block_name].logs[logname_dict['P velocity']].data * \
                 well.block[block_name].logs[logname_dict['Density']].data
        x_unit = '{} {}'.format(
            well.block[block_name].logs[logname_dict['P velocity']].header.unit,
                 well.block[block_name].logs[logname_dict['Density']].header.unit)
        y_data = well.block[block_name].logs[logname_dict['P velocity']].data / \
                 well.block[block_name].logs[logname_dict['S velocity']].data
        y_unit = '-'

        well_names.append(wname)
        # start plotting

        xp.plot(
            x_data,
            y_data,
            cdata=templates[wname]['color'],
            mdata=templates[wname]['symbol'],
            xtempl={'full_name': 'AI',
                    'unit': x_unit},
            ytempl={'full_name': 'Vp/Vs',
                    'unit': y_unit},
            mask=mask,
            fig=fig,
            ax=ax,
            **kwargs
        )
    ax.autoscale(True, axis='both')
    ax.set_title('{}, {}'.format(wi_name, desc))
    legend_elements = []
    for wname in well_names:
        legend_elements.append(
            Line2D([0], [0],
                   color=templates[wname]['color'],
                   lw=0, marker=templates[wname]['symbol'],
                   label=wname))

    this_legend = ax.legend(
        legend_elements,
        well_names,
        prop=FontProperties(size='smaller'),
        scatterpoints=1,
        markerscale=2,
        loc=1)

    if savefig is not None:
        fig.savefig(savefig)
