#!/usr/bin/env python
# coding: utf-8

# Necessary imports
import matplotlib.pyplot as plt
import utils.io as uio
from core.well import Project
from plotting import plot_rp as prp
from plotting import plot_logs as ppl

def main():
    # # Introduction
    #
    # ## Project table
    # The Excel sheet *project_table.xlsx*, in the *excels* folder of the install directory, is the important hub for
    # which wells, and well logs, to use.
    # ## Create a wells project

    wp = Project(
        name='MyProject',
        project_table='excels/project_table.xlsx',
        #working_dir='X:/Some/File path/On your/system'
    )


    # ## Load selected wells
    # For below lines to work, the wells: WELL_A, WELL_B and WELL_C should be selected in the project table
    wells = wp.load_all_wells()

    # ### Data structure
    # The above result is a dictionary of *Well* objects, so to pick out a specific well use
    this_well = wells['WELL_A']
    print(type(this_well))
    print(this_well.header)


    # To list the Blocks within a well, use
    print(this_well.block.keys())

    # To look at the content of this Block, you can start by pointing variable to this Block, so that it is easier to work with it
    this_block = this_well.block['Logs']

    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie', 'Volume': 'vcl'}


    # ## Load selected wells
    wells = wp.load_all_wells()


    # ### Data structure
    # The above result is a dictionary of *Well* objects, so to pick out a specific well use
    this_well = wells['WELL_A']
    print('The header of well {}:'.format(this_well.well))
    print(this_well.header)
    print('\nThe blocks in well {}:'.format(this_well.well))
    print(this_well.block.keys())

    # As can be seen does this will only contain one Block member, which has the default name *Logs*.
    # To look at the content of this Block, you can start by pointing variable to this Block, so that it is easier to work with it

    this_block = this_well.block['Logs']
    print('\nThe members of block {}:'.format(this_block.name))
    print(this_block.keys())


    # The *header* contain information about the start, stop and step values used
    print('\nThe header of block {}:'.format(this_block.name))
    print(this_block.header)


    # To extract for example the start value of this well, use
    start = this_block.header.strt.value
    print(start)


    # ## Load the project templates
    templates = uio.project_templates(wp.project_table)


    # ## Load working intervals
    # The working intervals for each well is specified in the *Working intervals* sheet of the project table.
    wis = uio.project_working_intervals(wp.project_table)


    # ## Define cutoffs
    # Cut offs that are used to classify the data (e.g. sands or shales).
    cutoffs_sands = {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    cutoffs_shales = {'Volume': ['>', 0.5], 'Porosity': ['<', 0.1]}


    # ## Plotting data
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie', 'Volume': 'vcl'}
    ppl.overview_plot(wells, log_table, wis, 'SAND E', templates, log_types=list(log_table.keys()))

    w = wells['WELL_A']
    w.depth_plot('P velocity', wis=wis)

    w.calc_mask({}, 'D sands', wis=wis, wi_name='SAND D')
    mask = w.block['Logs'].masks['D sands'].data
    w.depth_plot('P velocity', wis=wis, mask=mask, show_masked=False)

    # If you instead are interested in a mask applied on values, use the following
    w.calc_mask(cutoffs_sands, 'sands')
    mask = w.block['Logs'].masks['sands'].data
    w.depth_plot('P velocity', wis=wis, mask=mask, show_masked=True)
    # If you're interested in one specific working interval, the command
    ppl.overview_plot(wells, log_table, wis, 'SAND E', templates, log_types=list(log_table.keys()))
    # gives you an overview whether the specified logs are present in the requested working interval, and what depth the interval is at.

    this_well.depth_plot('P velocity', wis=wis)

    this_well.calc_mask({}, 'D sands', wis=wis, wi_name='SAND D')
    mask = this_well.block['Logs'].masks['D sands'].data
    this_well.depth_plot('P velocity', wis=wis, mask=mask, show_masked=False)

    this_well.calc_mask(cutoffs_sands, 'sands')
    mask = this_well.block['Logs'].masks['sands'].data
    this_well.depth_plot('P velocity', wis=wis, mask=mask, show_masked=True)

    prp.plot_rp(
        wells,
        log_table,
        wis,
        'SAND F',
        cutoffs_sands,
        templates,
        plot_type='AI-VpVs',
        show_masked=True,
        edge_color=False
    )

    plt.show()

if __name__ == '__main__':
    main()
    plt.show()
