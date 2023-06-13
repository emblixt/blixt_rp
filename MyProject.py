import os
import sys
import numpy as np
import matplotlib.pyplot as plt

working_dir = 'C:\\Users\\marten\\PycharmProjects\\blixt_rp'
# for working with development versions of the code use
sys.path.append(working_dir)
sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')

import blixt_rp.core.well as cw
import blixt_rp.rp.rp_core as rp
from blixt_rp.core.well import Project

# Constants and definitions
block_name = cw.def_lb_name

opt1 = {'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}
opt2 = {'ha': 'right', 'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}
opt3 = {'ha': 'right', 'va': 'bottom',
        'bbox': {'boxstyle': 'Circle, pad=0.2', 'facecolor': '0.9', 'alpha': 0.7, 'edgecolor': 'none'}}

text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}
text_style2 = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5, 'ec': 'none'}}

l_fonts = 16
t_fonts = 13

green = '#70AD47'  # oil
red = '#FF0000'  # gas
blue = '#8A12EE'  # brine

results_folder = os.path.join(working_dir, 'results_folder')

# useful_cutoffs():
cutoffs_sands = {'Volume': ['<', 0.3], 'Porosity': ['>', 0.1]}
cutoffs_shales = {'Volume': ['>', 0.6]}

# useful_log_tables():
virgin_logs = {
	'P velocity': 'vp', 'S velocity': 'vs', 'Density': 'den', 'Porosity': 'phit', 'Volume': 'vcl',
	'Resistivity': 'rdep'}


def new_project(_working_dir):
    # Create new project
    return Project(
        name=None,
        working_dir=_working_dir,
        project_table=os.path.join(_working_dir, 'excels\\project_table.xlsx'),
        tops_file=None,
        tops_type=None,
        log_to_stdout=False
    )


def init(_working_dir, new=False):
    if new:
        wp = new_project(_working_dir)
    else:
        wp = Project(load_from=os.path.join(_working_dir, 'MyProject_log.txt'))
    wells = wp.load_all_wells(unit_convert_using_template=True)
    templates = wp.load_all_templates()
    wis = wp.load_all_wis()

    # calculate and add well path
    for wname in list(wells.keys()):
        wells[wname].add_well_path(wp.project_table, verbose=False)
    # calculate and add twt
    for wname in list(wells.keys()):
        wells[wname].add_twt(wp.project_table, verbose=False)

    # Convert sonic to velocities in necessary wells
    for wname in list(wells.keys()):
        if wname == 'WELL_D':
            wells[wname].block[block_name].sonic_to_vel(vel_names=['vp', 'vs'])

    return wp, wells, templates, wis


def predefined_fluid_subs(wp, wells, templates, wis):
    """
    Args:
        wp:
        wells:
        templates:
        wis:

    Returns:

    """
    import blixt_rp.core.fluids as fluids
    import blixt_rp.core.minerals as minerals

    # Create and read the mineral and fluid mixes from the project excel table
    my_minerals = minerals.MineralMix()
    my_fluids = fluids.FluidMix()
    my_minerals.read_excel(wp.project_table)
    my_fluids.read_excel(wp.project_table)

    cutoffs = {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie',
                 'Volume': 'vcl'}
	tag = 'fs'  # tag the resulting logs after fluid substitution
	rp.run_fluid_sub(wells, log_table, my_minerals, my_fluids, cutoffs, wis, tag, templates=templates)