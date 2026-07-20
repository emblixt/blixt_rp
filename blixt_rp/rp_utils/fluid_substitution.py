"""
Outline:
    This is to be a new home for fluid substitution, but with all the "new" functions and methods that allows
    pint Quantities

    It should have two main functionalities:
    1. One 'advanced' function
        Copy and rewrite the run_fluid_sub() from rp_core.py, AND run_fluid_substitution.py,
        OR, at least use them as inspiration

    2. One 'naiv' method that is interactive and uses bokeh as front-end
           Use test_fluidsub() in fluids_new as inspiration
           There is one problem with using the CrossPlotter for visualizing the results, as the parameters both
           before and after fluid substitution need to have the same name for them to be plotted simultaneously.
           So we need a new 'Well' (or DataSource) to store the substituted results.

    The 'advanced' option takes the fluid, and mineral, properties and their respective mixtures, from the
    'project table' Excel file.
    It describes a Gassmann fluid substitution in multiple intervals for multiple wells, where the fluid properties
    are given in the sheet "Fluids" of the 'project table' Excel file, and the initial and final mixtures
    (for each well and each interval) of the fluids are given in the "Fluid mixtures" sheet:

      -The column 'Use' of the "Fluid mixtures" sheet is either 'Yes' or 'No'. If not 'Yes', the given well and interval
    combination is ignored.

      -The column 'Substitution order' of the "Fluid mixtures" sheet is either 'Initial' or 'Final'. Which describes the
    substitution order that goes from initial to final.

      -The column 'Well name' of the "Fluid mixtures" sheet contains the name of the well for which the fluid substitution
    is to be done

      -The column 'Interval name' of the "Fluid mixtures" sheet contains the name of the interval (specified elsewhere as
    top and bottom measured depth) for which the fluid substitution is to be done

      -The column 'Fluid name' of the "Fluid mixtures" sheet which of the fluids in the "Fluids" sheet to use

      -The "Fluid type" column determines which fluid ('Brine', 'Oil' or 'Gas') the Batzle and Wang method should calculate.
    Or it can be 'User specified', which means that no calculation is needed - it just takes the given Bulk, Shear moduli
    and density.

      -The column "Volume fraction" of the "Fluid mixtures" sheet, which determines the fraction of each fluid type,
    is either the name of a specific well log curve (e.g. SW = water saturation), a constant (e.g. 0,2),
    or "complement" (meaning that it occupies the rest of the pore volume. The total pore volume is of course 1.

    For the "Fluids" sheet of the 'project table' Excel file, we have the following columns:

      -'Name': a unique name of this specific fluid

      -'Bulk moduli', 'Shear moduli' and 'Density' can be given as fixed values in these columns when the column
      'Calculation method' is set to 'User specified'

      -'Calculation method' is either 'User specified' or 'Batzle and Wang'

      - The other columns are parameters that are used by the Batzle and Wang calculation
"""
import logging
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import unittest
import os, sys
import logging

from bokeh.io import output_file
from bokeh.models import ColumnDataSource
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\rp_utils', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp.rp_core as rp
from blixt_rp import Q_
from blixt_rp.core.project_new import Project
from blixt_rp.core.well_new import Well
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info
from blixt_rp.core.core import Intervals, LogTable, Cutoffs, CutoffRule
from blixt_rp.core.fluids_new import FluidMix
from blixt_rp.core.minerals_new import MineralMix

logger = logging.getLogger(__name__)

def run_fluid_substitution(
        wells: dict,
        log_table: LogTable,
        mineral_mix: MineralMix,
        fluid_mix: FluidMix,
        cutoffs: Cutoffs,
        working_intervals: Intervals,
        tag: str | None = None,
        verbose: bool = False
):
    """
    Runs the fluid substitution for the input wells, in the lithologies defined by the cutoffs and stratigraphies
    defined in the fluid_mix.

    :param wells:
    :param log_table:
    :param mineral_mix:
    :param fluid_mix:
    :param cutoffs:
    :param working_intervals:
    :param tag:
    :param verbose:
    :return:
    """

    if tag is None:
        tag = ''
    elif tag[0] != '_':
        tag = '_{}'.format(tag)

    # Calculate the elastic properties of the fluids.
    # It is done for each well and for each interval we want to do
    # fluid substitution in
    fluid_mix.calc_elastics(wells, debug=verbose)

    # TODO Insert mineral calc_elastics here?, or later?

    # Create a list of necessary logs for this fluid substitution
    necessary_logs = [log_table[_x] for _x in ['Porosity', 'Density', 'P velocity', 'S velocity']]
    # Lists the necessary logs in a fluid mix
    necessary_logs += fluid_mix.necessary_logs()
    # TODO Create a similar method for the mineral mix
    necessary_logs += mineral_mix.necessary_logs()

# Loop over all wells
    for w, well in wells.items():
        w = w.lower()

        #
        # Test if well is listed in the fluid mix
        if w not in fluid_mix.well_names():
            print_info('Well {} not listed in the fluid mixture. Skipping'.format(w), 'warning', logger)
            continue

        info_txt = 'Starting Gassmann fluid substitution on well {}'.format(w)
        print_info('{}'.format(info_txt), 'info', logger)

        #
        # test if necessary logs are present in the well
        skip_this_well = False
        warn_txt = ''
        for _x in necessary_logs:
            if _x.lower() not in [_y.lower() for _y in well.log_names()]:
                warn_txt += 'Log name {} not present in well {}\n'.format(log_table[_x], w)
                skip_this_well = True
        if skip_this_well:
            warn_txt += '  SKIPPING well {}'.format(w)
            print_info(warn_txt, 'warning', logger)
            continue

        #
        # Calculate initial values
        # TODO the below method does not exist, and maybe it should be a method of
        # mineral_mix rather than the well?
        k0_dict = well.calc_vrh_bounds(mineral_mix, param='k', wis=working_intervals, method='Voigt-Reuss-Hill', block_name=block_name)
        por = well.get_log_curve(log_table['Porosity'])
        vp_1 = well.get_log_curve(log_table['P velocity'])
        vs_1 = well.get_log_curve(log_table['S velocity'])
        rho_1 = well.get_log_curve(log_table['Density'])
        # TODO the below method does not exist, maybe it should be a function of fluid_mix instead?
        rho_f1_dict = well.calc_vrh_bounds(fluid_mix.fluids['initial'], param='rho', wis=working_intervals, method='Voigt', block_name=block_name)
        k_f1_dict = well.calc_vrh_bounds(fluid_mix.fluids['initial'], param='k', wis=working_intervals, method='Reuss', block_name=block_name)

        #
        # Final fluids
        rho_f2_dict = well.calc_vrh_bounds(fluid_mix.fluids['final'], param='rho', wis=working_intervals, method='Voigt', block_name=block_name)
        k_f2_dict = well.calc_vrh_bounds(fluid_mix.fluids['final'], param='k', wis=working_intervals, method='Reuss', block_name=block_name)

        # Run the fluid substitution separately in each working interval
        for wi in fluid_mix.interval_names():
            # TODO We could perhaps extend the 'calc_elastics' of the fluid and mineral mixes
            # TODO so that it calculates the VRH bounds too? Then we could something like
            # Not 'calculates the VRH bounds too' It should calculate the VRH bounds of the initial and final fluid
            # instead of each constituent fluid
            k_f1 = fluid_mix.get_fluids( subst_order='initial', well_name=w, wi_name=wi)[0].k
            rho_f1 = fluid_mix.get_fluids( subst_order='initial', well_name=w, wi_name=wi)[0].rho
            k_f2 = fluid_mix.get_fluids( subst_order='final', well_name=w, wi_name=wi)[0].k
            rho_f2 = fluid_mix.get_fluids( subst_order='final', well_name=w, wi_name=wi)[0].rho
            # TODO And similarly for the mineral mix
            k0 = mineral_mix.get_minerals(well_name=w, wi_name=wi)[0].k
            # TODO Instead of the current solution where we pick up everything from the dictionaries we
            # created above. E.G.
            k_f1 = k_f1_dict[wi]
            rho_f1 = rho_f1_dict[wi]
            k_f2 = k_f2_dict[wi]
            rho_f2 = rho_f2_dict[wi]
            k0 = k0_dict[wi]

            # TODO Below code is old, needs to be updated accordingly
            # calculate the mask for the given cut-offs, and for the given working interval
            well.calc_mask(cutoffs, wis=wis, wi_name=wi, name='this_mask', log_table=log_table,
                           log_type_input=log_type_input)
            mask = lb.masks['this_mask'].values

            # Do the fluid substitution itself
            _vp_2, _vs_2, _rho_2, _k_2 = gassmann_vel(
                vp_1.values, vs_1.values, rho_1.values, k_f1, rho_f1, k_f2, rho_f2, k0, por)

            # Add the fluid substituted results to the well
            for xx, yy in zip([vp_1, vs_1, rho_1], [_vp_2, _vs_2, _rho_2]):
                new_name = deepcopy(xx.name)
                new_name += '{}'.format(tag.lower())
                new_header = deepcopy(xx.header)
                new_header.name += '{}'.format(tag.lower())
                new_header.desc = 'Fluid substituted {}'.format(xx.name)
                mod_history = 'Calculated using Gassmann fluid substitution using following\n'
                mod_history += 'Mineral mixtures: {}\n'.format(mm.print_minerals(wname, wi))
                mod_history += 'Initial fluids: {}\n'.format(
                    fm.print_fluids('initial', wname, wi))
                mod_history += 'Final fluids: {}\n'.format(
                    fm.print_fluids('final', wname, wi))
                new_header.modification_history = mod_history
                new_data = deepcopy(xx.values)
                new_data[mask] = yy[mask]
                lb.add_log(new_data, new_name, xx.get_log_type(), new_header)


class   InteractiveFluidSub:
    def __init__(self,
                 well: Well,
                 log_table: LogTable,
                 cutoffs: Cutoffs,
                 rpt_keywords: dict | None = None
                 ):
        """
        Creates a new well, with fluid substituted vp, vs and rho.
        The names of the vp, vs and rho logs will have the same name as in the input (as defined by the log_table)
        to be able to plot it in the same CrossPlotter

        :param well:
        :param log_table:
            LogTable that contains the vp, vs and rho logs we want to apply the fluid substitution on, and other
            logs e.g. vsh and phie which we would like to plot and/or use in the cutoffs.

        :param cutoffs:
        :param rpt_keywords
            dict
            Dictionary which contains values used for e.g. brine and oil density and bulk moduli.
            See rpt_wrapper_new.py for default values
        """
        pass


class TestCases(unittest.TestCase):
    def test_fluid_sub(self):
        # Create a project
        wp = Project(
            name='FluidSub',
            project_table=os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx'),
            working_dir=os.path.join(project_dir, 'blixt_rp')
        )

        # Load wells (and then automatically all templates)
        wells = wp.load_all_wells(verbose=True)
        print(wp.get_well_names)
        w = wp.get_well('WELL_F')

        # Load working intervals
        wis = wp.load_all_wis()

        # Load fluids
        my_fluids = FluidMix()
        my_fluids.read_excel(wp.project_table, wis)
        print(my_fluids.print_all_fluids())

        # Load minerals
        my_mins = MineralMix()
        my_mins.read_excel(wp.project_table)
        print(my_mins.print_all_minerals())

        # Create cutoffs
        cutoffs = Cutoffs(
            cutoffs=[CutoffRule('Volume', '<', Q_(0.5, '')),
                     CutoffRule('Porosity', '>', Q_(0.1, ''))],
            name='sst'
        )

        # Create log table
        log_table = LogTable({'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry',
                     'Porosity': 'phie', 'Volume': 'vcl'})

        # Fluid substitution
        tag = 'fs'  # tag the resulting logs after fluid substitution
        run_fluid_substitution(
            {w.name: w},
            log_table,
            mineral_mix=my_mins,
            fluid_mix=my_fluids,
            cutoffs=cutoffs,
            working_intervals=wis,
            tag=tag,
            verbose=True
        )

