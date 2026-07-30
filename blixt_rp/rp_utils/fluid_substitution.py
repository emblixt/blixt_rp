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
"""
import logging
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import unittest
import os, sys
import logging
from typing import Literal, List, Any, Dict
from pathlib import Path
import matplotlib.pyplot as plt

from bokeh.io import output_file
from bokeh.models import ColumnDataSource
import pint
from urllib3.fields import format_header_param

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
from blixt_rp.rp.rp_core import gassmann_vel
from blixt_rp.core.fluids_new import FluidMix
from blixt_rp.core.minerals_new import MineralMix
from blixt_rp.core.fluids_and_minerals import load_substitution_cases

logger = logging.getLogger(__name__)

def run_fluid_substitution(
        wells: dict,
        project_table: str | Path,
        log_table: LogTable,
        lithology_cutoffs: Cutoffs,
        working_intervals: Intervals,
        pvt_table: dict | None = None,
        verbose: bool = False
):

    """

    This 'advanced' option takes the fluid, and mineral, properties and their respective mixtures, from the
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

    -The 'Tag' column makes it possible to run different fluid sub. scenarios in the same well & interval, e.g.
    'fs_oil_08' and 'fs_gas_09'

    A *fluid substitution case* is defined by a unique combination of Well name, Interval name and Tag

    -The column 'Fluid name' of the "Fluid mixtures" sheet determines which of the fluids in the "Fluids" sheet to use

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

    :param wells:
        dict
        Dictionary with well name: Well object as key: value pairs
    :param project_table:
        str | Path
        file path to the project table .xlsx file that contains the Fluid and Mineral specifications
    :param log_table:
        LogTable
        LogTable object contain the mapping between a log type (e.g. 'P velocity') and a specific log (e.g. 'vp_oil')
    :param lithology_cutoffs:
        Cutoffs
        The Cutoffs object contains a list of cut-off rules which are used to define the lithology the
        fluid substitution should be applied to (that is, preferably sands)

        Some mineral properties are calculated using cutoffs too (e.g. Shale), but those cutoffs are specified
        in the project_table.
        Because of these mineral-specific cutoffs, it is important to NOT apply the lithology_cutoffs of the fluid-substitution
        before we start with the calculations.
    :param working_intervals:
        Intervals
        Holds the information about the working intervals that are defined for the wells
    :param pvt_table:
        dict
        Dictionary that contains pressure, density and bulk modulus values from typically a PVT simulation.
        It can be used to calculate the elastic properties as a function of pressure for fluids (e.g. gas condensates)
        which can't be modelled accurately from Batzle & Wang
        E.G.
            pvt_table = dict(
                pressure = Q_([10., 15., 20., 25., 30., 35., 40., 45., 50.], 'MPa'),
                rho = Q_([0.12, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.64], 'g/cc'),
                k = Q_([0.020, 0.035, 0.055, 0.080, 0.110, 0.145, 0.185, 0.230, 0.28], 'GPa')
            )
    :param verbose:
    :return:
    """

    # Load all fluid substitution cases:
    fluid_subst_cases = load_substitution_cases(project_table)

    # Load all mineral mixtures
    mineral_subst_cases = load_substitution_cases(project_table, from_fluid_mixture=False)

    last_well = 'XXX'
    # Iterate over all fluid substitution cases:
    for fsc in fluid_subst_cases:
        print('\n-----------------------------------------------------------------------------' )
        mod_history = f'Running fluid substitution in well {fsc.well}, in interval {fsc.interval}, with tag={fsc.tag}\n'
        print(mod_history)

        if last_well != fsc.well.upper():
            new_well = True
        else:
            new_well = False

        # Check if this well | interval pair is among the mineral cases
        msc = None
        for _msc in mineral_subst_cases:
            if _msc.well == fsc.well and _msc.interval == fsc.interval:
                msc = _msc
        if msc is None:
            err_txt = f'There is no mineral case listed in {os.path.basename(project_table)} that matches well {fsc.well} and interval {fsc.interval}'
            print_info(err_txt, 'error', logger, 'IOError')

        # Check if the well exist
        if fsc.well.upper() not in list(wells.keys()):
            err_txt = f'There is no well {fsc.well} among the input wells'
            print_info(err_txt, 'error', logger, 'IOError')

        # prepare the well to be used as input for the fluid and mineral elastics calculation
        well = wells[fsc.well.upper()]

        # The dict method harmonizes the logs in this well.
        well_dict = well.dict()

        # Add the necessary keys needed to calculate the mineral properties
        for _new_key, _log_type in zip(['vp', 'vs', 'rho'], ['P velocity', 'S velocity', 'Density']):
            well_dict[_new_key] = well_dict[log_table[_log_type]]

        # Get the cut-off rule that corresponds to this specific well and interval pair
        interval_rule = working_intervals.get_cutoff_rule(fsc.interval, fsc.well)

        # Make a copy of the lithology_cutoffs so that you can modify this local version of it
        local_cutoffs = deepcopy(lithology_cutoffs)

        # Add the interval_rule to these cutoffs
        local_cutoffs.append([interval_rule])
        # And calculate the mask. This mask identifies the lithology defined by the input lithology_cutoffs (hopefully sands)
        # within this interval
        local_mask = local_cutoffs.create_mask(well_dict)

        # Calculate the interval mask given the interval rules and input data
        interval_cutoffs = Cutoffs([interval_rule])
        interval_mask = interval_cutoffs.create_mask(well_dict)

        # In the unit-ignorant version of the dictionary, used by the mineral elastic calculation,
        # apply the interval mask
        # (Remember that we cant apply the input lithology_cutoffs, as we then typically only keep sands and remove the shale)
        well_dict_wo_units = {_key: _value.magnitude[interval_mask] if isinstance(_value, Q_) else _value[interval_mask] for _key, _value in well_dict.items()}

        # Print the log parameters and their length before and after applying the mask
        if verbose:
            print(list(well_dict_wo_units.keys()))
        mod_history += ' -Interval rule: {}\n'.format(interval_rule)
        for _key in list(well_dict_wo_units.keys()):
            if _key == 'md':
                mod_history += ' -{}: length before & after mask; {} & {}\n'.format(_key,
                    len(well_dict[_key]), len(well_dict_wo_units[_key]))
                mod_history += ' -MD data now covers: {} to  {}\n'.format(well_dict_wo_units[_key][0], well_dict_wo_units[_key][-1])

        # Extract the burial depth to this specific substitution case
        bd = working_intervals.get_interval(fsc.interval, fsc.well).mid

        # Calculate the fluid elastic properties ('well_dict_wo_units' is not used in this calculation)
        mod_history += fsc.calc_elastics(well_dict_wo_units, bd, pvt_table=pvt_table, verbose=verbose)

        # Calculate the mineral elastic properties ('bd' is not used in this calculation)
        mod_history += msc.calc_elastics(well_dict_wo_units, bd, verbose=verbose)

        # Now that we have calculated the fluid and mineral elastics, we can apply the initial cutoff on these
        # results.
        # Because we previously masked out everything outside our interval, we now identify the lithology within
        # this interval that matches the initial cutoff (hopefully sands)
        # To do that, we first create a data set that just contain the data needed by the cutoff rule
        _data = {_param.lower(): well_dict_wo_units[_param.lower()] for _param in lithology_cutoffs.cutoff_params}
        # Then we calculate the mask
        lithology_mask = lithology_cutoffs.create_mask(_data)


        # check consistency of input data
        # if verbose:
        if False:
            print('INPUT:')
            print('  Vp: ', type(well_dict['vp']), len(well_dict['vp'][local_mask]))
            print('  Vs: ', type(well_dict['vs']), len(well_dict['vs'][local_mask]))
            print('  Rho: ', type(well_dict['rho']), len(well_dict['rho'][local_mask]))
            print('  Porosity: ', type(well_dict[log_table['Porosity']]), len(well_dict[log_table['Porosity']][local_mask]))
            print('  Initial bulk: ', type(fsc.initial_bulk_gpa), len(fsc.initial_bulk_gpa[lithology_mask]))
            print('  Initial density: ', type(fsc.initial_density_gcc), len(fsc.initial_density_gcc[lithology_mask]))
            print('  Final bulk: ', type(fsc.final_bulk_gpa), len(fsc.final_bulk_gpa[lithology_mask]))
            print('  Final density: ', type(fsc.final_density_gcc), len(fsc.final_density_gcc[lithology_mask]))
            print('  Mineral bulk: ', type(msc.final_bulk_gpa), len(msc.final_bulk_gpa[lithology_mask]))

        if verbose:
            print(mod_history)

        # Notice the use of the 'local_mask' on the well logs, while the 'lithology_mask' is used on the calculated
        # elastic properties
        # Double check that their length matches
        n_1 = len(well_dict['vp'][local_mask])
        n_2 = len(fsc.initial_bulk_gpa[lithology_mask])
        if n_1 != n_2:
            err_txt = 'Length of the masks does not match: {} vs. {}'.format(n_1, n_2)
            print_info(err_txt, 'error', logger, 'IOError')

        # Now run the fluid substitution
        _vp_2, _vs_2, _rho_2, _k_2 = gassmann_vel(
            well_dict['vp'][local_mask].to('m/s').magnitude,
            well_dict['vs'][local_mask].to('m/s').magnitude,
            well_dict['rho'][local_mask].to('g/cc').magnitude,
            fsc.initial_bulk_gpa[lithology_mask],
            fsc.initial_density_gcc[lithology_mask],
            fsc.final_bulk_gpa[lithology_mask],
            fsc.final_density_gcc[lithology_mask],
            msc.final_bulk_gpa[lithology_mask],
            well_dict[log_table['Porosity']][local_mask].magnitude
        )

        # Create new vp, vs and rho logs based on copies of the input logs, and replace the original
        # values with the fluid substituted values only where the combined mask of interval and cutoffs are True
        for _type, _result, _result_units in zip(['P velocity', 'S velocity', 'Density'], [_vp_2, _vs_2, _rho_2], ['m/s', 'm/s', 'g/cc']):
            _log = well.get_log_curve(log_table[_type])
            _units = _log.units

            # Create a new name for the fluid substituted log
            _suffix = 'fs_{}'.format(fsc.tag) if fsc.tag is not None else '{}_fs'.format(_log.name)
            _name = '{}_{}'.format(_log.name, _suffix)

            # If the well is new, the new fluid_substituted log should not exist
            if new_well and _name in well.get_log_names:
                err_txt = 'The log {} already exists in well {}'.format(_name, well.name)
                print_info(err_txt, 'error', logger, 'IOError')

            # Only copy the LogCurve if the new log doesn't exist within the well, else modify the existing substituted log
            if _name not in well.get_log_names:
                # Copy the existing _log
                fs_log = _log.copy(_suffix)
            else:
                # Get the fluid substituted log
                fs_log = well.get_log_curve(_name)

            # Now replace the original with the fluid substituted results
            fs_log.data[local_mask] = Q_(_result, _result_units).to(_units)

            fs_log.header.modification_history += mod_history

            # Add the new fluid substituted log to the well if it wasn't there before
            if _name not in well.get_log_names:
                well.add_log(fs_log, if_log_exists='ask')

        last_well = fsc.well.upper()


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
        pvt_table = dict(
            pressure = Q_([10., 15., 20., 25., 30., 35., 40., 45., 50.], 'MPa'),
            rho = Q_([0.12, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.64], 'g/cc'),
            k = Q_([0.020, 0.035, 0.055, 0.080, 0.110, 0.145, 0.185, 0.230, 0.28], 'GPa')
        )

        # Load wells (and then automatically all templates)
        wp.load_all_wells(verbose=True)
        wells = {_w.name: _w for _w in wp.wells}
        print(list(wells.keys()))

        # Load working intervals
        wis = wp.load_all_wis()

        # Create cutoffs
        cutoffs = Cutoffs(
            cutoffs=[CutoffRule('Volume', '<', Q_(0.5, '')),
                     CutoffRule('Porosity', '>', Q_(0.1, ''))],
            name='sst'
        )

        # Create log table
        log_table = LogTable({'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry',
                              'Porosity': 'phie', 'Volume': 'vcl'})

        # Take into account that the Cutoffs are created using LogTypes and not log names
        cutoffs = cutoffs.use_log_table(log_table)

        # print the names of the important logs prior to fluid substitution:
        print('Original:')
        for _log_type in log_table.log_types:
            for _wn, _w in wells.items():
                print(' {} | {}: {}'.format(_wn, _log_type, ', '.join([_l.name for _l in _w.get_logs_of_type(_log_type)])))

        run_fluid_substitution(wells, wp.project_table, log_table, cutoffs, wis, pvt_table=pvt_table, verbose=True)

        # print the names of the important logs prior to fluid substitution:
        print('Final:')
        for _log_type in log_table.log_types:
            for _wn, _w in wells.items():
                print(' {} | {}: {}'.format(_wn, _log_type, ', '.join([_l.name for _l in _w.get_logs_of_type(_log_type)])))

        for _w in wells.values():
            _file = os.path.join(project_dir, 'blixt_rp\\results_folder\\fs_plot_{}.html'.format(_w.name))
            _w.plot([
                ['vp_dry', 'vp_dry_fs_MyTag', 'vp_dry_fs_Cond'],
                ['vs_dry', 'vs_dry_fs_MyTag', 'vs_dry_fs_Cond'],
                ['rho_dry', 'rho_dry_fs_MyTag', 'rho_dry_fs_Cond']],
                _file,
                wis=wis)
