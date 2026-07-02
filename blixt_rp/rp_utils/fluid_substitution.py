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
from blixt_rp.core.well import Well
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info
from blixt_rp.core.core import Intervals, LogTable, Cutoffs, CutoffRule
from blixt_rp.core.fluids_new import FluidMix
from blixt_rp.core.minerals import MineralMix

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

    # Calculate the elastic properties of the fluids
    fluid_mix.calc_elastics(wells, working_intervals, debug=verbose)

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
        my_fluids.read_excel(wp.project_table)
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

