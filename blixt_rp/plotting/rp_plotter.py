"""
Plots log data from multiple wells, from specific intervals (or whole well) using cut offs or not
using standard modulus or other parameters used in rock physics (e.g. bulk & shear modulus, poisson ratio, vp/vs ratio
"""
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper, CheckboxEditor, Select)
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, TextInput
from bokeh.plotting import figure, show
from bokeh.plotting import column, row
from bokeh.io import output_file

import sys
import os.path
import unittest
import numpy as np
import matplotlib.pyplot as plt
import logging

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.utils import print_info, add_one, fix_well_name, cycle_colors, isnan
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_rp.core.core import Intervals, WorkingIntervalsTable
from blixt_rp.core.core import Cutoffs, ClassificationTable
from blixt_rp.plotting.cross_plotter import CrossPlotter
from blixt_rp import Q_

output_file(os.path.join(project_dir, 'blixt_rp\\results_folder\\plot.html'))
logger = logging.getLogger(__name__)

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\plotting',
    'test_data'))
las_fileA = os.path.join(test_file_dir, "Well A.las")


class RockPhysicsPlotter(CrossPlotter):
    """
    Class for handling the rock physics plot

    Intended use
    > data_sources = ...  # See cross_plotter.py DataSource for explanation
    > working_intervals = ...
    > cutoffs = ...
    > rp_plot = RockPhysicsPlotter(data_sources, working_intervals, cutoffs)
    >

    NOTE the data_sources must have
    """
    def __init__(self,
                 data_sources: dict,
                 working_intervals: Intervals | None = None,
                 cutoffs: Cutoffs | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 tools: list | None = None
                 ):
        """
        Plots the data in data_sources in a x vs. y cross plot, where x is the independent variable
        (typically TVD) and y is the dependent variable (e.g. Vp), and it tries to find the function
        f(x) that best matches y(x)

        :param data_sources:
            dict of DataSource objects
            e.g.
                {w.name: w.data_source() for w in project.wells}
            where project is a Project object from project_new.py

        :param working_intervals:
        :param cutoffs:
        :param width:
        :param height:
        :param tools:
        """

        if width is None:
            width = 900

        super().__init__(data_sources, working_intervals=working_intervals, cutoffs=cutoffs, width=width,
                         height=height, tools=tools)


    def draw(self, cds: ColumnDataSource, set_all_intervals_active: bool = False, verbose: bool = False):
        from bokeh.models import Button, Tooltip

        if verbose:
            cds.js_on_change('patching', CustomJS(
                args=dict(source=cds),
                code="""
                console.log('Patching event detected in RockPhysicsPlotter');
                """
            ))

            cds.js_on_change('data', CustomJS(
                args=dict(source=cds),
                code="""
                console.log('Data event detected in RockPhysicsPlotter');
                """
            ))

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis = super().draw(cds)
        style_table = self.add_data_source_style(cds)

        return (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask,
                ct_guis,
                wis_guis,
                style_table)

class TestCases(unittest.TestCase):

    def test_data_cds(self, unit_test=True):
        import blixt_rp.plotting.cross_plotter as xp
        test = xp.TestCases()

        ds1, ds2, wis, coffs = test.test_data()

        tp = RockPhysicsPlotter({str(_s.name):_s for _s in [ds1, ds2]},
                                cutoffs=coffs,
                                working_intervals=wis,
                                )

        d_cds = tp.cds

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table = tp.draw(d_cds)
        if unit_test:
            show(
                row(
                    column(
                        xplot,
                        row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask)
                    ),
                    column(style_table,
                           column(ct_guis[0], row(ct_guis[1], ct_guis[2], ct_guis[3])),  #,  ct_guis[4])),
                           column(wis_guis[0])  #, wis_guis[1])
                           )
                )
            )
        else:
            return xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table

    def test_well(self, unit_test=True):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable, CutoffRule, Cutoffs
        from blixt_rp.rp.rp_wrapper_new import data_source_moduli

        w = Well()
        w.read_las(las_fileA, True)

        lt1 = LogTable(name='ptb5', log_table={'P velocity': 'vp_ptb5', 'S velocity': 'vs_ptb5', 'Density': 'rho_ptb5',
                                               'Porosity': 'phie', 'Volume': 'vcl'})
        lt2 = LogTable(name='so08', log_table={'P velocity': 'vp_so08', 'S velocity': 'vs_so08', 'Density': 'rho_so08',
                                               'Porosity': 'phie', 'Volume': 'vcl'})
        lt3 = LogTable(name='sg08', log_table={'P velocity': 'vp_sg08', 'S velocity': 'vs_sg08', 'Density': 'rho_sg08',
                                               'Porosity': 'phie', 'Volume': 'vcl'})

        wds1 = w.data_source(log_table=lt1, verbose=False)
        wds2 = w.data_source(log_table=lt2, verbose=False)
        wds3 = w.data_source(log_table=lt3, verbose=False)

        data_source_moduli(wds1, lt1, verbose=False)
        data_source_moduli(wds2, lt2, verbose=False)
        data_source_moduli(wds3, lt3, verbose=False)

        rule1 = CutoffRule('phie', '>', Q_(0.1, ''))
        rule2 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])

        tp = RockPhysicsPlotter({str(_s.name):_s for _s in [wds1, wds2, wds3]},
                                cutoffs=cutoffs,
                                working_intervals=None,
                                )

        d_cds = tp.cds

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table = tp.draw(d_cds)
        if unit_test:
            show(
                row(
                    column(
                        xplot,
                        row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask)
                    ),
                    column(style_table,
                           column(ct_guis[0], row(ct_guis[1], ct_guis[2], ct_guis[3])),  # ,  ct_guis[4])),
                           column(wis_guis[0])  #, wis_guis[1])
                           )
                )
            )
        else:
            return xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table


    def test_well_with_rpt(self, unit_test=True):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable, CutoffRule, Cutoffs
        from blixt_rp.rp.rp_wrapper_new import (data_source_moduli, rpt_wrapper, rpt_to_moduli,
                                                rpt_phi_sw, return_rpt_keywords, RptVariableTable)

        w = Well()
        w.read_las(las_fileA, True)

        lt1 = LogTable(name='ptb5', log_table={'P velocity': 'vp_ptb5', 'S velocity': 'vs_ptb5', 'Density': 'rho_ptb5',
                                               'Porosity': 'phie', 'Volume': 'vcl'})
        lt2 = LogTable(name='so08', log_table={'P velocity': 'vp_so08', 'S velocity': 'vs_so08', 'Density': 'rho_so08',
                                                'Porosity': 'phie', 'Volume': 'vcl'})
        lt3 = LogTable(name='sg08', log_table={'P velocity': 'vp_sg08', 'S velocity': 'vs_sg08', 'Density': 'rho_sg08',
                                                'Porosity': 'phie', 'Volume': 'vcl'})

        wds1 = w.data_source(log_table=lt1, verbose=False)
        wds2 = w.data_source(log_table=lt2, verbose=False)
        wds3 = w.data_source(log_table=lt3, verbose=False)

        data_source_moduli(wds1, lt1, verbose=False)
        data_source_moduli(wds2, lt2, verbose=False)
        data_source_moduli(wds3, lt3, verbose=False)

        rule1 = CutoffRule('phie', '>', Q_(0.1, ''))
        rule2 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])

        tp = RockPhysicsPlotter({str(_s.name):_s for _s in [wds1, wds2, wds3]},
                          cutoffs=cutoffs,
                          working_intervals=None,
                          )

        d_cds = tp.cds

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table = tp.draw(d_cds)

        rpt_table = RptVariableTable()
        table_cds = rpt_table.cds

        phi = Q_(np.linspace(0.05, 0.35, 4))
        sw = [Q_(_x, '') for _x in [0.1, 0.5, 1.0]]
        annotations = ['SW=0.1', 'SW=0.5', 'SW=1.0']
        _vp, _vs, _rho = rpt_wrapper(phi, rpt_phi_sw, sw, return_rpt_keywords())
        _dict = rpt_to_moduli(_vp, _vs, _rho, annotations)

        rpt_lines_cds = ColumnDataSource(_dict)

        rpt_dt, add_row, var_select, delete_row, update_table, rpt_lines_cds = rpt_table.draw(table_cds, rpt_lines_cds, t=phi, rpt=rpt_phi_sw, constants=sw, rpt_annotations=annotations)

        # This doesn't work as intended. Add it to RockPhysicsPlotter
        x_menu.value = 'ai'
        y_menu.value = 'vp/vs'

        xplot.multi_line(xs=x_menu.value, ys=y_menu.value, line_color='colors', line_width=3, legend_field='labels',
                         source=rpt_lines_cds)

        if unit_test:
            show(
                row(
                    column(
                        xplot,
                        row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask)
                    ),
                    column(style_table,
                           column(ct_guis[0], row(ct_guis[1], ct_guis[2], ct_guis[3])),  #,  ct_guis[4])),
                           column(wis_guis[0])  # , wis_guis[1])
                           )
                )
            )
        else:
            return (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table,
                    rpt_dt, add_row, var_select, delete_row, update_table)


