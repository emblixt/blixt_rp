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
from typing import Callable

from blixt_rp.rp.rp_wrapper_new import data_source_moduli

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
                 wells: list,
                 log_tables: list,
                 working_intervals: Intervals | None = None,
                 cutoffs: Cutoffs | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 tools: list | None = None
                 ):
        """
        Plots the data in data_sources in a x vs. y cross plot,

        :param wells:
            list
            List of Well objects
            NOTE: ONLY TESTED WITH ONE WELL
        :param log_tables:
            list
            List of LogTable objects:
                LogTable(
                    name='so08',
                    log_table={'P velocity': 'vp_so08', 'S velocity': 'vs_so08', 'Density': 'rho_so08',
                               'Porosity': 'phie', 'Volume': 'vcl'}
                )
            Each LogTable must include P velocity, S velocity and Density, and these should point to the well
            logs inside the well that will be used to calculate the Rock Physics parameters (moduli).
            But the LogTable can also include other logs (e.g. Porosity and Volume shale) that are useful for
            display, in particular for CutoffRules.

        :param working_intervals:
        :param cutoffs:
        :param width:
        :param height:
        :param tools:
        """

        if width is None:
            width = 900

        # Calculate the elastic moduli for the well(s) for the different LogTables
        elastic_mod_sources = {}
        for _well in wells:
            for _log_table in log_tables:
                wds = _well.data_source(log_table=_log_table, verbose=False)
                data_source_moduli(wds, _log_table, verbose=False)
                elastic_mod_sources[wds.name] = wds

        self._rpt_table = None

        super().__init__(elastic_mod_sources, working_intervals=working_intervals, cutoffs=cutoffs, width=width,
                         height=height, tools=tools)

    def rpt_table(self,
                  rpt_variable: Q_ | None = None,
                  rpt_constants: Q_ | None = None,
                  rpt_function: Callable | None = None,
                  rpt_annotations: list | None = None,
                  ):
        """
        Creates the RPT table and sets up the initial RPT data

        :param rpt_variable:
            Pint Quantity with an np.array of length N
            values used to draw the rock physics template, preferably less than about 10 items long for creating
            nice plots
        :param rpt_constants:
            Pint Quantity with a list
            list of length M of constants (pint quantities) used to parametrize the rpt function
        :param rpt_function:
            function
            Rock physics template function of rpt_variable
            Should take a second argument which is used to parameterize the function
            e.g.
            def rpt(rpt_variable, rpt_constants, **rpt_keywords):
                return c*t + rpt_keywords.pop('zero_crossing', 0)
        :param rpt_annotations:
            list of same length as rpt_constants
            Used to annotate the rock physics template

        :return:
        """
        from blixt_rp.rp.rp_wrapper_new import return_rpt_keywords, rpt_wrapper, rpt_to_moduli, RptVariableTable

        if (rpt_variable is None) or (rpt_constants is None) or (rpt_function is None) or (rpt_annotations is None):
            return None, None, None, None, None, None, None

        self._rpt_table = RptVariableTable()
        table_cds = self._rpt_table.cds

        _vp, _vs, _rho = rpt_wrapper(rpt_variable, rpt_function, rpt_constants, return_rpt_keywords())
        _dict = rpt_to_moduli(_vp, _vs, _rho, rpt_annotations)
        rpt_lines_cds = ColumnDataSource(_dict)

        rpt_dt, add_row, var_select, delete_row, update_table, rpt_lines_cds = self._rpt_table.draw(
            table_cds,
            rpt_lines_cds,
            t=rpt_variable,
            rpt=rpt_function,
            constants=rpt_constants,
            rpt_annotations=rpt_annotations)

        return table_cds, rpt_lines_cds, rpt_dt, add_row, var_select, delete_row, update_table

    def draw(self,
             cds: ColumnDataSource,
             rpt_variable: Q_ | None = None,
             rpt_constants: Q_ | None = None,
             rpt_function: Callable | None = None,
             rpt_annotations: list | None = None,
             set_all_intervals_active: bool = False,
             verbose: bool = False):
        """
        Draws the cross plot and more

        :param rpt_variable:
            Pint Quantity with an np.array of length N
            values used to draw the rock physics template, preferably less than about 10 items long for creating
            nice plots
        :param rpt_constants:
            Pint Quantity with a list
            list of length M of constants (pint quantities) used to parametrize the rpt function
        :param rpt_function:
            function
            Rock physics template function of rpt_variable
            Should take a second argument which is used to parameterize the function
            e.g.
            def rpt(rpt_variable, rpt_constants, **rpt_keywords):
                return c*t + rpt_keywords.pop('zero_crossing', 0)
        :param rpt_annotations:
            list of same length as rpt_constants
            Used to annotate the rock physics template

        :return:
        """
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

        table_cds, rpt_lines_cds, rpt_dt, add_row, var_select, delete_row, update_table = self.rpt_table(
            rpt_variable=rpt_variable,
            rpt_constants=rpt_constants,
            rpt_function=rpt_function,
            rpt_annotations=rpt_annotations
        )

        # When there is no RPT table or plot, return results now, with rpt related results set to None
        if table_cds is None:
            return (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask,
                    ct_guis,
                    wis_guis,
                    style_table,
                    rpt_dt, add_row, var_select, delete_row, update_table)

        # We begin with the initial setup
        new_data = dict(rpt_lines_cds.data)
        if x_menu.value in list(new_data.keys()):
            new_data['xs'] = new_data[x_menu.value]
        else:
            new_data['xs'] = new_data['ai']  # Default back to plot AI, the axes will be mixed up though
        if y_menu.value in list(new_data.keys()):
            new_data['ys'] = new_data[y_menu.value]
        else:
            new_data['ys'] = new_data['ai']  # Default back to plot AI, the axes will be mixed up though
        rpt_lines_cds.data = new_data

        # The rpt_lines_cds needs to be updated when x_menu or y_menu is updated similar to how the crossplot
        # cds is updated
        _args_dict = dict(x_drop=x_menu, y_drop=y_menu, x_y_source=rpt_lines_cds)
        _code = """
            const x_param = x_drop.value;
            const y_param = y_drop.value;
            var new_data = x_y_source.data;
            
            // REPLACE DATA
            new_data['xs'] = x_y_source.data[x_param];
            new_data['ys'] = x_y_source.data[y_param];
            console.log('dropdown: ' + cb_obj.value, x_param);
            x_y_source.data = new_data;
            x_y_source.change.emit();
        """
        x_menu.js_on_change(
            'value',
            CustomJS(
                args=_args_dict,
                code=_code)
        )
        y_menu.js_on_change(
            'value',
            CustomJS(
                args=_args_dict,
                code=_code)
        )

        xplot.multi_line(xs='xs', ys='ys', line_color='colors', line_width=3, legend_field='labels',
                         source=rpt_lines_cds)

        return (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask,
                ct_guis,
                wis_guis,
                style_table,
                rpt_dt, add_row, var_select, delete_row, update_table)

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

        rule1 = CutoffRule('phie', '>', Q_(0.1, ''))
        rule2 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])

        tp = RockPhysicsPlotter([w],
                                log_tables=[lt1, lt2, lt3],
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

        # Parameters for plotting the rock physics template
        phi = Q_(np.linspace(0.05, 0.35, 4))
        sw = Q_([0.1, 0.5, 1.0], '')
        annotations = ['SW=0.1', 'SW=0.5', 'SW=1.0']

        rule1 = CutoffRule('phie', '>', Q_(0.1, ''))
        rule2 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])

        tp = RockPhysicsPlotter([w],
                                log_tables=[lt1, lt2, lt3],
                                cutoffs=cutoffs,
                                working_intervals=None,
                          )

        d_cds = tp.cds

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table, \
            rpt_dt, add_row, var_select, delete_row, update_table = tp.draw(d_cds,
                                                                            rpt_variable=phi,
                                                                            rpt_constants=sw,
                                                                            rpt_function=rpt_phi_sw,
                                                                            rpt_annotations=annotations)

        # The below lines have/should have been implemented in RockPhysicsPlotter
        # rpt_table = RptVariableTable()
        # table_cds = rpt_table.cds

        # _vp, _vs, _rho = rpt_wrapper(phi, rpt_phi_sw, sw, return_rpt_keywords())
        # _dict = rpt_to_moduli(_vp, _vs, _rho, annotations)

        # rpt_lines_cds = ColumnDataSource(_dict)

        # rpt_dt, add_row, var_select, delete_row, update_table, rpt_lines_cds = rpt_table.draw(table_cds, rpt_lines_cds, t=phi, rpt=rpt_phi_sw, constants=sw, rpt_annotations=annotations)

        # # This doesn't work as intended. Add it to RockPhysicsPlotter
        # x_menu.value = 'ai'
        # y_menu.value = 'vp/vs'

        # xplot.multi_line(xs=x_menu.value, ys=y_menu.value, line_color='colors', line_width=3, legend_field='labels',
        #                  source=rpt_lines_cds)

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


