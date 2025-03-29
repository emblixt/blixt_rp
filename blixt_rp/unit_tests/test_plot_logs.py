import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from bokeh.models import ColumnDataSource, Spinner
from bokeh.plotting import show, row
from bokeh.io import output_file

import bruges.rockphysics.anisotropy as bra
import pint.errors
from math import isclose

from .. import ureg, Q_

from blixt_rp.core.log_curve_new import LogCurve, Depth, is_equivalent, read_las
from blixt_rp.core.core import Template, LogTable, Cutoffs, CutoffRule
from blixt_rp.core.well_new import Well
import blixt_rp.plotting.plot_logs_new as bupp
from blixt_utils.plotting.log_plotter import LogColumn, Line, add_lines

output_file('C:\\Users\emb\\Documents\\plot.html')
test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"

n = 1500
# create a regularly sampled data set
data1 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth1 = Q_(np.linspace(24, 3430, n), 'm')

# create an irregularly sampled data set
data2 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth2 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'FT')

class TestPlot(unittest.TestCase):

    def test_plot_with_backus(self):
        log_table = LogTable({
            'Resistivity': ['rdep', 'rmed', 'rmic'],
            'Sonic': ['dt', 'dt_edit'],
            'P velocity': ['Vp_brine', 'Vp_Oil70', 'Vp_Gas70'],
            'S velocity': ['Vs_brine', 'Vs_Oil70', 'Vs_Gas70'],
            'Density': ['Rho_brine', 'Rho_Oil70', 'Rho_Gas70']
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table)
        for curve in well1.logs:
            if curve.name == 'rdep':
                _t = Template(**{'min': 0.1, 'max': 100., 'line_color': 'blue', 'scale': 'log'})
                curve.style = _t
            elif curve.name == 'rmed':
                curve.style = Template(**{'min': 0.1, 'max': 100., 'line_color': 'red', 'scale': 'log'})
            elif curve.name == 'rmic':
                curve.style = Template(**{'min': 0.1, 'max': 100., 'line_color': 'green', 'scale': 'log'})
            elif curve.name == 'dt':
                curve.style = Template(**{'min': 40., 'max': 180., 'line_color': 'blue'})
            elif curve.name == 'dt_edit':
                curve.style = Template(**{'min': 40., 'max': 180., 'line_color': 'red'})

        # Create a two-column plot
        plotter = bupp.plot_logs(well1, [['rdep', 'rmed', 'rmic'], ['vp_brine', 'vp_oil70', 'vp_gas70']],
                                 scales=['log', 'linear'])
        # Add a new column
        ai_column = LogColumn('AI plots')
        plotter.add_column(ai_column, keep_column_width=False)

        # "Realize" the figure
        grid = plotter.figure()

        # access the last figure in grid plot, and let it display the x-axis
        _p = grid.children[-1][0]
        _p.xaxis.visible = True

        #  Play with the Backus average
        ba_length = Spinner(title="Backus avg. window length", low=5, high=100, step=5., value=20, width=80)
        vp0 = well1.get_log_curve('vp_brine').values
        vs0 = well1.get_log_curve('vs_brine').values
        rho0 = well1.get_log_curve('rho_brine').values
        md = well1.get_log_curve('vp_brine').depth
        ba = bra.backus(vp0, vs0, rho0, ba_length.value, md.step().magnitude)
        line_source_dict = dict( ai=vp0 * rho0, ai_backus=ba[0] * ba[2], md=md.values )
        line_source = ColumnDataSource(line_source_dict)
        orig_template = Template(**dict(line_color='black', line_style = 'solid', line_width = 1, name = 'Orig AI'))
        orig_line = Line(x='ai', y='md', source=line_source, style=orig_template)
        backus_template = Template(**dict(line_color='red', line_style = 'solid', line_width = 2,
                                          name = 'AI {}m Backus'.format(ba_length.value),
                                          min=orig_line.x_range()[0], max=orig_line.x_range()[1]))
        backus_line = Line(x='ai_backus', y='md', source=line_source, style=backus_template)
        ai_column.lines = [backus_line, orig_line]
        add_lines(_p, ai_column)
        _p.legend.click_policy = 'hide'
        show(row(grid, ba_length))

    def test_interactive_edit(self):
        log_table = LogTable({
            'P velocity': 'Vp_brine',
            'S velocity': 'Vs_brine',
            'Density': 'Rho_brine'
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table)

        # Create a two-column plot
        plotter = bupp.plot_logs(well1, [['rdep', 'rmed', 'rmic'], ['vp_brine', 'vp_oil70', 'vp_gas70']],
                                 scales=['log', 'linear'])
        # Add a new column
        ai_column = LogColumn('AI plots')
        plotter.add_column(ai_column, keep_column_width=False)

        # "Realize" the figure
        grid = plotter.figure()

        # access the last figure in grid plot, and let it display the x-axis
        _p = grid.children[-1][0]
        _p.xaxis.visible = True

        #  Play with the Backus average
        ba_length = Spinner(title="Backus avg. window length", low=5, high=100, step=5., value=20, width=80)
        vp0 = well1.get_log_curve('vp_brine').values
        vs0 = well1.get_log_curve('vs_brine').values
        rho0 = well1.get_log_curve('rho_brine').values
        md = well1.get_log_curve('vp_brine').depth
        ba = bra.backus(vp0, vs0, rho0, ba_length.value, md.step().magnitude)
        line_source_dict = dict( ai=vp0 * rho0, ai_backus=ba[0] * ba[2], md=md.values )
        line_source = ColumnDataSource(line_source_dict)
        orig_template = Template(**dict(line_color='black', line_style = 'solid', line_width = 1, name = 'Orig AI'))
        orig_line = Line(x='ai', y='md', source=line_source, style=orig_template)
        backus_template = Template(**dict(line_color='red', line_style = 'solid', line_width = 2,
                                          name = 'AI {}m Backus'.format(ba_length.value),
                                          min=orig_line.x_range()[0], max=orig_line.x_range()[1]))
        backus_line = Line(x='ai_backus', y='md', source=line_source, style=backus_template)
        ai_column.lines = [backus_line, orig_line]
        add_lines(_p, ai_column)
        _p.legend.click_policy = 'hide'
        show(row(grid, ba_length))


