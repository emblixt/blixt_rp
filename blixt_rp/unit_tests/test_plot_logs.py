import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from bokeh.models import ColumnDataSource
from bokeh.plotting import show
from bokeh.io import output_file

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

    def test_plot(self):
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

        plotter = bupp.plot_logs(well1, [['rdep', 'rmed', 'rmic'], ['vp_brine', 'vp_oil70', 'vp_gas70']],
                                 scales=['log', 'linear'])
        ai_column = LogColumn('AI plots')
        plotter.add_column(ai_column, keep_column_width=False)
        grid = plotter.figure()
        # modify last figure in grid plot
        _p = grid.children[-1][0]
        _p.xaxis.visible = True

        #  Play with the Backus average
        line_source_dict = dict(
            ai=well1.get_log_curve('vp_brine').values * well1.get_log_curve('rho_brine').values,
            md=well1.get_log_curve('vp_brine').depth.values
        )
        line_source = ColumnDataSource(line_source_dict)
        this_line = Line(x='ai', y='md', source=line_source)
        ai_column.add_line(this_line)
        add_lines(_p, ai_column)
        show(grid)


