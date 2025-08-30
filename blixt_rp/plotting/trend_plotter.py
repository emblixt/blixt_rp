"""
Plots a single parameter (e.g. Vp) from multiple wells, from specific intervals (or whole well) using cut offs or not
and calculates the trend vs.TVD for that parameter
"""
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper, CheckboxEditor, Select)
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, TextInput
from bokeh.plotting import figure, show
from bokeh.plotting import column, row

import sys
import os.path
import unittest
import numpy as np
import matplotlib.pyplot as plt
import logging
from pint import Quantity as Q_

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

logger = logging.getLogger(__name__)

class TrendPlotter(CrossPlotter):
    """
    Class for handling the (depth) trend plots

    Intended use
    > data_sources = ...  # See cross_plotter.py DataSource for explanation
    > working_intervals = ...
    > cut_offs = ...
    > dt_plot = DepthTrendPlotter(data_sources, working_intervals, cut_offs)
    >
    """
    def __init__(self,
                 data_sources: dict,
                 x: str,
                 y: str,
                 working_intervals: Intervals | None = None,
                 cut_offs: Cutoffs | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 tools: list | None = None):
        """
        Plots the data in data_sources in a x vs. y cross plot, where x is the independent variable
        (typically TVD) and y is the dependent variable (e.g. Vp), and it tries to find the function
        f(x) that best matches y(x)

        :param data_sources:
            dict of DataSource objects
            e.g.
                {w.name: w.data_source() for w in project.wells}
            where project is a Project object from project_new.py

        :param x:
            str
            Name of the independent variable (e.g. 'tvd')
        :param y:
            str
            Name of the dependent variable (e.g. 'vp_oil')

        :param working_intervals:
        :param cut_offs:
        :param width:
        :param height:
        :param tools:
        """
        if width is None:
            width = 900
        # Remove all the data that are not to be included in the calculation of the trend (i.e. x and y)
        for _key, _val in data_sources.items():
            _this_data = _val.data
            for _k in list(_this_data.keys()):
                if not (_k == x or _k == y or _k == 'md'):
                    _ = _this_data.pop(_k)
            _val.data = _this_data

        super().__init__(data_sources, x, y, width, height, tools)
        self.working_intervals = working_intervals
        self.cut_offs = cut_offs
        self._interval_table = None
        if self.working_intervals is not None:
            self._interval_table = WorkingIntervalsTable(self.working_intervals)

    @property
    def interval_table(self):
        return self._interval_table

    def draw(self, source: ColumnDataSource, verbose: bool = False):
        from bokeh.models import Button, Div

        if self.interval_table is not None:
            wis_source = self.interval_table.source
            wis_table = self.interval_table.draw(wis_source)
        else:
            wis_source = None
            wis_table = Div(text='', width=10, height=10)

        # def apply_callback(attr, old, new):  # if calling it from wis_source.on_change
        # dict(new) is a dictionary representation of the new (modified) wis_source
        # print(dict(new))

        def apply_callback():
            if verbose:
                print('APPLY SUCCESSFUL:', wis_source.data['use'])
            # Take a copy of the source
            _dict = dict(source.data)
            mask = self.interval_table.create_mask(
                # ColumnDataSource(dict(new)),
                wis_source,
                Q_(source.data['md'], 'm'),
                list(source.data['source_name']))

            # Modify the mask in source
            _dict['mask'] = mask
            # Update the source
            source.data = _dict
            if verbose:
                print(source.data['md'][source.data['mask']][:5])
                print(source.data['md'][source.data['mask']][-5:])

        if wis_source is not None:
            apply = Button(label='Apply', button_type='success')
            apply.on_click(apply_callback)
        else:
            apply = Div(text='', width=10, height=10)

        if verbose:
            source.js_on_change('patching', CustomJS(
                args=dict(source=source),
                code="""
                console.log('Patching event detected in TrendPlotter');
                """
            ))

            source.js_on_change('data', CustomJS(
                args=dict(source=source),
                code="""
                console.log('Data event detected in TrendPlotter');
                """
            ))

        xplot, x_menu, y_menu, size_menu, color_menu = super().draw(source)

        return xplot, x_menu, y_menu, size_menu, color_menu, wis_table, apply

class TestCases(unittest.TestCase):

    def test_data(self):
        from blixt_rp.core.core import Template, StratUnit, Interval, Intervals
        from blixt_rp.plotting.cross_plotter import DataSource
        from pint import Quantity as Q_

        md1 = np.linspace(1000., 2000., 500)
        md2 = np.linspace(800., 2500., 800)
        l1_1 = np.random.normal(10., 1., 500)
        l1_2 = np.random.normal(10., 1., 800)
        l2_1 = np.random.normal(100., 1., 500) + np.linspace(-8, 8, 500)
        l2_2 = np.random.normal(100., 1., 800) + np.linspace(-10, 10, 800)
        l3 = np.random.normal(50., 1., 500)
        l4 = np.random.normal(70., 1., 800)

        template_md = Template(name='md', units='m', marker='circle', fill_color='red')
        template_one = Template(name='var_one', units='m', min=5., max=15., marker='circle', fill_color='red')
        template_two = Template(name='var_two', units='m/s', min=90., max=110., marker='square', fill_color='blue')
        template_three = Template(name='var_three', units='kg', min=10., max=80., marker='triangle', fill_color='yellow')
        template_four = Template(name='var_four', units='feet', min=30., max=90., marker='hex', fill_color='green')

        data_one = dict(md=md1, var_one=l1_1, var_two=l2_1, var_three=l3)
        data_two = dict(md=md2, var_one=l1_2, var_two=l2_2, var_four=l4)
        templates_one = {_x.name: _x for _x in [template_md, template_one, template_two, template_three]}
        templates_two = {_x.name: _x for _x in [template_md, template_one, template_two, template_four]}

        ds1 = DataSource(name='one', data=data_one, templates=templates_one)
        ds2 = DataSource(name='two', data=data_two, templates=templates_two)

        wis = Intervals(
            name='test', intervals=[
                Interval( well='one', top=Q_(1350, 'm'), base=Q_(1450, 'm'), interval_info=StratUnit('wi_1', 1)),
                Interval( well='one', top=Q_(1450, 'm'), base=Q_(1750, 'm'), interval_info=StratUnit('wi_2', 1)),
                Interval( well='two', top=Q_(1300, 'm'), base=Q_(1500, 'm'), interval_info=StratUnit('wi_1', 1)),
                Interval( well='two', top=Q_(1500, 'm'), base=Q_(1950, 'm'), interval_info=StratUnit('wi_2', 1))
            ] )

        return ds1, ds2, wis

    def test_data_source(self):

        ds1, ds2, wis = self.test_data()

        tp = TrendPlotter({str(_s.name):_s for _s in [ds1, ds2]},
                          x='md',
                          y='var_two',
                          working_intervals=wis)
        d_source = tp.source

        xplot, x_menu, y_menu, size_menu, color_menu, wis_table, apply = tp.draw(d_source)


        # tp.show_plot(d, 'C:\\Users\emb\Downloads\plot.html')
        return xplot, x_menu, y_menu, size_menu, color_menu, wis_table, apply
