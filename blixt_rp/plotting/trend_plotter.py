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
    > cutoffs = ...
    > dt_plot = DepthTrendPlotter(data_sources, working_intervals, cutoffs)
    >
    """
    def __init__(self,
                 data_sources: dict,
                 x: str,
                 y: str,
                 working_intervals: Intervals | None = None,
                 cutoffs: Cutoffs | None = None,
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
        :param cutoffs:
        :param width:
        :param height:
        :param tools:
        """
        if width is None:
            width = 900

        super().__init__(data_sources, x, y, working_intervals, cutoffs, width, height, tools)

    def draw(self, source: ColumnDataSource, verbose: bool = False):
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

        xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis = super().draw(source)
        # Passify the x_menu
        x_menu.disabled = True

        return (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask,
                ct_guis,
                wis_guis)

class TestCases(unittest.TestCase):

    def test_data_source(self):
        import blixt_rp.plotting.cross_plotter as xp
        test = xp.TestCases()

        ds1, ds2, wis, coffs = test.test_data()

        tp = TrendPlotter({str(_s.name):_s for _s in [ds1, ds2]},
                          x='md',
                          y='var_two',
                          cutoffs=coffs,
                          working_intervals=wis)

        d_source = tp.source

        return tp.draw(d_source)


