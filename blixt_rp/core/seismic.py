import pickle
import os
import sys
import unittest
from dataclasses import field

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from itertools import cycle
from typing import Literal
import bruges
from copy import deepcopy, copy
import inspect

import pint
from bokeh.models.plots import GridPlot
from bokeh.models import Tool
from scipy.optimize import least_squares

from bokeh.plotting import show, figure, column, row
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, LinearColorMapper, DataTable, TableColumn, PointDrawTool, CheckboxEditor, \
    LassoSelectTool
from bokeh.models import Span, CrosshairTool, HoverTool, LassoSelectTool, ColorBar, NumericInput, CustomJS, Select
from bokeh.models import NumberFormatter, Button, HTMLTemplateFormatter, StringEditor
from bokeh.events import SelectionGeometry

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp import Q_
import blixt_utils.misc.curve_fitting as mycf
import blixt_utils.io.io as uio
from blixt_utils.plotting.helpers import wiggle_plot
import blixt_utils.plotting.crossplot as xp
from blixt_rp.core.core import Template, CutoffRule, Cutoffs, LogTable, Header
from blixt_rp.core.log_curve_new import read_las
from blixt_utils.utils import print_info

# global variables
# output_file('C:\\Users\marte\Documents\plot.html')
output_file(os.path.join(project_dir, 'blixt_rp\\test_data\\plot.html'))
logger = logging.getLogger(__name__)
clrs = list(mcolors.BASE_COLORS.keys())
clrs.remove('w')
cclrs = cycle(clrs)  # "infinite" loop of the base colors
test_data_length = 1000

selected_cells = []

section_types = ['line section', 'avo section']

class VolumeOfInterest:
    """
    Simple class for holding the inline, xline and sample ranges for seismic subvolume
    """
    def __init__(self,
                 inline_range: range,
                 xline_range: range,
                 sample_range: range
    ):
        self.inline_range = inline_range
        self.xline_range = xline_range
        self.sample_range = sample_range

    def dict(self) -> dict:
        """
        Returns a dictionary useful for subvolume selection by xarray
        :return:
        """
        return dict(iline=self.inline_range, xline=self.xline_range, samples=self.sample_range)

    # practical shorts
    @property
    def i(self):
        return self.inline_range
    @property
    def j(self):
        return self.xline_range
    @property
    def k(self):
        return self.sample_range

    # useful lists
    @property
    def inlines(self):
        return list(self.inline_range)
    @property
    def xlines(self):
        return list(self.xline_range)
    @property
    def time_slices(self):
        return list(self.sample_range)

class AngleStack:
    """
    Utility class for holding information about one angle stack
    """
    def __init__(self,
                 name: str,
                 filename: str,
                 angle: float):
        self.name = name
        self.filename = filename
        self.angle = angle

    @property
    def title(self):
        return os.path.basename(self.filename).split('.')[0]

class SeismicTraces:
    """
    Simple class that contains N number of seismic traces,
    Each trace must have the same depth/time dimension

    """
    # TODO
    # TODO Suggested major re-write
    # Rename this to something like "SeismicSection", and make StochasticSyntheticTraces and
    # AvoAmplitudes methods of this class.
    # Add more attributes, e.g. "incident_angle", "chi_angle" which can be used for near, mid and far-stacks
    # and so on.
    # Also add a model as a non-mandatory attribute. so that the seismic section can be calculated from
    # this model
    # Also, make the output seismic sections from models.py of this type
    def __init__(self,
                 x: np.ndarray | None = None,
                 y: np.ndarray | None = None,
                 traces: np.ndarray | None = None,
                 cds: ColumnDataSource | None = None,
                 trace_type: str | None = None,
                 title: str | None = None
                 ):
        """

        :param x:
            np.ndarray of length M with values in the x direction
            Contains incidence angles when trace_type is 'avo',
            Chi angles when trace_type is 'eei', and index when trace_type is 'index'
        :param y:
            Numpy array of depth/time values, length N
        :param traces:
            np.ndarray of size MxN, contains M number of seismic traces, each of length N
        :param cds:
            ColumnDataSource({'value': [_seismic.traces.T]})
            Must have the key 'value'
        :param trace_type:
            string that describes the type of seismic variation in the x direction
            'avo': Incidence angle (AVO)
            'eei': Chi angle (Extended Elastic Impedance)
            'index': index along an Inline, Xline or arbitrary seismic line
        """
        if x is None:
            x = np.linspace(0,35, 128)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, test_data_length)
        self._y = y
        if trace_type is None:
            trace_type = 'avo'
        if trace_type not in ['avo', 'eei', 'index']:
            print_info('trace_type is not recognized: {}'.format(trace_type), 'error', logger, 'IOError')
        self._trace_type = trace_type

        self.cds = cds

        if traces is None and cds is None:
            _synts = StochasticSyntheticTraces(trace_length=test_data_length, n_traces=len(x))
            traces = _synts.get_traces(simulate_avo=self._trace_type=='avo')
            if title is None:
                title = 'Synthetic'
        self._traces = traces

        self._title = title

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def trace_type(self):
        return self._trace_type

    @property
    def traces(self):
        return self._traces

    @property
    def title(self):
        if self._title is None:
            return self.trace_type
        else:
            return '{}: {}'.format(self._title, self.trace_type)

    @title.setter
    def title(self, value: str):
        self._title = value

class StochasticSyntheticTraces:
    def __init__(self,
                 trace_length:int | None = None,
                 dt: float | None = None,
                 n_traces: int | None = None,
                 n_reflectors: int | None = None,
                 seed: int | None = None):

        if trace_length is None:
            trace_length = 3000
        if dt is None:
            dt = 0.001  # seconds
        if n_traces is None:
            n_traces = 128
        if n_reflectors is None:
            n_reflectors = 3

        x = np.linspace(0, n_traces - 1, n_traces)
        y = np.linspace(0, trace_length - 1, trace_length)
        traces = np.zeros((n_traces, trace_length))

        self.trace_length = trace_length
        self.dt = dt
        self.n_traces = n_traces
        self.seed = seed
        self.n_reflectors = n_reflectors
        self.rng = np.random.default_rng(self.seed)
        self.idx_refl = None

        self.w_length = 0.082  # Ricker wavelength in seconds
        self.f0 = 25.  # Hz

        self.traces = traces
        self.wavelet = None

    def get_traces(self, simulate_avo=False) -> np.ndarray:
        idx_refl = self.rng.integers(100, self.trace_length, self.n_reflectors)
        print('Depth index of reflectors in the synthetic traces: ', idx_refl)
        # self.idx_refl = self.trace_length - idx_refl  # Because we plot the data "upside down"
        self.idx_refl = idx_refl
        refl = np.zeros(self.trace_length)
        refl[idx_refl] = 2 * self.rng.random(self.n_reflectors) - 1

        # Convolve reflectivity model with a Ricker wavelet '''
        this_wavelet = bruges.filters.wavelets.ricker(self.w_length, self.dt, self.f0)
        self.wavelet = this_wavelet.amplitude
        this_trace = np.convolve(refl, self.wavelet, mode='same')
        for i in range(self.traces.shape[0]):
            if simulate_avo:
                _refl = deepcopy(refl)
                _refl[idx_refl[1]] = _refl[idx_refl[1]] * i*2/(self.traces.shape[0] - 1)
                this_trace=  np.convolve(_refl, self.wavelet, mode='same')
            self.traces[i,:] = this_trace

        return self.traces

class AvoAmplitudes:
    """
    Class for handling seismic amplitudes vs incident angle, or EEI vs. Chi angle for specific
    points in a seismic section
    """
    # from blixt_rp.core.models import LaminarModel, WedgeModel
    def __init__(self):
        from copy import copy
        empty_dict = dict(
            use=[],
            name=[],  # should match the names in point_cds from AvoAnalyzer
            color=[],
            marker=[],  # Use circle for base case, and other for the other cases
            angle=[],
            sin2theta=[],
            amplitude=[],
            chi_angle=[],
            eei_amplitude=[]
        )
        self.emtpy_dict = empty_dict
        self._cds_dict = copy(empty_dict)


    @property
    def cds(self) -> ColumnDataSource:
        """
        Column data source that can hold the avo and eei amplitudes at different angles with the following structure
        name:   use:    angle:   amplitude: ...
        Point1  True    15       -10
        Point1  True    25       -20
        Point1  True    35       -40
        Point2  True    15       100
        ...

        """
        return ColumnDataSource(self._cds_dict)

    def calc_amplitudes(self,
                            points_cds: ColumnDataSource,
                            section_type: str,
                            inc_angle_range: list | None = None,
                            seismic_cds: ColumnDataSource | None = None,
                            model = None
                        ) -> dict:
        """
        Calculates the Avo amplitudes from various seismic input sections (see "section_types") at the
        specified locations (point_cds) and returns and stores the values in self._cds_dict

        You can update the cds by simply calling:
        > cds.data = self.calc_amplitudes(...)
        after the points_cds or seismic_cds has changed

        :param points_cds:
            ColumnDataSource
            The points from where to extract the AVO amplitudes
            Must follow the standard from AvoAnalyzer.cds

        :param section_type:
            str
            String describing the seismic input
            One of the elements in section_types

        :param inc_angle_range:
            list
            Optional
            List of min and max of the incident angle in degrees.

        :param seismic_cds:
            ColumnDataSource
            Optional
            Needed to calculate the AVO amplitudes, depending on what section_type is set to.

        :param model:
            One of the model types in models.py.
            TODO Maybe we should restrict ourselves to LaminarModel and WedgeModel only
            Used to calculate the synthetic AVO / Chi response of the seismic section

        :return:
            dict
            follows the structure of self._cds_dict
        """
        from copy import copy
        from blixt_rp.core.models import calc_synth_cds, global_base_case_name
        # create markers for the different cases, but save the 'circle' for the base case
        markers = ['star', 'diamond', 'hex', 'inverted_triangle', 'plus', 'square', 'triangle']
        _dict = copy(self.emtpy_dict)

        if section_type not in section_types:
            section_type = 'avo section'

        #
        # Iterate over all points in points_cds and calculate the AVO response in each point
        #
        for _point_no in range(len(points_cds.data['name'])):

            if section_type == 'line section' and model is None:
                pass

            elif section_type in ['line section', 'avo section'] and model is not None:
                # Extract the trace index from the current point
                trace_index = int(points_cds.data['x'][_point_no])

                # Realize the model at the current trace index to get the elastics of the model
                elastics_dict = model.model_table.realize(model.resolution, trace_index)

                # Extract the depth index by matching the point depth with the depth of the realized model
                depth_index = np.argmin(
                    (elastics_dict['Base']['vp'].depth.magnitude - points_cds.data['y'][_point_no])**2)
                print('Depth index: ', depth_index)

                # Iterate over all incident angles
                for inc_angle in model.angles:

                    # Iterate over all cases
                    for _n, _case in enumerate(model.model.case_names):
                        # Create markers for the different cases
                        if _case.lower() == global_base_case_name.lower():
                            _marker = 'circle'
                        else:
                            _marker = markers[_n]
                        synths = calc_synth_cds(
                            elastics_dict[_case]['vp'],
                            elastics_dict[_case]['vs'],
                            elastics_dict[_case]['rho'],
                            elastics_dict[_case]['twt'],
                            model.dt, inc_angle, model.avo_or_eei, model.freq
                        )
                        _dict['use'].append(True)
                        _dict['name'].append('{} {}'.format(points_cds.data['name'][_point_no], _case))
                        _dict['color'].append(points_cds.data['color'][_point_no])
                        _dict['marker'].append(_marker)
                        _dict['angle'].append(inc_angle)
                        _dict['sin2theta'].append(np.nan)
                        _dict['amplitude'].append(synths.data['value'][0][depth_index])
                        _dict['chi_angle'].append(np.nan)
                        _dict['eei_amplitude'].append(np.nan)

            elif section_type == 'avo section' and model is None:
                # Extract the incident angles from the seismic section
                n_angles = seismic_cds.data['value'][0].shape[1]  # Remember that the data in seismic_cds is transposed
                inc_angles = np.linspace(inc_angle_range[0], inc_angle_range[1], n_angles, endpoint=True)

                # Iterate over the AVO incident angle.
                for i in range(n_angles):
                    _dict['use'].append(True)
                    _dict['name'].append(points_cds.data['name'][_point_no])
                    _dict['color'].append(points_cds.data['color'][_point_no])
                    _dict['marker'].append('circle')
                    _dict['angle'].append(inc_angles[i])
                    _dict['sin2theta'].append(np.nan)
                    _dict['amplitude'].append(seismic_cds.data['value'][0][points_cds.data['j'][_point_no], i])
                    _dict['chi_angle'].append(np.nan)
                    _dict['eei_amplitude'].append(np.nan)

            # elif section_type == 'avo section' and model is not None:
            #     pass

        self._cds_dict = _dict
        return _dict

    def draw(self, avo_cds) -> figure:
        p = figure(width=600, height=300, tools= "pan,wheel_zoom,box_zoom,reset")
        p.toolbar.logo = None
        p.scatter(x='angle', y='amplitude',
                  source=avo_cds, fill_color='color', marker='marker',
                  legend_group='name', line_color='black', size=10)
        return p

class AvoAnalyzer:
    """
    Add a "Point Draw Tool" to either:
    1. A "line section" which is a typical inline or cross-line section of either 'real' or 'synthetic' data
    2. An "avo section" which shows the (real or synthetic) amplitude response as a function of incidence angle (or chi angle)

    If the seismic is synthetic, the model (see models.py) must be provided.

    Take inspiration from avo_qc() in this script, and create_mc_avo_plot() in avo_chi_plotter.py

    """
    def __init__(self,
                 p: figure,
                 section_type: str,
                 extraction_points: list | None = None,
                 column_name: str | None = 'SYNTH_2D',
                 model = None
                 ):
        """
        Class designed to allow the user to add a point to a seismic section, and to get information about the
        AVO behavior in that point

        :param p:
            Bokeh.plotting.figure
            Figure that holds the seismic section
            Sometimes it is a simple bokeh figure, sometimes a gridplot (NOTE that we can't add a PointDrawTool
            to a gridplot when merge_tools=True was used in its creation), and more recently a bokeh Row consisting of a gridplot and a toolbar

        :param section_type:
            str
            'line section': Typical seismic section (e.g. an inline or cross-line section)
            'avo section': Seismic is provided as an amplitude response as a function of incidence angle or chi angle

        :param extraction_points:
            list
            Optional
            List of lists or two-tuples with the x, y data coordinates of where the AVO data should be
            extracted. If None, it automatically selects two points
            E.G. [[20, 3450], [20, 4675]]

        :param column_name:
            str
            When the figure p is delivered as a gridplot (from LogPlotter), we need to know the name of the
            column the seismic is drawn. See e.g. the LaminarModel class in models.py

        :param model:
            One of the model types in models.py.

            Mandatory if the seismic is not "real"
            Used to calculate the synthetic AVO / Chi response of the seismic section


        """
        from bokeh.models.plots import GridPlot
        from bokeh.plotting import Row
        from bokeh.plotting._figure import figure as _figure
        from blixt_rp.plotting.log_plotter import select_column

        #
        # Determine how the seismic "figure" is given
        #
        self.toolbar = None
        # As a simple bokeh figure
        if isinstance(p, _figure):
            self.figure = p
            self.grid = p

        # As a bokeh Row (see the method 'draw_ext_toolbar' in LogPlotter)
        elif isinstance(p, Row):
            self.figure = select_column(p.children[0], column_name)
            if self.figure is None:
                self.figure = select_column(p.children[0], 'SYNTH_VARIATION')
            self.grid = p.children[0]
            self.toolbar = p.children[1]

        # Or as a grid plot
        elif isinstance(p, GridPlot):
            self.figure = select_column(p, column_name)
            self.grid = p
            print_info('It is not possible to add a PointDrawTool consistently to a gridplot if merge_tools=True was used',
                       'warning', logger)
        else:
            raise IOError('Seismic section must be provided as a bokeh figure, gridplot or Row((gridplot, toolbar)')

        self.model = model

        #
        # Determine the seismic section type, and extract the data and array coordinates of the
        # seismic data
        #
        if section_type not in section_types:
            section_type = 'avo section'
        self.section_type = section_type
        self.extraction_points = extraction_points

        # Data coordinates of seismic image
        x0, y0, dw, dh = get_coords_of_seismic_figure(self.figure)
        self.x0 = x0
        self.y0 = y0
        self.dw = dw
        self.dh = dh

        # Size of seismic array
        n_cols, n_rows = get_size_of_seismic_figure(self.figure)
        self.n_cols = n_cols
        self.n_rows = n_rows

        print('From AvoAnalyzer:')
        print('  x0: {}, y0: {}, dw: {}, dh: {}'.format(x0, y0, dw, dh))
        print('  Number of columns: {}, Number of rows: {}'.format(n_cols, n_rows))

    @property
    def cds(self) -> ColumnDataSource:
        """
        Sets up the ColumnDataSource for the points we will add to the seismic section for which we will calculate
        the AVO behavior

        :return:
            ColumnDataSource
        """
        if self.extraction_points is not None:
            if len(self.extraction_points) > 4:
                raise NotImplementedError('Not capable of handling more than four extraction points')
            xs = [_point[0] for _point in self.extraction_points]
            ys = [_point[1] for _point in self.extraction_points]
        else:
            _dx = self.dw / 4.
            _dy = self.dh / 4.
            xs = [self.x0 + (_i + 1) * _dx for _i in range(2)]
            ys = [self.y0 + (_i + 1) * _dy for _i in range(2)]
        i_indxs = [np.argmin((np.linspace(self.x0, self.x0 + self.dw, self.n_cols) - _xs)**2) for _xs in xs]
        j_indxs = [np.argmin((np.linspace(self.y0, self.y0 + self.dh, self.n_rows) - _ys)**2) for _ys in ys]
        # print(self.n_cols*(xs[0] - self.x0)/self.dw, np.argmin( (np.linspace(self.x0, self.x0 + self.dw, self.n_cols) - xs[0])**2 ))
        # print(self.n_rows*(ys[0] - self.y0)/self.dh, np.argmin( (np.linspace(self.y0, self.y0 + self.dh, self.n_rows) - ys[0])**2 ))
        names = ['One', 'Two', 'Three', 'Four']
        colors = ['red', 'blue', 'green', 'yellow']

        _dict = dict(name=[], x=[], y=[], i=[], j=[], color=[],
                     # The Intercept calculated by linear fitting to the offset data, and through the analytical calculation
                     # when we have a elastic model
                     intercept=[], intercept_anal=[],
                     # The Gradient calculated by linear fitting to the offset data, and through the analytical calculation
                     # when we have a elastic model
                     gradient=[], gradient_anal=[],
                     quality=[])
        for _i in range(len(xs)):
            _dict['name'].append(names[_i])
            _dict['x'].append(xs[_i])
            _dict['y'].append(ys[_i])
            _dict['i'].append(i_indxs[_i])
            _dict['j'].append(j_indxs[_i])
            _dict['color'].append(colors[_i])
            _dict['intercept'].append(0.)
            _dict['intercept_anal'].append(0.)
            _dict['gradient'].append(0.)
            _dict['gradient_anal'].append(0.)
            _dict['quality'].append(None)

        return ColumnDataSource(_dict)

    def inc_angle_range(self) -> list:
        """
        Range (from min to max) of the incident angle in degrees
        :return:
            list
        """
        if self.section_type == 'avo section':
            return [self.x0, self.x0 + self.dw]
        else:
            return []

    def add_point_tool(self,
                       cds: ColumnDataSource):
        """
        Adds a PointDrawTool to the given figure.
        Adds the data coordinates and array indexes of the points to the input ColumnDataSource

        :param cds:
            ColumnDataSource
            The source of the points drawn or added to the current figure
        :return:
        """
        from blixt_rp.plotting.log_plotter import add_tool_to_layout

        _renderer = self.figure.scatter(x='x', y='y', fill_color='color', source=cds, line_color='black', size=10)
        draw_tool = PointDrawTool(renderers=[_renderer], empty_value='black')
        add_tool_to_layout(self.grid, draw_tool)

        # if isinstance(self.grid, GridPlot):
        #     for child in self.grid.children:
        #         for item in child:
        #             if hasattr(item, 'tools'):
        #                 item.add_tools(draw_tool)
        #                 item.toolbar.active_tap = draw_tool
        # else:
        #     self.figure.add_tools(draw_tool)
        #     self.figure.toolbar.active_tap = draw_tool

        callback = CustomJS(args=dict(points=cds, ncols=self.n_cols, nrows=self.n_rows,
                                      x0=self.x0, y0=self.y0, dw=self.dw, dh=self.dh),
                            code="""
                const xs = points.data.x;
                const ys = points.data.y;
                let indices = [];
                for (let i = 0; i < xs.length; i++) {
                    const x = xs[i];
                    const y = ys[i];
                    // Convert data coords → array indices
                    let col = Math.floor(ncols*(x-x0)/dw);       
                    points.data['i'][i] = col;
                    let row = Math.floor(nrows*(y-y0)/dh);
                    points.data['j'][i] = row;
                    //row = nrows - 1 - row;         // flip y-axis (NumPy indexing)

                    indices.push([col, row]);
                }

                console.log("Array indices:", indices);
                points.change.emit()
                console.log(points.data['i'])
                """)
        cds.js_on_change('data', callback)


    def add_points_table(self,
                         cds: ColumnDataSource):
        """

        :return:
        """
        formatter = NumberFormatter(format='0.0')
        template = """
                <div style="background:<%= 
                    (function color_from_val(){
                        return(color)
                        }()) %>; 
                    color: white"> 
                <%= value %>
                </div>
            """
        color_formatter = HTMLTemplateFormatter(template=template)
        columns = [
            TableColumn(field="name", title="Name", editor=StringEditor()),
            TableColumn(field="x", title="X", formatter=formatter),
            TableColumn(field="y", title="Y", formatter=formatter),
            TableColumn(field='color', title='Color', formatter=color_formatter)
        ]
        i_column =    TableColumn(field='intercept', title='I', formatter=formatter)
        i_anal_column =    TableColumn(field='intercept_anal', title='I*', formatter=formatter)
        g_column =    TableColumn(field='gradient', title='G', formatter=formatter)
        g_anal_column =    TableColumn(field='gradient_anal', title='G*', formatter=formatter)
        q_column =    TableColumn(field='quality', title='Qual.', formatter=NumberFormatter(format='0.00'))

        if self.section_type in section_types:
            [columns.append(_c) for _c in [i_column, g_column, q_column]]
        if self.model is not None:
            [columns.append(_c) for _c in [i_anal_column, g_anal_column]]

        table = DataTable(source=cds, columns=columns, editable=True, height=200)
        return table

    def draw(self, points_cds, seismic_cds):
        """
        Creates the bokeh user interfaces used to analyze the AVO / Chi behavior

        :return:
        """
        #
        # Use the extraction points, and let the user control them using the PointDrawTool
        #
        self.add_point_tool(points_cds)
        points_table = self.add_points_table(points_cds)

        #
        # Calculate the AVO / Chi amplitudes in the above given points
        #
        avo_amplitudes = AvoAmplitudes()
        _avo_dict = avo_amplitudes.calc_amplitudes(
            points_cds,
            self.section_type,
            self.inc_angle_range(),
            seismic_cds,
            self.model
        )
        # You can update the avo_cds by calling: avo_cds.data = avo_amplitudes.calc_amplitudes()
        avo_cds = avo_amplitudes.cds

        #
        # return the table, plot of AVO amplitudes and the AVO cds
        avo_figure = avo_amplitudes.draw(avo_cds)
        return points_table, avo_figure, avo_cds


def create_seismic_figure(
        _width: int,
        _height: int,
        _y_range_flipped: bool = True,
        _x_axis_visible: bool = True,
        _y_axis_visible: bool = True,
        _tools: list | None = None) -> figure:

    _w = Span(dimension="width", line_dash="dashed", line_width=1)
    _h = Span(dimension="height", line_dash="dashed", line_width=1)
    if _tools is None:
        _tools = "pan,wheel_zoom,box_zoom,reset"

    _p = figure(width=_width,
                height=_height,
                x_axis_location = 'above',
                tools=_tools)
    # style the plot
    _p.toolbar.logo = None
    _p.add_tools(CrosshairTool(overlay=[_w, _h]))
    _p.add_tools(HoverTool())
    hover = _p.select(dict(type=HoverTool))
    hover.tooltips = [("(x,y)", "($x, $y)"), ("Value", "@value")]

    _p.xaxis.visible = _x_axis_visible
    _p.xaxis.axis_label_text_font_size='10px'
    _p.xaxis.major_label_text_font_size='10px'
    _p.xaxis.axis_label_standoff=0

    _p.y_range.flipped = _y_range_flipped
    _p.yaxis.visible = _y_axis_visible

    return _p


def add_selection(_p, _x, _y, x_resamp=5, y_resamp=3, visible=False):
    """

    :param _p:
        bokeh figure
    :param _x:
        np.array
        Contains the x-values (e.g. inline or xline numbers) for seismic
    :param _y:
        np.array
        Contains the y-values (e.g. TWT or Z) for the seismic
    :param visible:
        bool
        If True, show a limited number of the fake scatter point data
    :return:
    """
    if x_resamp is None:
        x_resamp = 1
    if y_resamp is None:
        y_resamp = 1

    _p.add_tools(LassoSelectTool())

    # Add some empty FAKE scatter data that can be selected using the Lasso tool
    if visible:
        x_resamp = 40
        y_resamp = 20
        # _xx = np.array([list(_x[::40]) for _i in _y[::20]])
        # _yy = np.array([list(_y[::20]) for _i in _x[::40]])
    _xx = np.array([list(_x[::x_resamp]) for _i in _y[::y_resamp]])
    _yy = np.array([list(_y[::y_resamp]) for _i in _x[::x_resamp]])
    _yy = _yy.T
    fake_cds = ColumnDataSource(dict(x=_xx.ravel(),
                                        y=_yy.ravel()))
    if visible:
        _p.scatter(x='x', y='y', source=fake_cds,
                   fill_color='black', line_color=None, size=8.)
    else:
        _p.scatter(x='x', y='y', source=fake_cds,
                   fill_color=None, line_color=None, size=0.)

    selected_cds = ColumnDataSource(data=dict(x=[], y=[]))
    _p.scatter(x='x', y='y', source=selected_cds, fill_color='gray', fill_alpha=0.6, size=8.)

    draw_selected = CustomJS(args=dict(s1=fake_cds, s2=selected_cds), code="""
        const inds = cb_obj.indices;
        const d1 = s1.data;
        const d2 = s2.data;
        d2['x'] = [];
        d2['y'] = [];
        for (let i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]]);
            d2['y'].push(d1['y'][inds[i]]);
        }
        s2.change.emit();
    """)
    def callback(event):
        if event.final == True:
            global selected_cells
            already_selected = []
            selected_cells = []
            for _index in fake_cds.selected.indices:
                if _index in already_selected:
                    continue
                _i = closest(fake_cds.data['x'][_index], _x)
                _j = closest(fake_cds.data['y'][_index], _y)
                selected_cells.append((_i, _j))
                already_selected.append(_index)

    _p.select(LassoSelectTool).overlay.fill_color = 'lightblue'
    _p.select(LassoSelectTool).overlay.fill_alpha = 0.5
    _p.on_event(SelectionGeometry, callback)
    # _p.js_on_event(SelectionGeometry, draw_selected)
    fake_cds.selected.js_on_change('indices', draw_selected)


def test_add_selection(_p,
                       # _seismic_sources,
                       _x,
                       _y,
                       # _angles
                       # _angle_source
                       ):
    """

    :param _p:
        bokeh figure
    :param _seismic_sources
        dict
        Dictionary with a seismic data set for each offset
    :param _x:
        np.array
        Contains the x-values (e.g. inline or xline numbers) for seismic
    :param _y:
        np.array
        Contains the y-values (e.g. TWT or Z) for the seismic
    :param _angles:
        dict
        Dictionary with names (e.g. 'near', 'mid', 'far') as keys and center incident angle (in deg) as values
    :param _angle_source:
    Column data source with key 'avg_amp' that will contain the average amplitude at each offset within the
    select area
    :return:
    """
    _p.add_tools(LassoSelectTool())

    # Add some empty FAKE scatter data that can be selected using the Lasso tool
    _xx = np.array([list(_x[::40]) for _i in _y[::20]])
    _yy = np.array([list(_y[::20]) for _i in _x[::40]])
    _yy = _yy.T
    fake_cds = ColumnDataSource(dict(x=_xx.ravel(),
                                        y=_yy.ravel()))
    _p.scatter(x='x', y='y', source=fake_cds,
               fill_color='black', line_color=None, size=8.)


    def callback(event):
        if event.final == True:
            global selected_cells
            # for the _angle_sources to be updated we should change the whole .data property,
            # which is why we first take a copy of it
            # _new_data = copy(_angle_source.data)  # THis fails! Why?
            # print(fake_source.selected.indices)
            already_selected = []
            selected_cells = []
            # container = {_stack: [] for _stack in _angle_source.data['name']}
            # for _stack in list(_angles.keys()):
            #     selected_cells[_stack] = []
            for _index in fake_cds.selected.indices:
                if _index in already_selected:
                    continue
                _i = closest(fake_cds.data['x'][_index], _x)
                _j = closest(fake_cds.data['y'][_index], _y)
                # for _stack in _angle_source.data['name']:
                #     container[_stack].append(_seismic_sources[_stack][_i, _j].data)
                #     print(_seismic_sources[_stack][_i, _j].data)
                # for _stack in list(_angles.keys()):
                #     selected_cells[_stack].append((_i, _j))
                selected_cells.append((_i, _j))
                already_selected.append(_index)
            # for _a, _stack in enumerate(_angle_source.data['name']):
            #     # _angle_source.data['avg_amp'][_a] = np.nanmean(container[_stack])
            #     _new_data['avg_amp'][_a] = np.nanmean(container[_stack])
            # # _angle_source.change.emit()  # this does not work
            # _angle_source.data = dict(_new_data)
            # print('Selection triggered')
    # callback2 = CustomJS(args=dict(source=_angle_source), code="""
    # source.change.emit()
    # console.log('TEST')
    # """)

    _p.select(LassoSelectTool).overlay.fill_color = 'lightblue'
    _p.select(LassoSelectTool).overlay.fill_alpha = 0.5
    _p.on_event(SelectionGeometry, callback)
    # _p.js_on_event(SelectionGeometry, callback2)  # This makes no effect


def add_seismic_to_figure(
        _p: figure,
        _seismic_cds: ColumnDataSource,
        _x: np.array,
        _y: np.array,
        title: str | None = None
) -> NumericInput:
    """
    Adds the given seismic cds data to the figure and return a numeric input that controls the
    range of the color bar
    :param _p:
    :param _seismic_cds:
        Column data source with key 'value' containing the data
    :param _x:
    :param _y:
        X and Y coordinates of the seismic
    :param title:
        str
    :return:
    """
    min_val = np.nanmin(_seismic_cds.data['value'])
    max_val = np.nanmax(_seismic_cds.data['value'])
    _seismic_color_map = seismic_color_map(min_val=min_val, max_val=max_val)
    _p.image('value', source=_seismic_cds, color_mapper=_seismic_color_map,
             dh=_y[-1] - _y[0],
             dw=_x[-1] - _x[0],
             x=_x[0],
             y=_y[0]
             )

    if title is not None:
        _p.xaxis.axis_label = title
    color_bar = ColorBar(color_mapper=_seismic_color_map,
                         title_text_align='right',
                         label_standoff=3, major_label_text_font_size='10px')
    _p.add_layout(color_bar, 'right')

    orig_max_val = _seismic_color_map.high
    color_amp_factor = NumericInput(value=100, low=10, high=500,
                                    title="Boost color scale (10 - 500%)")
    color_amp_factor.js_on_change('value', CustomJS(
        args=dict(
            c_amp=color_amp_factor,
            c_map=_p._property_values['right'][0].color_mapper,
            max_val=orig_max_val),
        code="""
            c_map.high = max_val * c_amp.value/100.
            c_map.low = -1. * max_val * c_amp.value/100.
            console.log('test:', c_amp.value);
            """))

    return color_amp_factor

def avo_qc(
        angle_stacks: list,
        voi: VolumeOfInterest,
        line_direction: str = 'inline', # or 'xline'
        verbose: bool = False
):
    from blixt_utils.utils import print_info

    inlines = [str(_x) for _x in voi.inlines]
    inline0 = int(inlines[int(len(inlines)/2)])
    xlines = [str(_x) for _x in voi.xlines]
    xline0 = int(xlines[int(len(xlines)/2)])

    if line_direction == 'inline':
        seismic_cdss, x, y, suffix = _load_angle_stacks(angle_stacks, voi, inline0, None, verbose)
    else:
        seismic_cdss, x, y, suffix = _load_angle_stacks(angle_stacks, voi, None, xline0, verbose)

    if verbose:
        for _key in list(seismic_cdss.keys()):
            info_txt = '{}: {}'.format(_key, seismic_cdss[_key].shape)
            print_info(info_txt, 'info', logger)
    names = [_as.name for _as in angle_stacks]
    titles = {_name: angle_stacks[_i].title for _i, _name in enumerate(names)}
    angles = {_name: angle_stacks[_i].angle for _i, _name in enumerate(names)}

    # Create data set to hold different angle stacks
    def add_angle_table(_angle_stacks):
        formatter = NumberFormatter(format='0.0')
        _angle_cds = ColumnDataSource(dict(
            use=[True] * len(_angle_stacks),
            name=[_as.name for _as in _angle_stacks],
            angles=[_as.angle for _as in _angle_stacks],
            sin2theta=[(np.sin(_as.angle * np.pi / 180.)) ** 2 for _as in _angle_stacks],
            avg_amp=[0. for _as in _angle_stacks]
        ))
        table_columns = [
            TableColumn(field='use', title='Use',
                        editor=CheckboxEditor(),
                        width=30),
            TableColumn(field='name', title='Angle stack'),
            TableColumn(field='angles', title='Angle'),
            TableColumn(field='avg_amp', title='Average amp.', formatter=formatter)
        ]
        _angle_table = DataTable(
            source=_angle_cds,
            columns=table_columns,
            editable=True,
            height=150
        )

        return _angle_table, _angle_cds

    angle_table, angle_cds = add_angle_table(angle_stacks)

    p = create_seismic_figure(900, 600)
    current_cds = ColumnDataSource(dict(value=[seismic_cdss[names[0]].T]))
    # _test = current_cds.data['value'][0].data
    # print(_test.shape)

    c_amp = add_seismic_to_figure(p, current_cds, x, y,
                                 '{}, {}'.format(titles[names[0]], suffix))

    # Add possibility to select an area of interest
    add_selection(p, x, y, visible=False)

    # add points to extract Intercept and Gradient from
    points_table, points_cds = add_i_g_point(p, x, y, names)

    def calc_new_point_cds_dict(_point_cds, _angles)->dict:
        """

        :param _point_cds:
        :param _angles:
            dictionary with angles for each angle stack
        :return:
        """
        _out = dict(
            point=[],
            amp=[],
            sin2theta=[],
            color=[]
        )
        # Iterate over all points in _point_cds
        for _i in range(len(_point_cds.data['x'])):
            for _key in list(_angles.keys()):
                _out['point'].append(_i)
                _out['color'].append(_point_cds.data['color'][_i])
                _out['amp'].append(_point_cds.data[_key][_i])
                _out['sin2theta'].append( (np.sin(_angles[_key] * np.pi / 180.))**2 )
        return _out

    def calc_avo_curve_dict(_point_cds)->dict:
        sin2theta_range = [0, 0.4]
        _out = dict(
            point=[],
            sin2theta=[],
            y=[],
            color=[]
        )
        # Iterate over all points in _point_cds
        for _i in range(len(_point_cds.data['x'])):
            _out['point'].append(_i)
            _out['color'].append(_point_cds.data['color'][_i])
            _out['sin2theta'].append(sin2theta_range),
            _out['y'].append([_point_cds.data['i'][_i], _point_cds.data['i'][_i] +
                              sin2theta_range[1] * _point_cds.data['g'][_i]])
        return _out

    # Create figure to hold scatter points of offset amplitudes
    sp = figure(width=600, height=400)

    scatter_point_dict = calc_new_point_cds_dict(points_cds, angles)
    scatter_point_cds = ColumnDataSource(scatter_point_dict)

    avo_curve_dict = calc_avo_curve_dict(points_cds)
    avo_curve_cds = ColumnDataSource(avo_curve_dict)

    draw_calculated_avo_amps(sp, scatter_point_cds, angle_cds=angle_cds)
    draw_avo_curves(sp, avo_curve_cds)

    # Add interactivity
    select_stack = Select(title='Select angle stack to view', value=names[0], options=names)
    if line_direction == 'inline':
        select_direction = Select(title='Inline or Xline?', value='Inline', options=['Inline', 'Xline'], disabled=True)
        select_line = Select(title='Select line nr.', value=inline0, options=inlines)
    else:
        select_direction = Select(title='Inline or Xline?', value='Xline', options=['Inline', 'Xline'], disabled=True)
        select_line = Select(title='Select line nr.', value=xline0, options=xlines)
    # run_calc = Button(label='Calc. I & G', button_type='success')
    update_seismic = Button(label='Update', button_type='success')

    # callback functions
    title_callback = CustomJS(
        args=dict(title=titles, dir=select_direction, line=select_line, selected=select_stack, xaxis=p.xaxis[0]),
        code="""
        const stack = selected.value;
        const t = title[stack];
        const d = dir.value;
        const l = line.value;
        xaxis.axis_label = t + ':  ' + d + ': ' + l;
        console.log('Test ', t, d, l);
        """
    )
    update_direction_callback = CustomJS(
        args=dict(direction=select_direction, line=select_line, inlines=inlines, xlines=xlines,
                  inline0=inline0, xline0=xline0),
        code="""
        var test = "Unknown";
        if (direction.value === 'Inline') {
            test = "You choose Inline";
            line.options=inlines;
            line.value=inline0;
        } else {
            test = "You choose Xline";
            line.options=xlines;
            line.value=xline0;
        };
        console.log('Direction: ', test);
        """
    )
    # def switch_seismic_callback(attr, old, new):
    def switch_seismic_callback():
        global selected_cells
        if select_direction.value == 'Inline':
            _seismic_cdss, _x, _y, _suffix = _load_angle_stacks(angle_stacks, voi, int(select_line.value), None, verbose)
        else:
            _seismic_cdss, _x, _y, _suffix = _load_angle_stacks(angle_stacks, voi, None, int(select_line.value), verbose)

        # current_cds.data['value'] = [seismic_cdss[new].T]
        current_cds.data['value'] = [_seismic_cdss[select_stack.value].T]
        calc_offset_amplitudes(use_updated_seismic=_seismic_cdss)
        calc_avg_offset_amplitudes(_seismic_cdss, angle_cds, selected_cells, verbose=True)
        selected_cells = []

    def calc_i_g():
        # calculate intercept and gradient for each point
        # For the changes in point_cds to be detected by the browser, we need to
        # change the whole .data property
        # It is not a large data set, so we take a copy of it:
        new_data = deepcopy(points_cds.data)
        for _i in range(len(new_data['x'])):
            _angles = []
            _amps = []
            for _a, _stack in enumerate(names):
                if _stack != angle_cds.data['name'][_a]:
                    info_txt = 'Mix up of angle stacks. {} != {}'.format(_stack, angle_cds.data['name'][_a])
                    print_info(info_txt, 'error', logger, 'IOError')
                # print(_stack, angle_cds.data['name'][_a], angle_cds.data['use'][_a], angle_cds.data['angles'][_a])
                if angle_cds.data['use'][_a]:
                    _angles.append(angle_cds.data['angles'][_a])
                    _amps.append(new_data[_stack][_i])
            _angles = np.array(_angles)
            _amps = np.array(_amps)
            # print('XXX', _angles, _amps)
            # _angles = np.array([_as.angle for _as in angle_stacks])
            # _amps = np.array([new_data[_stack][_i] for _stack in names])
            _int, _grad, _qual = avo_ig(_amps, _angles)
            new_data['i'][_i] = _int
            new_data['g'][_i] = _grad
            new_data['q'][_i] = _qual
            if verbose:
                print('Point: {}, amps: {}, I: {}, G: {}, Q: {}'.format(
                    _i,
                    [str(new_data[_stack][_i]) for _stack in names],
                    _int, _grad, _qual))
        points_cds.data = dict(new_data)
        scatter_point_dict = calc_new_point_cds_dict(points_cds, angles)
        scatter_point_cds.data = scatter_point_dict

    def calc_offset_amplitudes(use_updated_seismic=None):
        _count = 0
        if use_updated_seismic is not None:
            _seismic_cdss = use_updated_seismic
        else:
            _seismic_cdss = seismic_cdss
        for _x, _y in zip(points_cds.data['x'], points_cds.data['y']):
            _i = closest(_x, x)
            _j = closest(_y, y)
            for _row, _stack in enumerate(names):
                if verbose:
                    print('{}: On {}, from point {}:{} add {}'.format(
                        angle_cds.data['use'][_row], _stack, _x, _y, _seismic_cdss[_stack][_i, _j].data))
                points_cds.data[_stack][_count] = _seismic_cdss[_stack][_i, _j].data
            _count += 1
        calc_i_g()

    def calc_avg_offset_amplitudes(_seismic_cdss, _angle_cds, _selected_cells, verbose=False):
        """
        Should calculate the average seismic amplitude for the selected cells for all angle stacks
        :param _seismic_cdss:
        :param _angle_cds:
        :param _selected_cells:
        :return:
        """
        _new_data = deepcopy(_angle_cds.data)
        container = {}
        for _i, _stack in enumerate(_new_data['name']):
            container[_stack] = []
            for _index in _selected_cells:
                container[_stack].append(_seismic_cdss[_stack][_index].data)
            if len(container[_stack]) > 0:
                # As pointed out by Ben, the average of a value oscillating around zero will be near zero,
                # _new_data['avg_amp'][_i] = np.nanmean(np.array(container[_stack]))
                # Try using the absolute value
                _new_data['avg_amp'][_i] = np.nanmean(np.abs(np.array(container[_stack])))
            else:
                _new_data['avg_amp'][_i] = 0.0
        _angle_cds.data = dict(_new_data)
        if verbose:
            for _i, _stack in enumerate(_angle_cds.data['name']):
                print('At {}, the avg. absolute amplitude of {} is calculated from {}'.format(_stack,
                                                                                     _angle_cds.data['avg_amp'][_i],
                                                                                     ['{:.2}'.format(np.abs(_amp)) for _amp in container[_stack]]))

    def changing_stacks_callback(attr, old, new):
        calc_i_g()

    def changing_points_callback(attr, old, new):
        avo_curve_dict = calc_avo_curve_dict(points_cds)
        avo_curve_cds.data = avo_curve_dict
        scatter_point_dict = calc_new_point_cds_dict(points_cds, angles)
        scatter_point_cds.data = scatter_point_dict

    # connect interactive input to callbacks
    # select_stack.js_on_change('value', title_callback)
    select_direction.js_on_change('value', update_direction_callback)
    # select_stack.on_change('value', switch_seismic_callback)
    update_seismic.on_click(switch_seismic_callback)
    update_seismic.js_on_click(title_callback)
    # run_calc.on_click(calc_offset_amplitudes)  # this uses the old seismic (e.g. inline = inline0)
    # run_calc.on_click(switch_seismic_callback)
    angle_cds.on_change('data', changing_stacks_callback)
    points_cds.on_change('data', changing_points_callback)

    return p, c_amp, select_direction, select_line, select_stack, update_seismic, points_table, angle_table, sp

def draw_calculated_avo_amps(
        p: figure,
        point_cds: ColumnDataSource,
        angle_cds=None,
):
    """
    Should draw the extracted input amplitude points
    Need to respond to changes in calculated points
    :param p:
        Bokeh figure to add the glyphs to
    :param point_cds:
        ColumnDataSource
        with keys: 'point', 'amp', 'sin2theta', 'color'
    :param angle_cds
        ColumnDataSource
        optional data cds with average values
        with keys: 'avg_amp', 'sin2theta'
    :return:
    """
    p.scatter(
        x='sin2theta',
        y='amp',
        fill_color='color',
        legend_field='point',
        source=point_cds,
        line_color='black',
        size=10
    )
    if angle_cds is not None:
        p.scatter(
            x='sin2theta',
            y='avg_amp',
            source=angle_cds,
            legend_label='AOI avg. amp.',
            marker='square',
            size=15,
            fill_color='gray',
            fill_alpha=0.6
        )
    p.xaxis.axis_label = 'Sin(theta)**2 (Offset)'
    p.yaxis.axis_label = 'Amplitude'
    p.legend.click_policy = 'hide'
    p.legend.location = 'top_right'
    p.legend.label_text_font_size = '8pt'


def draw_avo_curves(
        p: figure,
        line_cds: ColumnDataSource
):
    """
    Should draw the extracted AVO curves
    Need to respond to changes in calculated I & G
    :param p:
        Bokeh figure to add the glyphs to
    :param line_cds:
        ColumnDataSource
        with keys: 'point', 'y', 'sin2theta', 'color'
    :return:
    """
    p.multi_line(
        xs='sin2theta',
        ys='y',
        color='color',
        legend_field='point',
        source=line_cds
    )


def next_color():
    return next(cclrs)


def straight_line(x, a, b):
    return a*x + b


def avo_ig(amp, ang):
    """Calculates the Intercept and Gradient.
    
    Based on avo_IGn in
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/avo_attributes.ipynb
    which in turn is based on
    https://github.com/waynegm/OpendTect-External-Attributes/blob/master/Python_3/Jupyter/AVO_IG.ipynb

    """
    ang_rad = np.sin(np.radians(ang))**2  # converts incident angle to sin2theta
    m, resid, rank, singval= np.linalg.lstsq(np.c_[ang_rad,np.ones_like(ang_rad)], amp, rcond=None)
    # using only 2 angle stacks residuals  are not computed
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    if amp.shape[0]>2:
        qual = 1 - resid/(amp.shape[0] * np.var(amp,axis=0))
        return m[1],m[0],qual # intercept, gradient, quality factor
    else:
        return m[1],m[0], None # intercept, gradient


def pickle_test_data():
    filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\SEISMIC DATA\\KPSDM-NEAR_10deg_cropped.sgy"
    near, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\SEISMIC DATA\\KPSDM-MID_18deg_cropped.sgy"
    mid, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\SEISMIC DATA\\KPSDM-FAR_26deg_cropped.sgy"
    far, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

    inline = 6550
    xline = 27009
    near_line = near.sel(INLINE=inline)
    mid_line = mid.sel(INLINE=inline)
    far_line = far.sel(INLINE=inline)

    with open('C:\\Users\\marten\\Downloads\\angle_lines.dat', 'wb') as output:
        pickle.dump((twt, near_line, mid_line, far_line), output)

    near_trace = near.sel(INLINE=inline, XLINE=xline)
    mid_trace = mid.sel(INLINE=inline, XLINE=xline)
    far_trace = far.sel(INLINE=inline, XLINE=xline)

    with open('C:\\Users\\marten\\Downloads\\angle_traces.dat', 'wb') as output:
        pickle.dump((twt, near_trace, mid_trace, far_trace), output)


def interpolate_along_offset(offset_traces: np.ndarray, offset_angles: pint.Quantity, inc_angles: pint.Quantity):
    """
    Creates a "Pseudo gather"
    Interpolates the input offset_traces (near, mid, far, ...) in the offset direction so that we get a
    more densely sampled set of traces

    :param offset_traces:
        np.ndarray of shape (M, N)
        M is the number of traces, and N is the number of samples along the trace
    :param offset_angles:
        array like pint Quantity of length M which contains the center incidence angle for each trace
    :param inc_angles:
        array like pint Quantity of length J, with the incidence angles we would like to evaluate the seismic.
        The min & max of inc_angles should not be less / larger than the corresponding min / max of
        offset_angles. Else we would need to extrapolate
    :return:
        np.ndarray of size (J, N)
    """
    from scipy import interpolate

    # calculate sin2(theta)
    s2t_in = np.sin(offset_angles.to('rad').magnitude)**2
    s2t_out = np.sin(inc_angles.to('rad').magnitude)**2

    return interpolate.interp1d(s2t_in, offset_traces, axis=0, fill_value=None)(s2t_out)


def seismic_color_map(min_val=-1, max_val=1, n=256, symmetric=True) -> LinearColorMapper:
    # Construct cmap dictionary
    c_dict = {'red': ((0, 0.6314, 0.6314),
                      (0.33, 0, 0),
                      (0.4, 0.302, 0.302),
                      (0.5, 0.8, 0.8),
                      (0.6, 0.3804, 0.3804),
                      (0.667, 0.749, 0.749),
                      (1, 1, 1)),
              'green': ((0, 1, 1),
                        (0.33, 0, 0),
                        (0.4, 0.302, 0.302),
                        (0.5, 0.8, 0.8),
                        (0.6, 0.2706, 0.2706),
                        (0.667, 0, 0),
                        (1, 1, 1)),
              'blue': ((0, 1, 1),
                       (0.33, 0.749, 0.749),
                       (0.4, 0.302, 0.302),
                       (0.5, 0.8, 0.8),
                       (0.6, 0, 0),
                       (0.667, 0, 0),
                       (1, 0, 0))}

    if symmetric:
        if np.sign(min_val) != np.sign(max_val):
            max_magnitude = np.max([np.abs(min_val), np.abs(max_val)])
            min_val = np.sign(min_val) * max_magnitude
            max_val = np.sign(max_val) * max_magnitude

    cmap = mpl.colors.LinearSegmentedColormap('Seismic', c_dict).reversed()
    _colors = cmap(np.linspace(0, 1, n))
    return LinearColorMapper(palette=[mpl.colors.to_hex(_c) for _c in _colors], low=min_val, high=max_val)

def read_zgy(filename: str,
             inline_range: range,
             xline_range: range,
             sample_range: range):
    """

    :param filename:
    :param inline_range:
        range with integers corresponding to the inline numbers we want to read
        E.G. range(6204, 6264, 6)
    :param xline_range:
        range with integers corresponding to the xline numbers we want to read
        E.G. range(21688, 23600, 8)
    :param sample_range:
        range with integers corresponding to the sample numbers we want to read
        E.G. range(500, 1000, 20)
    :return:
    """
    import xarray as xr
    zgy = xr.open_dataset(filename)
    sub_vol = zgy.sel(
        iline=inline_range,
        xline=xline_range,
        samples=sample_range
    )
    zgy.close()
    return sub_vol


def add_i_g_point(_p, _x, _y, _angle_stack_names):
    _dx = (_x[-1] - _x[0]) / 4.
    _dy = (_y[-1] - _y[0]) / 4.
    xs = [_x[0] + (_i + 1) * _dx for _i in range(2)]
    ys = [_y[0] + (_i + 1) * _dy for _i in range(2)]
    _cds_dict = dict(
        # x=xs, y=ys, color=['red', 'blue', 'yellow'], i=[0., 0., 0.], g=[0., 0., 0.], q=[None, None, None])
        x=xs, y=ys, color=['red', 'blue'], i=[0., 0.], g=[0., 0.], q=[None, None])

    # Add a key: value pair for each angle stack to hold the amplitudes at each point
    for _name in _angle_stack_names:
        _cds_dict[_name] = [None, None]

    _cds = ColumnDataSource(_cds_dict)

    _renderer = _p.scatter(x='x', y='y', fill_color='color', source=_cds, line_color='black', size=10)
    formatter = NumberFormatter(format='0.0')

    # Try to color the cells of the 'color' column by their value
    template = """
            <div style="background:<%= 
                (function color_from_val(){
                    return(color)
                    }()) %>; 
                color: white"> 
            <%= value %>
            </div>
        """
    color_formatter = HTMLTemplateFormatter(template=template)

    _columns = [TableColumn(field="x", title="X", formatter=formatter),
                TableColumn(field="y", title="Y", formatter=formatter),
                TableColumn(field='color', title='Color', formatter=color_formatter),
                TableColumn(field='i', title='I', formatter=formatter),
                TableColumn(field='g', title='G', formatter=formatter),
                TableColumn(field='q', title='Qual.', formatter=NumberFormatter(format='0.00')),
                ]
    _table = DataTable(source=_cds, columns=_columns, editable=True, height=200)
    draw_tool = PointDrawTool(renderers=[_renderer], empty_value='black')
    _p.add_tools(draw_tool)
    _p.toolbar.active_tap = draw_tool
    return _table, _cds


    # switch_seismic_callback = CustomJS(
    #     args = dict(
    #         cs=current_cds,
    #         new_cds=[seismic_cdss[select_stack.value].T],  # this doesn't update the selected seismic!
    #         #  new_cds=seismic_cdss,
    #         selected=select_stack
    #     ),
    #     code="""
    #     const stack = selected.value;
    #     cs.data['value'] = new_cds;
    #     //cs.data['value'][0] = new_cds[stack].T;  //Can't use pythons transpose function in JavaScript
    #     //cs.data['value'][0] = new_cds[stack];
    #     cs.change.emit();
    #     console.log('test: ', stack, cs.data['value'][0,0][0]);
    #     """
    # )


def closest(_x0: float, _x: np.ndarray):
    if type(_x0) != float:
        _x0 = float(_x0)
    _i = np.argmin((_x - _x0)**2)
    # return _x[_i]
    return _i

def _load_angle_stacks(_angle_stacks, _voi, _inline, _xline, _verbose=False):
    _seismic_cdss = {}
    _x = None
    _y = None
    _line = None
    _suffix = None
    for _as in _angle_stacks:
        if _verbose:
            print_info('Loading {}'.format(_as.title), 'info', logger)
        zgy = read_zgy(_as.filename, _voi.i, _voi.j, _voi.k)
        if _inline is not None:
            _line = zgy.sel(iline=_inline)
            _x = zgy['data'].coords['xline'].data
            _suffix = 'Inline: {}'.format(_inline)
        elif _xline is not None:
            _line = zgy.sel(xline=_xline)
            _x = zgy['data'].coords['iline'].data
            _suffix = 'Xline: {}'.format(_xline)
        else:
            print_info("Either 'inline' or 'xline' must be set", 'error', logger, 'IOError')
        # _seismic_cdss[_as.name] = ColumnDataSource(dict(value=[_line.data.T]))
        _seismic_cdss[_as.name] = _line.data
        _y = zgy['data'].coords['samples'].data
    return _seismic_cdss, _x, _y, _suffix


def seismic_data_plot(_type:str) -> figure:
    """
    Creates a plot window with seismic data suitable for testing other functionality on
    :param _type:
        'seismic section',
        'avo section',
        'synthetic section'
    :return:
        Bokeh figure
    """
    p = figure()
    if _type == 'seismic section':
        near = AngleStack('near',
                          "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-NEARSTACK_MIG-FIN_16bit.zgy",
                          angle=10.)
        mid = AngleStack('mid',
                         "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-MIDSTACK_MIG-FIN_16bit.zgy",
                         angle=18.)
        far = AngleStack('far',
                         "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-FARSTACK_MIG_FIN_16bit.zgy",
                         angle=26.)
        names = ['near', 'mid', 'far']
        voi = VolumeOfInterest(
            range(4800, 5200, 1),
            range(32253, 32255, 1),
            range(3000, 4000, 4))

        xline = 32254
        seismic_cdss, x, y, suffix = _load_angle_stacks([near, mid, far], voi, None, xline, True)
        cds = ColumnDataSource(dict(value=[seismic_cdss[names[0]].T]))
        min_val = np.nanmin(cds.data['value'])
        max_val = np.nanmax(cds.data['value'])
        _seismic_color_map = seismic_color_map(min_val=min_val, max_val=max_val)
        p.image('value', source=cds,
                # TODO The seismic seems upside down!
                color_mapper=_seismic_color_map, dh=cds.data['value'][0].shape[1], dw=cds.data['value'][0].shape[0],
                x=0, y=-1. * cds.data['value'][0].shape[1])

    elif _type == 'avo section':
        _synts = StochasticSyntheticTraces(n_reflectors=10)
        synts = _synts.get_traces(simulate_avo=True)
        cds = ColumnDataSource({'value': [synts.T]})
        _seismic_color_map = seismic_color_map(min_val=np.min(synts), max_val=np.max(synts))
        p.image('value', source=cds,
                color_mapper=_seismic_color_map, dh=synts.shape[1], dw=synts.shape[0], x=0, y=0)

    return p

def picks_from_seismic_cds(seismic_cds: ColumnDataSource, feature: str, depth_index: int | list | None = None):
    """
    Extract amplitudes and depth indexes of selected features from the input seismic CDS
    :param seismic_cds:
            ColumnDataSource({'value': [traces.T]})
            where traces is a np.ndarray of size MxN, contains M number of seismic traces, each of length N
    :param feature:
        What seismic feature to pick:
            - 'extract' : Extract seismic amplitude at given depth index
            - 'global_max' :
            - 'global_min' :
            - 'nearest_max' :
            - 'nearest_min' :
            - 'nearest_max_above' :
            - 'nearest_min_above' :
            - 'nearest_max_below' :
            - 'nearest_min_below' :
    :param depth_index:
        Either a list of length M (one for each trace) depth indexes, or None
    :return:
        list of amplitudes, list of depth indexes from where the amplitudes are taken from
    """
    from scipy.signal import argrelextrema

    # TODO CLEAN UP, NOT SURE IF depth_index are just indexes or actually depth values!
    #

    data = seismic_cds.data['value'][0]
    if depth_index is None:
        if feature not in ['global_max', 'global_min']:
            print_info('Depth indexes are needed to extract amplitudes', 'error', logger, 'IOError')
    else:
        if isinstance(depth_index, int):
            depth_index = [depth_index] * data.shape[1]
        if len(depth_index) != data.shape[1]:
            print_info('Depth indexes size must match number of traces', 'error', logger, 'IOError')

    amps = []
    amp_inds = []

    if feature == 'global_max':
        for i in range(data.shape[1]):
            amps.append(np.nanmax(data[:, i]))
            amp_inds.append(np.nanargmax(data[:,i]))
    elif feature == 'global_min':
        for i in range(data.shape[1]):
            amps.append(np.nanmin(data[:, i]))
            amp_inds.append(np.nanargmin(data[:,i]))
    elif feature == 'extract':
        for i in range(data.shape[1]):
            amps.append(data[depth_index[i], i])
            amp_inds = depth_index
    elif feature in ['nearest_max', 'nearest_max_above', 'nearest_max_below']:
        for i in range(data.shape[1]):
            max_idx = argrelextrema(data[:, i], np.greater)[0]
            if feature == 'nearest_max':
                idx = closest(depth_index[i], np.array(max_idx))
                amps.append(data[max_idx[idx], i])
                amp_inds.append(max_idx[idx])
            if feature == 'nearest_max_above':
                filtered_idx = [x for x in max_idx if x <= depth_index[i]]
                if len(filtered_idx) < 1:
                    continue
                idx = closest(depth_index[i], np.array(filtered_idx))
                amps.append(data[filtered_idx[idx], i])
                amp_inds.append(filtered_idx[idx])
            if feature == 'nearest_max_below':
                filtered_idx = [x for x in max_idx if x >= depth_index[i]]
                if len(filtered_idx) < 1:
                    continue
                idx = closest(depth_index[i], np.array(filtered_idx))
                amps.append(data[filtered_idx[idx], i])
                amp_inds.append(filtered_idx[idx])
    elif feature in ['nearest_min', 'nearest_min_above', 'nearest_min_below']:
        for i in range(data.shape[1]):
            min_idx = argrelextrema(data[:, i], np.less)[0]
            if feature == 'nearest_min':
                idx = closest(depth_index[i], np.array(min_idx))
                amps.append(data[min_idx[idx], i])
                amp_inds.append(min_idx[idx])
            if feature == 'nearest_min_above':
                filtered_idx = [x for x in min_idx if x <= depth_index[i]]
                if len(filtered_idx) < 1:
                    continue
                idx = closest(depth_index[i], np.array(filtered_idx))
                amps.append(data[filtered_idx[idx], i])
                amp_inds.append(filtered_idx[idx])
            if feature == 'nearest_min_below':
                filtered_idx = [x for x in min_idx if x >= depth_index[i]]
                if len(filtered_idx) < 1:
                    continue
                idx = closest(depth_index[i], np.array(filtered_idx))
                amps.append(data[filtered_idx[idx], i])
                amp_inds.append(filtered_idx[idx])

    return amps, amp_inds

def get_coords_of_seismic_figure(p: figure):
    """
    A "seismic" figure, as created by "add_seismic_to_figure()" is assumed to only contain one glyph
    :param p:
    :return:
        x0, y0, dw, dh
        The data coordinates of the location (x0, y0) and data width (dw) and data height (dh) of the seismic figure
    """
    from bokeh.models import GlyphRenderer
    glyph_renderers = [r for r in p.renderers if isinstance(r, GlyphRenderer)]
    glyphs = [r.glyph for r in glyph_renderers]
    # srcs = [r.data_source for r in glyph_renderers]
    glyph = glyphs[0]
    return glyph.x, glyph.y, glyph.dw, glyph.dh

def get_size_of_seismic_figure(p: figure):
    """
    A "seismic" figure, as created by "add_seismic_to_figure()" is assumed to only contain one glyph
    :param p:
    :return:
        x0, y0, dw, dh
        The data coordinates of the location (x0, y0) and data width (dw) and data height (dh) of the seismic figure
    """
    from bokeh.models import GlyphRenderer
    glyph_renderers = [r for r in p.renderers if isinstance(r, GlyphRenderer)]
    # glyphs = [r.glyph for r in glyph_renderers]
    srcs = [r.data_source for r in glyph_renderers]
    test = srcs[0]
    test = test.data['value']
    return test[0].T.shape



class TestCases(unittest.TestCase):

    def test_data_for_avo_analyzer(self, section_type: str | None = None, synthetic: bool = False):
        from blixt_rp.plotting.log_plotter import select_column
        # Determine if input is of either section type 'real', 'avo' or 'synthetic'
        if section_type is None:
            section_type = 'avo section'
        model = None

        idx_refl = None
        refl_points = None
        if section_type == 'line section' and not synthetic:
            near = AngleStack('near',
                              "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-NEARSTACK_MIG-FIN_16bit.zgy",
                              angle=10.)
            mid = AngleStack('mid',
                             "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-MIDSTACK_MIG-FIN_16bit.zgy",
                             angle=18.)
            far = AngleStack('far',
                             "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-FARSTACK_MIG_FIN_16bit.zgy",
                             angle=26.)
            ufar = AngleStack('ufar',
                              "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-UFARSTACK_MIG-FIN_16bit.zgy",
                              angle=34.)
            angle_stacks = [near, mid, far, ufar]

            voi = VolumeOfInterest(
                range(4640, 5243, 1),
                range(32253, 32255, 1),
                range(2616, 5108, 4) )

            line_n = 32254
            line_direction = 'xline'  # or 'inline'
            seismic_cds = None
            p = None
            c_amp = None

        elif section_type == 'line section' and synthetic:
            from blixt_rp.core.models import WedgeModel
            model = WedgeModel(n_traces=51)
            (title, lf_table, add_row, delete_row, update, model_table, add_row_m, delete_row_m, update_m, grid, controls,
             seismic_cds) = model.draw_2d()
            p = grid
            c_amp = None
            refl_points = None

        elif section_type == 'avo section' and not synthetic:
            refl_points = []
            _synts = StochasticSyntheticTraces()
            synts = _synts.get_traces(simulate_avo=True)
            idx_refl = _synts.idx_refl
            seismic_cds = ColumnDataSource({'value': [synts.T]})
            p = create_seismic_figure(600, 400)
            c_amp = add_seismic_to_figure(p, seismic_cds, np.arange(40), np.arange(5000))
            if idx_refl is not None:
                depth = np.linspace(0., 5000., seismic_cds.data['value'][0].shape[0])
                for _j in idx_refl:
                    print('XXX, idx_refl: {}, in depth: {}, and in inverted depth {}'.format(_j, depth[_j], 5000. - depth[_j]))
                    refl_points.append([20., depth[_j]])

        elif section_type == 'avo section' and synthetic:
            from blixt_rp.core.models import LaminarModel
            model = LaminarModel()
            (title, lf_table, add_row, delete_row, update, _model_table,
             add_row_m, delete_row_m, update_m, grid, freq_slider, synth_cds_base) = model.draw()
            p = grid
            seismic_cds = synth_cds_base
            c_amp = None

        else:
            raise IOError('Wrong section type ({}) and synthetic ({}) provided'.format(section_type, synthetic))

        return p, seismic_cds, c_amp, refl_points, model



    def test_seismic_data_plot(self):
        # p = seismic_data_plot('seismic section')
        p = seismic_data_plot('avo section')
        show(p)

    def test_interpolate(self):
        from blixt_rp.plotting.log_plotter import LogColumn, LogPlotter
        las_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Tampen\\Wells\\34_3_3S_seismic.las"
        # las_file = "G:\\My Drive\\Work - Current and Recent\\GeoMind\\Clients\\AkerBP\\PL932 Kaldafjell AVO feasibility\\Wells\\34_3_3S_seismic.las"
        t = Template(**{'name': 'Seismic', 'units': 'dimensionless', 'line_color': 'b'})
        log_table = LogTable({'Seismic':
                                  ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2'
                                   ]})
        rename_logs = {
            'near': ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2'],
            'mid': ['CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2'],
            'far': ['CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2'],
            'ufar': ['CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2']
        }
        log_curves, well_info = read_las(las_file, log_table=log_table, template=t, rename_logs=rename_logs)

        offset_angles = pint.Quantity(np.array([10., 18., 26., 34.]), 'deg')
        traces = np.zeros((len(offset_angles), len(log_curves['near'])))
        traces[0,:] = log_curves['near'].values
        traces[1,:] = log_curves['mid'].values
        traces[2,:] = log_curves['far'].values
        traces[3,:] = log_curves['ufar'].values
        # fig1, ax1 = plt.subplots()
        # ax1.imshow(traces.T, aspect=1./1000)
        st_orig = SeismicTraces(x=offset_angles.magnitude,
                                 y=log_curves['near'].depth.values,
                                 # traces=traces,
                                 traces=None,
                                 cds=ColumnDataSource({'value': [traces.T]}),
                                 trace_type='avo'
                                 )
        lc_orig = LogColumn('orig',
                            seismic_traces=st_orig)

        new_angles = pint.Quantity(np.linspace(10, 34, 100), 'deg')
        new_traces = interpolate_along_offset(traces, offset_angles, new_angles)
        st_intrp = SeismicTraces(x=new_angles.magnitude,
                                 y=log_curves['near'].depth.values,
                                 # traces=new_traces,
                                 traces=None,
                                 cds=ColumnDataSource({'value': [new_traces.T]}),
                                 trace_type='avo')
        lc_intrp = LogColumn('interpolated', seismic_traces=st_intrp)

        plotter = LogPlotter(columns=[lc_orig, lc_intrp])
        grid = plotter.figure()
        show(grid)


        # fig2, ax2 = plt.subplots()
        # ax2.imshow(new_traces.T, aspect=1./100.)
        # plt.show()


    def test_synthetic_traces(self):
        _synts = StochasticSyntheticTraces()
        synts = _synts.get_traces(simulate_avo=True)
        print(synts.shape)
        cds = ColumnDataSource({'value': [synts.T]})
        print(np.min(synts))
        _seismic_color_map = seismic_color_map(min_val=np.min(synts), max_val=np.max(synts))

        p = figure()
        p.image('value', source=cds,
                color_mapper=_seismic_color_map, dh=synts.shape[1], dw=synts.shape[0], x=0, y=0)
        show(p)


    def test_ixg_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2)

        with open('C:\\Users\\marten\\Downloads\\angle_traces.dat', 'rb') as input:
            twt, near_trace, mid_trace, far_trace = pickle.load(input)

        i, g, q = avo_ig(np.array([near_trace.values.flatten(), \
                                   mid_trace.values.flatten(), \
                                   far_trace.values.flatten()]), [10., 18., 26.])

        res = least_squares(
            mycf.residuals,
            [1.,1.],
            args=(i, g),
            kwargs={'target_function': straight_line})
        trend_line = 'WS = {:.4f}*I {:.4f} - G'.format(*res.x)
        print(res.status)
        print(res.message)
        print(res.success)
        print(res.fun.shape)

        xp.plot(i, g, cdata=res.fun,
           ctempl = {'full_name': 'Residual', 'colormap': 'seismic', 'min': np.min(res.fun), 'max':np.max(res.fun)},
           title='IL: 6550, XL: 27009\n{}'.format(trend_line),
           fig=fig, ax=axes[0])

        x_new = np.linspace(*axes[0].get_xlim(), 50)
        axes[0].plot(x_new, straight_line(x_new, *res.x), c='r', label='_nolegend_')

        with open('C:\\Users\\marten\\Downloads\\angle_lines.dat', 'rb') as input:
            twt, near_line, mid_line, far_line = pickle.load(input)
        print('Line shape: {}'.format(near_line.values.shape))

        i, g, q = avo_ig(np.array([near_line.values.flatten(), \
                                   mid_line.values.flatten(), \
                                   far_line.values.flatten()]), [10., 18., 26.])

        res = least_squares(
                mycf.residuals,
                [1.,1.],
                args=(i, g),
                kwargs={'target_function': straight_line}
        )
        trend_line = 'WS = {:.4f}*I {:.4f} - G'.format(*res.x)
        print(res.status)
        print(res.message)
        print(res.success)
        print(res.fun.shape)

        xp.plot(i[::100], g[::100], cdata=res.fun[::100],
           ctempl = {'full_name': 'Residual', 'colormap': 'seismic', 'min': np.min(res.fun), 'max':np.max(res.fun)},
           title='IL: 6550\n{}'.format(trend_line),
           fig=fig, ax=axes[1],
           edge_color=False)

        x_new = np.linspace(*axes[1].get_xlim(), 50)
        axes[1].plot(x_new, straight_line(x_new, *res.x), c='r', label='_nolegend_')

        fig, ax = plt.subplots()
        ax.imshow(np.transpose(res.fun.reshape((401, 501))),
                  cmap='seismic',
                  vmin=np.min(res.fun),
                  vmax=np.max(res.fun),
                  interpolation='spline16',
                  aspect='auto',
                  extent=(0,400,3000,1000))

        #X, Y = np.meshgrid(np.arange(0,401,1), twt)
        #ax.pcolormesh(X, Y, np.transpose(res.fun.reshape((401, 501))),
        #          cmap='seismic',
        #          vmin=np.min(res.fun),
        #          vmax=np.max(res.fun),
        #          )
        #ax.invert_yaxis()
        ax.grid(True)
        ax.set_title('Inline: 6550, Residual')
        ax.set_xlabel('X lines')
        ax.set_ylabel('TWT [ms]')
        plt.show()


    def test_plot_amp_vs_offset(self):
        filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\Test_angle_stacks\\KPSDM-NEAR_10deg_cropped.sgy"
        near, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
        filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\Test_angle_stacks\\KPSDM-MID_18deg_cropped.sgy"
        mid, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
        filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\Test_angle_stacks\\KPSDM-FAR_26deg_cropped.sgy"
        far, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

        inline = 6550
        xline = 27009
        t0 = 1245
        near_trace = near.sel(INLINE=inline, XLINE=xline)
        near_max = np.max(np.abs(near_trace.values))
        mid_trace = mid.sel(INLINE=inline, XLINE=xline)
        mid_max = np.max(np.abs(mid_trace.values))
        far_trace = far.sel(INLINE=inline, XLINE=xline)
        far_max = np.max(np.abs(far_trace.values))
        angs = np.array([10., 18., 26.])
        fig, axes = plt.subplots(nrows=1, ncols=2)


        for a, t in zip(angs, [near_trace, mid_trace, far_trace]):
            wiggle_plot(axes[0], twt, t.values, zero_at=a, scaling=10. / max([near_max, mid_max, far_max]))

        axes[0].axhline(t0)

        ind = np.argmin(abs(near_trace.TWT.values - t0))

        amps = np.array([near_trace.values[ind], mid_trace.values[ind], far_trace.values[ind]])

        i, g, q = avo_ig(amps, angs)

        xp.plot(angs, amps, cdata='b', fig=fig, ax=axes[1])
        _angs = np.linspace(1, angs[-1]+10)
        axes[1].plot(_angs, i + g*np.sin(np.radians(_angs))**2)

        plt.show()


    def test_plot_line(self):
        filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\Test_angle_stacks\\KPSDM-NEAR_10deg_cropped.sgy"
        data, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
        print(header)
        uu={'add_colorbar':False,'robust':True,'interpolation':'spline16'}
        fig, ax = plt.subplots()
        data.plot.imshow(x='XLINE', y='TWT', yincrease=False, ax=ax, **uu)
        plt.show()


    def test_amp_spectra(self):
        from blixt_rp.core.wavelets import plot_cwt
        filename = "U:\\COMMON\\SAAS DEVELOPMENT\\TEST_DATA\\Test_angle_stacks\\KPSDM-NEAR_10deg_cropped.sgy"
        data, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

        inline = 6550
        xline = 27009
        sr0 = 0.004 # sample rate in seconds
        this_trace = data.sel(INLINE=inline, XLINE=xline)
        this_inline = data.sel(INLINE=inline)

        #fig, ax = plt.subplots()
        #ax.plot(this_trace.data, twt)
        #ax.invert_yaxis()
        #ax.set_title('IL {}, XL {}'.format(inline, xline))
        #ax.set_ylabel('TWT')


        #f1, amp1, f_peak1 = fullspec(this_inline.data, sr0)
        #f2, amp2 = ampspec(this_trace.data, sr0, smoothing='median')

        #plot_ampspecs([[f1, amp1, f_peak1], [f2, amp2]], ['Whole inline 6550', 'Smooth single trace'])

        #import pandas as pd
        #dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
        #df_nino = pd.read_table(dataset)
        #N = df_nino.shape[0]
        #t0=1871
        #sr0=0.25
        #time = np.arange(0, N) * sr0 + t0
        #signal = df_nino.values.squeeze()

        waveletname = 'cmor'
        desired_freqs = np.linspace(1, 1./(2 * sr0), 10)
        # scales, fs = freq2scale(desired_freqs, waveletname, sr0)
        scales = np.arange(1, 128)
        plot_cwt(twt / 1000., this_trace.values, scales, waveletname=waveletname, cmap='jet')
        #plot_cwt(time/1000., signal, desired_freqs)
        # try extracting the scales corresponding to above frequencies
        #scales, fs = freq2scale(desired_freqs, 'morl', sr0)
        ##widths = np.linspace(2, 80, 41)  # pywt.scale2frequency('morl', 2)/0.004 = 101.6 which is less than 125
        #cwtmatr, freqs = pywt.cwt(this_trace.data, scales, 'morl', sr0)

        #X, Y = np.meshgrid(freqs, twt)

        #fig2, ax2 = plt.subplots()
        #ax2.contourf(X, Y, abs(cwtmatr.transpose()),
        #           cmap='jet',
        #           #vmax=abs(cwtmatr).max(),
        #           #vmin=-abs(cwtmatr).max()
        #           vmax=np.max(abs(cwtmatr)),
        #           vmin=np.min(abs(cwtmatr))
        #           )
        #ax2.invert_yaxis()
        #ax2.set_title('IL {}, XL {}'.format(inline, xline))
        #ax2.set_ylabel('TWT')
        #ax2.set_xlabel('Frequency [Hz]')
        #ax2.grid(True)
        plt.show()


    def test_read_zgy(self):
        f = "R:\\3D\\UTM31\\Acquired Data\\CGG23M03_NVG22PH1_BRAGE\\CGG18M01_NVG_Final_Ki-PreSDM_Z_VVertical_5.3.0.zgy"
        zgy = read_zgy(f, range(6204, 6264, 6), range(21688, 23600, 8), range(500, 1000, 20))  # (10 x 239 x 25)
        print(zgy.keys())

        self.assertTrue(list(zgy.keys()) == ['cdp_x', 'cdp_y', 'data'])

        self.assertTrue(list(zgy['data'].coords.keys()) == ['iline', 'xline', 'samples'])
        self.assertTrue(list(zgy['cdp_x'].coords.keys()) == ['iline', 'xline'])

        # zgy['cdp_x/y'].coords[i/xline] contains the inline, xline numbers
        # E.G:
        self.assertTrue(zgy['cdp_y'].coords['xline'][0] == 21688)

        # The zgy['cdp_x/y'].data contains and numpy array of each i/xline x/y coordinates.
        # E.G:
        self.assertAlmostEqual(zgy['cdp_y'].data[0,0],6701691.7233520)

        # A specific trace can be extracted by
        trace = zgy['data'].data[5, 100]
        self.assertTrue(len(trace) == 25)

        # and its corresponding x and y coordinates
        self.assertAlmostEqual(zgy['cdp_x'].data[5, 100], 498723.94845768646)
        self.assertAlmostEqual(zgy['cdp_y'].data[5, 100], 6711691.625876557)

        # and its corresponding i/x line numbers
        self.assertTrue(zgy['data'].coords['iline'][5], 6234)
        self.assertTrue(zgy['data'].coords['xline'][100], 22488)

        # Whole inline and x-line can be extracted using
        inline_slice = zgy.sel(iline=6234)
        crossline_slice = zgy.sel(xline=22488)

    def test_create_traces_from_zgy(self):
        import blixt_rp.plotting.log_plotter as bupp
        f = "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01_NVG21PH1-EW_FINAL_KPSDM_T_FAR_STK_16bit.zgy"
        xline_ranges = range(32253, 32255, 1)
        inline_ranges = range(4640, 5243, 1)
        sample_range = range(2616, 5108, 4)
        zgy = read_zgy(f, inline_ranges, xline_ranges, sample_range)
        xline = zgy.sel(xline=32254)
        x = zgy['data'].coords['iline'].data
        y = zgy['data'].coords['samples'].data
        amps = xline.data
        seis_traces = SeismicTraces(
            x=x,
            y=y,
            traces=amps,
            trace_type='index',
            title='CGG22M01_NVG21PH1-EW_FINAL_KPSDM_T_FAR_STK xline: 32254'
        )
        # seis_traces = SeismicTraces(
        #     x=x,
        #     y=y,
        #     traces=amps.T,
        #     trace_type='index',
        #     title='CGG22M01_NVG21PH1-EW_FINAL_KPSDM_T_FAR_STK xline: 32254'
        # )
        cl = bupp.LogColumn(name='data', seismic_traces=seis_traces)
        # The LogColumn is not the best method for plotting a seismic line this way, as it seems
        # to cause problem for the PointDrawTool
        # lp = bupp.LogPlotter(width=1000, columns=[cl], add_tools=[PointDrawTool()])
        lp = bupp.LogPlotter(width=1000, columns=[cl])
        grid = lp.figure()
        # Try with a PointDrawTool
        cds = ColumnDataSource(dict(
            x=[4700, 4900, 5000], y=[3000, 3500, 4000], color=['red', 'green', 'yellow'] ) )
        renderer = grid.children[0][0].scatter(x='x', y='y', color='color', source=cds, size=10)
        columns = [TableColumn(field="x", title="x"),
                   TableColumn(field="y", title="y"),
                   TableColumn(field='color', title='color')]
        table = DataTable(source=cds, columns=columns, editable=True, height=200)
        draw_tool = PointDrawTool(renderers=[renderer], empty_value='black')
        grid.children[0][0].add_tools(draw_tool)
        grid.children[0][0].toolbar.active_tap = draw_tool
        show(column(grid, table))

    def test_create_seismic_figure(self):
        f = "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01_NVG21PH1-EW_FINAL_KPSDM_T_FAR_STK_16bit.zgy"
        inline_ranges = range(4640, 5243, 1)
        xline_ranges = range(32253, 32255, 1)
        sample_range = range(2616, 5108, 4)
        zgy = read_zgy(f, inline_ranges, xline_ranges, sample_range)
        xline = zgy.sel(xline=32254)
        x = zgy['data'].coords['iline'].data
        y = zgy['data'].coords['samples'].data
        amps = xline.data
        seismic_cds = ColumnDataSource(dict(value=[amps.T]))

        p = create_seismic_figure(900, 600)
        c_amp = add_seismic_to_figure(p, seismic_cds, x, y, title='CGG22M01_NVG21PH1-EW_FINAL_KPSDM_T_FAR_STK')
        # # Test how to access the limits of the colorbar ColorBar
        # print(inspect.getmembers(p._property_values['right'][0].color_mapper))
        # print(p._property_values['right'][0].color_mapper.high)
        # p._property_values['right'][0].color_mapper.high = 50000.



        # Add points
        table, point_cds = add_i_g_point(p, x, y, ['far'])

        show(column(p, row(table, c_amp)))

    def test_avo_qc(self):
        near = AngleStack('near',
                          "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-NEARSTACK_MIG-FIN_16bit.zgy",
                          angle=10.)
        mid = AngleStack('mid',
                          "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-MIDSTACK_MIG-FIN_16bit.zgy",
                          angle=18.)
        far = AngleStack('far',
                          "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-FARSTACK_MIG_FIN_16bit.zgy",
                          angle=26.)
        ufar = AngleStack('ufar',
                          "R:\\3D\\UTM31\\Acquired Data\\CGG22M01-NVG21PH1\\CGG22M01-NVG21PH1-NSRE-FINAL-KPSDM-T-UFARSTACK_MIG-FIN_16bit.zgy",
                          angle=34.)

        voi = VolumeOfInterest(
            range(4640, 5243, 1),
            range(32253, 32255, 1),
            range(2616, 5108, 4) )

        xline = 32254

        return avo_qc([near, mid, far, ufar], voi, line_direction='xline', verbose=True)
        # return avo_qc([near, ufar], voi, xline=xline, verbose=True)

    def test_closest(self):
        x = np.random.normal(10,2, 20)
        x = np.array([int(_x*10) for _x in x])
        # print(np.sort(x))
        print(x)
        print(closest(90.45, x))

    def test_picks(self):
        _synts = StochasticSyntheticTraces(trace_length=500)
        synts = _synts.get_traces(simulate_avo=True)
        cds = ColumnDataSource({'value': [synts.T]})
        _seismic_color_map = seismic_color_map(min_val=np.min(synts), max_val=np.max(synts))
        (m, n) = cds.data['value'][0].shape
        print(synts.shape, m, n )

        p = figure()
        p.image('value', source=cds,
                color_mapper=_seismic_color_map, dh=synts.shape[1], dw=synts.shape[0], x=0, y=0)

        markers = ['asterisk', 'circle', 'cross', 'dash', 'dot', 'square']
        colors = ['black', 'white', 'orange', 'green', 'yellow', 'brown']
        for i, feature in enumerate(['nearest_min']):
        # for i, feature in enumerate(['nearest_min', 'nearest_min_above', 'nearest_min_below']):
        # for i, feature in enumerate(['extract', 'global_max', 'global_min', 'nearest_max', 'nearest_max_above', 'nearest_max_below']):
        # for i, feature in enumerate(['extract', 'global_max', 'global_min', 'nearest_max']):
            if feature in ['global_max', 'global_min']:
                indxs = None
            else:
                indxs = 250

            amps, inds = picks_from_seismic_cds(cds, feature, indxs)
            print(feature, [('{:.2f}'.format(_a), str(_i)) for _a, _i in zip(amps, inds)])

            p.scatter(x=np.arange(n)[::2], y=inds[::2], marker=markers[i], line_color=colors[i],
                      fill_color=colors[i], size=7)

        show(p)

    def test_avo_analyzer(self):
        section_type = 'avo section'
        synthetic = True
        p, seismic_cds, c_amp, refl_points, model = self.test_data_for_avo_analyzer(section_type, synthetic)

        analyze_avo = AvoAnalyzer(p, section_type, refl_points, model=model)
        points_cds = analyze_avo.cds
        points_table, avo_figure, avo_cds = analyze_avo.draw(points_cds, seismic_cds)


        show(column(p, points_table, avo_figure))

