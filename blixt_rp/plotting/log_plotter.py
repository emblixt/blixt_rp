import unittest

import matplotlib as mpl

import bokeh.plotting
import numpy as np
from pandas import DataFrame, cut
import logging
import sys, os
from copy import deepcopy

from typing import Literal
# from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis,  Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper, CDSView, BooleanFilter)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, ColorBar, LogColorMapper
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor, Rect, HTMLTemplateFormatter)
from bokeh.layouts import gridplot
from bokeh.transform import transform
from bokeh.io import output_file

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))

# from blixt_utils.misc.templates import necessary_keys
from blixt_rp.core.core import Template, LogTable
from blixt_rp.core.seismic import SeismicTraces, seismic_color_map
from blixt_rp import Q_

logger = logging.getLogger(__name__)

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\plotting',
    'test_data'))

output_file(os.path.join(test_file_dir, 'plot.html'))

default_tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]

test_data_length = 1000

line_dashes = {
    '-': 'solid',
    '--': 'dashed',
    ':': 'dotted',
    '.-': 'dotdash',
    '-.': 'dashdot'}


class LogPlotter:
    """
    Class for plotting multiple well logs, from one well, in different "columns", all with a common y-axis (time
    or depth)
    """
    from blixt_rp.core.core import Cutoffs

    def __init__(self,
                 width: int = 300,
                 height: int = 600,
                 columns: list |None = None,
                 tools: list | None = None
                 ):
        """

        :param width:
        :param height:
        :param columns:
            list
            list of LogColumn objects
        :param tools:
        """
        from blixt_rp.core.core import ClassificationTable

        if columns is None:
            columns = []
        self._width = width
        self._height = height
        self._columns = columns
        # Used to store the original data before we start using cutoffs
        self.original_cds = None

        self.active_cutoffs = []

        if tools is None:
            tools = default_tools
        self._tools = tools


    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        self._width = w

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, h):
        self._height = h

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, l: list):
        self._columns = l

    def add_column(self, _column, keep_column_width=True):
        if keep_column_width:
            old_width = self.width
            old_column_width = self.column_width
            self.width = old_width + _column.rel_width * old_column_width
        self._columns.append(_column)

    def insert_column(self, _i, _column, keep_column_width=True):
        if keep_column_width:
            old_width = self.width
            old_column_width = self.column_width
            self.width = old_width + _column.rel_width * old_column_width
        self._columns.insert(_i, _column)

    @property
    def column_width(self):
        n_cols = sum([_column.rel_width for _column in self.columns])
        return int(self.width / n_cols)

    @property
    def logs(self) -> list:
        logs = []
        for _column in self.columns:
            for _line in _column.lines:
                if _line.x not in logs:
                    logs.append(_line.x)
            if _line.y not in logs:
                logs.append(_line.y)
        return logs

    @property
    def units(self) -> list:
        units = []
        for _column in self.columns:
            for _line in _column.lines:
                if _line.units not in units:
                    units.append(_line.units)
        # Also add units for the depth variables:
        units.append('m')
        units.append('ms')
        return units

    def add_settings(self, grid: gridplot, width=None):
        from blixt_rp.core.core import TemplatesTable

        code = """
            const data = source.data;
            const cols = Object.keys(data);
            const nrows = data[cols[0]].length;

            console.log('Plot has ${grid.children.length} columns', grid.children.length);
            console.log(`Iterating over ${nrows} rows and ${cols.length} columns:`);
            
            // Iterate over all columns in the gridplot
            for (let _c = 0; _c < grid.children.length; _c++) {
                // Iterate over all renderers
                for (const r of grid.children[_c][0].renderers) {
                    console.log('Plot column ', _c);
                    // If the renderer is a Line glyph
                    if (r.glyph && r.glyph.constructor.__name__ === 'Line') {
                        console.log('Found a line renderer:', r.name);
                        // Iterate over all lines in the settings table
                        for (let i = 0; i < nrows; i++) {
                            if (r.name === data['name'][i]) {
                                console.log(data['name'][i]);
                                r.glyph.line_color = data['line_color'][i];
                                r.glyph.line_width = data['line_width'][i];
                                r.glyph.line_dash = styles[data['line_style'][i]];
                            }
                        }
                    }
                }
            }

            // for (let i = 0; i < nrows; i++) {
            //     for (let j = 0; j < cols.length; j++) {
            //         const col = cols[j];
            //         const value = data[col][i];
            //         console.log(`Row ${i}, Column '${col}': ${value}`);
            //     }
            // }
        """

        templates = []
        prev_templates = []
        for _col in self.columns:
            for _line in _col.lines:
                if _line.name in prev_templates:
                    # skip templates already added:
                    continue
                templates.append(_line.style)
                prev_templates.append(_line.name)
        table = TemplatesTable(templates, width)
        cds = table.cds

        # js_on_change with 'patching' works
        source_callback = CustomJS(args=dict(source=cds, grid=grid, styles=line_dashes), code=code)
        cds.js_on_change('patching', source_callback)

        # returns a settings table
        return table.draw(cds)

    def add_cutoffs_table(
            self,
            common_cds: ColumnDataSource,
            cutoffs: Cutoffs | None = None,
            width: int | None = None):
        """

        :param common_cds:
            ColumnDataSource
            One common CDS for all line objects in the plot.
            Must have a field called 'mask'
            To create a cutoff, the mask can be dependent on the data in the other columns too.
            To make this work, we need to a CDS which is common for all Lines in the LogPlotter
        :param cutoffs:
            Cutoffs
            Object containing the rules used for masks and classification
        :param width:
            int
        :return:
        """
        from bokeh.models import Button, Div
        from blixt_rp.core.core import ClassificationTable

        # Save a copy of the original common_cds which we can use to restore the data after applying cutoffs
        self.original_cds = ColumnDataSource(dict(deepcopy(common_cds.data)))

        if cutoffs is not None:
            _cutoffs_table = ClassificationTable(rules=cutoffs.cutoffs, width=width)
        else:
            _cutoffs_table = None

        # Initialize the Cutoffs table and its controls
        table_cds = _cutoffs_table.cds
        table, add_row, delete_row, update, use = _cutoffs_table.draw(table_cds,
                                                                      parameters=self.logs,
                                                                      units=self.units)

        def apply_mask_function():
            _data = dict(common_cds.data)
            if _cutoffs_table is not None:
                _mask_ct = _cutoffs_table.create_mask(table_cds, common_cds, verbose=True)
                _active_cutoffs = _cutoffs_table.active_cutoffs(table_cds)
                print('XXX: Trying to apply mask using: ', _active_cutoffs)
            else:
                _mask_ct = np.ones(len(common_cds.data['mask']), dtype=bool)
                print('XXX: Trying to apply mask with all data unmasked')

            for _key in list(_data.keys()):
                if _key == 'mask':
                    continue
                _data[_key][~_mask_ct] = np.nan
            common_cds.data = _data

        def reset_mask_function():
            common_cds.data = dict(deepcopy(self.original_cds.data))

        # Add buttons that applies and resets the mask
        apply_mask = Button(label='Apply masks', button_type='success')
        apply_mask.on_click(apply_mask_function)
        reset_mask = Button(label='Reset masks', button_type='success')
        reset_mask.on_click(reset_mask_function)
        # This is a test to see if reset_mask can update the plot
        # reset_mask.js_on_click(CustomJS(args=args_dict, code=self.js_code()))
        # table_cds.js_on_change('patching', CustomJS(code="""console.log('Change in Common CDS');"""))

        return table, add_row, delete_row, update, apply_mask, reset_mask

    def draw(self, title: str | None = None) -> gridplot:
        children = []
        lines = []
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
        _h = Span(dimension="height", line_dash="dashed", line_width=0)
        for i, _column in enumerate(self.columns):
            _p = create_column_figure(_column, _w, _h, self.column_width, self.height,
                                      _y_range_flipped=i==0,
                                      _x_axis_visible=len(_column) > 0,
                                      _y_axis_visible=i==0,
                                      _tools=self._tools,
                                      _title=title)
            _p.name = _column.name
            children.append(_p)

        # Let the y-axis of all columns be controlled by the y-axis of the first column
        for i, child in enumerate(children):
            if i > 0:
                child.y_range = children[0].y_range

        return gridplot([[_child for _child in children]],
                        toolbar_location='right', merge_tools=True)


class LogColumn:
    """
    A class that can contain multiple well logs (lines) in the same "column"
    """

    def __init__(self,
                 name: str,
                 rel_width: float = 1.,
                 rel_header_height: float = 0.,
                 lines: list | None = None,
                 seismic_traces: SeismicTraces | None = None,
                 scale: str = 'linear'
                 ):
        """
        :param name:
            str
            Unique string that separates the different columns from each other
        :param rel_width:
            float
            Width relative to the width of one of N columns of equal width
            E.G. for a 400 pixel wide LogPlotter with 4 LogColumns with rel_width=1, each LogColumn will be 100 wide
        :param rel_header_height:
            float
            Height of header relative to the total height
        :param lines:
            list
            A list of Line objects
        :param seismic_traces:
            A SeismicTraces objects
        :param scale:
            str
            'linear' or 'log'
        """
        if lines is None:
            lines = []
        self._lines = lines
        self._seismic_traces = seismic_traces
        self._rel_width = rel_width
        self._rel_header_height = rel_header_height
        self._name = name
        if scale is None:
            scale = 'linear'
        self.scale = scale

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def rel_width(self):
        return self._rel_width

    @rel_width.setter
    def rel_width(self, rw):
        self._rel_width = rw

    @property
    def rel_header_height(self):
        return self._rel_header_height

    @rel_header_height.setter
    def rel_header_height(self, rh):
        self._rel_header_height = rh

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, l: list):
        for _line in l:
            if not isinstance(_line, Line):
                raise IOError('{} is not a Line object'.format(_line))
        self._lines = l

    def add_line(self, _line):
        if not isinstance(_line, Line):
            raise IOError('{} is not a Line object'.format(_line))
        self._lines.append(_line)

    @property
    def seismic_traces(self):
        return self._seismic_traces

    @seismic_traces.setter
    def seismic_traces(self, st: SeismicTraces):
        if not isinstance(st, SeismicTraces):
                raise IOError('{} is not a SeismicTraces object'.format(st))
        self._seismic_traces = st

    def __len__(self):
        if self.seismic_traces is not None:
            return 1
        else:
            return len(self.lines)


class Line:
    """
    Simple class for defining a line object
    """
    def __init__(self,
                 x: str,
                 y: str,
                 cds: ColumnDataSource,
                 style: Template | None = None
                 ):
        """

        :param x:
            string
            Name of X variable
        :param y:
            string
        :param cds:
            ColumnDataSource that can be updated interactively in bokeh.
            The x and y must be strings that exists within this cds. E.G.
            > cds = ColumnDataSource(dict(alpha=<some data>, beta=<some other data>))
            > Line(x='alpha', y='beta', cds=cds)
        :param style:
            Template
            Template object which we use to create the line arguments that goes to the bokeh.figure.line() method.
            Most important arguments are:
            line_color='black', line_style='solid,  line_width=1, name='name'
        """
        self._x = x
        self._y = y
        self.cds = cds
        if style is None:
            style = Template(**dict(line_color='black', line_style='solid', line_width=2, name='TEST',
                                    min=1000., max=13000.))
        self.style = style
        self._name = style.name

    def __len__(self):
        return len(self.cds.data[self.x])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def line_args(self):
        # handle linestyle
        # if self.style.line_style == '-':
        #     line_dash = 'solid'
        # elif self.style.line_style == '--':
        #     line_dash = 'dashed'
        # elif self.style.line_style == ':':
        #     line_dash = 'dotted'
        # elif self.style.line_style == '-.':
        #     line_dash = 'dotdash'
        if self.style.line_style in ['-', '--', ':', '.-', '-.']:
            line_dash = line_dashes[self.style.line_style]
        else:
            line_dash = 'solid'

        _d = {'line_color': 'blue' if self.style.line_color is None else self.style.line_color,
              'line_width': 1.0 if self.style.line_width is None else self.style.line_width,
              'line_dash': line_dash,
              'legend_label': self.style.name}
        return _d

    @property
    def min(self):
        if self.cds is None:
            return np.nanmin(self.x)
        else:
            return np.nanmin(self.cds.data[self.x])

    @property
    def max(self):
        if self.cds is None:
            return np.nanmax(self._x)
        else:
            return np.nanmax(self.cds.data[self.x])

    @property
    def units(self) -> str:
        units = ''
        if self.style is not None and self.style.units is not None:
            units = self.style.units
        return units


    def x_range(self, from_style=False):
        _min = None
        _max = None
        if from_style:
            if 'min' in list(self.style.keys()):
                _min = self.style.min
            if 'max' in list(self.style.keys()):
                _max = self.style.max

        if _min is None:
            _min = self.min
        if _max is None:
            _max = self.max
        return _min, _max

class WellPlotter(LogPlotter):
    """
    Class for plotting well logs from one well
    """
    from blixt_rp.core.well_new import Well
    from blixt_rp.core.core import Cutoffs
    def __init__(self,
                 well: Well,
                 column_content: list,
                 buffer: float | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 scales: list | None = None,
                 rel_widths: list | None = None
                 ):
        """
        Attempts to plot logs side by side, or together, in different "columns", utilizing the interactive plotting
        possibilities in bokeh
        :param well:
            well object, new format from well_new.py
        :param column_content:
            list
            List  of lists that contain the names of the logs to plot in each column
            E.G.
                [['rhob', 'neu'], ['vp', 'vs']]
        :param buffer:
            float
            distance in meters
            Log is plotted from top of working interval - buffer to base of working interval + buffer
            Default is 50 m
        :param width:
        :param height:
        :param scales:
            list of strings determining the x scale of each column
            ['log', 'linear'
        :param rel_widths:
            list of floats determining the relative width of each column
            [1., 2., 1.
        :return:
        """
        from blixt_utils.utils import print_info

        # This is necessary if we want to be able to use cutoffs to mask out data
        self.data_source = well.data_source()
        self.cds = self.data_source.cds

        # Check that the requested logs exists in the well
        for _col in column_content:
            # Iterate over a copy of the list by slicing!
            for _log in _col[:]:
                if _log not in well.get_log_names:
                    print_info('Log {} does not exist in {}'.format(_log, well.name), 'warning', logger)
                    # flat_list.remove(_log)
                    _col.remove(_log)
        if scales is None:
            scales = ['linear'] * len(column_content)
        if rel_widths is None:
            rel_widths = [1.] * len(column_content)
        if buffer is None:
            buffer = 50.
        if width is None:
            width = 800.
        if height is None:
            height = 1000.

        # Gather data that should be plotted in each of the columns
        plot_columns = []
        for _i, _c in enumerate(column_content):
            plot_columns.append(
                LogColumn(
                    'column_{}'.format(str(_i)),
                    rel_width=rel_widths[_i],
                    # lines=[well.get_log_curve(_l).get_line() for _l in _c],
                    lines=[Line(x=_l, y='depth', cds=self.cds, style=self.data_source.templates[_l]) for _l in _c],
                    scale=scales[_i])
            )
        super().__init__(width=width, height=height, columns=plot_columns, tools=None)

    def add_cutoffs_table(self, cutoffs: Cutoffs | None = None, width: int | None = None):
        return super().add_cutoffs_table(self.cds, cutoffs, width)

def create_column_figure(_column: LogColumn,
                         _w: Span | None,
                         _h: Span | None,
                         _width: int,
                         _height: int,
                         _y_range_flipped: bool,
                         _x_axis_visible: bool,
                         _y_axis_visible: bool,
                         _tools: list | None,
                         _title: str | None = None) -> bokeh.plotting.figure:
    """
    Creates a figure for one column

    :param _column:
        LogColumn
    :param _w:
        Span
        Width setting for the shared crosshair
    :param _h:
        Span
        Height setting for the shared crosshair
    :param _width:
        int
        Width of a default column in pixels
    :param _height:
        int
        height of figure in pixels
    :param _y_range_flipped:
        bool
        Set to True to flip the y-axis range
    :param _x_axis_visible:
        bool
        Set to True to show the x-axis
    :param _y_axis_visible:
        bool
        Set to True to show the y-axis
    :param _tools:
        list
        List of tools included in the toolbar, Except for CrossHairTool, which is added separately
    :return:
        Bokeh figure
    """
    if _w is None:
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
    if _h is None:
        _h = Span(dimension="height", line_dash="dashed", line_width=1)
    if _tools is None:
        _tools = "pan,wheel_zoom,box_zoom,reset"

    _p = figure(width=int(_column.rel_width * _width),
                height=int(_height * (1. - _column.rel_header_height)),
                x_axis_location = 'above',
                x_axis_type = _column.scale,
                tools=_tools,
                title=_title)  # , active_inspect=None)
    # style the plot
    _p.toolbar.logo = None
    _p.add_tools(CrosshairTool(overlay=(_w, _h)))
    hover = _p.select(dict(type=HoverTool))
    hover.tooltips = [("(x,y)", "($x, $y)"), ("Value", "@value")]

    if len(_column.lines) > 0:
        _p.x_range = Range1d(*_column.lines[0].x_range(from_style=True))

    _p.xaxis.visible = _x_axis_visible
    _p.xaxis.axis_label_text_font_size='10px'
    _p.xaxis.major_label_text_font_size='10px'
    _p.xaxis.axis_label_standoff=0

    # Add lines with log curves, if they exist in this column
    add_lines(_p, _column)

    # Add seismic, if it exists in this column
    add_seismic_traces(_p, _column)

    _p.y_range.flipped = _y_range_flipped
    _p.yaxis.visible = _y_axis_visible

    # add legends if they exists
    if len(_p.legend) > 0:
        _p.legend.click_policy = 'hide'
        _p.legend.location = 'top_right'
        _p.legend.label_text_font_size = '8pt'

    return _p


def add_lines(_p: bokeh.plotting.figure,
              _column: LogColumn,
              ):
    """
    Add lines to the column figure
    :param _p:
    :param _column:
    :return:
    """
    from blixt_utils.utils import print_info
    extra_axes = []
    for j, _line in enumerate(_column.lines):
        _legend_label = _line.line_args['legend_label']
        if j == 0:
            if _line.cds is None:
                print_info('Specifying a line with X and Y arrays is to be deprecated, Use a CDS',
                           'warning', logger)
                _p.line(x=_line.x, y=_line.y, name=_line.name,  **_line.line_args)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.cds, name=_line.name, **_line.line_args)
            _p.xaxis.axis_label = '{} [{}]'.format(_legend_label, _line.units)
            _p.x_range = Range1d(*_line.x_range(from_style=True))
            # print('XXX', _p.x_range.start, _p.x_range.end)
        else:
            _p.extra_x_ranges[_legend_label] = Range1d(*_line.x_range(from_style=True))
            if _line.cds is None:
                _p.line(x=_line.x, y=_line.y, name=_line.name, **_line.line_args, x_range_name=_legend_label)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.cds, name=_line.name, **_line.line_args, x_range_name=_legend_label)
            if _column.scale == 'log':
                this_ax = LogAxis(axis_label='{} [{}]'.format(_legend_label, _line.units), x_range_name=_legend_label,
                                     axis_label_text_font_size='10px',
                                     major_label_text_font_size = '10px',  # )  # ,
                                     axis_label_standoff=0)  # this only adds space between new axes and its label
            else:
                this_ax = LinearAxis(axis_label='{} [{}]'.format(_legend_label, _line.units), x_range_name=_legend_label,
                                     axis_label_text_font_size='10px',
                                     major_label_text_font_size = '10px',  # )  # ,
                                     axis_label_standoff=0)  # this only adds space between new axes and its label
            extra_axes.append(this_ax)
    for _ax in extra_axes:
        _p.add_layout(_ax, 'above')


def add_seismic_traces(_p: bokeh.plotting.figure,
                _column: LogColumn):
    """
    Similar to add_line(), but adds the seismic traces to the given column _column
    :param _p:
    :param _column:
    :return:
    """
    trace_cds = None
    _seismic_color_map = None
    if _column.seismic_traces is not None:
        _seismic = _column.seismic_traces
        if _seismic.cds is None:
            trace_cds = ColumnDataSource({'value': [_seismic.traces.T]})
            min_val = np.nanmin(_seismic.traces)
            max_val = np.nanmax(_seismic.traces)
        else:
            trace_cds = _seismic.cds
            min_val = np.nanmin(trace_cds.data['value'])
            max_val = np.nanmax(trace_cds.data['value'])
        _seismic_color_map = seismic_color_map(min_val=min_val, max_val=max_val)
        if trace_cds is not None:
            # _p.image('value', source=trace_cds, color_mapper=_seismic_color_map, dh=test_data_length, dw=128,
            #          x=0, y=0)
            _p.image('value', source=trace_cds, color_mapper=_seismic_color_map,
                     dh=np.max(_seismic.y) - np.min(_seismic.y),
                     dw=np.max(_seismic.x) - np.min(_seismic.x) + 1.,
                     x=np.min(_seismic.x),
                     y=np.min(_seismic.y)
                     )

            _p.xaxis.axis_label = _seismic.title
            # add color bar
            color_bar = ColorBar(color_mapper=_seismic_color_map,
                                 # title='Seismic',
                                 title_text_align = 'right',
                                 label_standoff=3, major_label_text_font_size='10px')
            _p.add_layout(color_bar, 'above')


def add_color_amp(_p):
    from bokeh.models import NumericInput
    orig_max_val = _p._property_values['above'][1].color_mapper.high
    color_amp_factor = NumericInput(value=100, low=10, high=500,
                                    title="Boost color scale (10 - 500%)")
    color_amp_factor.js_on_change('value', CustomJS(
        args=dict(
            c_amp=color_amp_factor,
            c_map=_p._property_values['above'][1].color_mapper,
            max_val=orig_max_val),
        code="""
            c_map.high = max_val * c_amp.value/100.
            c_map.low = -1. * max_val * c_amp.value/100.
            console.log('test:', c_amp.value);
            """))

    return color_amp_factor


def add_strat_table(_p: bokeh.plotting.figure,
                    stratigraphy: dict,
                    width: int | None = None,
                    column_index: int | None = None) -> bokeh.models.DataTable:
    # TODO Rewrite to reuse the WorkingIntervalsTable class
    """
    Adds the stratigraphic levels in the stratigraphy dictionary to the plot, and adds a table that
    control them.
    :param _p:
        Figure to add the stratigraphy too
    :param stratigraphy:
        Dictionary with at least these key : value pairs:
            'name' : list with the name of each level
            'top' : list of depths [m MD] to the top of each level
        Optional key : value pairs
            'base' : List of depths [m MD] to the base of each level
            'visible' : List of bool
            'level' : List of levels (e.g. 'Group', 'Formation', 'Member')
            'color' :
            'line_style' :
            'line_width' : List of integers
            'font_size' : List of strings, e.g. ['10px', ...]
    :param width:
        width in pixels
    :param column_index:
        int or None
        The figure _p has several columns, or children. If column_index is not None, it plots the graphical
        visualization of intervals (colored rectangles) in this column
    :return:
    """
    if width is None:
        width = 600
    code = """
        //Try to loop over all spans 
        const data = source.data;
        const mid_points = new Float64Array(data['top'].length);
        const heights = new Float64Array(data['top'].length);
        for (var i = 0; i < spans.length; i++) {
            const boolValue = data['visible'][i] === 1 ? true : false;
            spans[i].location = data['top'][i];
            spans[i].line_color = data['color'][i];
            spans[i].visible = boolValue;
            spans[i].line_dash = data['line_style'][i];
            spans[i].line_width = data['line_width'][i]; 
            mid_points[i] = 0.5 * (data['top'][i] + data['base'][i])
            heights[i] = data['base'][i] - data['top'][i]
        };
        source.data['mid'] = mid_points
        source.data['height'] = heights
        console.log('Data changed:', data['mid'][i]);
        source.change.emit();
        """
    mid_point_func = """
        // Find midpoint of each interval
        const top = top_data;
        const base = base_data;
        const mid_point = new Float64Array(top.length);
    for (var i = 0; i < top.length; i++) {
        mid_point[i] = 0.5 * (top[i] + base[i])
    };
    return mid_point
"""
    _necessary_keys = ['name', 'top']
    _optional_keys = ['base', 'visible', 'level', 'color', 'line_style', 'line_width', 'font_size']
    _current_keys = list(stratigraphy.keys())
    for _key in _necessary_keys:
        if _key not in _current_keys:
            raise IOError('Necessary key {} is lacking in stratigraphy dictionary'.format(_key))

    for _key in _optional_keys:
        # give default values to keys not listed among _current_keys
        if _key not in _current_keys:
            if _key == 'base':
                stratigraphy[_key] = stratigraphy['top']
            elif _key == 'visible':
                stratigraphy[_key] = [True] * len(stratigraphy['top'])
            elif _key == 'level':
                stratigraphy[_key] = [1 if np.mod(i, 2) == 0 else 2 for i in range(len(stratigraphy['top']))]
            elif _key == 'color':
                stratigraphy[_key] = ['blue'] * len(stratigraphy['top'])
            elif _key == 'line_style':
                stratigraphy[_key] = ['solid'] * len(stratigraphy['top'])
            elif _key == 'line_width':
                stratigraphy[_key] = [2] * len(stratigraphy['top'])
            elif _key == 'font_size':
                stratigraphy[_key] =['12px'] * len(stratigraphy['top'])

    # create extra source items
    stratigraphy['mid'] = list(0.5 * (np.array(stratigraphy['top']) + np.array(stratigraphy['base'])))
    stratigraphy['height'] = list(np.array(stratigraphy['base']) - np.array(stratigraphy['top']))

    stratigraphy_frame = DataFrame(stratigraphy)

    names = sorted(stratigraphy_frame['name'].unique())
    levels = sorted(stratigraphy_frame['level'].unique())
    line_widths = ['1', '2', '3', '4', '5']
    line_styles = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']

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
    formatter = HTMLTemplateFormatter(template=template)

    table_columns = [
        TableColumn(field='visible', title='Show',
                    editor=CheckboxEditor(),
                    width=30),
        TableColumn(field='name', title='Stratigraphy',
                    editor=SelectEditor(options=names),
                    formatter=StringFormatter(font_style='bold')),
        TableColumn(field='level', title='Level',
                    editor=IntEditor(step=1),
                    width=30),
        TableColumn(field='top', title='Top [m MD]',
                    editor=NumberEditor()),
        TableColumn(field='base', title='Base [m MD]',
                    editor=NumberEditor()),
        TableColumn(field='color', title='Color',
                    editor=StringEditor(),
                    formatter=formatter),
        TableColumn(field='line_style', title='Line style',
                    editor=SelectEditor(options=line_styles)),
        TableColumn(field='line_width', title='Line width',
                    # editor=IntEditor()),
                    editor = SelectEditor(options=line_widths),
                    width=30),
        TableColumn(field='font_size', title='Font size',
                    editor=StringEditor())

    ]

    strati_cds = ColumnDataSource(stratigraphy_frame)

    # mid_point = CustomJSTransform(func=mid_point_func)  # the bokeh transform method seems to only handle one input variable!

    def spans():
        _tmp = []
        for _i, _name in enumerate(strati_cds.data['name']):
            _tmp.append(Span(location=strati_cds.data['top'][_i], dimension='width',
                             line_width=strati_cds.data['line_width'][_i],
                             line_dash=strati_cds.data['line_style'][_i],
                             line_color=strati_cds.data['color'][_i]))
        return _tmp

    def strati_units():
        if column_index is not None:
            for _i, _name in enumerate(strati_cds.data['name']):
                glyph = Rect(x="level", y="mid", width=1, height="height", angle=0, fill_color="color",
                             fill_alpha='visible', line_width=0.)
                # glyph = Rect(x=0, y=transform('top', 'base'), width=10, height=100, angle=0, fill_color="color")
                _p.children[column_index][0].add_glyph(strati_cds, glyph)
            _p.children[column_index][0].xgrid.grid_line_color = None

    strati_units()

    # Add text

    # Tried to hide texts that overlap
    # _l = strati_cds.data['level']
    # _text_alpha = [float(_l[_i] != _l[_i + 1]) for _i in range(len(_l) - 1)] + [1.]
    text_glyph = Text(x=_p.children[0][0].x_range.start, y='top',
                      text='name', text_font_size='font_size', text_alpha='visible')
                      # text='name', text_font_size='font_size', text_alpha=_text_alpha)  # This didn't work, need to modify cds!
    _p.children[0][0].add_glyph(strati_cds, text_glyph)
    if column_index is not None:
        text_glyph = Text(x='level', y='mid',
                          text='name', text_font_size='font_size', text_font_style='bold',
                          text_alpha='visible', text_align='center', text_baseline='middle',
                          angle=90., angle_units='deg')
        _p.children[column_index][0].add_glyph(strati_cds, text_glyph)

    # text_glyph = None
    _spans = spans()
    for _span in _spans:
        for i, _child in enumerate(_p.children):
            _child[0].add_layout(_span)
    #         if i == 0:
    #             text_glyph = Text(x=_child[0].x_range.start, y='top',
    #                               text='name', text_font_size='font_size', text_alpha='visible')
    #             _child[0].add_glyph(strati_cds, text_glyph)
    #     if column_index is not None:
    #         for i, _child in enumerate(_p.children):
    #             if i == column_index:
    #                 text_glyph = Text(x='level', y='mid',
    #                                   text='name', text_font_size='font_size', text_font_style='bold',
    #                                   text_alpha='visible', text_align='center', text_baseline='middle',
    #                                   angle=90., angle_units='deg')
    #                 _child[0].add_glyph(strati_cds, text_glyph)

    span_callback = CustomJS(args=dict(source=strati_cds, spans=_spans), code=code)

    strati_cds.js_on_change('patching', span_callback)  # 'patching' is important!

    return DataTable(
        source=strati_cds,
        columns=table_columns,
        editable=True,
        width=width,
        index_position=-1,
        index_header='row index',
        index_width=60)


def select_column(grid: gridplot, column_name: str) -> figure:
    """
    Return the figure (column) named column_name of the gridplot grid
    :param grid:
        Bokeh gridplot
    :param column_name:
        str
        name of column
    :return:
        figure or None if not found
    """
    return grid.select_one({'name': column_name})


def select_line(grid: gridplot, line_name: str) -> bokeh.models.Line | None:
    """
    Return the figure (column) named column_name of the gridplot grid
    :param grid:
        Bokeh gridplot
    :param line_name:
        str
        name of line
    :return:
        Line or None if not found
    """
    from bokeh.models import Line
    _line = None
    for _i, _child in enumerate(grid.children):
        _line = _child[0].select_one({'name': line_name})
        if _line is not None:
            return _line
    return _line

def test_data():
    # Builds some line objects using a common CDS
    from blixt_rp.plotting import cross_plotter
    test = cross_plotter.TestCases()
    ds1, ds2, wis, cutoffs = test.test_data()
    cds = ds1.cds
    line1 = Line(x='var_one', y='md', cds=cds, style=ds1.templates['var_one'])
    line2 = Line(x='var_two', y='md', cds=cds, style=ds1.templates['var_two'])
    line3 = Line(x='var_three', y='md', cds=cds, style=ds1.templates['var_three'])
    return cds, line1, line2, line3, wis, cutoffs

def test_well_data():
    from blixt_rp.core.well_new import Well
    from blixt_rp.core.core import LogTable, CutoffRule, Cutoffs
    project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table_new.xlsx')
    las_file3 = os.path.join(test_file_dir, "Well F.las")
    log_table = LogTable({
        'P velocity': ['Vp_dry', 'Vp_Sg08'],
        'Porosity': ['PHIE'],
        'Volume': ['VSH']
    })
    rule1 = CutoffRule('vp_dry', '>', Q_(3000, 'm/s'))
    rule2 = CutoffRule('phie', '<', Q_(0.1, ''))
    rule3 = CutoffRule('vsh', '>', Q_(0.4, ''))
    well = Well()
    well.read_las(las_file3, log_table=log_table, template_file=project_table)
    return well, Cutoffs([rule1, rule2, rule3])


class TestCases(unittest.TestCase):
    def test_log_plotter1(self):
        lp = LogPlotter()
        print(lp.width)
        lp.width = 600
        print(lp.width)
        self.assertTrue(lp.width == 600)

    def test_line(self):
        cds, line1, line2, line3, wis, cutoffs = test_data()
        p1 = figure(title="Line example", x_axis_label="var_one", y_axis_label="md")
        p1.line(x=line1.x, y=line1.y, source=line1.cds, **line1.line_args)
        self.assertIsInstance(line1, Line)

        cds2 = ColumnDataSource(dict(deepcopy(cds.data)))
        cds2.data[line1.x][line1.cds.data[line1.x] > 10.1] = np.nan
        cds2.data[line1.y][line1.cds.data[line1.x] > 10.1] = np.nan
        p2 = figure(title="Masked line example", x_axis_label="x", y_axis_label="y")
        p2.line(x=line1.x, y=line1.y, source=cds2, **line1.line_args)

        show(row(p1, p2))

    def test_column_with_lines(self):
        _, line1, line2, line3, _, _ = test_data()
        c1 = LogColumn('c1', lines=[line1, line2, line3])

        p = create_column_figure(c1, None, None, 300, 600, True, True, True, None)
        show(p)
        # print(inspect.getmembers(p.children[0].y_range))
        self.assertTrue(True)

    def test_two_column_plot(self):
        _, line1, line2, line3, _, _ = test_data()
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2', lines=[line1, line2])
        c1 = LogColumn('c1', lines=[line1, line2, line3], rel_width=2)
        lp.columns = [c2, c1]
        show(lp.draw())
        self.assertTrue(True)

    def test_cutoffs(self, unit_test=True):
        common_cds, line1, line2, line3, _, cutoffs = test_data()
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2', lines=[line2])
        c1 = LogColumn('c1', lines=[line2, line3], rel_width=2)
        lp.columns = [c2, c1]
        print(common_cds.data.keys())
        grid = lp.draw()
        table, add_row, delete_row, update, apply_mask, reset_mask = lp.add_cutoffs_table(
            common_cds=common_cds, cutoffs=cutoffs)
        if unit_test:
            show(
                row(
                    grid, column(
                        table,
                        row(add_row, delete_row, update, apply_mask, reset_mask)
                    )
                )
            )
        else:
            return grid, table, add_row, delete_row, update, apply_mask, reset_mask

    def test_add_settings(self):
        _, line1, line2, line3, _, _ = test_data()
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2', lines=[line1, line2])
        c1 = LogColumn('c1', lines=[line1, line2, line3], rel_width=2)
        lp.columns = [c2, c1]
        grid = lp.draw()
        settings = lp.add_settings(grid)
        show(row(grid, settings))

    def test_two_column_plot_with_seismic(self):
        _, line1, line2, line3, seismic1 = test_data()
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2', lines=[line1, line2])
        c1 = LogColumn('c1', seismic_traces=seismic1, rel_width=2)
        lp.columns = [c2, c1]
        show(lp.draw())
        self.assertTrue(True)

    def test_link_table(self):
        _, line1, line2, line3, _ = test_data()
        lp = LogPlotter(width=800, height=1000)
        c3 = LogColumn('c3', rel_width=0.5)
        c2 = LogColumn('c2', lines=[line1, line2])
        c1 = LogColumn('c1', lines=[line2, line3, line1], rel_width=2)
        lp.columns = [c2, c3, c1]
        p = lp.draw()

        print(p.children)

        data_table = add_strat_table(
            p,
            {'name': ['Viking GP', 'Draupne FM', 'Another FM'],
             'top': [1000., 1400., 1630.], 'base': [2600., 1630., 1900.], 'color': ['red', 'blue', 'green'],
             'level': [0, 1, 1]},
            width=800,
            column_index=1
        )
        show(column(p, data_table))
        self.assertTrue(True)

    def test_well_plotter(self):
        well, cutoffs = test_well_data()
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        well_plot = WellPlotter(well, column_content)
        show(well_plot.draw())

    def test_well_plotter_with_cutoffs(self, unit_test=True):
        well, cutoffs = test_well_data()
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        well_plot = WellPlotter(well, column_content)
        table, add_row, delete_row, update, apply_mask, reset_mask = well_plot.add_cutoffs_table(cutoffs)
        grid = well_plot.draw()
        if unit_test:
            show(
                row(
                    grid, column(
                        table,
                        row(add_row, delete_row, update, apply_mask, reset_mask)
                    )
                )
            )
        else:
            return grid, table, add_row, delete_row, update, apply_mask, reset_mask

    def test_well_plotter_with_settings(self):
        well, cutoffs = test_well_data()
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        well_plot = WellPlotter(well, column_content)
        grid = well_plot.draw()
        settings_table = well_plot.add_settings(grid)
        show(row(grid, settings_table))



