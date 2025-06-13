import matplotlib as mpl

import bokeh.plotting
import numpy as np
from openpyxl.styles.builtins import title
from pandas import DataFrame
from typing import Literal
# from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis,  Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, ColorBar, LogColorMapper
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor, Rect, HTMLTemplateFormatter)
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.transform import transform

# from blixt_utils.misc.templates import necessary_keys
from blixt_rp.core.core import Template
from blixt_rp.core.seismic import SeismicTraces, seismic_color_map

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]

test_data_length = 1000


class LogPlotter:
    """
    Class for plotting multiple well logs, from one well, in different "columns", all with a common y-axis (time
    or depth)
    """

    def __init__(self,
                 width: int = 300,
                 height: int = 600,
                 columns: list |None = None,
                 add_tools: list | None = None
                 ):
        """

        :param width:
        :param height:
        :param columns:
            list
            list of LogColumn objects
        """
        if columns is None:
            columns = []
        self._width = width
        self._height = height
        self._columns = columns
        self._add_tools = add_tools


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

    def figure(self) -> gridplot:
        if self._add_tools is not None:
            my_tools = tools + self._add_tools
        else:
            my_tools = tools
        children = []
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
        _h = Span(dimension="height", line_dash="dashed", line_width=0)
        for i, _column in enumerate(self.columns):
            _p = create_column_figure(_column, _w, _h, self.column_width, self.height,
                                      _y_range_flipped=i==0,
                                      _x_axis_visible=len(_column) > 0,
                                      _y_axis_visible=i==0,
                                      _tools=my_tools)
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
                raise IOError('{} is not a SeismicTraces object'.format(_st))
        self._seismic_traces = st

    def __len__(self):
        return max([len(self.lines), 1])  # There is only one seismic_traces object

class Line:
    """
    Simple class for defining a line object
    """
    def __init__(self,
                 x=None,
                 y=None,
                 style: Template | None = None,
                 source: ColumnDataSource | None = None
    ):
        """

        :param x:
        :param y:
        :param style:
            Template
            Template object which we use to create the line arguments that goes to the bokeh.figure.line() method.
            Most important arguments are:
            line_color='black', line_style='solid,  line_width=1, name='name'
        :param source:
            ColumnDataSource that can be updated interactively in bokeh.
            If given, the x and y must be strings that exists within this source. E.G.
            > source = ColumnDataSource(dict(alpha=<some data>, beta=<some other data>))
            > Line(x='alpha', y='beta', source=source)
        """
        if x is None:
            x = np. random.normal(0., 1., test_data_length)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, len(self._x))
        self._y = y
        if style is None:
            style = Template(**dict(line_color='black', line_style = 'solid', line_width = 2, name = 'TEST',
                                    min=1000., max=13000.))
        self.style = style
        self.source = source

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def line_args(self):
        # handle linestyle
        if self.style.line_style == '-':
            line_dash = 'solid'
        elif self.style.line_style == '--':
            line_dash = 'dashed'
        elif self.style.line_style == ':':
            line_dash = 'dotted'
        elif self.style.line_style == '-.':
            line_dash = 'dotdash'
        else:
            line_dash = 'solid'

        _d = {'line_color': 'blue' if self.style.line_color is None else self.style.line_color,
              'line_width': 1.0 if self.style.line_width is None else self.style.line_width,
              'line_dash': line_dash,
              'legend_label': self.style.name}
        return _d

    @property
    def min(self):
        if self.source is None:
            return np.nanmin(self.x)
        else:
            return np.nanmin(self.source.data[self.x])

    @property
    def max(self):
        if self.source is None:
            return np.nanmax(self._x)
        else:
            return np.nanmax(self.source.data[self.x])

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


def create_column_figure(_column: LogColumn,
                         _w: Span | None,
                         _h: Span | None,
                         _width: int,
                         _height: int,
                         _y_range_flipped: bool,
                         _x_axis_visible: bool,
                         _y_axis_visible: bool,
                         _tools: list | None) -> bokeh.plotting.figure:
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
                tools=_tools)  # , active_inspect=None)
    # style the plot
    _p.toolbar.logo = None
    _p.add_tools(CrosshairTool(overlay=[_w, _h]))
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
    extra_axes = []
    for j, _line in enumerate(_column.lines):
        _legend_label = _line.line_args['legend_label']
        if j == 0:
            if _line.source is None:
                _p.line(x=_line.x, y=_line.y,  **_line.line_args)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.source, **_line.line_args)
            _p.xaxis.axis_label = _legend_label
            _p.x_range = Range1d(*_line.x_range(from_style=True))
            # print('XXX', _p.x_range.start, _p.x_range.end)
        else:
            _p.extra_x_ranges[_legend_label] = Range1d(*_line.x_range(from_style=True))
            if _line.source is None:
                _p.line(x=_line.x, y=_line.y, **_line.line_args, x_range_name=_legend_label)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.source, **_line.line_args, x_range_name=_legend_label)
            if _column.scale == 'log':
                this_ax = LogAxis(axis_label=_legend_label, x_range_name=_legend_label,
                                     axis_label_text_font_size='10px',
                                     major_label_text_font_size = '10px',  # )  # ,
                                     axis_label_standoff=0)  # this only adds space between new axes and its label
            else:
                this_ax = LinearAxis(axis_label=_legend_label, x_range_name=_legend_label,
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
    trace_source = None
    _seismic_color_map = None
    if _column.seismic_traces is not None:
        _seismic = _column.seismic_traces
        if _seismic.source is None:
            trace_source = ColumnDataSource({'value': [_seismic.traces.T]})
            min_val = np.nanmin(_seismic.traces)
            max_val = np.nanmax(_seismic.traces)
        else:
            trace_source = _seismic.source
            min_val = np.nanmin(trace_source.data['value'])
            max_val = np.nanmax(trace_source.data['value'])
        _seismic_color_map = seismic_color_map(min_val=min_val, max_val=max_val)
        if trace_source is not None:
            # _p.image('value', source=trace_source, color_mapper=_seismic_color_map, dh=test_data_length, dw=128,
            #          x=0, y=0)
            _p.image('value', source=trace_source, color_mapper=_seismic_color_map,
                     dh=np.max(_seismic.y) - np.min(_seismic.y),
                     dw=np.max(_seismic.x) - np.min(_seismic.x),
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


def add_strat_table(_p: bokeh.plotting.figure,
                    stratigraphy: dict,
                    width: int | None = None,
                    column_index: int | None = None) -> bokeh.models.DataTable:
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
                stratigraphy[_key] =['15px'] * len(stratigraphy['top'])

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

    strati_source = ColumnDataSource(stratigraphy_frame)

    # mid_point = CustomJSTransform(func=mid_point_func)  # the bokeh transform method seems to only handle one input variable!

    def spans():
        _tmp = []
        for _i, _name in enumerate(strati_source.data['name']):
            _tmp.append(Span(location=strati_source.data['top'][_i], dimension='width',
                             line_width=strati_source.data['line_width'][_i],
                             line_dash=strati_source.data['line_style'][_i],
                             line_color=strati_source.data['color'][_i]))
        return _tmp

    def strati_units():
        if column_index is not None:
            for _i, _name in enumerate(strati_source.data['name']):
                glyph = Rect(x="level", y="mid", width=1, height="height", angle=0, fill_color="color",
                             fill_alpha='visible', line_width=0.)
                # glyph = Rect(x=0, y=transform('top', 'base'), width=10, height=100, angle=0, fill_color="color")
                _p.children[column_index][0].add_glyph(strati_source, glyph)
            _p.children[column_index][0].xgrid.grid_line_color = None

    strati_units()

    text_glyph = None
    _spans = spans()
    for _span in _spans:
        for i, _child in enumerate(_p.children):
            _child[0].add_layout(_span)
            if i == 0:
                text_glyph = Text(x=_child[0].x_range.start, y='top',
                                  text='name', text_font_size='font_size', text_alpha='visible')
                _child[0].add_glyph(strati_source, text_glyph)
        if column_index is not None:
            for i, _child in enumerate(_p.children):
                if i == column_index:
                    text_glyph = Text(x='level', y='mid',
                                      text='name', text_font_size='font_size', text_font_style='bold',
                                      text_alpha='visible', text_align='center', text_baseline='middle',
                                      angle=90., angle_units='deg')
                    _child[0].add_glyph(strati_source, text_glyph)

    span_callback = CustomJS(args=dict(source=strati_source, spans=_spans), code=code)

    strati_source.js_on_change('patching', span_callback)  # 'patching' is important!

    return DataTable(
        source=strati_source,
        columns=table_columns,
        editable=True,
        width=width,
        index_position=-1,
        index_header='row index',
        index_width=60)


