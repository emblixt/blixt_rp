import matplotlib as mpl
from copy import deepcopy

import bokeh.plotting
import numpy as np
from openpyxl.styles.builtins import title
from pandas import DataFrame
from typing import Literal
from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS, LinearColorMapper)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, ColorBar, LogColorMapper
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor)
from bokeh.io import output_file
from bokeh.layouts import gridplot

import bruges

# from blixt_utils.misc.templates import necessary_keys

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]


