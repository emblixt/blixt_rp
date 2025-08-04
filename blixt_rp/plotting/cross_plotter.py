# ----------------------------------------------------------------------------------------------------------
# Filename: cross_plotter.py
#  Purpose: Cross plot data using the bokeh interactive plotting library
#           Similar to crossplot.py, but tried to modernize it
#
# ---------------------------------------------------------------------------------------------------
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
from copy import deepcopy

# from html5lib.constants import mathmlTextIntegrationPointElements
from scipy.stats import wilcoxon

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.utils import print_info

logger = logging.getLogger(__name__)

default_tools = [
    PanTool(),
    WheelZoomTool(),
    BoxZoomTool(),
    HoverTool(),
    CrosshairTool(),
    ResetTool(),
    SaveTool()
]

cnames = [tmp['color'] for tmp, j in zip(plt.rcParams['axes.prop_cycle'], range(20))]
markers = ['circle', 'diamond', 'hex', 'inverted_triangle', 'plus', 'square', 'star', 'triangle']


def add_x_y_to_source(
               source_dict: dict,
               x: None | str = None,
               y: None | str = None,
               size: None | float | str = None,
               color: None | str = None,
               marker: None | str = None,
               legend_group: None | str = None,
               mask: None | np.ndarray = None
               ) -> dict:
    """
    The idea is to let the input decide which variable that should be used on the x-axis, y-axis, size, color,
    marker and legend.
    This function adds the extra keys for 'x', 'y', ... to the original source dictionary

    E.G.
        if x = 'vp'
        WHEN
        source_dict = dict(vp=..., vs=..., rho=...)
        THEN
        x_y_source = dict(vp=..., vs=..., rho=..., x=source_dict['vp'], y=..., etc...)

    :param source_dict:
    :param x:
    :param y:
    :param size:
    :param color:
    :param marker:
    :param legend_group:
    :param mask:

    :return:
        dict
        transformed_source
        The dictionary has at least the following keys:
            'x', 'y', 'size', 'color', 'marker' and 'legend_group'
    """
    _vars = list(source_dict.keys())
    _n = len(source_dict[_vars[0]])
    _dict = {}

    # for _key in ['x', 'y', 'size', 'color', 'marker', 'legend_group']:
    #     if _key in _vars:
    #         warn_txt = 'The key {} is among the original keys of the input source and will be overwritten'.format(_key)
    #         print_info(warn_txt, 'warning', logger)


    if x is not None and x in _vars:
        _dict['x'] = source_dict[x]
    elif x is not None:
        _dict['x'] = x
    else:
        _dict['x'] = source_dict[_vars[0]]

    if y is not None and y in _vars:
        _dict['y'] = source_dict[y]
    elif y is not None:
        _dict['y'] = y
    else:
        _dict['y'] = source_dict[_vars[1]]

    if size is not None and size in _vars:
        _dict['size'] = source_dict[size]
    elif size is not None:
        _dict['size'] = [size] * _n
    else:
        _dict['size'] = source_dict['size']

    if color is not None and color in _vars:
        _dict['color'] = source_dict[color]
    elif color is not None:
        _dict['color'] = [color] * _n
    else:
        _dict['color'] = source_dict['color']

    if marker is not None and marker in _vars:
        _dict['marker'] = source_dict[marker]
    elif marker is not None:
        _dict['marker'] = [marker] * _n
    else:
        _dict['marker'] = source_dict['marker']

    if legend_group is not None and legend_group in _vars:
        _dict['legend_group'] = source_dict[legend_group]
    elif legend_group is not None:
        _dict['legend_group'] = [legend_group] * _n
    else:
        _dict['legend_group'] = source_dict['legend_group']

    if mask is not None:
        _dict['mask'] = mask
    else:
        _dict['mask'] = source_dict['mask']

    _dict['source_name'] = source_dict['source_name']

    source_dict.update(_dict)
    return source_dict


class DataSource:
    """
    Class containing a dictionary of data, E.G. for one well, which can easily generate a ColumnDataSource
    But can hold some extra attributes and methods that the ColumnDataSource itself cant have
    """
    def __init__(self,
                 name: None | str = None,
                 data: None | dict = None,
                 templates: None | dict = None,
                 mask: None | np.ndarray = None
                 ):
        """
        Container for data related to one well
        :param name:
            str
            Name of  well
        :param data:
            dict
            Dictionary of data, with variable names as keys.
            Remember that a ColumnDataSource requires that all variable has the same length
        :param templates:
            dict
            Dictionary of Template objects for each variable, and preferably also one Template for this
            DataSource itself.
        :param mask:
            np.ndarray of boolean values
            Must have same length as the data (values) in the data dictionary
            True means data is kept
            False means the data is masked out
        """
        from blixt_rp.core.core import Template
        self._name = name
        self._data = data
        if templates is None:
            templates = {}
        if data is not None:
            for _key in list(data.keys()):
                if _key not in list(templates.keys()):
                    templates[_key] = None
        self._templates = templates
        if mask is None:
            mask = np.array(np.ones(len(data[list(data.keys())[0]])), dtype=bool)
        self.mask = mask

    def keys(self):
        return self._data.keys()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if not isinstance(new_data, dict):
            raise IOError('Data must be input as a dictionary, not {}'.format(type(new_data)))
        self._data = new_data

    @property
    def templates(self):
        return self._templates

    @templates.setter
    def templates(self, new_templates):
        for _key in list(self.keys()):
            if _key not in list(new_templates.keys()):
                new_templates[_key] = None
        self._templates = new_templates

    @property
    def source(self) -> ColumnDataSource:
        return ColumnDataSource(data=self.data, name=self.name)

    @source.setter
    def source(self, new_source):
        if isinstance(new_source, ColumnDataSource):
            self.data = dict(new_source.data)
        elif isinstance(new_source, dict):
            self.data = new_source
        else:
            raise IOError('New source must be either ColumnDataSource or Dict, not {}'.format(
                type(new_source)
            ))

    def min_and_max(self):
        return {_key: [np.nanmin(self.data[_key]), np.nanmax(self.data[_key])] for _key in list(self.keys())}

    # def transform_source_data(self,
    #                      x: None | str = None,
    #                      y: None | str = None,
    #                      size: None | str = None,
    #                      color: None | str = None,
    #                      marker: None | str = None,
    #                      legend_group: None | str = None,
    #                      ) -> dict:
    #     """
    #     The idea is to let the input decide which parameter that should be used on the x-axis, y-axis, size, color,
    #     marker and legend
    #     E.G.
    #         if x = 'vp'
    #         WHEN
    #         self.data = dict(vp=..., vs=..., rho=...)
    #         THEN
    #         transformed_source_data = dict(x=self.data['vp'], y=..., etc...)

    #     :param x:
    #     :param y:
    #     :param size:
    #     :param color:
    #     :param marker:
    #     :param legend_group:

    #     :return:
    #         dict
    #         transformed_source
    #         The dictionary has at least the following keys:
    #             'x', 'y', 'size', 'color', 'marker' and 'legend_group'
    #     """

    #     _n = len(self.data[list(self.keys())[0]])
    #     _dict = {}

    #     if x is not None and x in list(self.keys()):
    #         _dict['x'] = self.data[x]
    #     elif x is not None:
    #         _dict['x'] = x
    #     else:
    #         pass

    #     if y is not None and y in list(self.keys()):
    #         _dict['y'] = self.data[y]
    #     elif y is not None:
    #         _dict['y'] = y
    #     else:
    #         pass

    #     if size is not None and size in list(self.keys()):
    #         _dict['size'] = self.data[size]
    #     elif size is not None:
    #         _dict['size'] = [size] * _n
    #     else:
    #         pass

    #     if color is not None and color in list(self.keys()):
    #         _dict['color'] = self.data[color]
    #     elif color is not None:
    #         _dict['color'] = [color] * _n
    #     else:
    #         pass

    #     if marker is not None and marker in list(self.keys()):
    #         _dict['marker'] = self.data[marker]
    #     elif marker is not None:
    #         _dict['marker'] = [marker] * _n
    #     else:
    #         pass

    #     if legend_group is not None and legend_group in list(self.keys()):
    #         _dict['legend_group'] = self.data[legend_group]
    #     elif legend_group is not None:
    #         _dict['legend_group'] = [legend_group] * _n
    #     else:
    #         pass

    #     _dict['mask'] = self.mask

    #     return _dict

# class ParamSelector:
#
#     def __init__(self,
#                  selector: str,
#                  options: list,
#                  ):
#         """
#         :param selector:
#             str
#             either 'x', 'y', 'size', 'color', 'marker' or 'legend_group'
#         :param options:
#         """
#         self.selector = selector
#         self.options = options
#         self.code = """
#         """
#
#     def drop_menu(self):
#         from bokeh.models import Select
#         return Select(title=self.selector, value=self.options[0], options=self.options)


class WorkingIntervalsTable:
    def __init__(self,
                 intervals,
                 width: None | int = None):
        """

        :param intervals:
            blixt_rp.core.core.Intervals object

        :param width:
        """
        if width is None:
            width = 600
        self.width = width
        self._intervals = intervals

    @property
    def intervals(self):
        return self._intervals

    @property
    def source(self) -> ColumnDataSource:
        """
        Returns a ColumnDataSource with the following columns:
            'use', 'name', 'level', 'wells', 'color'
        :return:
            ColumnDataSource
        """
        _well_names = self.intervals.well_names()
        _interval_names = self.intervals.interval_names()

        _dict = self.intervals.get_strat_units()
        _ = _dict.pop('desc')
        _ = _dict.pop('source')
        _dict['use'] = [True] * len(_dict['name'])

        # add a column with the well names that contains the interval of each row
        _wells = []
        for _name in _dict['name']:
            _wl = []
            for _wn in _well_names:
                if self.intervals.get_interval(_name, _wn) is not None:
                    _wl.append(_wn)
            _wells.append(_wl)

        _dict['wells'] = [', '.join(_w) for _w in _wells]

        return ColumnDataSource(_dict)

    def __dict__(self):
        return self.intervals.get_intervals_dict()

    def active_intervals_dict(self, source):
        """
        Returns a dictionary suitable for a ColumnDataSource which contains all intervals for all wells
        with their top (m MD) and base and "active" status (status determined by the 'source' input)

        :param source:
            ColumnDataSource
            The self.source of WorkingIntervalsTable
            It is used to set which intervals are active or not

        :return:
        {'interval': [...], 'well': [...], 'top': [...], 'base': [...], 'active': [...]}
        """
        _dict = dict(
            interval=[],
            well=[],
            top=[],
            base=[],
            active=[]
        )
        for _int_name in self.intervals.interval_names():
            for _wname in self.intervals.well_names():
                _int = self.intervals.get_interval(_int_name, _wname)
                if _int is not None:
                    _dict['interval'].append(_int_name)
                    _dict['well'].append(_wname)
                    _dict['top'].append(_int.top.magnitude)
                    _dict['base'].append(_int.base.magnitude)
                    # iterate over all working intervals in 'source' to see which are active or not
                    _active = False
                    for _i, _i_n in enumerate(source.data['name']):
                        if _i_n == _int_name and _wname in source.data['wells'][_i]:
                            _active = True
                    _dict['active'].append(_active)
        return _dict

    def table_columns(self):
        """
        Creates the columns that the table should use
        :return:
        """
        from bokeh.models import (SelectEditor, StringEditor, StringFormatter, IntEditor, TableColumn, CheckboxEditor,
                                  HTMLTemplateFormatter)

        # Try to color the cells of the 'color' column by their value
        colored_cell_template = """
                <div style="background:<%= 
                    (function color_from_val(){
                        return(color)
                        }()) %>; 
                    color: white"> 
                <%= value %>
                </div>
            """
        formatter = HTMLTemplateFormatter(template=colored_cell_template)
        table_columns = [
            TableColumn(field='use', title='Show',
                        editor=CheckboxEditor(),
                        width=30),
            TableColumn(field='name', title='Stratigraphy',
                        formatter=StringFormatter(font_style='bold')),
            TableColumn(field='level', title='Level',
                        editor=IntEditor(step=1),
                        width=30),
            TableColumn(field='wells', title='Wells'),
            TableColumn(field='color', title='Color',
                        editor=StringEditor(),
                        formatter=formatter)
        ]
        return table_columns


    def draw(self, source):
        from bokeh.models import DataTable

        active_intervals = ColumnDataSource(self.active_intervals_dict(source))

        active_callback = CustomJS(
            args=dict(act_int=active_intervals),
            code="""
                const data_active = act_int.data;
                console.log('active_intervals has been updated');
                for (let j = 0; j < data_active['interval'].length; j++) {
                    console.log('   - Interval: ' + data_active['interval'][j] + ' is active? ' + data_active['active'][j]);
                }
            """
        )

        source_callback = CustomJS(
            args=dict(source=source, act_int=active_intervals, cb=active_callback),
            code="""
                const data_table = source.data;
                var data_active = act_int.data;
                for (let i = 0; i < data_table['name'].length; i++) {
                    for (let j = 0; j < data_active['interval'].length; j++) {
                        if (data_table['name'][i] === data_active['interval'][j]) {
                            data_active['active'][j] = data_table['use'][i]
                            console.log('  Interval: ' + data_table['name'][i] + ' in well: ' + data_active['well'][j] + ' is active? ' + data_active['active'][j]);
                        }
                    }
                }
                act_int.data = data_active;
                cb.execute();
            """
        )

        source.js_on_change('patching', source_callback)  # 'patching' is necessary. Don't know what it means

        return DataTable(
            source=source,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            index_position=-1,
            index_header='index'
        )


class ClassificationTable:
    """
    Table with rules that classify the data.
    The rules can also be used to mask out data (cutoffs)

    """
    def __init__(self,
                 rules: None | list = None,
                 width: None | int = None):
        """

        :param rules:
            list
            List of blixt_rp.core.core.CutoffRules objects
        :param width:
            int
        """
        if width is None:
            width = 600
        self.width = width
        if rules is None:
            rules = []
        self._rules = rules
        self.use_status = [False for _i in range(len(self._rules))]
        self.cutoffs = []
        self.keys = ['use', 'name', 'log', 'units', 'operator', 'limits']

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, new_rules):
        self._rules = new_rules

    @property
    def classes(self):
        _classes = {}
        for _rule in self.rules:
            if _rule.name not in list(_classes.keys()):
                _classes[_rule.name] = [_rule]
            else:
                _classes[_rule.name].append(_rule)
        return _classes

    @property
    def source(self) -> ColumnDataSource:
        """
        Returns a ColumnDataSource with the following columns:
            'use', 'name', 'log', 'units', 'operator', 'limits'
        When 'use' is set to True, and setting the 'use_cutoffs' check box to active, the rule is used to cut out data (mask)
        :return:
            ColumnDataSource
        """
        # _dict = dict(use=[], name=[], log=[], units=[], operator=[], limits=[])
        _dict = {_x:[] for _x in self.keys}
        # for _rule in self.rules.cutoffs:
        for _rule in self.rules:
            _u = None
            if isinstance(_rule.limit, list):
                _r = ', '.join([str(_x.magnitude) for _x in _rule.limit])
                _u = str(format(_rule.limit[0].units, '~'))
            else:
                _r = str(_rule.limit.magnitude)
                _u = str(format(_rule.limit.units, '~'))
            _dict['use'].append(False)
            _dict['name'].append('-')
            _dict['log'].append(_rule.param)
            _dict['units'].append(_u)
            _dict['operator'].append(_rule.operator)
            _dict['limits'].append(_r)

        return ColumnDataSource(_dict)

    def table_columns(self, parameters: list, units: list):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        _operators = ['<', '<=', '>', '>=', '><', '==', '!=']
        table_columns = [
            TableColumn(field='use', title='Use as cutoff', editor=CheckboxEditor()),
            TableColumn(field='name', title='Class', editor=StringEditor()),
            TableColumn(field='log', title='Parameter', editor=SelectEditor(options=parameters)),
            TableColumn(field='units', title='Units', editor=SelectEditor(options=units)),
            TableColumn(field='operator', title='Operator', editor=SelectEditor(options=_operators)),
            TableColumn(field='limits', title='Limits (comma separated)', editor=StringEditor())
        ]
        return table_columns

    def draw(self,
             source: ColumnDataSource,
             parameters: list,
             units: list,
             classification_data_source: ColumnDataSource | None = None,
             color_menu: Select | None = None):
        """

        :param source:
        :param parameters:
        :param units:
        :param classification_data_source:
            data source that the classification is applied on
        :param color_menu:
            Select
            Dropdown menu that decides how to color the data.
            Only the option "Classification" will be listened to here
        :return:
        """
        from bokeh.models import DataTable, Button, CheckboxGroup
        from blixt_rp.core.core import CutoffRule
        from pint import Quantity as Q_
        import blixt_utils.misc.masks as masks

        orig_data = None
        # take a backup of the original data
        if classification_data_source is not None:
            orig_data = dict(classification_data_source.data)

        def update_table_callback():
            # How do I use input variables to a python call back?
            #  - It depends on which widget you use, see:
            #      docs.bokeh.org/en/latest/docs/user_guide/interaction/python_callbacks.html#ug-interaction-python-callbacks
            cutoffs = []
            rules = []
            use_status = []
            for i, use in enumerate(source.data['use']):
                if ',' in source.data['limits'][i]:
                    this_limit = [Q_(float(_s), source.data['units'][i]) for _s in source.data['limits'][i].split(',')]
                else:
                    this_limit = Q_(float(source.data['limits'][i]), source.data['units'][i])
                use_status.append(use)
                _rule = CutoffRule(
                    param=source.data['log'][i],
                    operator=source.data['operator'][i],
                    limit=this_limit,
                    name=source.data['name'][i]
                )
                rules.append(_rule)
                if use:
                    cutoffs.append(_rule)
            self.rules = rules
            self.use_status = use_status
            self.cutoffs = cutoffs

        def use_cutoffs_callback(attr, old, new):
            if len(new) == 1 and classification_data_source is not None:
                print('Use cutoffs, ', old, new)
                update_table_callback()
                if len(self.cutoffs) > 0:  # There are active cutoffs
                    new_data = dict(classification_data_source.data)
                    _mask = None
                    for _i, _rule in enumerate(self.cutoffs):
                        _this_mask = masks.create_mask(
                            classification_data_source.data[_rule.param],
                            _rule.operator,
                            _rule.limit
                        )
                        if _i == 0:  # First cutoff rule
                            _mask = _this_mask
                        else:  # Append new cutoff rule using AND
                            _mask = masks.combine_masks([_mask, _this_mask])
                    # apply mask
                    for _key in list(new_data.keys()):
                        new_data[_key] = new_data[_key][_mask]
                    classification_data_source.data = new_data
            else:
                print('Do not use cutoffs, ', old, new)
                classification_data_source.data = orig_data

        def add_row_function():
            new_data = dict(source.data)
            new_data['use'].append(False)
            new_data['name'].append('Class A')
            new_data['log'].append(parameters[0])
            new_data['units'].append(units[0])
            new_data['operator'].append('>')
            new_data['limits'].append('1.')
            source.data = new_data

        def delete_row_function():
            selected_index = source.selected.indices
            # new_data = dict(use=[], name=[], log=[], units=[], operator=[], limits=[])
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(source.data['use'])):
                if _i in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(source.data[_x][_i])
            source.selected.indices = []
            source.data = new_data

        def use_color_classification(attr, old, new):
            print(attr, old, new)
            if new == 'Classification':
                update_table_callback()
                if len(self.rules) > 0:
                    print(list(self.classes.keys()))
                    new_data = dict(classification_data_source.data)
                    # Add 'non-classified' to all data points first
                    new_data['legend_group'][:] = 'No class'
                    new_data['color'][:] = 'gray'
                    new_data['marker'][:] = 'dot'
                    # Add specific style to classified data
                    for _i, _class in enumerate(self.classes.keys()):  # iterate over all classes
                        _mask = None
                        for _j, _rule in enumerate(self.classes[_class]):  # iterate over all rules within this class
                            _this_mask = masks.create_mask(
                                classification_data_source.data[_rule.param],
                                _rule.operator,
                                _rule.limit
                            )
                            if _j == 0:
                                _mask = _this_mask
                            else:  # Append rule using AND
                               _mask = masks.combine_masks([_mask, _this_mask])
                        new_data['legend_group'][_mask] = _class
                        new_data['color'][_mask] = cnames[_i]
                        new_data['marker'][_mask] = markers[_i]
                    print('Classification ON. {} data points with classes {} '.format(
                         len(new_data['legend_group']), list(set(new_data['legend_group']))))
                    classification_data_source.data = new_data
            else:
                # TODO
                # This doesnt work properly. The classification doesn't reset
                print('Classification OFF. {} data points with classes {} '.format(
                    len(orig_data['legend_group']), list(set(list(orig_data['legend_group'])))))
                classification_data_source.data = orig_data

        dt = DataTable(
            source=source,
            columns=self.table_columns(parameters, units),
            editable=True,
            width=self.width,
            index_position=-1,
            index_header='index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_callback)

        use_cutoffs = CheckboxGroup(labels=['Use cutoffs'], active=[])
        use_cutoffs.on_change('active', use_cutoffs_callback)

        if (color_menu is not None) and (classification_data_source is not None):
            color_menu.on_change('value', use_color_classification)

        return dt, add_row, delete_row, update_table, use_cutoffs


class CrossPlotter:
    """
    Class for holding the cross plot

        # Intended usage
    > data_sources = ...  # See DataSource for explanation
    > xp = CrossPlotter(data_sources)
    > source = xp.source()
        # Now you can add any additional functionality to control the source
    > xplot, x_menu, y_menu, size_menu, ... = xp.draw(source)
        # Now you can show and draw the xplot and the different menus

    """
    def __init__(self,
                 data_sources: None | dict = None,
                 width: None | int = None,
                 height: None | int = None,
                 tools: None | list = None):
        """

        :param data_sources:
            dictionary of DataSource's
            All variables within one DataSource have the same length
            The same variable name can be reused across DataSource's
        :param width:
            int
        :param height:
            int
        :param tools:
        """

        if width is None:
            width = 600
        if height is None:
            height = 600
        self.width = width
        self.height = height
        self._data_sources = data_sources
        # if data_sources is not None:
        #     self._sources = {_key: _val.source for _key, _val in data_sources.items()}
        # else:
        #     self._sources = None
        if tools is None:
            tools = default_tools
        self._tools = tools

        self.fixed_sizes = ['1', '5', '10', '20']
        self.fixed_colors = ['Data set', 'Classification']

    def __len__(self):
        _len = 0
        for _key, _item in self._data_sources.items():
            _var = self.common_variables[0]
            _len += len(_item.source.data[_var])
        return _len

    @property
    def all_variables(self):
        _all = []
        for _source in self._data_sources.values():
            _all += list(_source.keys())
        return list(set(_all))

    @property
    def templates(self):
        _all = {}
        for _source in self._data_sources.values():
            for _key, _val in _source.templates.items():
                _all[_key] = _val
        return _all

    @property
    def common_variables(self):
        return list(set.intersection(
            *[set(list(_s.keys())) for _s in self._data_sources.values()]
        ))

    @property
    def source(self,
               x: None | str = None,
               y: None | str = None) -> ColumnDataSource:
        """
        Returns a merged CDS with all the common variables from all data_sources, plus the extra variables
        created by add_x_y_to_source
        :return:
        """
        if x is None:
            x = self.common_variables[0]
        if y is None:
            y = self.common_variables[1]
        _dict = {}
        _i = 0
        for _key, _item in self._data_sources.items():
            _len = None
            for _var in self.common_variables:
                _len = len(_item.source.data[_var])
                if _i == 0:
                    _dict[_var] = _item.source.data[_var]
                else:
                    _dict[_var] = np.append(_dict[_var], _item.source.data[_var])
            if _i == 0:
                _dict['source_name'] = np.array([_key] * _len)
                _dict['legend_group'] = np.array([_key] * _len)
                _dict['color'] = np.array([cnames[_i]] * _len)
                _dict['marker'] = np.array([markers[_i]] * _len)
                _dict['size'] = np.array([5.] * _len)
                _dict['mask'] = np.array([True] * _len)
            else:
                _dict['source_name'] = np.append(_dict['source_name'], np.array([_key] * _len))
                _dict['legend_group'] = np.append(_dict['legend_group'], np.array([_key] * _len))
                _dict['color'] = np.append(_dict['color'], np.array([cnames[_i]] * _len))
                _dict['marker'] = np.append(_dict['marker'], np.array([markers[_i]] * _len))
                _dict['size'] = np.append(_dict['size'], np.array([5.] * _len))
                _dict['mask'] = np.append(_dict['mask'], np.array([True] * _len))
            _i += 1

        _dict = add_x_y_to_source(_dict, x, y)
        return ColumnDataSource(_dict)

    # def x_y_source(self,
    #                source_dict: dict,
    #                x: None | str = None,
    #                y: None | str = None,
    #                size: None | float | str = None,
    #                color: None | str = None,
    #                marker: None | str = None,
    #                legend_group: None | str = None,
    #                mask: None | np.ndarray = None
    #                ) -> dict:
    #     """
    #     The idea is to let the input decide which parameter that should be used on the x-axis, y-axis, size, color,
    #     marker and legend
    #     E.G.
    #         if x = 'vp'
    #         WHEN
    #         source_dict = dict(vp=..., vs=..., rho=...)
    #         THEN
    #         x_y_source = dict(x=source_dict['vp'], y=..., etc...)

    #     :param source_dict:
    #     :param x:
    #     :param y:
    #     :param size:
    #     :param color:
    #     :param marker:
    #     :param legend_group:
    #     :param mask:

    #     :return:
    #         dict
    #         transformed_source
    #         The dictionary has at least the following keys:
    #             'x', 'y', 'size', 'color', 'marker' and 'legend_group'
    #     """
    #     _n = len(self)
    #     _vars = list(source_dict.keys())
    #     _dict = {}

    #     if x is not None and x in _vars:
    #         _dict['x'] = source_dict[x]
    #     elif x is not None:
    #         _dict['x'] = x
    #     else:
    #         pass

    #     if y is not None and y in _vars:
    #         _dict['y'] = source_dict[y]
    #     elif y is not None:
    #         _dict['y'] = y
    #     else:
    #         pass

    #     if size is not None and size in _vars:
    #         _dict['size'] = source_dict[size]
    #     elif size is not None:
    #         _dict['size'] = [size] * _n
    #     else:
    #         _dict['size'] = source_dict['size']

    #     if color is not None and color in _vars:
    #         _dict['color'] = source_dict[color]
    #     elif color is not None:
    #         _dict['color'] = [color] * _n
    #     else:
    #         _dict['color'] = source_dict['color']

    #     if marker is not None and marker in _vars:
    #         _dict['marker'] = source_dict[marker]
    #     elif marker is not None:
    #         _dict['marker'] = [marker] * _n
    #     else:
    #         _dict['marker'] = source_dict['marker']

    #     if legend_group is not None and legend_group in _vars:
    #         _dict['legend_group'] = source_dict[legend_group]
    #     elif legend_group is not None:
    #         _dict['legend_group'] = [legend_group] * _n
    #     else:
    #         _dict['legend_group'] = source_dict['legend_group']

    #     if mask is not None:
    #         _dict['mask'] = mask
    #     else:
    #         _dict['mask'] = source_dict['mask']

    #     _dict['source_name'] = source_dict['source_name']
    #     return _dict

    def js_code(self, console_only=False):
        console_only_code = """
            const x_param = x_drop.value;
            console.log('dropdown: ' + cb_obj.value, x_param);
        """
        code1 = """
            const x_param = x_drop.value;
            const y_param = y_drop.value;
            const s_param = s_drop.value;
            //const c_param = c_drop.value;
            //const m_param = m_drop.value;
            //const l_param = l_drop.value;
            const min = min_max[s_param][0];
            const max = min_max[s_param][1];
            const s = [];
            function size(arr1) {
                let s = [];
                for (let i = 0; i < arr1.length; i++) {
                  s.push(10 + 70 * (arr1[i] - min) / (max - min));
                      }
                return s;
            }
            // REPLACE DATA
            for (let i = 0; i < data_keys.length; i++) {
                data_sources[i].data['x'] = orig_data_sources[i].data[x_param];
                data_sources[i].data['y'] = orig_data_sources[i].data[y_param]
                if (min_max[s_param][0] === 'constant') {
                        let length = orig_data_sources[i].data[x_param].length;
                        data_sources[i].data['size'] = Array(length).fill(Number(s_param));
                    } else {
                        data_sources[i].data['size'] = size(orig_data_sources[i].data[s_param]);
                    }
                }
            //
            for (let i = 0; i < data_keys.length; i++) {
                data_sources[i].change.emit();
                }

            // X AXIS
            xaxis.axis_label = x_param;
            if (templates[x_param]['min'] === null || templates[x_param]['min'] === undefined) {
                    console.log('No ' + x_param + ' min value. Use min: ' + min_max[x_param][0]);
                    x_range.start = min_max[x_param][0];
                } else {
                    console.log(x_param + ' min value ' + templates[x_param]['min']);
                    x_range.start = templates[x_param]['min'];
                }
            if (templates[x_param]['max'] === null || templates[x_param]['max'] === undefined) {
                    console.log('No ' + x_param + ' max value. Use max: ' + min_max[x_param][1]);
                    x_range.end = min_max[x_param][1];
                } else {
                    console.log(x_param + ' max value ' + templates[x_param]['max']);
                    x_range.end = templates[x_param]['max'];
                }
            x_range.change.emit()

            // Y AXIS
            yaxis.axis_label = y_param;
            if (templates[y_param]['min'] === null || templates[y_param]['min'] === undefined) {
                    console.log('No ' + y_param + ' min value. Use min: ' + min_max[y_param][0]);
                    y_range.start = min_max[y_param][0];
                } else {
                    console.log(y_param + ' min value ' + templates[y_param]['min']);
                    y_range.start = templates[y_param]['min'];
                }
            if (templates[y_param]['max'] === null || templates[y_param]['max'] === undefined) {
                    console.log('No ' + y_param + ' max value. Use max: ' + min_max[y_param][1]);
                    y_range.end = min_max[y_param][1];
                } else {
                    console.log(y_param + ' max value ' + templates[y_param]['max']);
                    y_range.end = templates[y_param]['max'];
                }
            y_range.change.emit()

            title.text = 'XXX';
            //
            console.log('dropdown: ' + cb_obj.value, x_param);
            """
        code2 = """
            const x_param = x_drop.value;
            const y_param = y_drop.value;
            const s_param = s_drop.value;
            //const c_param = c_drop.value;
            //const m_param = m_drop.value;
            //const l_param = l_drop.value;
            const min = min_max[s_param][0];
            const max = min_max[s_param][1];
            const s = [];
            function size(arr1) {
                let s = [];
                for (let i = 0; i < arr1.length; i++) {
                  s.push(10 + 70 * (arr1[i] - min) / (max - min));
                      }
                return s;
            }
            // REPLACE DATA
            x_y_source.data['x'] = source.data[x_param];
            x_y_source.data['y'] = source.data[y_param];
            if (min_max[s_param][0] === 'constant') {
                    let length = source.data[x_param].length;
                    x_y_source.data['size'] = Array(length).fill(Number(s_param));
                } else {
                    x_y_source.data['size'] = size(source.data[s_param]);
                }
            //
            x_y_source.change.emit();

            // X AXIS
            xaxis.axis_label = x_param;
            if (templates[x_param]['min'] === null || templates[x_param]['min'] === undefined) {
                    console.log('No ' + x_param + ' min value. Use min: ' + min_max[x_param][0]);
                    x_range.start = min_max[x_param][0];
                } else {
                    console.log(x_param + ' min value ' + templates[x_param]['min']);
                    x_range.start = templates[x_param]['min'];
                }
            if (templates[x_param]['max'] === null || templates[x_param]['max'] === undefined) {
                    console.log('No ' + x_param + ' max value. Use max: ' + min_max[x_param][1]);
                    x_range.end = min_max[x_param][1];
                } else {
                    console.log(x_param + ' max value ' + templates[x_param]['max']);
                    x_range.end = templates[x_param]['max'];
                }
            x_range.change.emit()

            // Y AXIS
            yaxis.axis_label = y_param;
            if (templates[y_param]['min'] === null || templates[y_param]['min'] === undefined) {
                    console.log('No ' + y_param + ' min value. Use min: ' + min_max[y_param][0]);
                    y_range.start = min_max[y_param][0];
                } else {
                    console.log(y_param + ' min value ' + templates[y_param]['min']);
                    y_range.start = templates[y_param]['min'];
                }
            if (templates[y_param]['max'] === null || templates[y_param]['max'] === undefined) {
                    console.log('No ' + y_param + ' max value. Use max: ' + min_max[y_param][1]);
                    y_range.end = min_max[y_param][1];
                } else {
                    console.log(y_param + ' max value ' + templates[y_param]['max']);
                    y_range.end = templates[y_param]['max'];
                }
            y_range.change.emit()

            title.text = 'XXX';
            //
            console.log('dropdown: ' + cb_obj.value, x_param);
            """
        code3 = """
            const x_param = x_drop.value;
            const y_param = y_drop.value;
            const s_param = s_drop.value;
            //const c_param = c_drop.value;
            //const m_param = m_drop.value;
            //const l_param = l_drop.value;
            const min = min_max[s_param][0];
            const max = min_max[s_param][1];
            const s = [];
            var new_data = x_y_source.data;
            
            function size(arr1) {
                let s = [];
                for (let i = 0; i < arr1.length; i++) {
                  s.push(10 + 70 * (arr1[i] - min) / (max - min));
                      }
                return s;
            }
            // REPLACE DATA
            new_data['x'] = x_y_source.data[x_param];
            new_data['y'] = x_y_source.data[y_param];
            if (min_max[s_param][0] === 'constant') {
                    let length = x_y_source.data[x_param].length;
                    x_y_source.data['size'] = Array(length).fill(Number(s_param));
                } else {
                    x_y_source.data['size'] = size(x_y_source.data[s_param]);
                }
            //
            x_y_source.change.emit();
                
            // X AXIS
            xaxis.axis_label = x_param;
            if (templates[x_param]['min'] === null || templates[x_param]['min'] === undefined) {
                    console.log('No ' + x_param + ' min value. Use min: ' + min_max[x_param][0]);
                    x_range.start = min_max[x_param][0];
                } else {
                    console.log(x_param + ' min value ' + templates[x_param]['min']);
                    x_range.start = templates[x_param]['min'];
                }
            if (templates[x_param]['max'] === null || templates[x_param]['max'] === undefined) {
                    console.log('No ' + x_param + ' max value. Use max: ' + min_max[x_param][1]);
                    x_range.end = min_max[x_param][1];
                } else {
                    console.log(x_param + ' max value ' + templates[x_param]['max']);
                    x_range.end = templates[x_param]['max'];
                }
            x_range.change.emit()
            
            // Y AXIS
            yaxis.axis_label = y_param;
            if (templates[y_param]['min'] === null || templates[y_param]['min'] === undefined) {
                    console.log('No ' + y_param + ' min value. Use min: ' + min_max[y_param][0]);
                    y_range.start = min_max[y_param][0];
                } else {
                    console.log(y_param + ' min value ' + templates[y_param]['min']);
                    y_range.start = templates[y_param]['min'];
                }
            if (templates[y_param]['max'] === null || templates[y_param]['max'] === undefined) {
                    console.log('No ' + y_param + ' max value. Use max: ' + min_max[y_param][1]);
                    y_range.end = min_max[y_param][1];
                } else {
                    console.log(y_param + ' max value ' + templates[y_param]['max']);
                    y_range.end = templates[y_param]['max'];
                }
            y_range.change.emit()
            
            title.text = 'XXX';
            //
            console.log('dropdown: ' + cb_obj.value, x_param);
            x_y_source.data = new_data;
            """
        if console_only:
            return console_only_code
        else:
            return code3

    def drop_down_menus(self):
        from bokeh.models import Select
        _x_menu =  Select(title='X axis', value=self.common_variables[0], options=self.common_variables)
        _y_menu =  Select(title='Y axis', value=self.common_variables[1], options=self.common_variables)
        _size_menu =  Select(title='Size', value=20,
                             options= self.fixed_sizes + self.common_variables)
        # TODO The selectors below needs fixing before they can be used properly
        _color_menu =  Select(title='Color axis', value=self.common_variables[0],
                              options=self.fixed_colors + self.common_variables)
        _marker_menu =  Select(title='Marker', value=self.common_variables[0], options=self.common_variables)
        _legend_menu =  Select(title='Legend', value=self.common_variables[0], options=self.common_variables)
        return _x_menu, _y_menu, _size_menu, _color_menu, _marker_menu, _legend_menu

    def min_and_max(self):
        _out = {}
        for _key in self.all_variables:
            _min = 1E6; _max = -1E6
            for _source in self._data_sources.values():
                if _key not in list(_source.keys()):
                    continue
                if _source.min_and_max()[_key][0] < _min:
                    _min = _source.min_and_max()[_key][0]
                if _source.min_and_max()[_key][1] > _max:
                    _max = _source.min_and_max()[_key][1]
            _out[_key] = [_min, _max]
        for _key in self.fixed_sizes:
            _out[_key] = ['constant', 'constant']
        return _out

    def fig(self):
        xplot = figure(width=self.width, height=self.height, tools=self._tools)
        xplot.toolbar.active_inspect = None
        xplot.toolbar.logo = None
        return xplot

    def draw(self, source):

        xplot = self.fig()

        # set up drop down menus
        x_menu, y_menu, size_menu, color_menu, marker_menu, legend_menu = self.drop_down_menus()

        # print('XXX:')
        # for _key, _val in self.min_and_max().items():
        #     print(' ', _key, _val)

        # Determine x and y variable
        x_var = x_menu.value
        y_var = y_menu.value
        size_var = size_menu.value

        # x_y_source_dict = self.x_y_source(
        #     dict(self.source.data),
        #     x=x_var,
        #     y=y_var,
        #     size=size_var)
        # x_y_source = ColumnDataSource(x_y_source_dict)

        xplot.scatter(x='x', y='y', source=source, fill_color='color', marker='marker',
                      legend_group='legend_group', size='size', fill_alpha=0.5, line_color=None)

        # Axes
        for _var, _axis in zip([x_var, y_var], [xplot.xaxis, xplot.yaxis]):
            # _axis.axis_label_text_font_size = '10px'
            # _axis.major_label_text_font_size = '10px'
            _axis.axis_label_standoff = 0
            _axis.axis_label = '{} [{}]'.format(
                self.templates[_var].name, self.templates[_var].units)
        for _var, _range in zip([x_var, y_var], [xplot.x_range, xplot.y_range]):
            if self.templates[_var].min is not None:
                _range.start = self.templates[_var].min
            if self.templates[_var].max is not None:
                _range.end = self.templates[_var].max

        # Legend
        if len(xplot.legend) > 0:
            xplot.legend.click_policy = 'hide'
            xplot.legend.location = 'top_right'
            # xplot.legend.label_text_font_size = '8pt'

        # # Test templates:
        # # print(list(self.templates.keys()))
        # _templates = {_key: _val.get_as_dict() for _key, _val in self.templates.items()}
        # # for _var in [x_var, y_var]:
        # for _var in list(self.templates.keys()):
        #     # print(list(_templates.keys()))
        #     print(_var, list(_templates[_var].keys()))
        #     print(_var, list(_templates[_var][_var].keys()))
        #     # print(_var, _templates[_var]['min'], _templates[_var]['max'])

        # arrange call backs
        args_dict = dict(x_drop=x_menu,
                         y_drop=y_menu,
                         s_drop=size_menu,
                         x_range=xplot.x_range,
                         y_range=xplot.y_range,
                         templates={_key: _val.get_as_dict()[_key] for _key, _val in self.templates.items()},
                         # data_keys=[_key for _key in list(sources.keys())],
                         # data_sources=[_val for _val in list(sources.values())],
                         # orig_data_sources=[_val.source for _val in list(self._data_sources.values())],
                         x_y_source=source,
                         source=self.source,
                         min_max=self.min_and_max(),
                         # # cmap=exp_cmap,
                         title=xplot.title,
                         xaxis=xplot.xaxis[0],
                         yaxis=xplot.yaxis[0],
                         # # bar=bar
                         )

        x_menu.js_on_change(
            'value',
            CustomJS(
                args=args_dict,
                code=self.js_code(console_only=False)))

        y_menu.js_on_change(
            'value',
            CustomJS(
                args=args_dict,
                code=self.js_code()))

        size_menu.js_on_change(
            'value',
            CustomJS(
                args=args_dict,
                code=self.js_code()))

        return xplot, x_menu, y_menu, size_menu, color_menu

    def show_plot(self, source, out_file):
        # This is a short cut to give a quick view of the cross plot
        from bokeh.io import output_file
        output_file(out_file)
        xplot, x_menu, y_menu, size_menu, color_menu = self.draw(source)
        show(column(xplot, row(x_menu, y_menu, size_menu, color_menu)))


class TestCases(unittest.TestCase):
    def test_data(self):
        from blixt_rp.core.core import Template
        md1 = np.linspace(1000., 2000., 500)
        md2 = np.linspace(800., 2500., 800)
        l1_1 = np.random.normal(10., 1., 500)
        l1_2 = np.random.normal(10., 1., 800)
        l2_1 = np.random.normal(100., 1., 500)
        l2_2 = np.random.normal(100., 1., 800)
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

        return data_one, templates_one, data_two, templates_two

    def test_data_source(self):

        data_one, templates_one, data_two, templates_two = self.test_data()

        ds1 = DataSource(data=data_one, templates=templates_one)
        ds2 = DataSource(data=data_two, templates=templates_two)
        xp = CrossPlotter(dict(one=ds1, two=ds2))
        # print(xp.all_variables)
        # print(xp.common_variables)
        # print(xp.templates)
        print(len(xp))
        d = xp.source
        _dict = dict(d.data)
        for _key, _item in _dict.items():
            print(_key, _item[:5], _item[-5:], len(_item))
        print('XXX')
        _d = add_x_y_to_source(_dict, x='var_one', y='var_two')
        for _key, _item in _d.items():
            print(_key, _item[:5], _item[-5:], len(_item))
        # xp.show_plot('C:\\Users\marte\Downloads\plot.html')
        xp.show_plot(d, 'C:\\Users\emb\Downloads\plot.html')

    def test_intervals(self):
        from blixt_rp.core.core import Intervals
        from bokeh.io import output_file
        output_file('C:\\Users\\emb\\Downloads\\plot.html')
        #project_table = "C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"
        project_table = "C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"
        wis = Intervals()
        wis.read_blixt_tops(project_table)
        wis.keep_wells(['WELL_B', 'WELL_C', 'WELL_F'])
        wis_table = WorkingIntervalsTable(wis)
        print(wis_table.source.data['use'])
        table = wis_table.draw(wis_table.source)
        show(table)

    def test_from_well(self):
        from blixt_rp.core.project_new import Project
        from blixt_rp.core.core import LogTable, Intervals

        # project_table = "C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"
        project_table = "C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"

        project = Project(
            name='testing',
            # working_dir='C:\\Users\\marte\\PycharmProjects\\blixt_rp',
            working_dir='C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp',
            project_table=project_table
        )
        log_table = LogTable({'Density': 'rho_dry', 'P velocity': 'vp_dry', 'S velocity': 'vs_dry',
                              'Porosity': 'PHIE', 'Volume': 'VCL'})

        wis = Intervals()
        wis.read_blixt_tops(project.project_table)

        project.load_all_wells(log_table=log_table)
        # print([(w.name, list(w.style.keys()), w.style.full_name) for w in project.wells])
        xp = CrossPlotter(
            {w.name: w.data_source() for w in project.wells}
        )
        source = xp.source
        xp.show_plot(source, 'C:\\Users\\emb\\Downloads\\plot.html')

    def test_classification_table(self):
        from pint import Quantity as Q_
        from bokeh.io import output_file
        from bokeh.plotting import column
        from blixt_rp.core.core import CutoffRule
        # output_file('C:\\Users\\emb\\Downloads\\plot.html')
        output_file('C:\\Users\\marte\\Downloads\\plot.html')

        rule1 = CutoffRule('param1', '>', Q_(100, 'm'))
        rule2 = CutoffRule('param2', '<', Q_(10, 'm'))
        rule3 = CutoffRule('param3', '==', Q_(1000, 'm'))
        rule5 = CutoffRule('param5', '><', [Q_(10, 'm'), Q_(1000, 'm')])
        ct = ClassificationTable([rule1, rule2, rule3, rule5])
        # ct = ClassificationTable()
        table_source = ct.source
        # print(ct.source.data)
        table, add_row, delete_row, update, use = ct.draw(table_source, ['log A', 'log B'], units=['m', 'km'])

        # show(column(table, add_row, delete_row, update, use))
        return table, add_row, delete_row, update, use

    def test_data_classification(self):
        import blixt_utils.misc.masks as masks
        data_one, templates_one, data_two, templates_two = self.test_data()
        print(templates_one.keys())

        ds1 = DataSource(data=data_one, templates=templates_one)
        ds2 = DataSource(data=data_two, templates=templates_two)
        params = []
        units = []
        for data_set in [ds1, ds2]:
            for _n, _t in data_set.templates.items():
                if _n not in params:
                    params.append(_n)
                    units.append(_t.units)

        xp = CrossPlotter(dict(one=ds1, two=ds2))
        d_source = xp.source

        xplot, x_menu, y_menu, size_menu, color_menu = xp.draw(d_source)

        ct = ClassificationTable()
        t_source = ct.source
        table, add_row, delete_row, update, use = ct.draw(
            t_source,
            params,
            units=units,
            classification_data_source=d_source,
            color_menu=color_menu
        )

        def test_use(attr, old, new):
            print('TEST: ', ct.cutoffs)

        # TODO
        # Add a functionality that modifies the legend_group of the source
        # based on the classification

        # This shows that you can have several call back functions attached one one event
        use.on_change('active', test_use)

        return xplot, x_menu, y_menu, size_menu, color_menu,  table, add_row, delete_row, update, use

