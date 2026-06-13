"""
Collection of objects and methods used throughout blixt_rp

"""
import os, sys
from copy import deepcopy

import numpy as np

from .. import Q_
import pint
import pandas as pd
import logging

from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper, CheckboxEditor, Select)
from bokeh.models import (DataTable, PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, CrosshairTool,
                          HoverTool, TextInput)
from bokeh.plotting import figure, show
from bokeh.plotting import column, row
from datetime import datetime
import matplotlib as mpl

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.utils import add_one, fix_well_name, cycle_colors, isnan, print_info
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info

logger = logging.getLogger(__name__)


class LogTable(dict):
    def __init__(self,
                 log_table: dict | None = None,
                 name: str | None = None
    ):
        """
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
        we also support the "multi log" log table, if you want to access specific, but multiple, logs for
        a log type.
                log_table = {
                   'P velocity': ['vp_virg', 'vp_brine', 'vp_oil', 'vp_gas'],
                   'S velocity': ['vs_virg', 'vs_brine', 'vs_oil', 'vs_gas'],
                   'Density': ['rhob_virg', 'rhob_brine', 'rhob_oil', 'rhob_gas']}

        :param name:
        :param log_table:
            dict
        """
        self.multi_log = False
        self.name = name
        if log_table is not None and len(log_table) > 0:
            # Make sure log names are in lowercase letters
            for _key, _val in log_table.items():
                if isinstance(_val, list):
                    log_table[_key] = [_x.lower() for _x in _val]
                else:
                    log_table[_key] = _val.lower()

            super().__init__(log_table)
            if isinstance(list(log_table.values())[0], list):
                self.multi_log = True

    # def __setitem__(self, key, value):
    #     print('YOU ARE HERE')
    #     if isinstance(value, list) and not self.multi_log:
    #         print_info('Not allowed to a add a list of log names to a normal LogTable', 'error', logger, 'ioerror')
    #     elif isinstance(value, str) and self.multi_log:
    #         print_info('Not allowed to a add a single string (log name) to a multi_log LogTable', 'error', logger, 'ioerror')
    #     else:
    #         self.__dict__[key] = value
    #         # setattr(self, key, value)

    @property
    def invert(self) -> dict:
        if self.multi_log:
            out = {}
            for k, v in self.items():
                for _v in v:
                    out[_v] = k
            return out
            # return {v[0]: k for k, v in self.items()}
        return {v: k for k, v in self.items()}

    def from_invert(self, inv_dict):
        self.multi_log = True
        tmp = {}
        for v in inv_dict.values():
            tmp[v] = []
        for k, v in inv_dict.items():
            tmp[v].append(k)
        for k in list(tmp.keys()):
            self.__setitem__(k, tmp[k])


    @property
    def dict(self) -> dict:
        return {k: v for k, v in self.items()}

    @property
    def log_names(self)-> list:
        """
        Returns list of all log names
        :return:
        """
        log_names = []
        for key in list(self.keys()):
            if self.multi_log:
                log_names += self[key]
            else:
                log_names.append(self[key])
        return log_names

    @property
    def log_types(self) -> list:
        """
        Returns a list of the associated log type for each log
        :return:
        """
        log_types = []
        for key in list(self.keys()):
            if self.multi_log:
                log_types += [key] * len(self[key])
            else:
                log_types.append(key)
        return log_types

    # def un_translate(self, rename_logs: dict):
    #     """
    #     The LogTable often contains translated log name(s), which is necessary as the same LogTable should be
    #     able to be used across multiple las files, which all use different names for the 'same' log.
    #     E.G. PHIE is called CPI_PHIE in one las file, and XXX_PHIE in another

    #     This method returns a new LogTable which uses the 'un-translated' log names, and leaves the original LogTable
    #     untouched

    #     :param rename_logs:
    #         dict
    #         E.G.
    #             {'phie': 'CPI_PHIE'}
    #             where the key is the wanted well log name, and the value is the well log names
    #             to translate from
    #             OR
    #             {'phie': ['CPI_PHIE', 'PHIE_TEST']}
    #             NOTE
    #             Because the value list (e.g. ['CPI_PHI', 'PHIE_TEST'] can contain several log names (BUT rarely do so),
    #             and we can't know which one to use, we take the first item. But raise a warning
    #     :return
    #         LogTable with the 'un-translated' log names
    #     """
    #     un_translated_log_table = deepcopy(self)
    #     if rename_logs is not None:
    #         for _log_name, _log_type in zip(self.log_names, self.log_types):
    #             if _log_name.lower() in [_l.lower() for _l in list(rename_logs.keys())]:
    #                 if isinstance(rename_logs[_log_name.lower()], list) and len(rename_logs[_log_name.lower()]) > 1:
    #                     warn_txt = 'We bluntly assume you want to un-translate using the first name in the list:'
    #                     warn_txt += ' {}, and not the second: {} (and so on)'.format(
    #                         rename_logs[_log_name][0], rename_logs[_log_name][1])
    #                     print_info(warn_txt, 'warning', logger)

    #                 # Do the un-translation. Different for multi log LogTables
    #                 if self.multi_log:
    #                     _log_list = [_l.lower() for _l in un_translated_log_table[_log_type]]
    #                     _i = _log_list.index(_log_name.lower())
    #                     _log_list.pop(_i)
    #                     if isinstance(rename_logs[_log_name.lower()], list):
    #                         _log_list.insert(_i, rename_logs[_log_name.lower()][0].lower())
    #                     else:
    #                         _log_list.insert(_i, rename_logs[_log_name.lower()].lower())
    #                     un_translated_log_table[_log_type] = _log_list
    #                 else:
    #                     un_translated_log_table[_log_type] = rename_logs[_log_name][0].lower()
    #     return un_translated_log_table

    def keep(self, logs_to_keep):
        """
        Because LogTables are used to select which logs that are to be read from a las file, but that they also
        are meant as a selection for multiple wells and las files, we need a way to only keep the log names that
        are present in one las file

        :param logs_to_keep:
            list
            List of log names that should be kept
            e.g. ['vsh', 'phie']

        :return:
            LogTable

        """
        # TODO We should add a translate_log or rename variable (optional) which 'untranslates" the logs_to_keep
        # OR
        # Can we rewrite the whole THING, and the let renaming occur in the reading of the las file?
        _dict = {}
        if self.multi_log:
            for _key in list(self.keys()):
                keep = False
                _these_params = []
                for _param in logs_to_keep:
                    if _param in self[_key]:
                        keep = True
                        _these_params.append(_param)
                if keep:
                    _dict[_key] = _these_params
        else:
            for _key in list(self.keys()):
                if self[_key] in logs_to_keep:
                    _dict[_key] = self[_key]
        return LogTable(_dict)



class CutoffRule:
    """
    Class for rules for cutoffs
    NOTE, a parameter (param) is masked OUT if its value is NOT within the given rules and limits
    This behaviour is guaranteed through the function create_mask in blixt_utils.misc.masks.py
    EG:
    > t = np.arange(10)
    > mask = create_mask(t,'>=',8)
    > print(t[mask])
    >      array([8, 9])
    """
    def __init__(self,
                 param: str,
                 operator: str | None,
                 limit: pint.Quantity | list | str,
                 name: str | None = None):
        """

        :param param:
            name of the parameter (e.g. 'Vsh') OR log type  (e.g. 'Volume')
        :param operator:
            string or None
            representing the masking operation
            '<':  masked_less
            '<=': masked_less_equal
            '>':  masked_greater
            '>=': masked_greater_equal
            '><': masked_inside
            '==': masked_equal
            '!=': masked_not_equal
        :param limit:
            pint.Quantity | list | str
            If limit is a string, the CutoffRule becomes an interval cutoff, where the string is the name of
            the interval we limit the data to.
        :param name:
            Name of the rule
            Useful for classifying data according to a set of rules sharing the same name
        """
        from blixt_utils.utils import print_info
        raise_unit_error = False
        self.interval_cutoff = False

        # instantiate 'param'
        self.param = param

        self.__name__ = name
        self.name = name

        # instantiate 'operator'
        if operator is None and not isinstance(limit, str):
            error_txt = 'If no operator is provided, the limit must be the name of an interval, not {}'.format(limit)
            print_info(error_txt, 'error', logger=logger, raiser='IOError')
        if isinstance(operator, str):
            if operator not in [ '<', '<=', '>', '>=', '><', '==', '!=']:
                error_txt = 'Could not match "{}" with any valid operator'.format(operator)
                print_info(error_txt, 'error', logger=logger, raiser='IOError')

        self.operator = operator

        # instantiate 'limit'
        if isinstance(limit, list):
            for _item in limit:
                if not isinstance(_item, pint.Quantity):
                    raise_unit_error = True
        elif isinstance(limit, str):
            self.interval_cutoff = True
        elif not isinstance(limit, pint.Quantity):
            raise_unit_error = True
        if raise_unit_error:
            error_txt = 'Limits must be provided as a pint.Quantity (has units)'
            print_info(error_txt, 'error', logger=logger, raiser='IOError')

        self.limit = limit

    def __str__(self):
        _param = '' if self.param is None else self.param
        _operator = '' if self.operator is None else self.operator
        if isinstance(self.limit, list):
            _limits = '[{}]'.format(' ,'.join([str(m.magnitude) for m in self.limit]))
        elif isinstance(self.limit, pint.Quantity):
            _limits = str(self.limit.magnitude)
        else:
            _limits = self.limit
        return '{}: {} {}'.format( _param, _operator, _limits)


class ClassificationTable:
    """
    Table with rules that classify the data.
    The rules can also be used to mask out data (cutoffs)

    """
    def __init__(self,
                 rules: list | None = None,
                 width: int | None = None):
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
        self.height = 100
        if rules is None:
            rules = []
        self._rules = rules  # List of all rules
        self.use_status = [False for _i in range(len(self._rules))]
        self.cutoffs = []  # List of all rules to be used as cutoffs
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
    def cds(self) -> ColumnDataSource:
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
            _dict['name'].append(_rule.name)
            _dict['log'].append(_rule.param)
            _dict['units'].append(_u)
            _dict['operator'].append(_rule.operator)
            _dict['limits'].append(_r)

        return ColumnDataSource(_dict)

    def create_mask(self, rules_cds: ColumnDataSource, data_cds: ColumnDataSource,
                    verbose: bool = False) -> np.array:
        """
        Creates a mask for the data_cds using the rules in rules_cds and returns a boolean mask

        NOTE the data_cds does not contain information about the units of the data, so we need to make sure the
        data and the rules are having the same units

        :param rules_cds:
            ColumnDataSource
            The cds of this class (self.cds)

        :param data_cds:
            ColumnDataSource
            The data that to be masked or classified
            The data cds is in the format specified by the CrossPlotter class
        :return:
        """
        import blixt_utils.misc.masks as masks
        cutoffs = []
        rules = []
        use_status = []
        for i, use in enumerate(rules_cds.data['use']):
            if ',' in rules_cds.data['limits'][i]:
                this_limit = [Q_(float(_s), rules_cds.data['units'][i]) for _s in rules_cds.data['limits'][i].split(',')]
            else:
                this_limit = Q_(float(rules_cds.data['limits'][i]), rules_cds.data['units'][i])
            use_status.append(use)
            _rule = CutoffRule(
                param=rules_cds.data['log'][i],
                operator=rules_cds.data['operator'][i],
                limit=this_limit,
                name=rules_cds.data['name'][i]
            )
            rules.append(_rule)
            if use:
                cutoffs.append(_rule)
        # self.rules = rules
        # self.use_status = use_status
        # self.cutoffs = cutoffs

        _mask = np.ones(len(data_cds.data['mask']), dtype=bool)
        if len(cutoffs) > 0:  # There are active cutoffs
            for _i, _rule in enumerate(cutoffs):
                if verbose:
                    print('XXX1: Creating cutoffs for param {}'.format(_rule.param) )
                _this_mask = masks.create_mask(
                    data_cds.data[_rule.param],
                    _rule.operator,
                    _rule.limit
                )
                if _i == 0:  # First cutoff rule
                    _mask = _this_mask
                else:  # Append new cutoff rule using AND
                    _mask = masks.combine_masks([_mask, _this_mask])

        return _mask

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

    def active_cutoffs(self, cds: ColumnDataSource) -> list:
        _list = []
        for i, _use in enumerate(cds.data['use']):
            if _use:
                _list.append('{}{}{}'.format(
                    cds.data['log'][i], cds.data['operator'][i], cds.data['limits'][i]))
        return _list

    def draw(self,
             cds: ColumnDataSource,
             parameters: list | None = None,
             units: list | None = None,
             color_menu: Select | None = None,
             verbose: bool = False):
        """

        :param cds:
            ColumnDataSource
            Source containing the rules
        :param parameters:
            list
            List of (log) parameters (variables) on which the rules are applied
            E.G. 'Vsh'
        :param units:
            list
            List of units used for the different parameters
        :param color_menu:
            Select
            Dropdown menu that decides how to color the data.
            Only the option "Classification" will be listened to here
        :return:
            dt, add_row, delete_row, update_table, use_cutoffs
        """
        from bokeh.models import DataTable, Button, CheckboxGroup, Div
        from blixt_rp.core.core import CutoffRule
        import blixt_utils.misc.masks as masks

        orig_data = None

        if parameters is None:
            # Use the parameters defined in the cds
            parameters = cds.data['log']
            parameters = list(set(parameters))
        if units is None:
            # Use the units defined in the cds
            units = cds.data['units']
            units = list(set(units))

        def update_table_callback():
            # How do I use input variables to a python call back?
            #  - It depends on which widget you use, see:
            #      docs.bokeh.org/en/latest/docs/user_guide/interaction/python_callbacks.html#ug-interaction-python-callbacks
            cutoffs = []
            rules = []
            use_status = []
            for i, use in enumerate(cds.data['use']):
                if ',' in cds.data['limits'][i]:
                    this_limit = [Q_(float(_s), cds.data['units'][i]) for _s in cds.data['limits'][i].split(',')]
                else:
                    this_limit = Q_(float(cds.data['limits'][i]), cds.data['units'][i])
                use_status.append(use)
                _rule = CutoffRule(
                    param=cds.data['log'][i],
                    operator=cds.data['operator'][i],
                    limit=this_limit,
                    name=cds.data['name'][i]
                )
                rules.append(_rule)
                if use:
                    cutoffs.append(_rule)
            self.rules = rules
            self.use_status = use_status
            self.cutoffs = cutoffs

        # def use_cutoffs_callback(attr, old, new):
        #     # Test using the create_mask() function instead
        #     if data_cds is not None:
        #         # Take a copy of the original data, and original mask
        #         _dict = dict(data_cds.data)
        #         orig_mask = data_cds.data['mask']

        #         if len(new) == 1:  # 'Use cutoffs' is active
        #             if verbose:
        #                 print('Use cutoffs, ', old, new)
        #             mask = masks.combine_masks([orig_mask, self.create_mask(cds, data_cds)])
        #         else:
        #             if verbose:
        #                 print('Reset cutoffs, ', old, new)
        #             # mask = np.ones(len(data_cds.data['mask']), dtype=bool)
        #             mask = orig_mask


        #         # Modify the mask in the data_cds
        #         _dict['mask'] = mask
        #         # Update the cds
        #         data_cds.data = _dict

            # if len(new) == 1 and data_cds is not None:
            #     print('Use cutoffs, ', old, new)
            #     update_table_callback()
            #     if len(self.cutoffs) > 0:  # There are active cutoffs
            #         new_data = dict(data_cds.data)
            #         _mask = None
            #         for _i, _rule in enumerate(self.cutoffs):
            #             _this_mask = masks.create_mask(
            #                 data_cds.data[_rule.param],
            #                 _rule.operator,
            #                 _rule.limit
            #             )
            #             if _i == 0:  # First cutoff rule
            #                 _mask = _this_mask
            #             else:  # Append new cutoff rule using AND
            #                 _mask = masks.combine_masks([_mask, _this_mask])
            #         # apply mask
            #         for _key in list(new_data.keys()):
            #             new_data[_key] = new_data[_key][_mask]
            #         data_cds.data = new_data
            # else:
            #     print('Do not use cutoffs, ', old, new)
            #     data_cds.data = orig_data

        def add_row_function():
            new_data = dict(cds.data)
            new_data['use'].append(False)
            new_data['name'].append('Class A')
            new_data['log'].append(parameters[0])
            new_data['units'].append(units[0])
            new_data['operator'].append('>')
            new_data['limits'].append('1500.')
            cds.data = new_data

        def delete_row_function():
            selected_index = cds.selected.indices
            # new_data = dict(use=[], name=[], log=[], units=[], operator=[], limits=[])
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(cds.data['use'])):
                if _i in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        # def use_color_classification(attr, old, new):
        #     print(attr, old, new)
        #     if new == 'Classification':
        #         update_table_callback()
        #         if len(self.rules) > 0:
        #             print(list(self.classes.keys()))
        #             new_data = dict(data_cds.data)
        #             # Add 'non-classified' to all data points first
        #             new_data['legend_group'][:] = 'No class'
        #             new_data['color'][:] = 'gray'
        #             new_data['marker'][:] = 'dot'
        #             # Add specific style to classified data
        #             for _i, _class in enumerate(self.classes.keys()):  # iterate over all classes
        #                 _mask = None
        #                 for _j, _rule in enumerate(self.classes[_class]):  # iterate over all rules within this class
        #                     _this_mask = masks.create_mask(
        #                         data_cds.data[_rule.param],
        #                         _rule.operator,
        #                         _rule.limit
        #                     )
        #                     if _j == 0:
        #                         _mask = _this_mask
        #                     else:  # Append rule using AND
        #                         _mask = masks.combine_masks([_mask, _this_mask])
        #                 new_data['legend_group'][_mask] = _class
        #                 new_data['color'][_mask] = cnames[_i]
        #                 new_data['marker'][_mask] = markers[_i]
        #             print('Classification ON. {} data points with classes {} '.format(
        #                 len(new_data['legend_group']), list(set(new_data['legend_group']))))
        #             data_cds.data = new_data
        #     else:
        #         # TODO
        #        # This doesnt work properly. The classification doesn't reset
        #         print('Classification OFF. {} data points with classes {} '.format(
        #             len(orig_data['legend_group']), list(set(list(orig_data['legend_group'])))))
        #         data_cds.data = orig_data

        title = Div(text =
                    """
                    <div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">
                        Classifications
                    </div>
               """)

        dt = DataTable(
            source=cds,
            columns=self.table_columns(parameters, units),
            editable=True,
            width=self.width,
            height=self.height,
            index_position=-1,
            index_header='index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_callback)

        # use_cutoffs = CheckboxGroup(labels=['Use cutoffs'], active=[])
        # use_cutoffs.on_change('active', use_cutoffs_callback)
        # use_cutoffs = Div(text='', width=10, height=10)

        # if (color_menu is not None) and (data_cds is not None):
        #     color_menu.on_change('value', use_color_classification)

        return column(title, dt, sizing_mode='stretch_width'), add_row, delete_row, update_table  #, use_cutoffs


class Cutoffs:
    def __init__(self,
                 cutoffs: list | None = None,
                 name: str | None = None
                 ):
        """

        :param name:
        :param cutoffs:
            List of CutOffRules
        """
        from blixt_utils.utils import print_info
        self.name = name
        if cutoffs is None:
            cutoffs = []
        cutoff_names = []
        for _key in cutoffs:
            if not isinstance(_key, CutoffRule):
                error_txt = 'Cutoffs must be provided as a CutoffRule'
                print_info(error_txt, 'error', logger=logger, raiser='IOError')
            if _key.param in cutoff_names:
                warn_txt = '{} is repeated and last occurrence is ignored'.format(_key.param)
                print_info(warn_txt, 'warning', logger=logger)
            cutoff_names.append(_key.param)
        self.cutoffs = cutoffs

    @property
    def cutoff_params(self):
        return [_x.param for _x in self.cutoffs]

    def __len__(self):
        return len(self.cutoffs)

    def __str__(self):
        return ', '.join([str(m) for m in self.cutoffs])

    def append(self, new_cutoffs):
        from blixt_utils.utils import print_info
        if isinstance(new_cutoffs, list):
            for i, co in enumerate(new_cutoffs[:]):
                if co.param in self.cutoff_params:
                    warn_txt = '{} is repeated and last occurrence is ignored'.format(co.param)
                    print_info(warn_txt, 'warning', logger=logger)
                    _ = new_cutoffs.pop(i)
            self.cutoffs = self.cutoffs + new_cutoffs
        elif isinstance(new_cutoffs, CutoffRule):
            if new_cutoffs.param in self.cutoff_params:
                warn_txt = '{} is repeated and last occurrence is ignored'.format(new_cutoffs.param)
                print_info(warn_txt, 'warning', logger=logger)
            else:
                self.cutoffs.append(new_cutoffs)

    def get_dict(self):
        """
        Returns a dictionary with the rules in a format used by many functions
        NOTE that the units of the limits are ignored
        :return:
        dict(<param1>=[<operator>: <limit>], ...)
        """
        return_dict = {}
        for rule in self.cutoffs:
            this_list = [rule.operator]
            if isinstance(rule.limit, pint.Quantity):
                this_list.append(rule.limit.magnitude)
            elif isinstance(rule.limit, list):
                this_list.append([rule.limit[0].magnitude, rule.limit[1].magnitude])
            elif isinstance(rule.limit, str):
                this_list.append(rule.limit)

            return_dict[rule.param] = this_list
        return return_dict

    def use_log_table(self, log_table: LogTable):
        """
        Returns a new CutOffs object where the parameter names have been converted from 'log type' (e.g. 'P velocity')
         to 'log names' (e.g. 'vp_brine')

        :param log_table:
        :return:
        """
        if log_table.multi_log:
            raise NotImplementedError('Multi log LogTables are not supported (Yet)')
        new_rules = deepcopy(self.cutoffs)
        for _rule in new_rules:
            try:
                _rule.param = log_table[_rule.param]
            except KeyError:
                # warn_txt = 'Log type {} is not found in log_table: {}'.format(
                #     _rule.param, ', '.join(list(log_table.keys())))
                # print_info(warn_txt, 'warning', logger=logger)
                continue
        return Cutoffs(cutoffs=new_rules)


class Template:
    """
    Template class
    """
    def __init__(self,
                 full_name: str | None = None,
                 name: str | None = None,
                 well: str | None = None,
                 units: str | None = None,
                 min: float | None = None,
                 max: float | None = None,
                 colormap: str | None = None,
                 center: float | None = None,
                 bounds: list | None = None,
                 scale: str | None = None,
                 line_color: str | None = None,
                 line_style: str | None = None,
                 line_width: float | None = None,
                 fill_color: str | None = None,
                 marker: str | None = None
                 ):

        self.full_name = full_name
        self.name = name
        self.well = well
        self.units = units
        self.min = min
        self.max = max
        self.colormap = colormap
        self.center = center
        self.bounds = bounds
        self.scale = scale
        self.line_color = line_color
        self.line_style = line_style
        self.line_width = line_width
        self.fill_color = fill_color
        self.marker = marker

    def keys(self):
        return self.__dict__.keys()

    def __str__(self):
        """
        Return better readable string representation of template object.
        """
        keys = list(self.keys())
        try:
            i = max([len(k) for k in keys])
        except ValueError:
            # no keys
            return ''
        pattern = "%%%ds: %%s" % i
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def get_from_project(self,
                         project_file: str,
                         log_type: str):
        """

        :param project_file:
            str
            Full pathname of project xlsx file
        :param log_type:
            str
            Name of the log type, e.g. "S Velocity"
        :return:
        """
        table = pd.read_excel(project_file, header=1, sheet_name='Templates', engine='openpyxl')
        all_templates = templates_from_table(table)
        for key in list(all_templates[log_type].keys()):
            self.__setattr__(key, all_templates[log_type][key])

        # # Also add the style settings for the wells
        # table = pd.read_excel(project_file, header=1, sheet_name='Well settings', engine='openpyxl')
        # all_templates = templates_from_table(table, well_style=True)
        # for key in list(all_templates[log_type].keys()):
        #     self.__setattr__(key, all_templates[log_type][key])

    def dict(self):
        return {self.name: self.__dict__}


class TemplatesTable:
    def __init__(self,
                 templates: list,
                 width: int | None = None):
        """

        :param templates:
            list
            List of Template objects
        :param width:
        """
        if width is None:
            width = 600
        self.width = width
        self.height = 200
        if templates is None:
            templates = []
        self._templates = templates
        self.keys = ['name', 'line_color', 'line_width', 'line_style']
        self.line_styles = {'-': 'solid', '--': 'dashed', ':': 'dotted', '-.': 'dashdot', '.-': 'dotdash'}

        _dict = {_key: [] for _key in self.keys}
        # for _t in self._templates:
        for _t in self.templates:
            for _key in self.keys:
                _val = _t.__dict__[_key]
                if _val is None:
                    if _key == 'line_color':
                        _val = 'blue'
                    elif _key == 'line_width':
                        _val = 1.
                    elif _key == 'line_style':
                        _val = '-'
                _dict[_key].append(_val)

        self._cds = ColumnDataSource(_dict)

    @property
    def templates(self):
        return self._templates

    @templates.setter
    def templates(self, new_templates):
        self._templates = new_templates

    @property
    def cds(self) -> ColumnDataSource:
        return self._cds

    @cds.setter
    def cds(self, new_cds):
        if isinstance(new_cds, ColumnDataSource):
            self._cds = new_cds
        elif isinstance(new_cds, dict):
            self._cds = ColumnDataSource(new_cds)
        else:
            raise IOError('New cds must be either ColumnDataSource or Dict, not {}'.format(
                type(new_cds)
            ))

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, IntEditor, HTMLTemplateFormatter)
        colored_cell_template = """
                <div style="background:<%= 
                    (function color_from_val(){
                        return(line_color)
                        }()) %>; 
                    color: white"> 
                <%= value %>
                </div>
            """
        formatter = HTMLTemplateFormatter(template=colored_cell_template)

        table_columns = [
            TableColumn(field='name', title='Name', editor=SelectEditor(options=[_t.name for _t in self.templates])),
            TableColumn(field='line_color', title='Line color', editor=StringEditor(), formatter=formatter),
            TableColumn(field='line_width', title='Line width', editor=IntEditor(step=1)),
            TableColumn(field='line_style', title='Line style',
                        editor=SelectEditor(options=list(self.line_styles.keys())))
        ]
        return table_columns

    def draw(self,
             cds: ColumnDataSource) -> DataTable:
        """
        Returns a table

        :param cds:
            ColumnDataSource
            Source for style used for plotting data
        :return:
        """

        return DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
            index_position=-1,
            index_header='index'
        )

class StratUnit(object):
    """
    Contains information about one specific stratigraphic unit (Group, Formation, Member, ...)
    Must have a unique name
    """

    def __init__(self,
                 name: str,
                 level: int,
                 desc=None,
                 source=None,
                 color=None
                 ):
        """

        :param name:
        :param level:
            int
            High number indicates higher order intervals (e.g. a sub interval, "Stage" or "Member"), while lower number indicate
            lower order hierarchies (e.g. "Era" or "Group")
            Set it to 0 when unknown or uncertain
            This integer is used in log_plotter.py to draw rectangles to represent the stratigraphic unit
        :param desc:
        :param source:
            str
            E.G. <User name>, or  "sodir", ...
        :param color:
        """
        if name is None:
            raise IOError('An Interval must have a name')
        self.name = name
        self.level = level
        self.desc = desc
        self.source = source
        self.color = color

    def __str__(self):
        return print_function(self)

    def __getitem__(self, item):
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()


class Interval:
    """
   Creates a relation between a strat. unit and a well, and the top and base of the strat. unit in that well
    """
    def __init__(self,
                 well: str,
                 top: pint.Quantity,
                 base: pint.Quantity,
                 interval_info: StratUnit,
                 depth_type: str | None = None):
        """

        :param well:
        :param top:
        :param base:
        :param interval_info:
        :param depth_type:
            str
        """
        import blixt_rp.core.log_curve_new as brlc
        self.well = well
        if depth_type is None:
            depth_type = 'md'
        self.top = brlc.handle_depth(top, depth_units=None, depth_type=depth_type)
        self.base = brlc.handle_depth(base, depth_units=None, depth_type=depth_type)
        self.interval_info = interval_info

    def __str__(self):
        return print_function(self)
    @property
    def name(self):
        return self.interval_info.name

    # Dont know if a need setters?
    # @name.setter
    # def name(self, value):
    #     self.interval_info.name = value

    @property
    def level(self):
        return self.interval_info.level

    @property
    def color(self):
        return self.interval_info.color

    @property
    def source(self):
        return self.interval_info.source

    @property
    def desc(self):
        return self.interval_info.desc

    @property
    def thickness(self):
        return self.base - self.top

    @property
    def mid(self):
        return 0.5 * (self.top + self.base)

    def distance_to_top(self, this_depth: pint.Quantity):
        """
        Distance to top of interval.
        Positive values when this_depth is deeper than top
        :param this_depth:
            pint.Quantity
        :return:
            pint.Quantity
        """
        return -1. * (self.top - this_depth)  # putting self.top ensures the result uses the same units as self.top

    def distance_to_base(self, this_depth: pint.Quantity):
        """
        Distance to base of interval.
        Positive values when this_depth is shallower than base
        :param this_depth:
            pint.Quantity
        :return:
            pint.Quantity
        """
        return self.base - this_depth


class Intervals(object):
    """
    Object that holds a number of working intervals / tops for many wells
    """
    def __init__(self,
                 name: str | None = None,
                 intervals: list | None = None,
                 verbose=False):
        """
        :param name:
            str
        :param intervals:
            list list(Interval)
            List of all single intervals
        :param verbose:
            bool
        """
        self.name = name
        if intervals is None:
            self.intervals = []
        elif isinstance(intervals, list):
            self.intervals = intervals
        else:
            raise IOError('Intervals must be a list (of intervals)')

    def __len__(self):
        return len(self.intervals)

    def __str__(self):
        return print_function(self)

    def interval_names(self):
        return list(set([_x.name for _x in self.intervals]))

    def keep_wells(self, well_names: list):
        for _interval in self.intervals[:]:
            if _interval.well not in well_names:
                self.intervals.remove(_interval)

    def well_names(self):
        return list(set([_x.well for _x in self.intervals]))

    def get_interval(self, interval_name, well_name):
        for _interval in self.intervals:
            if _interval.name.lower() == interval_name.lower() and _interval.well.lower() == well_name.lower():
                return _interval
        return None

    def get_well_depth_range(self,  well_name):
        min_top = 1E6
        max_base = -1E6
        for _interval in self.intervals:
            if _interval.well != well_name:
                continue
            if _interval.top.magnitude <= min_top:
                min_top = _interval.top.magnitude
            if _interval.base.magnitude >= max_base:
                max_base = _interval.base.magnitude
        return min_top,  max_base

    def get_well_intervals(self, well_name) -> list:
        _levels = []
        for _interval in self.intervals:
            if _interval.well == well_name:
                _levels.append(_interval)
        return _levels

    def add_interval(self, interval: Interval):
        self.intervals.append(interval)

    def get_strat_units(self) -> dict:
        """
        Returns a dictionary which contains all the StratUnit content
        :return:
        """
        strat_unit_dict = {'name': [], 'level': [], 'desc': [], 'source': [], 'color': []}
        for _interval in self.intervals:
            if _interval.name in strat_unit_dict['name']:  # avoid repeating
                continue
            else:
                strat_unit_dict['name'].append(_interval.name)
                strat_unit_dict['level'].append(_interval.level)
                strat_unit_dict['desc'].append(_interval.desc)
                strat_unit_dict['source'].append(_interval.source)
                strat_unit_dict['color'].append(_interval.color)
        return strat_unit_dict

    def get_intervals_dict(self,
                           well_name: str | None = None, show_only_intervals: list | None = None) -> dict:
        """
        Returns a dictionary with all intervals

        :param well_name:
            str or None
            if well_name is not None, it filters out all other wells

        :param show_only_intervals:
            list or None
            If None, all intervals are set visible
            If list of interval names, only these are set visible
        :return:
        """
        # intervals_dict = {'well': [], 'name': [], 'top MD [m]': [], 'base MD [m]': [],
        intervals_dict = {'well': [], 'name': [], 'top': [], 'base': [],
                          'level': [], 'color': [], 'source': [], 'visible': [], 'note': []}
        for _interval in self.intervals:
            if well_name is not None:
                if well_name.upper() != _interval.well.upper():
                    continue
            intervals_dict['well'].append(_interval.well.upper())
            intervals_dict['name'].append(_interval.name)
            # intervals_dict['top MD [m]'].append(_interval.top.to('m').magnitude)
            # intervals_dict['base MD [m]'].append(_interval.base.to('m').magnitude)
            intervals_dict['top'].append(_interval.top.to('m').magnitude)
            intervals_dict['base'].append(_interval.base.to('m').magnitude)
            intervals_dict['level'].append(_interval.level)
            intervals_dict['color'].append(_interval.color)
            intervals_dict['source'].append(_interval.source)
            if show_only_intervals is None:
                intervals_dict['visible'].append(True)
            else:
                if _interval.name.lower() in [xx.lower() for xx in show_only_intervals]:
                    intervals_dict['visible'].append(True)
                else:
                    intervals_dict['visible'].append(False)
            intervals_dict['note'].append(_interval.desc)
        return intervals_dict

    def get_cutoff_rule(self, interval_name: str, well_name: str) -> CutoffRule:
        _interval = self.get_interval(interval_name, well_name)
        if _interval is None:
            _top, _base = self.get_well_depth_range(well_name)
            return CutoffRule('md', '><', [Q_(_top, 'm'), Q_(_base, 'm')])
        print(_interval.top, _interval.base)
        return CutoffRule('md', '><', [_interval.top, _interval.base])

    def write_to_excel(self, file_name, intervals_sheet, interval_info_sheet, append: bool = False):
        import openpyxl
        int_rows = 0
        int_info_rows = 0
        int_header = True
        info_header = True
        if not os.path.isfile(file_name):
            wb = openpyxl.Workbook()
            wb.save(filename=file_name)
        elif append:
            # Figure out the row numbers where we can start appending data
            wb = openpyxl.load_workbook(file_name)
            try:
                sheet = wb[intervals_sheet]
                int_rows = sheet.max_row
                int_header = False
            except KeyError:   # this sheet doesn't exist, so we create it later
                pass
            try:
                sheet = wb[interval_info_sheet]
                int_info_rows = sheet.max_row
                info_header = False
            except KeyError:   # this sheet doesn't exist, so we create it later
                pass
            wb.close()

        # First collect and write the interval info
        interval_info_dict = self.get_strat_units()
        df = pd.DataFrame(interval_info_dict)
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(
                writer, sheet_name=interval_info_sheet, startcol=0, startrow=int_info_rows, index=False,
                header=info_header)

        # Then the individual intervals in each well
        intervals_dict = self.get_intervals_dict()
        df = pd.DataFrame(intervals_dict)
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(
                writer, sheet_name=intervals_sheet, startcol=0, startrow=int_rows, index=False,
                header=int_header)

    def read_sodir_tops(self, file_name, testing=True):
        df = pd.read_excel(file_name, engine='openpyxl')
        for i, well_name in enumerate(df['Wellbore name']):
            well_name = fix_well_name(well_name)
            _top = df['Top depth [m]'][i]
            _base = df['Bottom depth [m]'][i]
            _name = df['Lithostrat. unit'][i]
            _level = df['Level'][i]
            if _level == 'GROUP':
                l = 0
            elif _level == 'FORMATION':
                l = 1
            else:
                l = 2

            _interval_info = StratUnit(_name, l, source='SoDir')
            _interval = Interval(well_name, Q_(float(_top), 'm'), Q_(float(_base), 'm'),
                                 interval_info=_interval_info)

            self.add_interval(_interval)

            if testing and (i > 40):
                break

    def read_blixt_tops(self, file_name: str, intervals_sheet: str = 'Working intervals',
                        interval_info_sheet: str = 'Stratigraphic units'):
        # see if there is a sheet with interval info
        strat_units = None
        if interval_info_sheet in list(pd.read_excel(file_name, engine='openpyxl', sheet_name=None)):
            strat_units = {}
            strat_units_table = pd.read_excel(file_name, engine='openpyxl', sheet_name=interval_info_sheet, header=4)
            for _i, _name in enumerate(strat_units_table['Name']):
                if isinstance(strat_units_table['Use'][_i], float) or ('no' in strat_units_table['Use'][_i].lower()):  # skip strat. units that are not in Use
                    continue
                strat_units[_name] = StratUnit(_name,
                                               strat_units_table['Level'][_i],
                                               desc=strat_units_table['Description'][_i],
                                               source=strat_units_table['Source'][_i],
                                               color=strat_units_table['Color'][_i],
                                               )

        df = pd.read_excel(file_name, engine='openpyxl', sheet_name=intervals_sheet, header=4)
        for i, well_name in enumerate(df['Given well name']):
            interval_name = df['Interval name'][i]
            this_strat_unit = None
            if strat_units is None:
                this_level = get_level_from_name(interval_name, source='sodir')
                this_strat_unit = StratUnit(interval_name, this_level, source='blixt')
            else:
                if interval_name not in list(strat_units.keys()):  # only read the strat. units which are in Use
                    continue
                this_strat_unit = strat_units[interval_name]
            _interval = Interval(
                well=well_name.upper(),
                top=Q_(float(df['Top depth'][i]), 'm'),
                base=Q_(float(df['Base depth'][i]), 'm'),
                interval_info=this_strat_unit
            )

            self.add_interval(_interval)

    def bokeh_plot(self,
                   well_name: str,
                   p: figure):
        pass


class WorkingIntervalsTable:
    def __init__(self,
                 intervals,
                 width: int | None = None,
                 set_all_active: bool = True):
        """

        :param intervals:
            blixt_rp.core.core.Intervals object

        :param width:
        """
        if width is None:
            width = 600
        self.width = width
        self.height = 300
        self._intervals = intervals
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
        if set_all_active:
            _dict['use'] = [True] * len(_dict['name'])
        else:
            _dict['use'] = [False] * len(_dict['name'])

        # add a column with the well names that contains the interval of each row
        _wells = []
        for _name in _dict['name']:
            _wl = []
            for _wn in _well_names:
                if self.intervals.get_interval(_name, _wn) is not None:
                    _wl.append(_wn)
            _wells.append(_wl)

        _dict['wells'] = [', '.join(_w) for _w in _wells]

        self._cds =  ColumnDataSource(_dict)

    @property
    def intervals(self):
        return self._intervals

    @property
    def cds(self) -> ColumnDataSource:
        return self._cds

    @cds.setter
    def cds(self, new_cds):
        if isinstance(new_cds, ColumnDataSource):
            self._cds = new_cds
        elif isinstance(new_cds, dict):
            self._cds = ColumnDataSource(new_cds)
        else:
            raise IOError('New cds must be either ColumnDataSource or Dict, not {}'.format(
                type(new_cds)
            ))

    def __dict__(self):
        return self.intervals.get_intervals_dict()

    def active_intervals_dict(self, cds: ColumnDataSource) -> dict:
        """
        Returns a dictionary suitable for a ColumnDataSource which contains all intervals for all wells
        with their top (m MD) and base and "active" status (status determined by the 'cds' input)

        :param cds:
            ColumnDataSource
            The self.cds of WorkingIntervalsTable
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
                    _dict['top'].append(_int.top.to('m').magnitude)
                    _dict['base'].append(_int.base.to('m').magnitude)
                    # iterate over all working intervals in 'cds' to see which are active or not
                    _active = False
                    for _i, _i_n in enumerate(cds.data['name']):
                        if _i_n == _int_name and _wname in cds.data['wells'][_i] and cds.data['use'][_i]:
                            _active = True
                    _dict['active'].append(_active)
        return _dict

    def active_intervals(self, cds: ColumnDataSource) -> list:
        _list = []
        for _i, _i_n in enumerate(cds.data['name']):
            if cds.data['use'][_i]:
                _list.append(_i_n)
        return _list

    def create_mask(self, cds: ColumnDataSource, md: pint.Quantity, wells: list | None = None, verbose:bool = False) -> np.array:
        """
        Returns a boolean mask which is True within the intervals that are set Active in the table.
        :param cds:
            ColumnDataSource
            The internal cds property of self.
            Needed as input to active_intervals_dict()
        :param md:
            pint.Quantity
            Array of measured depth with units
        :param wells:
            list
            If provided, must have the same length as the md array
            Our data in cross_plotter.py have merged data from several wells into one array,
            so this list contains the name of the well from which the corresponding md value
            comes from
        :return:
        """
        _intervals = self.active_intervals_dict(cds)
        _mask = np.zeros(len(md), dtype=bool)  # initial mask with all False
        any_active_wis = False
        if wells is None:
            for i, _md in enumerate(md.to('m')):
                for j, _active in enumerate(_intervals['active']):
                    if _active:
                        if (_md.magnitude >= _intervals['top'][j]) and (_md.magnitude < _intervals['base'][j]):
                           _mask[i] = True
                           any_active_wis = True
        else:
            for i, _md in enumerate(md.to('m')):
                for j, _active in enumerate(_intervals['active']):
                    if _active and (_intervals['well'][j] == wells[i]):
                        # print('XXX', _active, _intervals['well'][j], wells[i])
                        if (_md.magnitude >= _intervals['top'][j]) and (_md.magnitude < _intervals['base'][j]):
                            _mask[i] = True
                            any_active_wis = True

        if not any_active_wis:  # No masking was done, need to set all to True
            _mask = np.ones(len(md), dtype=bool)

        return _mask

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

    def draw(self,
             cds: ColumnDataSource,
             verbose: bool = False
             ):
        """
        Returns a table and some buttons

        :param cds:
            ColumnDataSource
            Source for all working intervals
        :param verbose:
        :return:
        """
        from bokeh.models import DataTable, CheckboxGroup, Div
        import blixt_utils.misc.masks as masks


        active_intervals = ColumnDataSource(self.active_intervals_dict(cds))

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
            args=dict(source=cds, act_int=active_intervals, cb=active_callback),
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
                // source.data = data_table;
                // cb.execute();
            """
        )

        cds.js_on_change('patching', source_callback)  # 'patching' is necessary. Don't know what it means

        # def use_wis_callback(attr, old, new):
        #     if verbose:
        #         print('APPLY SUCCESSFUL:', source.data['use'])
        #     if data_source is not None:
        #         # Take a copy of the original data, and original mask
        #         _dict = dict(data_source.data)
        #         orig_mask = data_source.data['mask']

        #         if len(new) == 1:  # 'Use cutoffs' is active
        #             _mask = self.create_mask(
        #                 source,
        #                 Q_(data_source.data['md'], 'm'),
        #                 list(data_source.data['source_name']))
        #             mask = masks.combine_masks([orig_mask, _mask])
        #         else:  # Reset mask
        #             mask = orig_mask

        #         # Modify the mask in source
        #         _dict['mask'] = mask
        #         # Update the source
        #         data_source.data = _dict

        #         if verbose:
        #             print(data_source.data['md'][data_source.data['mask']][:5])
        #             print(data_source.data['md'][data_source.data['mask']][-5:])

        # if data_source is not None:
        #     use_wis = CheckboxGroup(labels=['Apply working intervals'], active=[])
        #     use_wis.on_change('active', use_wis_callback)
        # else:
        #     use_wis = Div(text='', width=10, height=10)
        # use_wis = Div(text='', width=10, height=10)


        title = Div(text =
                    """
                    <div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">
                        Working intervals
                    </div>
               """)

        dt = DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
            index_position=-1,
            index_header='index'
        )

        return column(title, dt, sizing_mode='stretch_width')  # , use_wis


class Header(AttribDict):
    """
    Class for well header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a Well object.
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every variable import and export modules
    :param
        header: Dictionary containing meta information of a single
        Well object.
    """
    defaults = {
        'name': None,
        'well': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
        'orig_filename': None,
        'modification_date': None,
        'modification_history': '',
        'note': '',
        'log_type': None
    }

    def __init__(self, header=None):
        """
        """
        if header is None:
            header = {}
        # super(Header, self).__init__(header)
        super().__init__(header)

    def __setitem__(self, key, value):
        """
        """
        # keys which shouldn't be modified
        if key in ['creation_date', 'modification_date']:
            pass
        else:
            # all other keys
            super(Header, self).__setitem__(key, value)

            super(Header, self).__setitem__(
                'modification_date',  datetime.now().isoformat())

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Header object.
        """
        # keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        try:
            i = max([len(k) for k in keys])
        except ValueError:
            # no keys
            return ''
        pattern = "%%%ds: %%s" % (i)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class LithoFluid:
    """
    Class containing the elastic properties, and the statistics describing it, for one single "litho fluid"
    (an element that describes a specific lithology with a specific fluid content)

    Vp, Vs and Rho can be given as functions to serve as input to a quasi 2D seismic model.
    They are then functions of an integer input "i" that parameterizes the elastic properties in the "lateral"
    direction
    """
    import matplotlib as mpl
    from typing import Callable
    def __init__(self,
                 name: str | None = None,
                 vp: float | pint.Quantity | Callable[[int], Q_] | None = None,
                 vs: float | pint.Quantity | Callable[[int], Q_] | None = None,
                 rho: float | pint.Quantity | Callable[[int], Q_] | None = None,
                 vp_std_dev: float | pint.Quantity | None = None,
                 vs_std_dev: float | pint.Quantity | None = None,
                 rho_std_dev: float | pint.Quantity | None = None,
                 vp_vs_cc: float | pint.Quantity | None = None,
                 vp_rho_cc: float | pint.Quantity | None = None,
                 vs_rho_cc: float | pint.Quantity | None = None,
                 default: str | None = None
                 ):
        """

        :param name:
        :param vp:
            The sonic speed, pressure velocity, of the lithofluid
            It can be given as a function (FunctionType) which is useful when using a LithoFluid as input to a
            quasi 2D model to describe a velocity change in the lateral direction (Vp is parameterized by an index)
            E.G.
                vp(_i) = pint.Quantity(3000. + _i *5), 'm/s')
        :param vs:
        :param rho:
        :param vp_std_dev:
        :param vs_std_dev:
        :param rho_std_dev:
        :param vp_vs_cc:
        :param vp_rho_cc:
        :param vs_rho_cc:
        :param default:
        """
        from typing import Callable

        self.name = name
        if isinstance(vp, Callable) or isinstance(vs, Callable) or isinstance(rho, Callable):
            self.type = 'quasi 2D'
        else:
            self.type = '1D'

        # Avoid integer input
        vp = float(vp) if isinstance(vp, int) else vp
        vs = float(vs) if isinstance(vs, int) else vs
        rho = float(rho) if isinstance(rho, int) else rho
        vp_std_dev = float(vp_std_dev) if isinstance(vp_std_dev, int) else vp_std_dev
        vs_std_dev = float(vs_std_dev) if isinstance(vs_std_dev, int) else vs_std_dev
        rho_std_dev = float(rho_std_dev) if isinstance(rho_std_dev, int) else rho_std_dev



        # Give default units and values (of shale) if not specified
        self.vp = Q_(vp, 'm/s') if isinstance(vp, float) else (Q_(3445, 'm/s') if vp is None else vp)
        self.vs = Q_(vs, 'm/s') if isinstance(vs, float) else (Q_(1767, 'm/s') if vs is None else vs)
        self.rho = Q_(rho, 'g/cm**3') if isinstance(rho, float) else (Q_(2.6, 'g/cm**3') if rho is None else rho)
        self.vp_std_dev = Q_(vp_std_dev, 'm/s') if isinstance(vp_std_dev, float) else (Q_(208, 'm/s') if vp_std_dev is None else vp_std_dev)
        self.vs_std_dev = Q_(vs_std_dev, 'm/s') if isinstance(vs_std_dev, float) else (Q_(171, 'm/s') if vs_std_dev is None else vs_std_dev)
        self.rho_std_dev= Q_(rho_std_dev, 'g/cm**3') if isinstance(rho_std_dev, float) else (Q_(0.036123123, 'g/cm**3') if rho_std_dev is None else rho_std_dev)
        self.vp_vs_cc = Q_(vp_vs_cc) if isinstance(vp_vs_cc, float) else (Q_(0.89) if vp_vs_cc is None else vp_vs_cc)
        self.vp_rho_cc = Q_(vp_rho_cc) if isinstance(vp_rho_cc, float) else (Q_(0.08) if vp_rho_cc is None else vp_rho_cc)
        self.vs_rho_cc = Q_(vs_rho_cc) if isinstance(vs_rho_cc, float) else (Q_(-0.07) if vs_rho_cc is None else vs_rho_cc)

        if default is not None:
            if default == 'shale':
                self.name = default
                self.vp = Q_(3445, 'm/s')
                self.vs = Q_(1767, 'm/s')
                self.rho = Q_(2.6, 'g/cm**3')
                self.vp_std_dev = Q_(208, 'm/s')
                self.vs_std_dev = Q_(171, 'm/s')
                self.rho_std_dev = Q_(0.036123123, 'g/cm**3')
                self.vp_vs_cc = Q_(0.89)
                self.vp_rho_cc = Q_(0.08)
                self.vs_rho_cc = Q_(-0.07)
            elif default == 'brine_sst':
                self.name = default
                self.vp = Q_(3675, 'm/s')
                self.vs = Q_(2086, 'm/s')
                self.rho = Q_(2.4, 'g/cm**3')
                self.vp_std_dev = Q_(260, 'm/s')
                self.vs_std_dev = Q_(188, 'm/s')
                self.rho_std_dev = Q_(0.092, 'g/cm**3')
                self.vp_vs_cc = Q_(0.92)
                self.vp_rho_cc = Q_(0.62)
                self.vs_rho_cc = Q_(0.65)
            elif default == 'oil_sst':
                self.name = default
                self.vp = Q_(3587, 'm/s')
                self.vs = Q_(2097, 'm/s')
                self.rho = Q_(2.3, 'g/cm**3')
                self.vp_std_dev = Q_(287, 'm/s')
                self.vs_std_dev = Q_(187, 'm/s')
                self.rho_std_dev = Q_(0.098, 'g/cm**3')
                self.vp_vs_cc = Q_(0.91)
                self.vp_rho_cc = Q_(0.58)
                self.vs_rho_cc = Q_(0.63)
            elif default == 'gas_sst':
                self.name = default
                self.vp = Q_(3535, 'm/s')
                self.vs = Q_(2127, 'm/s')
                self.rho = Q_(2.26, 'g/cm**3')
                self.vp_std_dev = Q_(307, 'm/s')
                self.vs_std_dev = Q_(181, 'm/s')
                self.rho_std_dev = Q_(0.12, 'g/cm**3')
                self.vp_vs_cc = Q_(0.89)
                self.vp_rho_cc = Q_(0.49)
                self.vs_rho_cc = Q_(0.58)
            else:
                raise IOError('default must be either "shale", "brine_sst", "oil_sst" or "gas_sst", not {}'.format(default))


    def to_sums_dict(self, index: int | None = None) -> dict:
        """
        Returns a dictionary which is compatible with the RokDoc "sums and average" file format
        :return:
        """
        from typing import Callable
        if isinstance(self.vp, Callable):
            if index is None:
                index = 0
            return_dict = {'VpMean': self.vp(index).to('m/s').magnitude,
                           'VsMean': self.vs(index).to('m/s').magnitude,
                           'RhoMean': self.rho(index).to('g/cm**3').magnitude,
                           'VpStdDev': self.vp_std_dev.to('m/s').magnitude,
                           'VsStdDev': self.vs_std_dev.to('m/s').magnitude,
                           'RhoStdDev': self.rho_std_dev.to('g/cm**3').magnitude,
                           'VpVsCorrCoef': self.vp_vs_cc.magnitude,
                           'VpRhoCorrCoef': self.vp_rho_cc.magnitude,
                           'VsRhoCorrCoef': self.vs_rho_cc.magnitude}
        else:
            return_dict = {'VpMean': self.vp.to('m/s').magnitude,
                           'VsMean': self.vs.to('m/s').magnitude,
                           'RhoMean': self.rho.to('g/cm**3').magnitude,
                           'VpStdDev': self.vp_std_dev.to('m/s').magnitude,
                           'VsStdDev': self.vs_std_dev.to('m/s').magnitude,
                           'RhoStdDev': self.rho_std_dev.to('g/cm**3').magnitude,
                           'VpVsCorrCoef': self.vp_vs_cc.magnitude,
                           'VpRhoCorrCoef': self.vp_rho_cc.magnitude,
                           'VsRhoCorrCoef': self.vs_rho_cc.magnitude}
        return return_dict



    def from_excel(self,
                   excel_file: str,
                   name: str,
                   avg_type: str | None = None):
        """
        Reads litho fluid properties from a RokDoc 'sums and averages' excel file
        :param excel_file:
        :param name:
        :param avg_type:
        :return:
        """
        from blixt_utils.io.io import read_sums_and_averages
        postfix = ''
        if avg_type is None:
            avg_type = 'mean'
        if avg_type == 'mean':
            postfix = 'Mean'
        elif avg_type == 'median':
            postfix = 'Median'
        elif avg_type == 'mode':
            postfix = 'Mode'
        else:
            raise IOError('avg_type can be either "mean", "median" or "mode", not {}'.format(avg_type))

        all_data = read_sums_and_averages(excel_file)

        self.name = name
        self.vp = Q_(all_data[name]['Vp'+postfix], 'm/s')
        self.vs = Q_(all_data[name]['Vs'+postfix], 'm/s')
        self.rho= Q_(all_data[name]['Rho'+postfix], 'g/cm**3')
        self.vp_std_dev = Q_(all_data[name]['VpStdDev'], 'm/s')
        self.vs_std_dev = Q_(all_data[name]['VsStdDev'], 'm/s')
        self.rho_std_dev= Q_(all_data[name]['RhoStdDev'], 'g/cm**3')
        self.vp_vs_cc = Q_(all_data[name]['VpVsCorrCoef'])
        self.vp_rho_cc = Q_(all_data[name]['VpRhoCorrCoef'])
        self.vs_rho_cc= Q_(all_data[name]['VsRhoCorrCoef'])

    def plot(self,
             ax = None,
             label: str | None = None,
             color: str | None = None):
        """
        Plot this litho fluid in a AI vs Vp/Vs plot with its statistics
        :param ax:
            matplotlib axes
        :param label:
        :param color:
        :return:

        """
        import matplotlib.pyplot as plt
        import blixt_rp.rp_utils.avo_monte_carlo as havo
        from blixt_utils.plotting import helpers as uhelp
        from blixt_utils.plotting import crossplot as xp
        if label is None:
            label = self.name
        if color is None:
            color = 'b'
        if ax is None:
            fig, ax = plt.subplots()

        n_samps = 1000
        vp, vs, rho = havo.elastics_from_stats(self.to_sums_dict(), n_samps)

        x_data = vp * rho
        x_unit = 'g/cc m/s'
        y_data = vp / vs
        y_unit = '-'
        xtempl = {'full_name': 'AI', 'unit': x_unit}
        ytempl = {'full_name': 'Vp/Vs', 'unit': y_unit}


        _ = xp.plot(
            x_data,
            y_data,
            cdata=color,
            cbar=False,
            xtempl=xtempl,
            ytempl=ytempl,
            edge_color=False,
            ax=ax
        )
        uhelp.confidence_ellipse(x_data, y_data, ax, n_std=1.0, edgecolor='k')
        uhelp.confidence_ellipse(x_data, y_data, ax, n_std=2.0, edgecolor='k', linestyle='--')

    def to_model_layer(self,
                       name: str | None = None,
                       case: str | None = None,
                       color: str | None = None,
                       thickness: float | pint.Quantity = Q_(25., 'm')):
        from blixt_rp.core.models import Layer
        if name is None:
            name = self.name
        if isinstance(thickness, float):
            thickness = Q_(thickness, 'm')
        return Layer(name, case, color, thickness, self)

    def fluid_sub(self):
        # TODO
        # new method that should return a new LithoFluid where the vp, vs and rho values are updated from the
        # original using a given fluid substitution
        pass


class LithoFluids:
    """
    Class containing the properties of the different litho fluids that can be modelled
    """
    def __init__(self,
                 litho_fluids: list | None = None
                 ):
        if litho_fluids is None:
            litho_fluids = []
        self._litho_fluids = litho_fluids

    @property
    def litho_fluids(self):
        return self._litho_fluids

    @litho_fluids.setter
    def litho_fluids(self, value: list):
        self._litho_fluids = value

    def __add__(self, other: LithoFluid):
        self._litho_fluids.append(other)

    def __len__(self):
        return len(self.litho_fluids)

    def __getitem__(self, item):
        for _lf in self.litho_fluids:
            if _lf.name.lower() == item.lower():
                return _lf
        print_info('No litho fluid named {} is found'.format(item), 'warning', logger)
        return None

    def from_excel(self,
                   excel_file: str,
                   avg_type: str | None = None):
        """
        Reads litho fluid properties from a RokDoc 'sums and averages' excel file, and adds them to the list of
        LithoFluids
        :param excel_file:
        :param avg_type:
        :return:
        """
        from blixt_utils.io.io import read_sums_and_averages
        postfix = ''
        if avg_type is None:
            avg_type = 'mean'
        if avg_type == 'mean':
            postfix = 'Mean'
        elif avg_type == 'median':
            postfix = 'Median'
        elif avg_type == 'mode':
            postfix = 'Mode'
        else:
            raise IOError('avg_type can be either "mean", "median" or "mode", not {}'.format(avg_type))

        all_data = read_sums_and_averages(excel_file)

        all_litho_fluids = []
        for _name in list(all_data.keys()):
            all_litho_fluids.append(
                LithoFluid(
                    name=_name,
                    vp=Q_(all_data[_name]['Vp'+postfix], 'm/s'),
                    vs=Q_(all_data[_name]['Vs'+postfix], 'm/s'),
                    rho=Q_(all_data[_name]['Rho'+postfix], 'g/cm**3'),
                    vp_std_dev=Q_(all_data[_name]['VpStdDev'], 'm/s'),
                    vs_std_dev=Q_(all_data[_name]['VsStdDev'], 'm/s'),
                    rho_std_dev=Q_(all_data[_name]['RhoStdDev'], 'g/cm**3'),
                    vp_vs_cc=Q_(all_data[_name]['VpVsCorrCoef']),
                    vp_rho_cc=Q_(all_data[_name]['VpRhoCorrCoef']),
                    vs_rho_cc=Q_(all_data[_name]['VsRhoCorrCoef'])
                )
            )
            self.litho_fluids = all_litho_fluids


class LithoFluidsTable:
    """
    Returns a table of "litho-fluids" that are useful for populating a model
    """
    def __init__(self,
                 litho_fluids: LithoFluids | None = None,
                 advanced: bool = True,
                 width: int | None = None,
                 height: int | None = None):
        """

        :param litho_fluids:
            LithoFluids object
        :param width:
            int
        """
        if width is None:
            width = 700
        self.width = width
        if height is None:
            height = 135
        self.height = height
        if litho_fluids is None:
            litho_fluids = LithoFluids()
        self.litho_fluids = litho_fluids
        if advanced:
            self.keys = ['name', 'vp', 'vs', 'rho',
                         'vp_std_dev', 'vs_std_dev', 'rho_std_dev',
                         'vp_vs_cc', 'vp_rho_cc', 'vs_rho_cc']
        else:
            self.keys = ['name', 'vp', 'vs', 'rho']
        self.advanced = advanced
        self.extract_index = 0

    @property
    def cds(self) -> ColumnDataSource:
        from typing import Callable
        _dict = {_x:[] for _x in self.keys}
        for _litho_fluid in self.litho_fluids.litho_fluids:
            for _key in self.keys:
                if _key == 'name':
                    _dict['name'].append(_litho_fluid.name)
                else:
                    if isinstance(_litho_fluid.__dict__[_key], Callable):
                        _dict[_key].append(_litho_fluid.__dict__[_key](self.extract_index).magnitude)
                    else:
                        _dict[_key].append(_litho_fluid.__dict__[_key].magnitude)
        return ColumnDataSource(_dict)

    @property
    def litho_fluid_names(self):
        _litho_fluid_names = []
        for _litho_fluid in self.litho_fluids.litho_fluids:
            _this_name = _litho_fluid.name
            if _this_name not in _litho_fluid_names:
                _litho_fluid_names.append(_this_name)
        return _litho_fluid_names

    def table_columns(self):
        from bokeh.models import (NumberEditor, NumberFormatter, StringEditor, TableColumn, StringFormatter)

        # HTML template to mark cells with functions. Uses the color_flag field
        template = """
            <div style="<%= color_flag %>">
                <%= value %>
            </div>
        """

        column_names = [_s.capitalize().replace('_', ' ') for _s in self.keys]
        table_columns = []
        for i, column_key in enumerate(self.keys):
            if column_key == 'name':
                _editor = StringEditor()
                _formatter = StringFormatter(font_style='bold')
            else:
                _editor = NumberEditor()
                _formatter = NumberFormatter(format='0.0[0]')
            table_columns.append(
                TableColumn(
                    field=column_key,
                    title=column_names[i],
                    editor=_editor,
                    formatter=_formatter

                )
            )

        return table_columns

    def draw(self,
             cds: ColumnDataSource):
        from bokeh.models import DataTable, Button, Div

        def add_row_function():
            new_data = dict(cds.data)
            lf = LithoFluid(default='brine_sst')
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append('brine sst')
                else:
                    new_data[_key].append(lf.__dict__[_key])
            cds.data = new_data

        def delete_row_function():
            selected_index = cds.selected.indices
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(cds.data['name'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        def update_table_function():
            from typing import Callable
            new_data = dict(cds.data)
            _litho_fluids = []
            for _i in range(len(new_data['name'])):
                # print(' - Iteration: {} of {}'.format( _i, len(new_data['name'])))
                _dict = {}
                if self.advanced:
                    keys = ['vp', 'vs', 'rho', 'vp_std_dev', 'vs_std_dev', 'rho_std_dev', 'vp_vs_cc', 'vs_rho_cc']
                    units = ['m/s', 'm/s', 'gram/cm^3', 'm/s', 'm/s', 'gram/cm^3', '', '', '']
                else:
                    keys = ['vp', 'vs', 'rho']
                    units = ['m/s', 'm/s', 'gram/cm^3']
                if _i < len(self.litho_fluids): # Within the number of original LFs in self.litho_fluids
                    # Check if there are any parameters that are functions
                    for _key, _unit in zip(keys, units):
                        old_param = self.litho_fluids.litho_fluids[_i].__dict__[_key]
                        if isinstance(old_param, Callable):
                            # When it is a function, don't update the litho-fluid
                            print('Litho-fluid parameter {} on row {} is a function. No update'.format(
                                _key, _i))
                            _dict[_key] = old_param
                            new_data[_key][_i] = old_param(self.extract_index).magnitude
                        else:
                            # When the old_param is a constant, replace it with the value extracted from the table
                            _dict[_key] = Q_(new_data[_key][_i], _unit)
                else:  # Outside the boundary of the original set of lithofluids
                    _dict = {_key: Q_(new_data[_key][_i], _unit) for _key, _unit in zip(keys, units)}

                this_litho_fluid = LithoFluid(name=new_data['name'][_i], **_dict)
                _litho_fluids.append(this_litho_fluid)

            self.litho_fluids.litho_fluids = _litho_fluids

            cds.data = new_data

        title = Div(text =
                    """
                    <div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">
                        Litho-fluids
                    </div>
               """)

        dt = DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
            index_position = -1,
            index_header = 'index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_function)

        return column(title, dt, sizing_mode='stretch_width'), add_row, delete_row, update_table


def cutoffs_string(cutoffs_list):
    msk_str = ''
    for key in cutoffs_list:
        msk_str += '{}: {} [{}]'.format(
            key.param, key.operator, ', '.join([str(m) for m in key.limit])) if \
            isinstance(key.limit, list) else \
            '{}: {} {}, '.format(
                key.param, key.operator, key.limit)
    if (len(msk_str) > 2) and (msk_str[-2:] == ', '):
        msk_str = msk_str.rstrip(', ')
    return msk_str


def templates_from_table(table: pd.DataFrame | dict, well_style=False) -> dict:
    """
    Returns a dictionary of logtype: template_dict
    :param table:
        pandas.DataFrame
        E.G.
            table = pd.read_excel(filename, header=1, sheet_name='Templates', engine='openpyxl')
    :param well_style:
        bool
        If true, read the style settings for the wells, and not for the logs
    :return:
        dict
    """
    return_dict = {}
    if well_style:
        for i, ans in enumerate(table['Given well name']):
            if not isinstance(ans, str):
                continue
            _ans = ans.upper().strip()
            return_dict[_ans] = {}
            return_dict[_ans]['full_name'] = _ans
            return_dict[_ans]['name'] = _ans
            return_dict[_ans]['well'] = _ans
            return_dict[_ans]['fill_color'] = None if isnan(table['Color'][i]) else table['Color'][i]
            return_dict[_ans]['marker'] = None if isnan(table['Symbol'][i]) else table['Symbol'][i]
        return return_dict

    for i, ans in enumerate(table['Log type']):
        if not isinstance(ans, str):
            continue
        return_dict[ans] = {}
        return_dict[ans]['full_name'] = ans
        return_dict[ans]['units'] = None if isnan(table['unit'][i]) else table['unit'][i]
        return_dict[ans]['min'] = None if isnan(table['min'][i]) else table['min'][i]
        return_dict[ans]['max'] = None if isnan(table['max'][i]) else table['max'][i]
        return_dict[ans]['center'] = None if isnan(table['center'][i]) else table['center'][i]
        return_dict[ans]['colormap'] = None if isnan(table['colormap'][i]) else table['colormap'][i]
        return_dict[ans]['bounds'] = None if isnan(table['bounds'][i]) else table['bounds'][i]
        return_dict[ans]['line_color'] = None if isnan(table['line color'][i]) else table['line color'][i]
        return_dict[ans]['line_style'] = None if isnan(table['line style'][i]) else table['line style'][i]
        return_dict[ans]['line_width'] = None if isnan(table['line width'][i]) else table['line width'][i]
        return_dict[ans]['marker'] = None if isnan(table['marker'][i]) else table['marker'][i]
    return return_dict


def print_function(my_object):
    keys = list(my_object.__dict__.keys())
    try:
        i = max([len(k) for k in keys])
    except ValueError:
        # no keys
        return ''
    pattern = "%%%ds: %%s" % i
    head = [pattern % (k, str(my_object.__dict__[k])) for k in keys]
    return "\n".join(head)


def get_level_from_name(_name: str, source: str | None = None) -> int:
    if source is None:
        source = 'sodir'  # Norwegian "Sokkel direktoratet"
    this_level = 2
    if source == 'sodir':
        # Try to extract level based on name of interval
        if ' gp' in _name.lower():
            this_level = 0
        elif ' fm' in _name.lower():
            this_level = 1
    return this_level
