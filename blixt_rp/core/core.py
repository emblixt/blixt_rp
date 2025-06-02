"""
Collection of objects and methods used throughout blixt_rp

"""
import os, sys

import numpy as np

from .. import ureg, Q_
import pint
import pandas as pd
import logging

from bokeh.plotting import figure
from datetime import datetime

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.utils import print_info, add_one, fix_well_name, cycle_colors, isnan
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info

logger = logging.getLogger(__name__)


class LogTable(dict):
    def __init__(self,
                 # name: str | None = None,
                 log_table: dict | None = None):
        """
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
        we also support the "multi log" log table, if you want to access specific, but multiple, logs for
        a log type. Notice that these can NOT be inverted.
                log_table = {
                   'P velocity': ['vp_virg', 'vp_brine', 'vp_oil', 'vp_gas'],
                   'S velocity': ['vs_virg', 'vs_brine', 'vs_oil', 'vs_gas'],
                   'Density': ['rhob_virg', 'rhob_brine', 'rhob_oil', 'rhob_gas']}

        :param name:
        :param log_table:
            dict
        """
        self.multi_log = False
        # self.name = name
        if log_table is not None:
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

    # TODO Create function to build a LogTable from the output 'logs' of result = uio.project_wells_new()


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
                 limit: pint.Quantity | list | str ):
        """

        :param param:
            name of the parameter
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
        """
        raise_unit_error = False
        self.interval_cutoff = False

        # instantiate 'param'
        self.__name__ = param
        self.param = param

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


class Cutoffs:
    def __init__(self,
                 # name: str | None = None,
                 cutoffs: list | None = None,
                 log_table: LogTable | None = None
                 ):
        """

        :param name:
        :param log_table:
        :param cutoffs:
            List of CutOffRules
        """
        # self.name = name
        self.log_table = log_table
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
    def cutoff_names(self):
        return [_x.param for _x in self.cutoffs]

    def __len__(self):
        return len(self.cutoffs)

    def __str__(self):
        return ', '.join([str(m) for m in self.cutoffs])

    def append(self, new_cutoffs):
        if isinstance(new_cutoffs, list):
            for co in new_cutoffs:
                if co.param in self.cutoff_names:
                    warn_txt = '{} is repeated and last occurrence is ignored'.format(co.param)
                    print_info(warn_txt, 'warning', logger=logger)
            self.cutoffs = self.cutoffs + new_cutoffs
        elif isinstance(new_cutoffs, CutoffRule):
            if new_cutoffs.param in self.cutoff_names:
                warn_txt = '{} is repeated and last occurrence is ignored'.format(new_cutoffs.param)
                print_info(warn_txt, 'warning', logger=logger)
            self.cutoffs.append(new_cutoffs)

    def get_dict(self):
        return_dict = {}
        for rule in self.cutoffs:
            this_list = [rule.operator]
            if isinstance(rule.limit, pint.Quantity):
                this_list.append(rule.limit.magnitude)
            elif isinstance(rule.limit, list):
                this_list.append([rule.limit[0].magnitude, rule.limit[1].magnitude])
            return_dict[rule.param] = this_list
        return return_dict


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

    def get_as_dict(self):
        return {self.name: self.__dict__}


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

    def well_names(self):
        return list(set([_x.well for _x in self.intervals]))

    def get_interval(self, interval_name, well_name):
        for _interval in self.intervals:
            if _interval.name == interval_name and _interval.well == well_name:
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
                           well_name: str | None = None) -> dict:
        """
        Returns a dictionary with all intervals
        if well_name is not None, it filters out all other wells
        :return:
        """
        # intervals_dict = {'well': [], 'name': [], 'top MD [m]': [], 'base MD [m]': [],
        intervals_dict = {'well': [], 'name': [], 'top': [], 'base': [],
                          'level': [], 'color': [], 'source': [], 'note': []}
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
            intervals_dict['note'].append(_interval.desc)
        return intervals_dict

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
            return_dict[_ans]['full_name'] = ans
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
