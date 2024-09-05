"""
Class to handle (working) intervals and tops
"""
import numpy as np
import sys
import os
import logging
import pint

from .. import ureg, Q_

logger = logging.getLogger(__name__)

# Add to path to avoid having to install libraries, useful in development
project_dir = os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core','')
sys.path.append(os.path.join(str(project_dir), 'blixt_utils'))

from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.core.log_curve_new import handle_coords
from blixt_utils.utils import print_info, add_one, fix_well_name, cycle_colors


class IntervalInfo(object):
    """
    Contains information about one specific interval (Group, Formation, Member, ...)
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
            High number indicates that high up in the hierarchy (e.g. "Era" or "Group") while lower (even negative)
            numbers indicate that is a sub interval (e.g. "Stage" or "Member")
            Set it to None when unknown or uncertain
        :param desc:
        :param source:
            str
            E.G. <User name>, or  "sodir", ...
        :param color:
        :param wells:
            dict
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


class SingleInterval(IntervalInfo):
    """
    Class for handling the top and base values of an interval for one well
    All distances (in MD, TVD, OWT, TWT) are counted positive downwards, like MD. So a top at 1000 m (MD or TVD) lies
    above a top at 2000 m, and similar for OWT or TWT (ms)
    """
    def __init__(self, name: str, level: int,
                 top: pint.Quantity, base: pint.Quantity,
                 desc=None, source=None, color=None,
                 uid=None, well=None,
                 coord_type='md'):
        """

        :param uid:
            str
            Unique name for this interval in this well
            (an interval, like Spekk Fm, can occur several times in a well)
        :param name:
            str
            Interval name, e.g. "Spekk Fm"
        :param well:
            str
            Name of well it belongs to
        :param top:
            float or pint.Quantity
        :param base:
            float or pint.Quantity
        :param units:
            str
        :param coord_type:
            str
        """
        super().__init__(name, level, desc, source, color)

        if uid is None:
            uid = name
        self.uid = uid
        self.well = well
        self.top = handle_coords(top, coord_units=None, coord_type=coord_type)
        self.base = handle_coords(base, coord_units=None, coord_type=coord_type)

        if self.top > self.base:
            raise ValueError('Top ({:.2}) must be smaller than Base ({:.2})'.format(
                self.top.magnitude, self.base.magnitude))

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


class Well(object):
    """
    Contains information about which intervals that exists in one well
    """
    def __init__(self,
                 name: str,
                 intervals: dict):
        if intervals is None:
            intervals = {}
        self.intervals = intervals
        self.name = name

    def __len__(self):
        return len(self.intervals)

    def __str__(self):
        return print_function(self)

    @property
    def get_levels(self):
        min_level = 1E6
        max_level = -1E6
        for uid in self.interval_uids():
            this_level = self.intervals[uid].level
            if this_level <= min_level:
                min_level = this_level
            elif this_level >= max_level:
                max_level = this_level
        return min_level, max_level

    @property
    def get_depth_range(self):
        min_top = 1E6
        max_base = -1E6
        for uid in self.interval_uids():
            this_top = self.intervals[uid].top.magnitude
            this_base = self.intervals[uid].base.magnitude
            if this_top <= min_top:
                min_top = this_top
            elif this_base >= max_base:
                max_base = this_base
        return min_top, max_base

    def interval_uids(self):
        """
        List of unique names of all intervals
        :return:
        """
        return list(self.intervals.keys())

    def interval_names(self):
        return [_int.name for _int in self.intervals.values()]

    def add_interval(self, interval: SingleInterval):
        if interval.uid in self.interval_uids():
            print_info('Interval {}, with UID {}, already exist in {}. Need to update UID'.format(
                interval.name, interval.uid, self.name), 'info', logger)
            interval.uid = add_one(interval.uid)
        if interval.well != self.name:
            print_info('Interval {} belongs to well {}. Not {}. Skipping'.format(
                interval.name, interval.well, self.name), 'warning', logger)
            return

        self.intervals.__setitem__(interval.uid, interval)

    def get_interval(self, interval_name):
        return self.intervals.__getitem__(interval_name)

    def plot_intervals(self, level, ax, depth_range=None):
        """

        :param level:
        :param ax:
        :param depth_range:
            two tuple

        :return:
        """
        ccl = cycle_colors()
        if depth_range is None:
            depth_range = self.get_depth_range
        _y = np.linspace(depth_range[0], depth_range[1], 1000)
        ax.plot(np.ones(len(_y)), _y, lw=0)
        i = 0
        for interval in self.intervals.values():
            if interval.level != level:
                continue
            if (depth_range[1] <= interval.mid.magnitude) or (interval.mid.magnitude <= depth_range[0]):
                continue
            top = interval.top.magnitude
            base = interval.base.magnitude
            ax.axhline(y=top, color='k', lw=0.5)
            ax.axhline(y=base, color='k', lw=0.5)
            aboves = _y > interval.top.magnitude
            belows = _y < interval.base.magnitude
            selection = aboves & belows
            ax.fill_betweenx(_y, np.ones(len(_y)), where=selection, color=next(ccl))
            ax.text((0.3 + np.mod(i, 2) * 0.4) * sum(ax.get_xlim()), interval.mid.magnitude, interval.name,
                    weight='bold', rotation='vertical', va='center')
            i += 1


class Intervals(object):
    """
    Object that holds a number of working intervals / tops for many wells
    """
    def __init__(self,
                 name=None,
                 desc=None,
                 wells=None,
                 intervals=None,
                 verbose=False):
        """

        :param name:
            str
        :param desc:
            str
        :param wells:
            dict
            Dictionary with "well_name: Well" as "key: value" pairs
        :param intervals:
            dict
            Dictionary with information about each unique interval with
            "interval_name: IntervalInfo" as "key: value" pairs
        :param verbose:
            bool
        """
        self.name = name
        self.desc = desc

        if wells is None:
            self.wells = {}
        elif isinstance(wells, dict):
            self.wells = wells
        else:
            raise IOError('Wells must be a dictionary (of wells)')

        if intervals is None:
            self.intervals = {}
        elif isinstance(intervals, dict):
            self.intervals = intervals
        else:
            raise IOError('Intervals must be a dictionary (of intervals)')

    def __len__(self):
        return len(self.intervals)

    def __str__(self):
        return print_function(self)

    def interval_names(self):
        return list(self.intervals.keys())

    def add_interval_info(self, interval: IntervalInfo):
        if interval.name in self.interval_names():
            raise NotImplementedError('Overwrite or append to functionality not implemented yet')
        self.intervals.__setitem__(interval.name, interval)

    def get_interval_info(self, interval_name):
        self.intervals.__getitem__(interval_name)

    def get_well(self, item):
        return self.wells.__getitem__(item)

    def well_names(self):
        return list(self.wells.keys())

    def add_well(self, well: Well):
        if well.name in self.well_names():
            raise NotImplementedError('Overwrite or append to functionality not implemented yet')
        self.wells.__setitem__(well.name, well)

    def add_single_interval(self, interval: SingleInterval):
        if interval.well not in self.well_names():
            self.add_well(Well(interval.well, {}))
        if interval.name not in self.interval_names():
            self.add_interval_info(IntervalInfo(interval.name, interval.level, desc=interval.desc,
                                                source=interval.source, color=interval.color))
        this_well = self.get_well(interval.well)
        this_well.add_interval(interval)

    def write_to_excel(self, file_name, intervals_sheet, interval_info_sheet):
        import pandas as pd
        # First collect and write the interval info
        interval_info_dict = {'name': [], 'level': [], 'desc': [], 'source': [], 'color': []}
        for i, interval in enumerate(self.intervals.values()):
            for _key in list(interval_info_dict.keys()):
                interval_info_dict[_key].append(interval.__getitem__(_key))
        df = pd.DataFrame(interval_info_dict)
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(
                writer, sheet_name=interval_info_sheet, startcol=0, startrow=0, index=False, header=True)

        # Then the individual intervals in each well
        intervals_dict = {'well': [], 'name': [], 'top MD [m]': [], 'base MD [m]': [],
                          'level': [], 'source': [], 'note': []}
        for well in self.wells.values():
            for uid in well.interval_uids():
                this_interval = well.get_interval(uid)
                intervals_dict['well'].append(well.name)
                intervals_dict['name'].append(this_interval.name)
                intervals_dict['level'].append(this_interval.level)
                intervals_dict['top MD [m]'].append(this_interval.top.magnitude)
                intervals_dict['base MD [m]'].append(this_interval.base.magnitude)
                intervals_dict['source'].append(this_interval.source)
                intervals_dict['note'].append(None)

        df = pd.DataFrame(intervals_dict)
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(
                writer, sheet_name=intervals_sheet, startcol=0, startrow=0, index=False, header=True)

        return intervals_dict

    def read_sodir_tops(self, file_name):
        import pandas as pd
        df = pd.read_excel(file_name, engine='openpyxl')
        for i, well_name in enumerate(df['Wellbore name']):
            well_name = fix_well_name(well_name)
            _top = df['Top depth [m]'][i]
            _base = df['Bottom depth [m]'][i]
            _name = df['Lithostrat. unit'][i]
            _level = df['Level'][i]
            if _level == 'GROUP':
                l = 3
            elif _level == 'FORMATION':
                l = 2
            else:
                l = 0

            _interval = SingleInterval(_name, l, Q_(float(_top), 'm'), Q_(float(_base), 'm'),
                                       uid=_name, well=well_name)

            self.add_single_interval(_interval)

            if i > 40:
                break


def print_function(my_object):
    keys = list(my_object.__dict__.keys())
    try:
        i = max([len(k) for k in keys])
    except ValueError:
        # no keys
        return ''
    pattern = "%%%ds: %%s" % (i)
    head = [pattern % (k, str(my_object.__dict__[k])) for k in keys]
    return "\n".join(head)


