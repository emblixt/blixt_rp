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


class SingleInterval(object):
    """
    Class for handling the top and base values of an interval for one well
    All distances (in MD, TVD, OWT, TWT) are counted positive downwards, like MD. So a top at 1000 m (MD or TVD) lies
    above a top at 2000 m, and similar for OWT or TWT (ms)
    """
    def __init__(self, uid: str, well: str, name: str, top: pint.Quantity, base: pint.Quantity, coord_type='md'):
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
        self.uid = uid
        self.well = well
        self.name = name
        self.top = handle_coords(top, coord_units=None, coord_type=coord_type)
        self.base = handle_coords(base, coord_units=None, coord_type=coord_type)

        if self.top > self.base:
            raise ValueError('Top ({:.2}) must be smaller than Base ({:.2})'.format(
                self.top.magnitude, self.base.magnitude))

    def __str__(self):
        return print_function(self)

    @property
    def thickness(self):
        return self.base - self.top

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

    def keys(self):
        return self.__dict__.keys()


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

    def interval_uids(self):
        """
        List of unique names of all intervals
        :return:
        """
        return list(self.intervals.keys())

    def interval_names(self):
        return [_int.name for _int in self.intervals.items()]

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

    def plot_intervals(self, level, ax, depth_range=[1000., 3000.]):
        ccl = cycle_colors()
        _y = np.linspace(depth_range[0], depth_range[1], 1000)
        ax.plot(np.ones(len(_y)), _y, lw=0)
        for i, interval in enumerate(self.intervals.items()):
            ax.axhline(y=interval.top.magnitude, color='k', lw=0.5)
            ax.axhline(y=interval.base.magnitude, color='k', lw=0.5)
            aboves = _y > interval.top.magnitude
            belows = _y < interval.base.magnitude
            selection = aboves & belows
            ax.fill_betweenx(_y, np.ones(len(_y)), where=selection, color=next(ccl))


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
        :param intervals:
            dict
            Dictionary with information about each uniw
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

    def get_interval(self, item):
        return self.intervals.__getitem__(item)

    def interval_names(self):
        return list(self.intervals.keys())

    def add_interval(self, interval: IntervalInfo):
        if interval.name in self.interval_names():
            raise NotImplementedError('Overwrite or append to functionality not implemented yet')
        self.intervals.__setitem__(interval.name, interval)

    def get_well(self, item):
        return self.wells.__getitem__(item)

    def well_names(self):
        return list(self.wells.keys())

    def add_well(self, well: Well):
        if well.name in self.well_names():
            raise NotImplementedError('Overwrite or append to functionality not implemented yet')
        self.wells.__setitem__(well.name, well)

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

            _interval = SingleInterval(_name, well_name, _name,
                                       Q_(float(_top), 'm'), Q_(float(_base), 'm'))

            if _interval.name not in self.interval_names():
                _interval_info = IntervalInfo(_name, level=l, source='sodir')
                self.add_interval(_interval_info)

            if well_name in self.well_names():
                well = self.get_well(well_name)
                well.add_interval(_interval)
            else:
                self.add_well(Well(well_name, {}))

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


