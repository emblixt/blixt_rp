"""
Class to handle (working) intervals and tops
"""
import sys
import os

# Add to path to avoid having to install libraries, useful in development
project_dir = os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core','')
sys.path.append(os.path.join(str(project_dir), 'blixt_utils'))

from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.core.log_curve_new import handle_coords


class Intervals(object):
    """
    Object that holds a number of working intervals / tops for many wells
    """
    def __init__(self,
                 name=None,
                 desc=None,
                 intervals=None,
                 coord_type='md',
                 verbose=False):
        """

        :param name:
            str
        :param desc:
            str
        :param intervals:
            dict
        :param coord_type:
            str
            'md', 'tvd', 'twt', 'owt'
        :param verbose:
            bool
        """
        self.name = name
        self.desc = desc

        if intervals is None:
            self.intervals = {}
        elif isinstance(intervals, dict):
            self.intervals = intervals
        else:
            raise IOError('Intervals must be a dictionary (of intervals)')

        if coord_type.lower() not in ['md', 'tvd', 'twt', 'owt']:
            raise IOError("coord_type must be either 'md', 'tvd', 'owt' or 'twt'")
        else:
            self.coord_type = coord_type.lower()

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, item):
        return self.intervals.__getitem__(item)

    def __setitem__(self, key, value):
        self.intervals.__setitem__(key, value)

    def __str__(self):
        keys = list(self.keys())

    def keys(self):
        return self.__dict__.keys()

    def interval_names(self):
        return list(self.intervals.keys())


class Interval(AttribDict):
    pass


class WellInterval(object):
    """
    Class for handling the top and base values of an interval for one well
    """
    def __init__(self, name, top, base, units='meter', coord_type='md'):
        """

        :param name:
        :param top:
        :param base:
        :param units:
        :param coord_type:
        """
        self.name = name
        self.top = handle_coords(top, coord_units=units, coord_type=coord_type)
        self.base = handle_coords(base, coord_units=units, coord_type=coord_type)

        if self.top > self.base:
            raise ValueError('Top ({:.2}) must be smaller than Base ({:.2})'.format(
                self.top.magnitude, self.base.magnitude))

    @property
    def thickness(self):
        return self.base - self.top