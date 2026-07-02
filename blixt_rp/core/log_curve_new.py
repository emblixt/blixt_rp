# -*- coding: utf-8 -*-
"""
Module for handling LogCurve objects
:copyright:
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import sys
import os.path
from datetime import datetime
import logging
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from pandas import DataFrame
import xarray as xr
import pint
# from pint import UnitRegistry
from math import isclose, ceil

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.core import Template, Cutoffs, LogTable
from blixt_rp.core.param import Param
from blixt_utils.signal_analysis.signal_analysis import smooth as _smooth
from blixt_utils.misc.curve_fitting import residuals, linear_function, calculate_depth_trend
from blixt_rp.rp_utils.definitions import allowed_depth_formats
from blixt_utils.misc.convert_data import convert
from blixt_utils.misc import masks as msks
from blixt_utils.misc import templates as tmplts
import blixt_rp.rp_utils.definitions as rud
from blixt_utils.io.io import read_general_ascii_GENERAL as read_file


from .. import ureg, Q_
import pint

logger = logging.getLogger(__name__)


class Depth(object):
    """
    Class handling depth or time data, that holds the "depth" of something along a well path
    It can either be 'md' or 'tvd' in depth domain, and 'twt' or 'owt' in time domain.
    Depth is counted positive downwards.
    """
    def __init__(self,
                 depth: float | np.ndarray | pint.Quantity,
                 units: str | None = None,
                 depth_type: str = 'md',
                 verbose: bool = False):
        """
        if depth is a pint.Quantity, units are ignored.
        if depth is not a pint.Quantity, and depth_units is None, an Error is raised

        :param depth:
            flt, int, np.ndarray or pint.Quantity
        :param units:
            str
            Any unit handled by pint.
            Extensions to pint default units should be added here:
            blixt_rp/units_to_pint.txt
            (jetbrains://pycharm/navigate/reference?project=PycharmProjects&path=blixt_rp/blixt_rp/units_to_pint.txt)
        :param depth_type:
            str
            'md', 'tvd', 'owt', or 'twt'
            'tvd' refers to True Vertical Depth relative to Kelly Bushing (tvd_kb)
        :param verbose:
        """

        if verbose:
            print(depth_type, units)
        _depth = handle_depth(depth, depth_units=units, depth_type=depth_type, verbose=verbose)

        self.depth = _depth
        self.depth_type = depth_type.lower()

    def __len__(self):
        return len(self.depth)

    @property
    def values(self):
        return self.depth.magnitude

    @property
    def magnitude(self):
        return self.depth.magnitude

    @property
    def units(self) -> pint.Unit:
        return self.depth.units

    @property
    def domain(self):
        if self.depth_type in ['md', 'tvd']:
            return 'depth'
        else:
            return 'time'

    @property
    def top(self) -> pint.Quantity:
        return self.depth[0]

    @property
    def base(self) -> pint.Quantity:
        return self.depth[-1]

    @property
    def mid(self) -> pint.Quantity:
        return 0.5 * (self.top + self.base)

    @property
    def is_evenly_spaced(self) -> bool:
        """
        Tests if the dimension of a LogCurve is evenly spaced

        :return:
            bool
            True if evenly sampled
        """
        arr = self.values
        if len(arr) <= 2:
            return True

        diffs = np.diff(arr)
        return np.allclose(diffs, diffs[0])

    def step(self, ignore_gaps: bool = False) -> pint.Quantity | None:
        """
        If evenly spaced it returns the step in depth/time, else None

        :param ignore_gaps
            bool
            Finds the most likely step (increment in depth/time) of log data.
            Useful if there are gaps in an evenly spaced data set
            (e.g. it has been cleaned to remove NaN's).
        :return:
        """
        if self.is_evenly_spaced:
            return self.depth[1] - self.depth[0]
        elif ignore_gaps:
            steps, counts = np.unique(np.diff(self.values), return_counts=True)
            return Q_(steps[np.argmax(counts)], self.units)
        else:
            return None

    def unique_depths(self, other):
        """
        Compare the depth values with the 'other' and returns a sorted Depth object with
        only the unique depth values
        """
        if not is_equivalent(self.units, other.units):
            raise IOError('Units of the two Depth objects are not the same: {}, {}'.format(
                self.units, other.units
            ))
        all_depth_values = np.append(self.values, other.values)

        unique_depth_values = np.array(list(set(list(all_depth_values))))
        unique_depth_values.sort()
        return Depth(Q_(unique_depth_values, self.units))


    # I'm trying to "protect" the depth_type from being updated, BUT that stops me from setting it in the first place!
    # def __setattr__(self, key, value):
    #     if key in ['depth_type']:
    #         pass
    #     elif key in ['depth']:
    #         self.__setattr__(key, handle_depth(value, depth_type=self.depth_type, depth_units=self.units))
    #     else:
    #         self.__setattr__(key, value)


class LogCurve(object):
    """
    This log curve object contains a log data & depth pair
    It uses pint to take care of units

    The units of the depth will be automatically converted to m or ms, depending on if the log is in
    depth or time

    If units are specified in the style, the LogCurve will automatically try to convert the log data units
    to this unit.
    So by specifying the style to:
    > style = {'units': 'km/s'}
    when the log data is given in 'm/s', it will be automatically converted to 'km/s'

    """
    from blixt_rp.core.core import Template, Header
    def __init__(self,
                 name: str,
                 log_data: pint.Quantity,
                 depth: Depth,
                 log_type: str | None = None,
                 well: str | None = None,
                 style: Template | dict | None = None,
                 header: Header | dict | None = None):
        """
        The basic building block of all well related data.
        Should hold both regularly sampled log curves and irregularly sampled core data

        :param name:
            str
            Name of log
        :param log_data:
            pint.Quantity
            np.ndarray inside a pint.Quantity
        :param depth:
            Depth
            Contains the depth data, either 'md', 'tvd', 'owt' or 'twt'

            Must be of same length as log_data
        :param log_type:
            str
            Name of the log type, e.g. "P velocity", as specified in the project table .xlsx file
        :param well:
            str
            Name of the well this data belongs to
        :param style:
            Template object or dict
        :param header:
            Header object or dict
        """
        from blixt_rp.core.core import Template, Header
        from blixt_utils.utils import print_info
        if len(log_data) != len(depth):
            raise IOError('Length of log_data ({}) must equal length of depth ({})'.format(
                len(log_data), len(depth)))
        if not isinstance(depth, Depth):
            raise IOError('Depth must be a Depth object')
        # self.name = name
        self._name = name
        self.data = log_data
        self.depth = depth
        # self.well = well
        self._well = well

        if style is None:
            style = Template()
        elif isinstance(style, dict):
            style = Template(**style)
        elif isinstance(style, Template):
            # This case is handled later in init()
            pass
        else:
            raise TypeError('style must be either a dict or a Template, not {}'.format(type(style)))

        if header is None:
            self.header = Header(header={})
        elif isinstance(header, dict):
            self.header = Header(header=header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise TypeError('header must be either a dict or a Header, not {}'.format(type(header)))

        # Make sure the values of 'name' and 'well' are aligned.
        if self._name is not None:
            self.header.name = self._name  # self.name wins over self.header.name
            style.name = self._name
        if self._well is not None:
            self.header.well = self._well  # self.well wins over self.header.well
            style.well = self._well

        if log_type is None:
            if self.header.log_type is not None:
                self.log_type = self.header.log_type
            else:
                self.log_type = None
        else:
            if (self.header.log_type is not None) and (self.header.log_type != log_type):
                warn_txt = 'Log type in header ({}), does not match given log type ({})'.format(
                    self.header.log_type, log_type)
                warn_txt += '\nFix inconsistency and try again'
                print_info(warn_txt, 'warning', logger)
                raise IOError(warn_txt)
            else:
                self.log_type = log_type
                if self.header.log_type is None:
                    self.header.log_type = log_type

        # If the style Template contains a unit, which is not None, then we should convert the data to this
        # unit if it is not already using those units.
        if style.units is not None:
            # print('XXX Units in template for {} is {}, while units in las file is {}'.format(name, style.units, self.data.units))
            if not is_equivalent(self.data.units, ureg.Unit(style.units)):
                # print(' XXX Units are not the same, try to convert')
                try:
                    info_txt = '{}: {}: Convert from {} to {}'.format(
                        self.well, self.name, str(self.data.units), style.units
                    )
                    self.data = self.data.to(style.units)
                    print_info(info_txt, 'info', logger)
                except pint.DimensionalityError:
                    warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                        self.well, self.name, str(self.data.units), style.units
                    )
                    print_info(warn_txt, 'warning', logger)
                    #  Then continue using the original unit
                    style.units = str(self.units)
        else:
            style.units = str(self.data.units)

        # If the style is 'borrowed' from another LogCurve, we need to handle the different scenarios:
        if self._name is None and style.name is None:  # Do nothing
            pass
        elif self._name is None:  # Take name from style
            self._name = style.name
        else:  # Rename the names given in the style
            style.name = self._name
            style.full_name = self._name

        self._style = style

    from blixt_rp.core.core import Template

    def __len__(self):
        return len(self.data)

    def __setattr__(self, key, value):
        # Wanted the units of the log curve to be updated if the style contain units, but the code doesn't work
        # if key == 'style':
        #     if 'units' in list(value.keys()):
        #         if value['units'] is not None:
        #             # self.units = value['units']
        #             print('SEEING UNITS')
        if key == 'log_type':
            self.header.log_type = value
        super().__setattr__(key, value)

    @property
    def name(self):
        if self._name is not None:
            self.header.name = self._name
        return self.header.name

    @name.setter
    def name(self, value):
        old_name = self._name
        self._name = value
        self.header.name = value
        self.header.modification_date = datetime.now().isoformat()
        self.header.modification_history += '\nLog name changed from {} to {}'.format(
            old_name, value)

    @property
    def well(self):
        if self._well is not None:
            self.header.well = self._well
        return self.header.well

    @well.setter
    def well(self, value):
        old_name = self._well
        self._well = value
        self.header.well = value
        self.header.modification_date = datetime.now().isoformat()
        self.header.modification_history += '\nWell name changed from {} to {}'.format(
            old_name, value)

    @property
    def top(self) -> pint.Quantity:
        return self.depth.depth[0]

    @property
    def base(self) -> pint.Quantity:
        return self.depth.depth[-1]

    @property
    def depth_type(self):
        return self.depth.depth_type

    @property
    def depth_units(self) -> pint.Unit:
        return self.depth.units

    @depth_units.setter
    def depth_units(self, to_units):
        """ Converts the depth units to 'to_units'"""
        from blixt_utils.utils import print_info
        try:
            cnv_txt = 'Convert depth from {} to {}'.format(
                str(self.depth.units), to_units
            )
            info_txt = '{}: {}: {}'.format(
                self.well, self.name, cnv_txt
            )
            self.depth.depth = self.depth.depth.to(to_units)
            print_info(info_txt, 'info', logger)
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += '\n{}'.format(cnv_txt)
        except pint.DimensionalityError:
            warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                self.well, self.name, str(self.depth.depth.units), to_units
            )
            print_info(warn_txt, 'warning', logger)

    @property
    def units(self) -> pint.Unit:
        return self.data.units

    @units.setter
    def units(self, to_units: str):
        """
        Converts the log data units to 'to_units', and changes the unit in the style too
        """
        from blixt_utils.utils import print_info
        try:
            cnv_txt = 'Convert from {} to {}'.format(
                str(self.data.units), to_units
            )
            info_txt = '{}: {}: {}'.format(
                self.well, self.name, cnv_txt
            )
            self.data = self.data.to(to_units)
            self.style.units = to_units
            print_info(info_txt, 'info', logger)
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += '\n{}'.format(cnv_txt)
        except pint.DimensionalityError:
            warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                self.well, self.name, str(self.data.units), to_units
            )
            print_info(warn_txt, 'warning', logger)

    @property
    def values(self):
        return self.data.magnitude

    @property
    def is_evenly_spaced(self):
        return self.depth.is_evenly_spaced

    @property
    def min(self):
        return np.nanmin(self.data.magnitude)

    @property
    def max(self):
        return np.nanmax(self.data.magnitude)

    @property
    def mean(self):
        return np.nanmean(self.data.magnitude)

    @property
    def median(self):
        return np.nanmedian(self.data.magnitude)

    @property
    def std(self):
        return np.nanstd(self.data.magnitude)

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style_template: Template | dict | None):
        from blixt_rp.core.core import Template
        if style_template is None:
            self._style = Template()
        elif isinstance(style_template, dict):
            self._style = Template(**style_template)
        elif isinstance(style_template, Template):
            self._style = style_template
        else:
            raise TypeError('style must be either a dict or a Template, not {}'.format(type(style_template)))
        if 'units' in list(self._style.keys()):
            if self._style.units is not None:
                self.units = self._style.units

    def step(self, ignore_gaps=False):
        return self.depth.step(ignore_gaps=ignore_gaps)

    def __lt__(self, other):
        if len(self.data) < len(other.data):
            return True
        else:
            return False

    def __le__(self, other):
        if len(self.data) <= len(other.data):
            return True
        else:
            return False

    def __gt__(self, other):
        if len(self.data) > len(other.data):
            return True
        else:
            return False

    def __eq__(self, other):
        if len(self.data) == len(other.data):
            return True
        else:
            return False

    def __ge__(self, other):
        if len(self.data) >= len(other.data):
            return True
        else:
            return False

    def __str__(self):
        return '{}: {}'.format(self.name, str(self.header))

    # def take_units_from_style(self):
    #     take_units_from_style(self)

    def copy(self, suffix='copy'):
        copied_log_curve = deepcopy(self)
        if suffix is not None and len(suffix) > 0:
            copied_log_curve.name = self.name + '_' + suffix
            copied_log_curve.header.name = self.name + '_' + suffix
        copied_log_curve.header.modification_date = datetime.now().isoformat()
        copied_log_curve.header.modification_history += \
            '\nCopy of {}'.format(self.name)
        return copied_log_curve

    def velocity_from_sonic(self, name: str | None = None):
        from blixt_utils.utils import print_info
        if self.log_type not in ['Sonic', 'Shear sonic']:
            warn_txt = 'Log type must be "Sonic" or "Shear sonic" to calculate velocity'
            print_info(warn_txt, 'warning', logger)
            return
        velocity = 1. / self.data.to('s/m')
        style = deepcopy(self.style)
        style.units = 'm/s'
        header = deepcopy(self.header)
        header.modification_date = datetime.now().isoformat()
        header.modification_history += '\nCalculated from sonic {}'.format(self.name)
        if self.log_type == 'Sonic':
            if name is None:
                name = 'Vp'
            log_type = 'P velocity'
            header.log_type = 'P velocity'
        else:
            if name is None:
                name = 'Vs'
            log_type = 'S velocity'
            header.log_type = 'S velocity'
        return LogCurve(
            name=name,
            log_data=velocity,
            depth=self.depth,
            log_type=log_type,
            well=self.well,
            style=style,
            header=header
        )

    def down_sample(self, step: int, suffix: str | None = 'down_sampled', verbose=False):
        """
        Interpolates the data to match the depth parameter (sampling) of input log_curve,
        and returns a new log_curve

        :param step:
            int
            Takes every 'step'th sample of the log curve.
            Similar to a[::step] in Python slicing notation
        :param suffix:
            str
        :param verbose:
            Bool

        :return
            LogCurve object
        """

        # Do the interpolation
        old_x = self.depth.values
        old_y = self.values
        new_x = old_x[::step]
        new_y = old_y[::step]


        header = deepcopy(self.header)
        header.modification_date = datetime.now().isoformat()
        header.modification_history += '\nDown sampled to every {}th sample based on {}'.format(step, self.name)


        if suffix is not None:
            new_name = self.name + '_' + suffix
        else:
            new_name = self.name
        return LogCurve(
            new_name,
            Q_(new_y, self.units),
            Depth(Q_(new_x, self.depth_units)),
            self.log_type,
            header=header,
            style=self.style
        )

    def make_evenly_spaced(self, step):
        """
        Interpolates the LogCurve so that it becomes evenly sampled with a step length of step
        :param step:
            pint Quantity
        :return:
            Nothing
            It modifies the original LogCurve
        """
        old_x = self.depth.values
        old_y = self.values

        # create a new depth parameter which is evenly sampled with given step size
        new_x = np.arange(
            self.depth.values[0],
            self.depth.values[-1],
            step.to(self.depth.units).magnitude
        )

        # Do the interpolation
        new_y = _interpolate(old_x, old_y, new_x)

        # Replace the original data in the LogCurve
        replace_data(self,
                     log_data=Q_(new_y, self.units),
                     depth=Depth(Q_(new_x, self.depth.units)),
                     overwrite=True,
                     info_txt='Resampled to match the given step {}'.format(str(step))
                     )


    def take_sampling_from(self, log_curve, suffix: str | None = 'resampled', verbose=False):
        """
        Interpolates the data to match the depth parameter (sampling) of input log_curve,
        and returns a new log_curve

        :param log_curve:
            LogCurve object
        :param suffix:
            str
        :param verbose:
            Bool

        :return
            LogCurve object
        """
        # TODO
        # Make sure this works correctly for log curves of different sampling and start and stop!
        if self.depth_type != log_curve.depth_type:
            raise ValueError('The two depth formats (depth_type) are not the same: {} != {}'.format(
                self.depth_type, log_curve.depth_type))

        # Do the interpolation
        old_x = self.depth.values
        old_y = self.values
        new_x = log_curve.depth.values
        if self.log_type in ['TVD', 'One-way time', 'Two-way time']:
            new_y = _interpolate(old_x, old_y, new_x, fill_value='extrapolate')
        else:
            new_y = _interpolate(old_x, old_y, new_x)

        header = deepcopy(self.header)
        header.modification_date = datetime.now().isoformat()
        header.modification_history += '\nResampled to match {}'.format(log_curve.name)

        if suffix is not None:
            new_name = self.name + '_' + suffix
        else:
            new_name = self.name
        return LogCurve(
            new_name,
            Q_(new_y, self.units),
            Depth(Q_(new_x, self.depth_units)),
            self.log_type,
            header=header,
            style=self.style
        )

    def to_twt(self, twt: pint.Quantity, dt: pint.Quantity, verbose: bool | None = False):
        """
        Returns a LogCurve object with a regularly sampled TWT as Depth, with the
        log curve data resampled to match the range of the input TWT
        :param twt:
            The twt data for the depth-time relation
        :param dt:
            The sample rate of the new twt
        :param verbose:
        :return:
            LogCurve object where Depth is a regularly sampled TWT array and with original LogCurve data resampled
            to match this new Depth
        """
        info_txt = 'Converting log {}, in {} domain to Time domain with a regularly sampled TWT Depth (dt: {})'.format(
            self.name, self.depth_type, dt)
        if verbose:
            print(info_txt)
            print(' Input twt: {}-{}'.format(np.nanmin(twt.magnitude), np.nanmax(twt.magnitude)))
        twt, data = _to_twt(twt, self.values, dt, verbose=verbose)
        if verbose:
            print(' Output twt: {}-{}'.format(np.nanmin(twt.magnitude), np.nanmax(twt.magnitude)))
            print(' Output data: {}-{}'.format(np.nanmin(data), np.nanmax(data)))

        return  replace_data(self,
                             Q_(data, self.units),
                             Depth(twt, depth_type='twt'),
                             overwrite=False,
                             info_txt=info_txt)

    def get_twt_from_owt(self):
        """
        Returns a Two-way time (TWT) LogCurve based on a One-way time (OWT) LogCurve
        :return:
        """
        from blixt_utils.utils import print_info
        if self.log_type != 'One-way time':
            warn_text = 'Two-way time can only be calculated from a One-way time log curve, not log_type {}'.format(
                self.log_type)
            print_info(warn_text, 'warning', logger)
            return None
        twt = replace_data(self, self.data * 2., self.depth, False,
                            'TWT calculated from OWT')
        twt.log_type = 'Two-way time'
        return twt

    def clean_data(self, nans=True, infinites=True, overwrite=False):
        """
        Removes data that are Nan and/or infinite
        Creates gaps in the data so that it is no longer regularly sampled
        :param nans:
        :param infinites:
        :param overwrite:
        :return:
        """
        if nans:
            _nans = np.isnan(self.data)
        if infinites:
            _infs = np.isinf(self.data)

        if nans & infinites:
            valid_indxs = ~_nans & ~_infs
            info_txt = 'Cleaned data from Nans and Infinites'
        elif infinites:
            valid_indxs = ~_infs
            info_txt = 'Cleaned data from Infinites'
        elif nans:
            valid_indxs = ~_nans
            info_txt = 'Cleaned data from Nans'
        else:
            return

        return replace_data(
            self,
            Q_(self.values[valid_indxs], self.units),
            Depth(Q_(self.depth.values[valid_indxs], self.depth_units)),
            # self.data[valid_indxs],
            # self.depth.depth[valid_indxs],
            overwrite,
            info_txt)

    def fill_gaps(self, method=None, extrapolate=False, overwrite=False,
                  fill_values=None, mask=None, suffix: str | None = None):
        """

        :param method:
        :param extrapolate:
        :param overwrite:
        :param fill_values:
            np.ndarray of same size as self.values
            Only used when method is set to 'fill_with_values'
        :param mask:
            Boolean np.ndarray of same size as self.values
            A false value indicates that the data should be masked out and replaced by the
            fill_value process
        :param suffix:
            str
        :return:
        """
        if method is None:
            method = 'interpolate'
        if mask is not None:
            self.data[~mask] = np.nan

        info_txt = 'Gaps have been filled using {}'.format(method)
        if extrapolate:
            info_txt += ' with extrapolation'

        if method == 'interpolate':
            fill_value = None
            if extrapolate:
                fill_value = 'extrapolate'
            tmp = self.clean_data()
            out = _interpolate(tmp.depth.values, tmp.values, self.depth.values, fill_value=fill_value)
        elif method == 'fill_with_values':
            out = deepcopy(self.values)
            nans = np.isnan(out)
            out[nans] = fill_values[nans]
        else:
            raise NotImplementedError("Method '{}' has not been implemented".format(method))

        return replace_data(
            self,
            Q_(out, self.units),
            self.depth,
            overwrite,
            info_txt,
            suffix=suffix
        )

    def smooth(self,
               window_len: pint.Quantity,
               method='median',
               discrete_intervals=None,
               mask=None,
               mask_desc=None,
               verbose=False,
               overwrite=False,
               suffix: str | None = None,
               **kwargs
               ):
        """
        :param window_len:
            pint.Quantity
            Length of smoothing window in depth or time
        :param method:
        :param discrete_intervals:
            list
            list of depth values (pint.Quantities or same unit as depth in LogCurve)
            at which the smoothened log are allowed discrete jumps.
            Typically the depth of the boundaries between two intervals (formations) in a well.
        :param mask:
            np.ndarray bool
            False values are masked out
        :param mask_desc:
        :param verbose:
        :param overwrite:
        :param suffix:
            str
        :param kwargs:
            keyword arguments passed on to _smooth
        :return:
            Smoothened LogCurve object when overwrite is False, else
        """
        from blixt_utils.utils import print_info
        if not self.is_evenly_spaced:
            warn_txt = 'Only evenly spaced data can be smoothened'
            print_info(warn_txt, 'warning', logger)
            return
        if not isinstance(window_len, pint.Quantity):
            # Simply assume that it is given in the correct units
            # window_len = Q_(window_len, self.depth_units)
            w_len = ceil(window_len / self.step().magnitude)
        else:
            w_len = ceil(window_len.to(self.depth_units).magnitude / self.step().magnitude)

        #if mask is not None:
        #    raise NotImplementedError('Smoothing masked log curves is not yet implemented')
        if mask is None:
            mask = np.array(np.ones(len(self.values)), dtype=bool)  # All True values -> all data is included

        if method == 'median' and np.mod(w_len, 2) == 0:
            w_len += 1  # avoid even length windows
        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')

            if isinstance(discrete_intervals[0], pint.Quantity):
                # Do unit conversion
                discrete_intervals = [_x.to(self.depth_units) for _x in discrete_intervals]
            else:
                # assume the intervals are given in correct units
                discrete_intervals = [Q_(_x, self.depth_units) for _x in discrete_intervals]

            disc_txt = 'allowing discrete jumps at {} '.format(
                ', '.join('{:.2}'.format(_x) for _x in discrete_intervals))
            # check if the proposed jump depths are within the depth range of the LogCurve

            # calculate the indexes of where the smoothened log are allowed discrete jumps.
            discrete_indexes = [np.argmin((self.depth.values[mask] - _z.magnitude)**2) for _z in discrete_intervals]

            out = np.zeros(0)
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    out = np.append(out, _smooth(
                        self.values[:discrete_indexes[i]], w_len, method=method, **kwargs
                    )
                                    )
                elif len(discrete_intervals) == i:  # last section
                    out = np.append(out, _smooth(
                        self.values[discrete_indexes[-1]:], w_len, method=method, **kwargs
                    )
                                    )
                else:
                    out = np.append(out, _smooth(
                        self.values[discrete_indexes[i-1]:discrete_indexes[i]],
                        w_len, method=method, **kwargs))

        else:
            disc_txt = ''
            out = _smooth(self.values[mask], w_len, method=method, **kwargs)

        info_txt = 'Data smoothened {}using a {} window of length {}'.format(disc_txt, method, window_len)
        return replace_data(
            self,
            Q_(out, self.units),
            Depth(Q_(self.depth.values[mask], self.depth_units)),
            overwrite,
            info_txt,
            suffix=suffix
        )

    def despike(self, max_clip: pint.Quantity, window_len: pint.Quantity | None = None, overwrite: bool = False,
                suffix: str | None = None):
        """
        Lower max_clip and longer window length clips away more data
        Note that gaps in the data can cause the despike method to miss out around the gap, so you can use fill_value
        first to improve

        :param max_clip:
        :param window_len:
        :param overwrite:
        :param suffix:
        :return:
        """
        # TODO By using the magnitudes we avoided a problem with user defined units (e.g. API), but
        # we need to add a test that the units are same for the max_clip and the data
        if window_len is None:
            window_len = Q_(2., 'm')
        if not isinstance(window_len, pint.Quantity):
            # Assume it is given in same unit as the depth data
            window_len = Q_(window_len, self.depth_units)
        if not isinstance(max_clip, pint.Quantity):
            max_clip = Q_(max_clip, self.units)

        info_txt = 'Despike with max clip {} and a window length of {}'.format(max_clip, window_len)
        _smooth = self.smooth(window_len)
        # spikes = np.where(self.data - _smooth.data > max_clip)[0]
        # spukes = np.where(_smooth.data - self.data > max_clip)[0]
        spikes = np.where(self.data.magnitude - _smooth.data.magnitude > max_clip.magnitude)[0]
        spukes = np.where(_smooth.data.magnitude - self.data.magnitude > max_clip.magnitude)[0]
        # out = deepcopy(self.data)
        out = deepcopy(self.data.magnitude)
        # out[spikes] = _smooth.data[spikes] + max_clip  # Clip at the max allowed diff
        # out[spukes] = _smooth.data[spukes] - max_clip  # Clip at the min allowed diff
        out[spikes] = _smooth.data[spikes].magnitude + max_clip.magnitude  # Clip at the max allowed diff
        out[spukes] = _smooth.data[spukes].magnitude - max_clip.magnitude  # Clip at the min allowed diff
        return replace_data(
            self,
            Q_(out, self.units),
            self.depth,
            overwrite,
            info_txt,
            suffix
        )

    def calc_mask(self,
                  cutoffs: Cutoffs,
                  name: str | None =None,
                  log_table: LogTable | None = None,
                  verbose: bool | None = False):
        """
        Based on the different cutoffs in the 'cutoffs' dictionary a mask is created.
        In the resulting mask, a False value indicates that the data is masked out

        When log_table is given, both the log type and log name need to match to return a mask
        Else, it tries to match the log type of the current log curve, and then the log name, with the
        keys in the 'cutoffs' dictionary.

        When a key in 'cutoffs' match either 'Depth', 'md', 'owt' or 'twt', and is the correct depth type
        of the log curve, the mask will work on the depth (I.E. a 'md' cut off will not work on a log curve
        with 'twt' as depth_type)

        :param cutoffs:
            CutOffs object
            dictionary with log name as keys, and list with mask operator and limits as values
            E.G. {'md': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
                OR
            use the log type instead of log name
            E.G. {'Depth': ['><', [2100, 2200]], 'Porosity': ['>', 0.1]}

            In the above examples, the Depth values within 2100 - 2200 are True,
            and Porosity values above 0.1 are also True
        :param name:
            str
            name of the mask
        :param log_table:
            dict
            Dictionary of log type: log name as "key: value" pairs that specify which log to use for each log type
            E.G.
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
        :param verbose:
            bool

        :return:
            LogCurve object
        """
        from blixt_utils.utils import mask_string
        from blixt_utils.utils import print_info

        green_flag = True
        final_mask = None
        masks = []

        if not isinstance(cutoffs, Cutoffs):
            raise IOError('Cutoffs must be specified as CutOffs() object, not {}'.format(type(cutoffs)))

        # mask_description = mask_string(cutoffs, None)
        mask_description = str(cutoffs)
        if len(cutoffs) == 0:  # no cutoffs, mask is all true
            final_mask = np.array(np.ones(len(self.data)))
            mask_description = 'All true mask for empty cutoffs'

        if log_table is not None:  # See if this log curve matches the log table criteria
            green_flag = False
            # for _log_type, _log_name in log_table.items():
            for _log_type, _log_name in zip(log_table.log_types, log_table.log_names):
                if (_log_type.lower() == self.log_type.lower()) and (_log_name.lower() == self.name.lower()):
                    green_flag = True
            if (not green_flag) and verbose:
                warn_txt = "The given log table does not match the current logcurve '{}', of type '{}', "\
                    "for well {}".format(
                        self.name, self.log_type, self.well
                )
                print_info(warn_txt, 'warning', logger)

        # for lname in list(cutoffs.keys()):
        # for lname in cutoffs.cutoff_names:
        for _cutoff in cutoffs.cutoffs:
            lname = _cutoff.param
            if lname.lower() in ['depth', 'md', 'twt', 'owt']:  # Create mask on Depth
                if lname.lower() == 'depth':  # For the special case of using Depth instead of md in the cutoffs
                    this_lname = 'md'
                else:
                    this_lname = lname.lower()
                if self.depth_type.lower() == this_lname:
                    # True when depth type is same as in cutoffs
                    masks.append(
                        msks.create_mask(
                            # self.depth.values, cutoffs[lname][0], cutoffs[lname][1]
                            self.depth, _cutoff.operator, _cutoff.limit
                        )
                    )
                    if verbose:
                        info_txt = "Creating mask based on Depth '{}', for well {}, based on {}".format(
                            self.depth_type, self.well, mask_description
                        )
                        print_info(info_txt, 'info', logger)
            if green_flag and (lname.lower() == self.name.lower()) or (lname.lower() == self.log_type.lower()):
                masks.append(
                    msks.create_mask(
                        # self.values, cutoffs[lname][0], cutoffs[lname][1])
                        self.data, _cutoff.operator, _cutoff.limit)
                )
                if verbose:
                    info_txt = "Creating mask based on '{}', for well {}, based on {}".format(
                        self.name, self.well, mask_description
                    )
                    print_info(info_txt, 'info', logger)
        if len(masks) > 0:
            # Combine all masks
            final_mask = msks.combine_masks(masks)

        if np.sum(final_mask) < 1:
            warn_txt = 'All values in log {}, in well {}, are masked out using {}'.format(
                self.name, self.well, mask_description)
            print_info(warn_txt, 'warning', logger)
        elif verbose:
            info_txt = '{} of {} are inside mask, for log {} in well {}, using {}'.format(
                np.sum(final_mask), len(final_mask), self.name, self.well, mask_description
            )
            print_info(info_txt, 'info', logger)

        return LogCurve(
            # create_data_array(
            #     name=name,
            #     data=final_mask,
            #     data_units=self.units,
            #     depth=self.depth,
            #     coord_units=self.coord_units,
            #     depth_type=self.depth_type,
            #     verbose=verbose
            # ),
            name,
            Q_(final_mask, 'dimensionless'),
            self.depth,
            log_type='Mask',
            well=self.well,
            header={
                'name': name,
                'well': self.well,
                'log_type': 'Mask',
                'desc': mask_description
            }
        )

    def calc_depth_trend(self,
                         trend_function=None,
                         x0=None,
                         loss=None,
                         mask=None,
                         mask_desc=None,
                         discrete_intervals=None,
                         down_weight_outliers=False,
                         verbose=False
                         ):
        """
        Calculates the depth trend for this log data by fitting the provided trend_function to the
        log data as a function of the depth

        Returns a list of optimized parameters for the trend function, with one list of parameters for each of
        the intervals

        :param trend_function:
            function
            Function to calculate the residual against
                residual = target_function(depth, *x) - self.data
            trend_function takes x as arguments, and depth as independent variable,
            E.G. for a linear target function:
            def trend_function(depth, a, b):
                return a*depth + b
            Default is the linear function
        :param x0:
            list
            List of parameters values used initially in the trend_function in the optimization
            Defaults to [1., 1.]
        :param loss:
            str
            Keyword passed on to least_squares() to determine which loss function to use
        :param mask:
            boolean numpy array of length N
            A False value indicates that the data is masked out
        :param mask_desc:
            str
            Description of what cutoffs the mask is based on.
        :param discrete_intervals:
            list
            list of depth values (pint.Quantities)
            at which the trend function are allowed discrete jumps
            Typically the depth of the boundaries between two intervals (formations) in a well.
        :param down_weight_outliers:
            bool
            If True, a weighting is calculated to minimize the impact of data far away from the median.
        :param verbose:
        :return:
            list of arrays with optimized parameters,
            e.g.
            [ array([a0, b0, ...]), array([a1, b1, ...]), ...]
            where [a0, b0, ...] are the output from least_squares for the first interval, and so on
        """
        from scipy.optimize import least_squares

        if trend_function is None:
            trend_function = linear_function
        if x0 is None:
            x0 = [1., 1.]
        # if loss is None:
        #     loss = 'cauchy'
        if (mask is not None) and (mask_desc is None):
            mask_desc = 'UNKNOWN'
        if discrete_intervals is not None:
            # Do unit conversion
            discrete_intervals = [_x.to(self.depth_units).magnitude for _x in discrete_intervals]

            disc_txt = 'allowing discrete jumps at {} {} '.format(
                ', '.join('{:.2}'.format(_x) for _x in discrete_intervals), self.depth_units)

        if False:
            raise NotImplementedError("calculate_depth_trend() has changed its output, It now returns the whole 'res' instead of 'res.x'")

        results = calculate_depth_trend(
            self.values,
            self.depth.values,
            trend_function,
            x0,
            None,
            mask,
            discrete_intervals,
            down_weight_outliers,
            ax=None,
            verbose=verbose,
            xlabel='{} [{}]'.format(self.name, str(self.units)),
            ylabel='{} [{}]'.format(self.depth_type, str(self.depth_units))
        )

        return results

    def de_trend(self, to_depth: pint.Quantity, trend_function, *params, suffix=None, verbose=False):
        """
        Projects the log data to the given 'to_depth' depth along the provided trend_function
        :param to_depth:
            pint.Quantity
            Depth to which the log data is to be projected to
        :param trend_function:
            Callable function
            Function that describes the depth trend in the log data
            f(depth, *params)
        :param params:
            paramaters that is used in the trend function
            eg. f = a * depth + b
        :param suffix:
            str
        :return:
            LogCurve object with de-trended data
        """
        if trend_function is None:
            trend_function = linear_function
        out = self.values + trend_function(to_depth.to(self.depth_units).magnitude, *params) - \
              trend_function(self.depth.values, *params)
        info_txt = 'Log de-trended to {} depth using {} with parameters [{}]'.format(
            to_depth, trend_function.__name__, ', '.join('{:}'.format(_x) for _x in params))

        if verbose:
            # TODO Here the self.min & max function have lost their units?!
            x0 = self.min.magnitude if isinstance(self.min, pint.Quantity) else self.min
            x1 = self.max.magnitude if isinstance(self.max, pint.Quantity) else self.max
            fig, ax = plt.subplots()
            self.plot(ax=ax)
            ax.plot(self.depth.values, out, 'k--')
            ax.legend(['Original data', 'de-trended data'])
            ax.vlines(to_depth.to(self.depth_units).magnitude, x0, x1, colors='b', ls='--')

        return replace_data(
            self,
            Q_(out, self.units),
            self.depth,
            False,
            info_txt,
            suffix=suffix
        )

    def plot(self, ax=None, **kwargs):
        set_active = False
        if ax is None:
            set_active = True
            fig, ax = plt.subplots()
        ax.plot(self.depth.values, self.values, **kwargs)
        ax.set_xlabel('{} [{}]'.format(self.depth_type, str(self.depth_units)))
        ax.set_ylabel('{} [{}]'.format(self.name, str(self.units)))
        if set_active:
            plt.show()

    def get_dataframe(self):
        return DataFrame({'depth': self.depth.values, 'data': self.values})

    @property
    def cds(self):
        return ColumnDataSource( {str(self.name): self.data.magnitude, 'depth': self.depth.magnitude} )

    def get_line(self):
        """
        Returns a (blixt_utils specific) Line object, which can be directly used in plot_logs_new.py
        :return:
        """
        from blixt_rp.plotting.log_plotter import Line
        self.style.name = self.name
        # _cds = ColumnDataSource(
        #     {str(self.name): self.data.magnitude, 'depth': self.depth.magnitude}
        # )
        # return Line(
        #     x=self.data.magnitude,
        #     y=self.depth.magnitude,
        #     style=self.style
        #     # line_args = {
        #     #     'line_color': 'blue' if self.style.line_color is None else self.style.line_color,
        #     #     'line_dash': 'solid' if self.style.line_style is None else self.style.line_style,
        #     #     'line_width': 1. if self.style.line_width is None else self.style.line_width,
        #     #     'legend_label': self.name
        #     # }
        # )
        return Line(
            x=self.name,
            y='depth',
            style=self.style,
            cds=self.cds
        )



def _interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, **kwargs):
    """
    Returns interpolated version of y(x), but now for x_new.

    kwargs
    keyword arguments passed on to scipy interp1d

    """
    from scipy import interpolate

    bounds_error = kwargs.pop('bounds_error', False)
    # fill_value = kwargs.pop('fill_value', 'extrapolate')
    fill_value = kwargs.pop('fill_value', None)

    bool_input = False
    if y.dtype == np.dtype('bool'):
        bool_input = True
        # convert boolean array to ints
        y = y * 1

    _out = interpolate.interp1d(x, y, bounds_error=bounds_error, fill_value=fill_value, **kwargs)(x_new)

    if bool_input:
        return _out > 0.5
    else:
        return _out


def find_depth_parameter(parameters, only_md=False):
    accepted_depth_keys = rud.rename_well_logs['depth']
    # Find depth parameter
    depth_key = None
    depth_type = None
    for _key in parameters:
        # for _key, _units in zip(log_names, log_units):
        if _key.lower() in [_x.lower() for _x in accepted_depth_keys]:
            depth_key = _key
            depth_type = 'md'
        elif _key.lower() == 'tvd' and not only_md:
            depth_key = _key
            depth_type = 'tvd'
        elif _key.lower() == 'owt' and not only_md:
            depth_key = _key
            depth_type = 'owt'
        elif _key.lower() == 'twt' and not only_md:
            depth_key = _key
            depth_type = 'twt'

    return depth_key, depth_type


def read_las(file_name: str, verbose: bool = False, encoding: str = 'UTF8',
             log_table = None, template: Template | str | None = None,
             rename_logs: dict | None = None,
             only_md: bool = True) -> (dict, dict):
    """
    Returns a LogCurve object for each, or selected, log in las file, packed in a dict

    :param file_name:
    :param verbose:
    :param encoding:
    :param log_table:
            LogTable
            Object which contains which log types, and associated and which log(s) to use for each log type
            When this is specified, we only load those logs that are listed
    :param template:
        str or Template object
        if str, it is the full filename of .xlsx file that contains templates of each log type.
    :param rename_logs:
        dict
        E.G.
        {'depth': ['DEPT', 'MD']}
        where the key is the wanted well log name, and the value list is a list of well log names to translate from
    :param only_md:
        bool
        If true it will only search and return logs with depth type = MD,
        The intended use is in las files where there is both md and twt logs, we will select the MD log for depth
    :return:
        tuple with two dicts
        first dict contains a log_name: LogCurve "key: value" pair for log in the las file
        second dict contains the standard info about the well and curves extracted from the las file
    """
    import pandas as pd
    from blixt_utils.utils import fix_well_name
    from blixt_utils.utils import print_info
    from blixt_utils.io.io import well_reader
    from blixt_rp.core.core import Template, templates_from_table

    with open(file_name, "r", encoding=encoding) as f:
        lines = f.readlines()
    # null_val, generated_keys, well_dict = well_reader(lines, file_format='las')
    null_val, generated_keys, well_dict = well_reader(lines, file_format='las', rename_logs=rename_logs)

    well_name = fix_well_name(well_dict['well_info']['well']['value'])

    # Find depth parameter
    depth_key, depth_type = find_depth_parameter(generated_keys, only_md=only_md)

    depth_units = well_dict['curve'][depth_key]['unit']

    if depth_key is None:
        warn_txt = 'No valid depth parameter was found in {}, which contains: {}'.format(
            # os.path.basename(file_name), ', '.join(log_names)
            os.path.basename(file_name), ', '.join(generated_keys)
        )
        print_info(warn_txt, 'warning', logger)
    if (depth_units is None) or (depth_units == ''):
        warn_txt = 'No valid depth unit was found in {} for depth {}'.format(
            os.path.basename(file_name), depth_key
        )
        print_info(warn_txt, 'warning', logger)
    depth_units = fix_units_for_pint(depth_units)

    # Only read those logs we requested in the log_table
    only_these_logs = generated_keys  # This contains all the logs
    log_types = [None] * len(only_these_logs)
    if log_table is not None:
        only_these_logs = log_table.log_names
        log_types = log_table.log_types

    # Find and extract data
    data = well_dict.pop('data')
    output = {}
    if isinstance(template, str):
        if verbose:
            print('read_las(): Template is given as a string')
        table = pd.read_excel(template, header=1, sheet_name='Templates', engine='openpyxl')
        template_dict = templates_from_table(table)
    elif isinstance(template, Template):
        print('read_las(): Template is given as a Template object')
        template_dict = template.dict()
    else:
        print('read_las(): No Template is given')
        template_dict = None
    for _key, _log_type in zip(only_these_logs, log_types):
        style = None
        if template_dict is not None:
            if _log_type in list(template_dict.keys()):
                style = Template(**template_dict[_log_type])
                style.name = _key
        try:
            data_units = well_dict['curve'][_key.lower()]['unit']
        except KeyError as e:
            warn_txt = '{}: Log {} is not found in {}'.format(
                e, _key, os.path.basename(file_name)
            )
            print_info(warn_txt, 'warning', logger)
            continue
        # if (data_units is None) or (data_units == ''):
        if data_units is None:
            warn_txt = 'No valid data unit was found in {} for data key {}. Skipping reading it'.format(
                os.path.basename(file_name), _key
            )
            print_info(warn_txt, 'warning', logger)
            continue
        data_units = fix_units_for_pint(data_units)

        # # Rename logs
        # log_name = None
        # if rename_logs is not None:
        #     for new_log_name in list(rename_logs.keys()):
        #         if _key.lower() in [_l.lower() for _l in rename_logs[new_log_name]]:
        #             log_name = new_log_name
        # if log_name is None:
        #     log_name = _key.lower()

        output[_key.lower()] = LogCurve(
        # output[log_name] = LogCurve(
            _key.lower(),
            # log_name,
            Q_(data[_key.lower()], data_units),
            Depth(Q_(data[depth_key], depth_units), depth_type=depth_type),
            _log_type,
            well_name,
            style=style,
            header={'orig_filename': file_name})

    return output, well_dict


def read_general_ascii(file_name: str,
                       separator: str,
                       data_begins_on_row: int,
                       var_names=None,
                       var_columns=None,
                       var_units=None,
                       var_types: list | None = None,
                       verbose: bool = False,
                       encoding: str = 'UTF8') -> dict | None:
    """

    :param file_name:
    :param separator:
        str
        'space', 'tab', ',', ';', ...
    :param data_begins_on_row:
        int
        Pythonic row number (starting at 0) of the first line of data
    :param var_names:
        int or list
        Either (pythonic) row number of where the variable names can be read in the file (NOTE: must have
        the same number of items and separator as the data for it to work)
        OR
        list of variables names. Its length must match the number of data columns or length of var_columns

        If None: generic variable names are created automatically
    :param var_columns:
        list
        list of integers that indicate the column numbers (pythonic) to load for each variable listed in var_names
        if var_names is an integer, all variables are loaded and this parameter is ignored
    :param var_units:
        int or list
        Either (pythonic) row number of where the units can be read in the file (NOTE: must have
        the same number of items and separator as the data for it to work)
        OR
        list of units. Its length must match the number of data columns

        If None: empty units are assigned each variable
    :param var_types:
    :param verbose:
    :param encoding:
        str
        Name of encoding to use when reading the data
        'utf-8-sig' seems useful
    :return:
    """

    from blixt_utils.utils import print_info
    output = {}
    data, units = read_file(file_name,
                            separator,
                            data_begins_on_row,
                            var_names,
                            var_columns,
                            var_units,
                            encoding)
    if var_types is None:
        var_types = [None] * len(data)
    # Find and locate necessary measured depth (MD) parameter
    depth_key, depth_type = find_depth_parameter(list(data.keys()), only_md=True)
    if depth_key is None:
        print_info('No MD log is identified', 'warning', logger)
        return None

    depth_units = fix_units_for_pint(units[depth_key])
    # print('XXX1', depth_key, depth_type, depth_units)
    depth = Depth(Q_(data[depth_key], depth_units), depth_type=depth_type, verbose=False)

    for i, _key in enumerate(list(data)):
        output[_key] = LogCurve(
            _key.lower(),
            Q_(data[_key], fix_units_for_pint(units[_key])),
            depth,
            header={'orig_filename': file_name, 'log_type': var_types[i]}
        )

    return output


def fix_units_for_pint(unit):

    if '^3' in unit:
        pass
    elif '**3' in unit:
        pass
    elif '3' in unit:
        unit = unit.replace('3', '^3')

    if '^2' in unit:
        pass
    elif '**2' in unit:
        pass
    elif '2' in unit:
        unit = unit.replace('2', '^2')

    if '_' in unit:
        unit = unit.replace('_', ' ')

    if unit == 'unitless':
        unit = 'dimensionless'

    if unit == '':
        unit = 'dimensionless'

    return unit


def translate_units_for_pint(unit):
    """
    BECAUSE OF THE PROBLEMS WITH PINT-XARRAY AND UNITS. WE WILL STOP USING XARRAY AND THIS FUNCTION
    Make units understandable for pint-xarray
    I think there is a bug in pint-xarray, which stops it from taking custom units into account
    To alleviate this, we do a simplistic translation here

    You need to add the following lines to: "<PATH TO site-packages>\\pint\\default_en.txt"

        ## Additions
        API = 1 * dimensionless = api

        ohmm = [resistivity] = ohm * meter = Ωm

        @alias degC = degc
        @alias ohm = Ohm
        @alias ohmm = Ohmm
        @alias foot = FT

    :param unit:
        str
    :return:
        str
    """
    from blixt_utils.utils import print_info
    print_info('This function will be deprecated', 'warning', logger)
    if '^3' in unit:
        pass
    elif '3' in unit:
        unit = unit.replace('3', '^3')
    if '^2' in unit:
        pass
    elif '2' in unit:
        unit = unit.replace('2', '^2')
    if '_' in unit:
        unit = unit.replace('_', ' ')

    # if unit in ['US/F']:
    #     return 'us/ft'
    # elif unit in ['Ohmm', 'ohmm']:
    #     return 'ohm m'
    # elif unit in ['']:
    #     pass
    # else:
    #     return unit
    return unit


def handle_depth(
        depth: int | float | np.ndarray | pint.Quantity,
        depth_type: str = 'md',
        depth_units: str | None = None,
        verbose: bool = False) -> pint.Quantity:
    """
    Based on the input, handle_depth returns a pint.Quantity in the accepted
    units

    if depth is a pint.Quantity, depth_units are ignored.

    if depth is not a pint.Quantity, and depth_units is None, an Error is raised

    :param depth:
        flt, int, np.ndarray or pint.Quantity
    :param depth_type:
        str
        'md', 'tvd', 'owt', or 'twt'
    :param depth_units:
        str
        Any unit handled by pint.
        Extensions to pint default units should be added here:
        blixt_rp/units_to_pint.txt
        (jetbrains://pycharm/navigate/reference?project=PycharmProjects&path=blixt_rp/blixt_rp/units_to_pint.txt)
    :param verbose:
    :return:
        pint.Quantity
    """
    if depth_type.lower() not in ['md', 'tvd', 'twt', 'owt']:
        raise IOError('Depth type must be either md, tvd, twt or owt. Not {}'.format(depth_type))

    if not isinstance(depth, pint.Quantity):
        if depth_units is None:
            raise IOError('Units must be specified')
        depth_units = fix_units_for_pint(depth_units)
        # Assign units to the data through pint.Quantity
        depth = Q_(depth, depth_units)

    # Convert depth to the accepted units of meter or millisecond
    if depth_type.lower() in ['md', 'tvd']:
        return depth.to('meter')
    else:
        return depth.to('millisecond')


def handle_coords_old(coords, coord_units: str, coord_type: str, verbose=False):
    """
    Function that assures that coordinates are of the right unit and type
    :param coords:
        np.ndarray or float
    :param coord_units:

    :param coord_type:
    :param verbose:
        bool
    :return:

    """
    from blixt_utils.utils import print_info
    coord_type = coord_type.lower()
    if coord_type not in ['md', 'tvd', 'twt', 'owt']:
        raise IOError('Dimension type must be either md, tvd, twt or owt. Not {}'.format(coord_type))

    coord_units = translate_units_for_pint(coord_units)

    # automatically convert the coordinate units to meter or ms
    if coord_type in ['md', 'tvd']:
        if coord_units.lower() in ['f', 'ft', 'feet', 'foot']:
            # _coords = depth * ureg.foot  # attach units to the coordinates
            # _coords = _coords.to(ureg.meter)  # convert to meter
            _coords = Q_(coords, 'foot')  # attach units to the coordinates
            _coords = _coords.to('meter')  # convert to meter
            if verbose:
                info_txt = 'Converting depth from {} to meter'.format(coord_units)
                print_info(info_txt, 'info', logger)
            coords = _coords.magnitude
            coord_units = 'meter'
            return coords, coord_units, coord_type
        elif coord_units.lower() in ['m', 'meter']:
            return coords, coord_units.lower(), coord_type
        elif not is_equivalent(coord_units, 'meter'):
            raise IOError('MD coordinate type not compatible with unit [{}]'.format(coord_units))
        else:
            raise IOError('Unknown combination of MD coordinate type and unit {}'.format(coord_units))
    else:
        if coord_units.lower() in ['s', 'sec', 'second']:
            #_coords = depth * ureg.second  # attach units to the coordinates
            # _coords = _coords.to(ureg.millisecond)  # convert to milliseconds
            _coords = Q_(coords, 'second')  # attach units to the coordinates
            _coords = _coords.to('millisecond')  # convert to milliseconds
            if verbose:
                info_txt = 'Converting depth from {} to ms'.format(coord_units)
                print_info(info_txt, 'info', logger)
            coords = _coords.magnitude
            coord_units = 'millisecond'
            return coords, coord_units, coord_type
        elif coord_units.lower() in ['ms', 'millisecond']:
            return coords, coord_units.lower(), coord_type
        elif not is_equivalent(coord_units, 'millisecond'):
            raise IOError('TWT or OWT coordinate type not compatible with unit [{}]'.format(coord_units))
        else:
            raise IOError('Unknown combination of TWT or OWT coordinate type and unit {}'.format(coord_units))


def create_data_array(
        name: str,
        data: np.ndarray,
        data_units: str,
        coords: np.ndarray,
        coord_units: str,
        coord_type: str,
        verbose=False):
    """
    This data pair object use the xarray library in combination with pint to take
    care of units
    A workaround because pint doesn't work with units: https://xarray.dev/blog/introducing-pint-xarray
    is to use the pint_xarray library

    Returns a xarray.DataArray with units
    > log_values = _create_data_array(name, data, data_units, depth, coord_units, depth_type)

    It automatically converts the coordinate units to meter or ms

    The data is accessed through
    > print(log_values.name)
    > log_values.data.magnitude  # returns the data without coordinates
    > print(log_values.data.units) # prints the units

    The depth_type of the result is seen using
    > print(list(log_values.depth.keys())[0])
    The units of the coordinates
    > print(log_values.depth[depth_type].units)
    And the coordinate (index) themselves  can be accessed by
    > log_values.depth[depth_type].data

    And the data can be plotted using
    > log_value.plot()

    You can convert units of data and/or coordinates by
    > log_values.pint.to(
    >     {'<NAME>': '<TO_UNIT>',
           '<COORD_TYPE>': '<TO_ANOTHER_UNIT>'}
    > )

    :param name:
        str
    :param data:
        np.ndarray
    :param data_units:
        str
        A unit supported by pint
    :param coords:
        np.ndarray
        same length as data
    :param coord_units:
        str
        A unit supported by pint
    :param coord_type:
        str
        Either 'md', 'twt' or 'owt'
    """

    data_units = translate_units_for_pint(data_units)

    coords, coord_units, coord_type = handle_coords_old(coords, coord_units, coord_type)

    coords_data = xr.DataArray(coords, dims=coord_type)
    log_values = xr.DataArray(name=name, data=data, dims=[coord_type], coords={coord_type: coords_data})
    # attach units to the xarray
    log_values = log_values.pint.quantify({name: data_units, coord_type: coord_units})
    return log_values


def replace_data(
        logcurve_in: LogCurve,
        log_data: pint.Quantity,
        depth: Depth,
        overwrite: bool,
        info_txt,
        suffix=None
) -> LogCurve | None:
    """

    :param logcurve_in:
    :param log_data:
    :param depth:
    :param overwrite:
        bool
        If True it overwrites the input data and returns nothing
        If False it creates a new version of the input data, replace the data and returns a LogCurve
    :param info_txt:
    :param suffix:
    :return:
    """
    if len(log_data) != len(depth):
        raise IOError('Length of log_data ({}) must equal length of depth ({})'.format(
            len(log_data), len(depth)))
    # new_d_arr_coords = xr.DataArray(coordinates, dims=logcurve_in.depth_type)
    # new_d_arr = xr.DataArray(name=logcurve_in.name, data=data, depth={logcurve_in.depth_type: new_d_arr_coords})
    # new_d_arr = new_d_arr.pint.quantify(
    #     {logcurve_in.name: logcurve_in.units, logcurve_in.depth_type: logcurve_in.coord_units}
    # )

    output = None
    if overwrite:
        logcurve_in.header.modification_date = datetime.now().isoformat()
        logcurve_in.header.modification_history += '\n{}'.format(info_txt)
        logcurve_in.data = log_data
        logcurve_in.depth = depth
    else:
        output = logcurve_in.copy(suffix)
        output.data = log_data
        output.depth = depth
        output.header.modification_date = datetime.now().isoformat()
        output.header.modification_history += '\n{}'.format(info_txt)
    return output


def is_equivalent(first: pint.Unit, second: pint.Unit, verbose: bool = False) -> bool:
    """
    Test if two units are equivalent
    :param first:
    :param second:
    :return:
    """
    # verbose = True
    if verbose:
        from blixt_utils.utils import print_info
        print_info('Testing if the units {} and {} are equivalent'.format(first, second), 'info', logger)
    try:
        factor = ureg.convert(1, first, second)
    except pint.DimensionalityError:
        return False
    return isclose(factor, 1)


def _to_twt(
        twt: pint.Quantity,
        data: np.ndarray,
        dt: pint.Quantity = Q_(1, 'millisecond'),
        axis: int = -1,
        verbose: bool | None = False
):
    """
    Converts the input depth domain data to a regularly sampled array in TWT
    :param twt:
        The twt data for the depth-time relation
    :param data:
        A N-D array of real values in the depth domain
        The length of the 'axis' axis must be the same as twt
    :param dt:
        The sample rate of the new twt
    :param axis:
       The axis corresponding to the depth axis, which shall be converted to the time domain
    :return:
        two tuple
        The regularly sampled twt values
        The data resampled to match the regularly sampled twt
    """
    from scipy.interpolate import interp1d
    if data.shape[axis] != twt.shape[0]:
        raise IOError('Depth array and TWT array must have same length')
    new_twt = np.arange(
        np.nanmin(twt).to('millisecond').magnitude,
        np.nanmax(twt).to('millisecond').magnitude,
        dt.to('millisecond').magnitude
    )
    return Q_(new_twt, 'millisecond'), _interpolate(twt.to('millisecond').magnitude, data, new_twt, kind='linear', axis=axis)

def _to_depth(
        time_depth_twt: pint.Quantity,
        data_twt: pint.Quantity,
        data: np.ndarray,
        axis: int = -1
):
    """
    Converts the data, which is supposed to be regularly sampled in TWT, to the depth domain, which is irregularly
    sampled in TWT
    :param time_depth_twt:
        The twt data for the depth-time relation
    :param data_twt:
        The twt data (values) for the given data
    :param data:
        A N-D array of real values, regularly sampled in the time domain
        The length of the 'axis' axis must be the same as data_twt
    :param axis:
       The axis corresponding to the depth axis, which shall be converted to the time domain
    :return:
        the data resampled to match the time (time_depth_twt) of the time-depth pair
    """
    from scipy.interpolate import interp1d
    if data.shape[axis] != data_twt.shape[0]:
        raise IOError('Data array and TWT array must have same length')
    return _interpolate(
        data_twt.to('millisecond').magnitude,
        data,
        time_depth_twt.to('millisecond').magnitude,
        kind='linear',
        axis=axis, fill_value='extrapolate')

# def take_units_from_style(log_curve: LogCurve):
#     # If the style Template contains a unit, which is not None, then we should convert the data to this
#     # unit if it is not already using those units.
#     if 'style' in list(log_curve.__dict__.keys()):
#         if log_curve.style.units is not None:
#             # print('Units in template is {}'.format(log_curve.style.units))
#             if not is_equivalent(log_curve.data.units, ureg.Unit(log_curve.style.units)):
#                 # print('Units are not the same, try to convert')
#                 try:
#                     info_txt = '{}: {}: Convert from {} to {}'.format(
#                         log_curve.well, log_curve.name, str(log_curve.data.units), log_curve.style.units
#                     )
#                     log_curve.data = log_curve.data.to(log_curve.style.units)
#                     print_info(info_txt, 'info', logger)
#                 except pint.DimensionalityError:
#                     warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
#                         log_curve.well, log_curve.name, str(log_curve.data.units), log_curve.style.units
#                     )
#                     print_info(warn_txt, 'warning', logger)
#                     #  Then continue using the original unit
#                     log_curve.style.units = str(log_curve.units)
#         else:
#             log_curve.style.units = str(log_curve.data.units)


def test_interpolate():
    x = np.array([0, 1, 1.5, 2., 2.5, 3., 4., 5., 8., 9.])
    y = np.exp(-x / 3.0)
    x_new = np.arange(0, 12, 1)
    y_new = _interpolate(x, y, x_new)
    plt.plot(x, y, 'o', x_new, y_new, '-')
    mask = np.ma.masked_inside(x, 1, 4).mask
    new_mask = _interpolate(x, mask, x_new)
    plt.plot(x_new[new_mask], y_new[new_mask])

    mask = np.ma.masked_greater(y, 0.7).mask
    new_mask = _interpolate(x, mask, x_new)
    plt.plot(x_new[new_mask], y_new[new_mask])

    plt.show()


def test_xarray():
    """
    This is testing a new way of storing all the data from a Project (multiple wells) in a xarray. Sorted
    so that it easy to perform calculations like depth trends and statistics on it.

    It should basically consist of a DataSet with two DataArrays of dimension (n, i, w, l) where
        n: number of unique depth values over all wells and intervals
        i: number of working intervals / formations
        w: number of wells
        l: number of logs
    One DataArray for log values, and one for depth values

    BUT I think we need to drop this idea as the number of unique depth values can become very large when the number of
    wells increase, and then most of the DataArray with log values should contain NaN's 
    :return:
    """
    lc1 = LogCurve(
        'log1',
        Q_(np.random.rand(11), 'm/s'),
        Depth(Q_(np.linspace(1000, 2000, 11)))
    )
    lc2 = LogCurve(
        'log2',
        Q_(np.random.rand(9), 'm/s'),
        Depth(Q_(np.linspace(1000, 2000, 9)))
    )

    depths = lc1.depth.unique_depths(lc2.depth)

    lc1_depth_indexes = [int(np.argmin(np.sqrt((depths.values - _x)**2))) for _x in lc1.depth.values]
    lc2_depth_indexes = [int(np.argmin(np.sqrt((depths.values - _x)**2))) for _x in lc2.depth.values]

if __name__ == '__main__':
    test_interpolate()
