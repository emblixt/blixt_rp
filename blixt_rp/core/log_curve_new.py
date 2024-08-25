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
from pandas import DataFrame
import xarray as xr
import pint
# from pint import UnitRegistry
from math import isclose, ceil

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.basename(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.param import Param
from blixt_rp.core.template_new import Template
from blixt_rp.core.header_new import Header
from blixt_utils.signal_analysis.signal_analysis import smooth as _smooth
from blixt_utils.misc.curve_fitting import residuals, linear_function
from blixt_rp.rp_utils.definitions import allowed_depth_formats
from blixt_utils.misc.convert_data import convert
from blixt_utils.misc import masks as msks
from blixt_utils.misc import templates as tmplts
import blixt_rp.rp_utils.definitions as rud
from blixt_utils.utils import print_info
# from blixt_utils.io.io import read_general_ascii_GENERAL as read_file, project_wells_new


from .. import ureg, Q_
import pint
import pint_xarray

logger = logging.getLogger(__name__)


class Coordinate(object):
    """
    Class handling depth or time data, that holds the "depth" of something along a well path
    It can either be 'md' or 'tvd' in depth domain, and 'twt' or 'owt' in time domain
    """
    def __init__(self, coords, units=None, coord_type='md', verbose=False):
        """

        :param coords:

        :param units:
        :param coord_type:
        :param verbose:
        """

        if verbose:
            print(coord_type, units)
        _coords = handle_coords(coords, coord_units=units, coord_type=coord_type, verbose=verbose)

        self.coords = _coords
        self.coord_type = coord_type.lower()

    @property
    def data(self):
        return self.coords.magnitude

    @property
    def units(self):
        return str(self.coords.units)

    @property
    def domain(self):
        if self.coord_type in ['md', 'tvd']:
            return 'depth'
        else:
            return 'time'

    # I'm trying to "protect" the coord_type from being updated, BUT that stops me from setting it in the first place!
    # def __setattr__(self, key, value):
    #     if key in ['coord_type']:
    #         pass
    #     elif key in ['coords']:
    #         self.__setattr__(key, handle_coords(value, coord_type=self.coord_type, coord_units=self.units))
    #     else:
    #         self.__setattr__(key, value)


class LogCurve(object):
    """
    Class handling logs of a well
    """

    def __init__(self,
                 name,
                 data,
                 start,
                 stop,
                 step=Param(name='', value=None),
                 log_type=None,
                 block=None,
                 well=None,
                 regular_sampling=True,
                 style=None,
                 unit=None,
                 header=None):
        """
        Initiates the most basic building block of a Well object, the LogCurve, which should hold one log

        :param name:
            str
        :param log_type:
            str
        :param block:
            str
        :param well:
            str
        :param start / stop:
            Param
            holds the start / stop depth in measured depth, MD, with units and description
        :param step:
            Param
            Holds the increment for regularly sampled (along MD) well logs, with units and description.
            None for irregularly sampled well logs (core data, etc. )
        :param regular_sampling:
            bool
            True for ordinary well logs that has a regular sampling in MD.
        :param style:
            dict
            If present, it should contain a dictionary with keys typical for a template. See
            blixt_utils.misc.templates.necessary_keys
        :param data:
            numpy.ndarray
            Containing the log data for this log.
            The reason why we don't use a Param for this numerical value(s) too, is that the LogCurve object has
            properties like unit etc.

        """
        _data_sanity_checks(data)
        self.name = name
        self.block = block
        self.well = well
        if header is None:
            header = {}

        if 'name' not in list(header.keys()) or header['name'] is None:
            header['name'] = name

        if 'well' not in list(header.keys()) or header['well'] is None:
            header['well'] = well

        if 'log_type' not in list(header.keys()) or header['log_type'] is None:
            header['log_type'] = log_type
        elif (log_type is None) and ('log_type' in list(header.keys())) and (header['log_type'] is not None):
            log_type = header['log_type']
        self.log_type = log_type

        if 'start' not in list(header.keys()) or header['start'] is None:
            header['start'] = start
        elif (start.value is None) and ('start' in list(header.keys())) and (header['start'] is not None):
            start = header['start']
        if (start is not None) and not isinstance(start, Param):
            raise TypeError('start value must be given as a Param object with name, value, units & description')
        self.start = start

        if 'stop' not in list(header.keys()) or header['stop'] is None:
            header['stop'] = stop
        elif (stop.value is None) and ('stop' in list(header.keys())) and (header['stop'] is not None):
            stop = header['stop']
        if (stop is not None) and not isinstance(stop, Param):
            raise TypeError('stop value must be given as a Param object with name, value, units & description')
        self.stop = stop

        if 'step' not in list(header.keys()) or header['step'] is None:
            header['step'] = step
        elif (step.value is None) and ('step' in list(header.keys())) and (header['step'] is not None):
            step = header['step']
        if (step is not None) and not isinstance(step, Param):
            raise TypeError('step value must be given as a Param object with name, value, units & description')
        self.step = step

        if 'regular_sampling' not in list(header.keys()) or header['regular_sampling'] is None:
            header['regular_sampling'] = regular_sampling
        self.regular_sampling = regular_sampling

        if not self.regular_sampling:
            self.step.value = None

        if 'style' not in list(header.keys()) or header['style'] is None:
            header['style'] = style
        elif (style is None) and ('style' in list(header.keys())) and (header['style'] is not None):
            style = header['style']
        elif style is None:
            style = tmplts.return_empty_template()
        self.style = style

        if 'unit' not in list(header.keys()) or header['unit'] is None:
            header['unit'] = unit
        elif (unit is None) and ('unit' in list(header.keys())) and (header['unit'] is not None):
            unit = header['unit']
        self.unit = unit

        if not isinstance(header, dict):
            raise TypeError('Input header must be a dictionary')
        self.data = data

    def __len__(self):
        return len(self.data)

    def copy(self, suffix='copy'):
        copied_log_curve = deepcopy(self)
        copied_log_curve.name = self.name + '_' + suffix
        copied_log_curve.header.name = self.name + '_' + suffix
        copied_log_curve.header.modification_date = datetime.now().isoformat()
        copied_log_curve.header.modification_history += \
            '\nCopy of {}'.format(self.name)
        return copied_log_curve

    def get_depth(self):
        if (not self.regular_sampling) or (self.start is None) or (self.stop is None):
            return None
        else:
            return np.linspace(self.start.value, self.stop.value, len(self.data))

    def depth_plot(self,
                   mask=None,
                   mask_desc=None,
                   wis=None,
                   ax=None,
                   savefig=None,
                   **kwargs):
        """
        Plots log as a function of MD.

        :param mask:
            boolean numpy array of same length as self.data
        :param mask_desc:
            str
            Text describing which criteria the mask is based on
        :param wis:
            dict
            dictionary of working intervals
        :param ax:
            matplotlib.axes._subplots.AxesSubplot object
        :param savefig:
            str
            full path name of file to save plot to
        :param kwargs:
            keyword arguments passed on to plot
        :return:
            matplotlib.lines.line2D
            Line2D object with the current line
        """
        _savefig = False
        if savefig is not None:
            _savefig = True

        if (mask is not None) and (mask_desc is None):
            mask_desc = 'UNKNOWN'
        if mask is None:
            mask = np.array(np.ones(len(self.data)), dtype=bool)  # All True values -> all data is included

        # set up plotting environment
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        else:
            fig = ax.get_figure()
        x1 = deepcopy(self.data)
        x1[~mask] = np.nan
        ax.plot(
            self.data,
            self.get_depth(),
            lw=0.5,
            alpha=0.5,
            c='gray'
        )
        this_line, = ax.plot(
            x1,
            self.get_depth(),
        )
        ax.set_ylim(self.stop.value, self.start.value)
        if self.style is not None:
            try:
                ax.set_xlim(self.style['min'], self.style['max'])
            except KeyError:
                print('No min/max plot range in style')
        title_txt = 'Well {}'.format(self.well)
        if mask_desc is not None:
            title_txt += ', using mask: {}'.format(mask_desc)
        ax.set_title(title_txt)
        ax.set_ylabel('Depth, MD [{}]'.format(self.start.unit))
        ax.set_xlabel('{} [{}]'.format(self.name, self.unit))

        if _savefig:
            fig.savefig(savefig)
        else:
            plt.show()

        return this_line

    def smooth(self, window_len=None, method='median',
               discrete_intervals=None,
               verbose=False, overwrite=False, **kwargs):
        """
        :param discrete_intervals:
            list
            list of indexes at which the smoothened log are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        """
        if method == 'median' and np.mod(window_len, 2) == 0:
            window_len += 1  # avoid even length windows
        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')
            out = np.zeros(0)
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    out = np.append(out, _smooth(
                        self.data[:discrete_intervals[i]], window_len, method=method, **kwargs
                    )
                                    )
                elif len(discrete_intervals) == i:  # last section
                    out = np.append(out, _smooth(
                        self.data[discrete_intervals[-1]:], window_len, method=method, **kwargs
                    )
                                    )
                else:
                    out = np.append(out, _smooth(
                        self.data[discrete_intervals[i - 1]:discrete_intervals[i]],
                        window_len, method=method, **kwargs))

        else:
            out = _smooth(self.data, window_len, method=method, **kwargs)

        if verbose:
            print('Max diff between smoothed and original version:', np.nanmax(self.data - out))
            fig, ax = plt.subplots()
            ax.plot(self.data, c='black', label='Original')
            ax.plot(out, c='r', label='{}, window length {}'.format(method, window_len))
            if discrete_intervals is not None:
                for xx in discrete_intervals:
                    ax.axvline(xx, 0, 1, ls='--')
            ax.set_title(self.name)
            ax.legend()
        if overwrite:
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += \
                '\nSmoothened using {}, with window length {}'.format(method, window_len)
            self.data = out
            return None
        else:
            return out

    def despike(self, max_clip, window_len=None):
        if window_len is None:
            window_len = 13  # Around 2 meters in most wells
        smooth = self.smooth(window_len)
        spikes = np.where(self.data - smooth > max_clip)[0]
        spukes = np.where(smooth - self.data > max_clip)[0]
        out = deepcopy(self.data)
        out[spikes] = smooth[spikes] + max_clip  # Clip at the max allowed diff
        out[spukes] = smooth[spukes] - max_clip  # Clip at the min allowed diff
        return out

    def calc_depth_trend(self,
                         depth,
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
        log data as a function of the provided depth data

        Returns a list of optimized parameters for the trend function, with one list of parameters for each of
        the intervals

        :param depth:
            numpy.ndarray of length N
            The independent variable, could be MD, TVD, max burial depth, ...

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
            list of indexes (in the unmasked data) at which the trend are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        :param down_weight_outliers:
            bool
            If True, a weighting is calculated to minimize the impact of data far away from the median.

        :param verbose:
            bool
            If True plots are created and more information is printed

        return
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
        if loss is None:
            loss = 'cauchy'
        if (mask is not None) and (mask_desc is None):
            mask_desc = 'UNKNOWN'
        if mask is None:
            mask = np.array(np.ones(len(self.data)), dtype=bool)  # All True values -> all data is included
        verbosity_level = 0
        fig, ax = None, None
        results = []
        if verbose:
            verbosity_level = 2
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(self.data, depth, lw=0.5, c='grey')
            if discrete_intervals is not None:
                for i in discrete_intervals:
                    ax.axhline(depth[i], 0, 1, ls='--')

        def do_the_fit(_mask):
            weights = None

            # Test if there are NaN values in input data, which needs to be masked out
            if np.any(np.isnan(self.data[_mask])):
                nan_mask = np.isnan(self.data)
                _mask = msks.combine_masks([~nan_mask, _mask])

            # TODO
            #  do a similar test for infinite data: if np.any(np.isinf(_data))

            if down_weight_outliers:
                weights = 1. - np.sqrt((self.data[_mask] - np.median(self.data[_mask])) ** 2)
                weights[weights < 0] = 0.

            return least_squares(residuals, x0, args=(depth[_mask], self.data[_mask]),
                                 kwargs={'target_function': trend_function, 'weight': weights},
                                 loss=loss, verbose=verbosity_level)

        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    this_depth_mask = msks.create_mask(depth, '<', depth[discrete_intervals[i]])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', depth[discrete_intervals[-1]])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [depth[discrete_intervals[i - 1]], depth[discrete_intervals[i] - 1]])

                combined_mask = msks.combine_masks([mask, this_depth_mask])
                res = do_the_fit(combined_mask)
                results.append(res.x)
                if verbose:
                    this_depth = np.linspace(depth[this_depth_mask][0], depth[this_depth_mask][-1], 10)
                    ax.plot(self.data[combined_mask], depth[combined_mask])
                    ax.plot(trend_function(this_depth, *res.x), this_depth, c='b')
        else:
            res = do_the_fit(mask)
            results.append(res.x)
            if verbose:
                this_depth = np.linspace(depth[0], depth[-1], 10)
                ax.plot(self.data[mask], depth[mask])
                ax.plot(trend_function(this_depth, *res.x), this_depth, c='b')

        if verbose:
            ax.set_ylim(ax.get_ylim()[::-1])
            title_txt = 'Well {}'.format(self.well)
            if mask_desc is not None:
                title_txt += ', using mask: {}'.format(mask_desc)
            ax.set_title(title_txt)
            ax.set_xlabel(self.name)

        return results

    def apply_trend_function(self,
                             depth,
                             trend_parameters,
                             trend_function=None,
                             discrete_intervals=None,
                             verbose=False
                             ):
        """
        Uses the result from calc_depth_trend() to calculate the trend over the whole well.
        Which is trivial when discrete_intervals is None

        :param depth:
            numpy.ndarray of length N
            The independent variable, could be MD, TVD, max burial depth, ...

        :param trend_parameters:
            list
            List of arrays with optimized parameters of the trend_function
            This list is the result from calc_depth_trend()

        :param trend_function:
            function
            Function to calculate the residual against
                residual = target_function(depth, *x) - self.data
            trend_function takes x as arguments, and depth as independent variable,
            E.G. for a linear target function:
            def trend_function(depth, a, b):
                return a*depth + b
            Default is the linear function

        :param discrete_intervals:
            list
            list of indexes (in the unmasked data) at which the trend are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        :param verbose:
            bool
            If True plots are created and more information is printed

        return
            np.ndarray of the trend, with same length as the original log
        """
        if trend_function is None:
            trend_function = linear_function
        fig, ax = None, None
        output = np.zeros(0)
        if verbose:
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(self.data, depth, lw=0.5, c='grey')
            if discrete_intervals is not None:
                for i in discrete_intervals:
                    ax.axhline(depth[i], 0, 1, ls='--')

        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')
            if len(trend_parameters) != len(discrete_intervals) + 1:
                raise IOError('Number of trends must be one more than the number of interval boundaries')
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    this_depth_mask = msks.create_mask(depth, '<', depth[discrete_intervals[i]])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', depth[discrete_intervals[-1]])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [depth[discrete_intervals[i - 1]], depth[discrete_intervals[i] - 1]])
                output = np.append(output, trend_function(depth[this_depth_mask], *trend_parameters[i]))
        else:
            output = trend_function(depth, *trend_parameters[0])

        if verbose:
            ax.plot(output, depth)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_title(self.well)
            ax.set_xlabel(self.name)

        return output


class LogCurve2dNew(object):
    """
    This log curve object use the xarray library to create a log curve with a 1D dimensional coordinate (index)
    attached to it. In combination with pint to take care of units

    Example:
    > lc = LogCurve2dNew(
    >     create_data_array(
    >         name='TEST',
    >         data=np.ndarray,
    >         data_units='m/s',
    >         coords=np.ndarray,
    >         coord_units='m',
    >         coord_type='md'
    >     )
    >     log_type='P velocity',
    >     well='A Well',
    >     style=None,
    >     header=None
    > )
    > # Convert data to km/s
    > lc.data = lc.data.pint.to('km/s')


    The units of the coordinates will be automatically converted to m or ms, depending on if the log is in
    depth or time

    If units are specified in the style, the LogCurve will automatically try to convert the data units
    to this unit. So the example conversion above could have been carried out by specifying the style
    > style = {'units': 'km/s'}

    """

    def __init__(self,
                 data_array,
                 log_type=None,
                 well=None,
                 style=None,
                 header=None):
        """
        The basic building block of all well related data.
        Should hold both regularly sampled log curves and irregularly sampled core data

        :param data_array:
            xr.DataArray
            should contain an 1D DataArray with name, data and a coordinate dimension (index) which is either md, twt or owt
            see _create_data_array for the basic construct
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
        self.d_arr = data_array
        self.name = None if (data_array is None) else str(data_array.name)
        self.well = well

        if style is None:
            self.style = Template(template={})
        elif isinstance(style, dict):
            self.style = Template(template=style)
        elif isinstance(style, Template):
            self.style = style
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

        # If the style Template contains a unit, which is not None, then we should convert the data to this
        # unit if it is not already using those units.
        if self.style.units is not None:
            # print('Units in template is {}'.format(self.style.units))
            if not is_equivalent(data_array.data.units, ureg.Unit(self.style.units)):
                # print('Units are not the same, try to convert')
                try:
                    info_txt = '{}: {}: Convert from {} to {}'.format(
                        self.well, self.name, str(data_array.data.units), self.style.units
                    )
                    self.d_arr = data_array.pint.to(self.style.units)
                    print_info(info_txt, 'info', logger)
                except pint.DimensionalityError:
                    warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                        self.well, self.name, str(data_array.data.units), self.style.units
                    )
                    print_info(warn_txt, 'warning', logger)
        else:
            self.style.units = None if (data_array is None) else str(data_array.data.units)

        if log_type is None:
            if self.header.log_type is not None:
                self.log_type = self.header.log_type
            else:
                self.log_type = ''
        else:
            if (self.header.log_type is not None) and (self.header.log_type != log_type):
                warn_txt = 'Log type in header ({}), does not match given log type ({})'.format(
                    self.header.log_type, log_type)
                warn_txt += '\nFix inconsistency and try again'
                print_info(warn_txt, 'warning', logger)
                raise IOError(warn_txt)
            else:
                self.log_type = log_type

    def __len__(self):
        return None if (self.d_arr is None) else len(self.d_arr)

    @property
    def coord_type(self):
        return None if (self.d_arr is None) else list(self.d_arr.coords.keys())[0]

    @property
    def coord_units(self):
        return None if (self.d_arr is None) else str(self.d_arr.coords[self.coord_type].units)

    @property
    def units(self):
        return None if (self.d_arr is None) else str(self.d_arr.data.units)

    @property
    def coords(self):
        return None if (self.d_arr is None) else self.d_arr.coords[self.coord_type].data

    @property
    def values(self):
        return None if (self.d_arr is None) else self.d_arr.data.magnitude

    @property
    def is_evenly_spaced(self):
        """
        Tests if the dimension of a LogCurve is evenly spaced

        :return:
            bool
            True if evenly sampled
        """
        arr = self.coords
        if len(arr) <= 2:
            return True

        diffs = np.diff(arr)
        return np.allclose(diffs, diffs[0])

    # @property
    def step(self, ignore_gaps=False):
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
            return np.diff(self.coords)[0]
        elif ignore_gaps:
            steps, counts = np.unique(np.diff(self.coords), return_counts=True)
            return steps[np.argmax(counts)]
        else:
            return None

    def __lt__(self, other):
        if len(self.d_arr) < len(other.d_arr):
            return True
        else:
            return False

    def __le__(self, other):
        if len(self.d_arr) <= len(other.d_arr):
            return True
        else:
            return False

    def __gt__(self, other):
        if len(self.d_arr) > len(other.d_arr):
            return True
        else:
            return False

    def __eq__(self, other):
        if len(self.d_arr) == len(other.d_arr):
            return True
        else:
            return False

    def __ge__(self, other):
        if len(self.d_arr) >= len(other.d_arr):
            return True
        else:
            return False

    def __str__(self):
        return '{}: {}'.format(self.name, str(self.header))

    def copy(self, suffix='copy'):
        copied_log_curve = deepcopy(self)
        if len(suffix) > 0:
            copied_log_curve.name = self.name + '_' + suffix
            copied_log_curve.header.name = self.name + '_' + suffix
        copied_log_curve.header.modification_date = datetime.now().isoformat()
        copied_log_curve.header.modification_history += \
            '\nCopy of {}'.format(self.name)
        return copied_log_curve

    def convert_to(self, to_units):
        """ Converts the data units to 'to_units', and changes the unit in the style too"""
        try:
            cnv_txt = 'Convert from {} to {}'.format(
                str(self.d_arr.data.units), to_units
            )
            info_txt = '{}: {}: {}'.format(
                self.well, self.name, cnv_txt
            )
            self.d_arr = self.d_arr.pint.to(to_units)
            self.style.units = to_units
            print_info(info_txt, 'info', logger)
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += '\n{}'.format(cnv_txt)
        except pint.DimensionalityError:
            warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                self.well, self.name, str(self.d_arr.data.units), to_units
            )
            print_info(warn_txt, 'warning', logger)

    def convert_coords_to(self, to_units):
        """ Converts the coordinate units to 'to_units'"""
        try:
            cnv_txt = 'Convert coordinates from {} to {}'.format(
                str(self.d_arr.coords[self.coord_type].units), to_units
            )
            info_txt = '{}: {}: {}'.format(
                self.well, self.name, cnv_txt
            )
            self.d_arr = self.d_arr.pint.to({self.coord_type: to_units})
            print_info(info_txt, 'info', logger)
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += '\n{}'.format(cnv_txt)
        except pint.DimensionalityError:
            warn_txt = '{}: {}: Pint cant convert from {} to {}'.format(
                self.well, self.name, str(self.d_arr.coords[self.coord_type].units), to_units
            )
            print_info(warn_txt, 'warning', logger)

    def take_sampling_from(self, log_curve, verbose=False):
        """
        Interpolates the data to match the depth parameter (sampling) of input log_curve,
        and returns a new log_curve

        :param log_curve:
            LogCurve2dNew object
        :param verbose:
            Bool

        :return
            LogCurve2dNew object
        """
        if self.coord_type != log_curve.coord_type:
            raise ValueError('The two depth formats (coord_type) are not the same: {} != {}'.format(
                self.coord_type, log_curve.coord_type))

        # # Create copy of original that we will modify
        # result = self.copy(suffix='')
        #
        # # If coord_units doesn't match convert to the units of the input log_curve
        # if result.coord_units != log_curve.coord_units:
        #     result.convert_coords_to(log_curve.coord_units)

        # Do the interpolation
        old_x = self.coords
        old_y = self.values
        new_x = log_curve.coords
        new_y = _interpolate(old_x, old_y, new_x)

        info_txt = 'Resampled to match {}'.format(log_curve.name)
        return replace_data_array(
            self,
            new_y,
            new_x,
            False,
            info_txt
        )

    def clean_data(self, nans=True, infinites=True, overwrite=False):
        if nans:
            _nans = np.isnan(self.d_arr.data.magnitude)
        if infinites:
            _infs = np.isinf(self.d_arr.data.magnitude)

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

        return replace_data_array(self,
               self.values[valid_indxs],
               self.coords[valid_indxs],
               overwrite,
               info_txt)

    def fill_gaps(self, method=None, extrapolate=False, overwrite=False,
                  fill_values=None):
        """

        :param method:
        :param extrapolate:
        :param overwrite:
        :param fill_values:
            np.ndarray of same size as self.values
            Only used when method is set to 'fill_with_values'
        :return:
        """
        info_txt = 'Gaps have been filled using {}'.format(method)
        if extrapolate:
            info_txt += ' with extrapolation'
        if method is None:
            method = 'interpolate'

        if method == 'interpolate':
            fill_value = None
            if extrapolate:
                fill_value = 'extrapolate'
            tmp = self.clean_data()
            out = _interpolate(tmp.coords, tmp.values, self.coords, fill_value=fill_value)
        elif method == 'fill_with_values':
            out = deepcopy(self.values)
            nans = np.isnan(out)
            out[nans] = fill_values[nans]
        else:
            raise NotImplementedError("Method '{}' has not been implemented".format(method))

        return replace_data_array(
            self,
            out,
            self.coords,
            overwrite,
            info_txt,
            suffix='fill_gaps'
        )

    def smooth(self,
               window_len: pint.Quantity,
               method='median',
               discrete_intervals=None,
               mask=None,
               mask_desc=None,
               verbose=False,
               overwrite=False,
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
        :param kwargs:
            keyword arguments passed on to _smooth
        :return:
            Smoothened LogCurve2dNew object when overwrite is False, else
        """
        if not self.is_evenly_spaced:
            warn_txt = 'Only evenly spaced data can be smoothened'
            print_info(warn_txt, 'warning', logger)
            return
        if not isinstance(window_len, pint.Quantity):
            # Simpy assume that it is given in the correct units
            w_len = ceil(window_len / self.step())
        else:
            w_len = ceil(window_len.to(self.coord_units).magnitude / self.step())

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
                print('Discrete intervals identified as list of Quantities')
                # Do unit conversion
                discrete_intervals = [_x.to(self.coord_units).magnitude for _x in discrete_intervals]
            else:
                # assume the intervals are given in correct units
                pass

            disc_txt = 'allowing discrete jumps at {} {} '.format(
                ', '.join('{:.2}'.format(_x) for _x in discrete_intervals), self.coord_units)
            # check if the proposed jump depths are within the depth range of the LogCurve

            # calculate the indexes of where the smoothened log are allowed discrete jumps.
            discrete_indexes = [np.argmin((self.coords[mask] - _z)**2) for _z in discrete_intervals]

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

        info_txt = 'Data smoothened {}using a {} window of length {}'.format(disc_txt, method, w_len)
        return replace_data_array(
            self,
            out,
            self.coords[mask],
            overwrite,
            info_txt,
            suffix='smooth'
        )

    def calc_mask(self,
                  cutoffs,
                  name=None,
                  log_table=None,
                  verbose=False):
        """
        Based on the different cutoffs in the 'cutoffs' dictionary a mask is created.
        In the resulting mask, a False value indicates that the data is masked out

        When log_table is given, both the log type and log name need to match to return a mask
        Else, it tries to match the log type of the current log curve, and then the log name, with the
        keys in the 'cutoffs' dictionary.

        When a key in 'cutoffs' match either 'Depth', 'md', 'owt' or 'twt', and is the correct coordinate
        of the log curve, the mask will work on the coordinate (I.E. a 'md' cut off will not work on a log curve
        with 'twt' coordinates)

        :param cutoffs:
            dict
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
            Dictionary of log type: log name key: value pairs that specify which log to use for each log type
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
            LogCurve2dNew object
        """
        from blixt_utils.utils import mask_string

        green_flag = True
        final_mask = None
        masks = []

        if not isinstance(cutoffs, dict):
            raise IOError('Cutoffs must be specified as dict, not {}'.format(type(cutoffs)))

        mask_description = mask_string(cutoffs, None)
        if len(cutoffs) == 0:  # no cutoffs, mask is all true
            final_mask = np.array(np.ones(len(self.d_arr)))
            mask_description = 'All true mask for empty cutoffs'

        if log_table is not None:  # See if this log curve matches the log table criteria
            green_flag = False
            for _log_type, _log_name in log_table.items():
                if (_log_type.lower() == self.log_type.lower()) and (_log_name.lower() == self.name.lower()):
                    green_flag = True
            if (not green_flag) and verbose:
                warn_txt = "The given log table does not match the current logcurve '{}', of type '{}', "\
                    "for well {}".format(
                        self.name, self.log_type, self.well
                )
                print_info(warn_txt, 'warning', logger)

        for lname in list(cutoffs.keys()):
            if lname.lower() in ['depth', 'md', 'twt', 'owt']:  # Create mask on coordinate
                if lname.lower() == 'depth':  # For the special case of using Depth instead of md in the cutoffs
                    this_lname = 'md'
                else:
                    this_lname = lname.lower()
                if self.coord_type.lower() == this_lname:
                    # True when coordinate type is same as in cutoffs
                    masks.append(
                        msks.create_mask(
                            self.coords, cutoffs[lname][0], cutoffs[lname][1]
                        )
                    )
                    if verbose:
                        info_txt = "Creating mask based on coordinate '{}', for well {}, based on {}".format(
                            self.coord_type, self.well, mask_description
                        )
                        print_info(info_txt, 'info', logger)
            if green_flag and (lname.lower() == self.name.lower()) or (lname.lower() == self.log_type.lower()):
                masks.append(
                    msks.create_mask(
                        self.values, cutoffs[lname][0], cutoffs[lname][1])
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

        return LogCurve2dNew(
            create_data_array(
                name=name,
                data=final_mask,
                data_units=self.units,
                coords=self.coords,
                coord_units=self.coord_units,
                coord_type=self.coord_type,
                verbose=verbose
            ),
            log_type='Mask',
            well=self.well,
            header={
                'name': name,
                'well': self.well,
                'log_type': 'Mask',
                'desc': mask_description
            }
        )

    def read(self,
             log_name,
             file_name,
             file_format,
             verbose=False,
             **kwargs):
        if file_format == 'las':
            # print('test: ', Q_(100, 'Ohmm'))  # Works!
            self.d_arr = create_data_array(*_read_las(log_name, file_name, verbose))
            self.name = log_name.lower()
            self.header.orig_filename = file_name


class LogCurve2D(object):
    """
    Class handling logs of a well
    Each log is a pair of depth data and log data

    """
    import xarray

    def __init__(self,
                 name: str,
                 data: np.ndarray,
                 depth: Param,
                 log_type=None,
                 well=None,
                 style=None,
                 unit=None,
                 header=None):
        """
        Initiates the most basic building block of a Well object, the LogCurve, which should hold one log

        :param name:
            str
        :param data:
            numpy.ndarray
            Containing the log data for this log.
            The reason why we don't use a Param for this numerical value(s) too, is that the LogCurve object has
            properties like unit etc.
        :param depth:
            Param
            Containing the depth data for this log.
            Must have same length as data
            If depth has no name it is assumed to be MD, and if no unit neither it is assumed to be in m
        :param well:
            str
        :param style:
            dict
            If present, it should contain a dictionary with keys typical for a template. See
            blixt_utils.misc.templates.necessary_keys

        """
        _data_sanity_checks(data)
        depth = _depth_sanity_checks(depth)
        if len(data) != len(depth):
            raise ValueError('Length of data ({}) and depth ({}) variables must be the same'.format(
                len(data), len(depth)))
        self.name = name
        self.data = data
        self.depth = depth
        self.well = well
        if header is None:
            header = {}

        if not isinstance(header, dict):
            raise TypeError('Input header must be a dictionary')

        if 'name' not in list(header.keys()) or header['name'] is None:
            self.header['name'] = name

        if 'well' not in list(header.keys()) or header['well'] is None:
            self.header['well'] = well

        if 'log_type' not in list(header.keys()) or header['log_type'] is None:
            self.header['log_type'] = log_type
        elif (log_type is None) and ('log_type' in list(header.keys())) and (header['log_type'] is not None):
            log_type = header['log_type']
        self.log_type = log_type

        # if 'style' not in list(header.keys()) or header['style'] is None:
        #     header['style'] = style
        if (style is None) and ('style' in list(header.keys())) and (header['style'] is not None):
            style = header['style']
            _ = header.pop('style')
        elif style is None:
            style = tmplts.return_empty_template()
        self.style = style

        if 'unit' not in list(header.keys()) or header['unit'] is None:
            header['unit'] = unit
        elif (unit is None) and ('unit' in list(header.keys())) and (header['unit'] is not None):
            unit = header['unit']
            # _ = header.pop('unit')
        self.unit = unit

    def __len__(self):
        return len(self.data)

    def copy(self, suffix='copy'):
        copied_log_curve = deepcopy(self)
        copied_log_curve.name = self.name + '_' + suffix
        copied_log_curve.header.name = self.name + '_' + suffix
        copied_log_curve.header.modification_date = datetime.now().isoformat()
        copied_log_curve.header.modification_history += \
            '\nCopy of {}'.format(self.name)
        return copied_log_curve

    def valid_depth(self, z, verbose=False):
        # TODO
        # This will not work for Z values that are increasingly negative with depth.
        # Will it help to use abs(z) < abs(min(depth)) ??
        if z < np.min(self.depth.value):
            if verbose:
                print('WARNING. depth {} is less than depth range of {}'.format(z, self.name))
            return False
        elif z > np.max(self.depth.value):
            if verbose:
                print('WARNING. depth {} is greater than depth range of {}'.format(z, self.name))
            return False
        else:
            return True

    def convert_depth_to(self, to_unit, verbose=False):
        if verbose:
            plt.plot(self.depth.value, self.data, c='b')
        success, self.depth.value = convert(self.depth.value, self.depth.unit, to_unit)
        if success:
            self.depth.unit = to_unit
        if verbose:
            plt.plot(self.depth.value, self.data, 'y')
            plt.show()

    def convert_data_to(self, to_unit):
        success, self.data = convert(self.data, self.unit, to_unit)
        if success:
            self.unit = to_unit

    def depth_plot(self,
                   mask=None,
                   mask_desc=None,
                   wis=None,
                   ax=None,
                   savefig=None,
                   **kwargs):
        """
        Plots log as a function of MD.

        :param mask:
            boolean numpy array of same length as self.data
        :param mask_desc:
            str
            Text describing which criteria the mask is based on
        :param wis:
            dict
            dictionary of working intervals
        :param ax:
            matplotlib.axes._subplots.AxesSubplot object
        :param savefig:
            str
            full path name of file to save plot to
        :param kwargs:
            keyword arguments passed on to plot
        :return:
            matplotlib.lines.line2D
            Line2D object with the current line
        """
        _savefig = False
        if savefig is not None:
            _savefig = True

        if (mask is not None) and (mask_desc is None):
            mask_desc = 'UNKNOWN'
        if mask is None:
            mask = np.array(np.ones(len(self.data)), dtype=bool)  # All True values -> all data is included

        # set up plotting environment
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        else:
            fig = ax.get_figure()
        x1 = deepcopy(self.data)
        x1[~mask] = np.nan
        ax.plot(
            self.data,
            self.depth.value,
            lw=0.5,
            alpha=0.5,
            c='gray'
        )
        this_line, = ax.plot(
            x1,
            self.depth.value,
        )
        ax.set_ylim(ax.get_ylim()[::-1])
        if self.style is not None:
            try:
                ax.set_xlim(self.style['min'], self.style['max'])
            except KeyError:
                print('No min/max plot range in style')
        title_txt = 'Well {}'.format(self.well)
        if mask_desc is not None:
            title_txt += ', using mask: {}'.format(mask_desc)
        ax.set_title(title_txt)
        ax.set_ylabel('{} [{}]'.format(self.depth.name, self.depth.unit))
        ax.set_xlabel('{} [{}]'.format(self.name, self.unit))

        if _savefig:
            fig.savefig(savefig)
        else:
            plt.show()

        return this_line

    def take_sampling_from(self, log_curve, verbose=False):
        return _take_sampling_from(self, log_curve, verbose=verbose)

    def smooth(self, window_len=None, method='median',
               discrete_intervals=None,
               verbose=False, overwrite=False, **kwargs):
        """
        :param discrete_intervals:
            list
            list of depth values (same unit as depth in LogCurve) at which the smoothened log are allowed
            discrete jumps.
            Typically the depth of the boundaries between two intervals (formations) in a well.

        """
        if method == 'median' and np.mod(window_len, 2) == 0:
            window_len += 1  # avoid even length windows
        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')

            # check if the proposed jump depths are within the depth range of the LogCurve

            # calculate the indexes of where the smoothened log are allowed discrete jumps.
            discrete_indexes = [np.argmin((self.depth.value - _z)**2) for _z in discrete_intervals]

            out = np.zeros(0)
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    out = np.append(out, _smooth(
                            self.data[:discrete_indexes[i]], window_len, method=method, **kwargs
                        )
                    )
                elif len(discrete_intervals) == i:  # last section
                    out = np.append(out, _smooth(
                            self.data[discrete_indexes[-1]:], window_len, method=method, **kwargs
                        )
                    )
                else:
                    out = np.append(out, _smooth(
                            self.data[discrete_indexes[i-1]:discrete_indexes[i]],
                            window_len, method=method, **kwargs))

        else:
            out = _smooth(self.data, window_len, method=method, **kwargs)

        if verbose:
            #print('Max diff between smoothed and original version:', np.nanmax(self.data - out))
            fig, ax = plt.subplots()
            ax.plot(self.data, c='black', label='Original')
            ax.plot(out, c='r', label='{}, window length {}'.format(method, window_len))
            if discrete_intervals is not None:
                for xx in discrete_intervals:
                    ax.axvline(xx, 0, 1, ls='--')
            ax.set_title(self.name)
            ax.legend()
        if overwrite:
            self.header.modification_date = datetime.now().isoformat()
            self.header.modification_history += \
                '\nSmoothened using {}, with window length {}'.format(method, window_len)
            self.data = out
            return None
        else:
            return out

    def despike(self, max_clip, window_len=None):
        if window_len is None:
            window_len = 13  # Around 2 meters in most wells
        smooth = self.smooth(window_len)
        spikes = np.where(self.data - smooth > max_clip)[0]
        spukes = np.where(smooth - self.data > max_clip)[0]
        out = deepcopy(self.data)
        out[spikes] = smooth[spikes] + max_clip  # Clip at the max allowed diff
        out[spukes] = smooth[spukes] - max_clip  # Clip at the min allowed diff
        return out

    def calc_depth_trend(self,
                         depth,
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
        log data as a function of the provided depth data

        Returns a list of optimized parameters for the trend function, with one list of parameters for each of
        the intervals

        :param depth:
            numpy.ndarray of length N
            The independent variable, could be MD, TVD, max burial depth, ...

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
            list of indexes (in the unmasked data) at which the trend are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        :param down_weight_outliers:
            bool
            If True, a weighting is calculated to minimize the impact of data far away from the median.

        :param verbose:
            bool
            If True plots are created and more information is printed

        return
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
        if loss is None:
            loss = 'cauchy'
        if (mask is not None) and (mask_desc is None):
            mask_desc = 'UNKNOWN'
        if mask is None:
            mask = np.array(np.ones(len(self.data)), dtype=bool)  # All True values -> all data is included
        verbosity_level = 0
        fig, ax = None, None
        results = []
        if verbose:
            verbosity_level = 2
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(self.data, depth, lw=0.5, c='grey')
            if discrete_intervals is not None:
                for i in discrete_intervals:
                    ax.axhline(depth[i], 0, 1, ls='--')

        def do_the_fit(_mask):
            weights = None

            # Test if there are NaN values in input data, which needs to be masked out
            if np.any(np.isnan(self.data[_mask])):
                nan_mask = np.isnan(self.data)
                _mask = msks.combine_masks([~nan_mask, _mask])

            # TODO
            #  do a similar test for infinite data: if np.any(np.isinf(_data))

            if down_weight_outliers:
                weights = 1. - np.sqrt((self.data[_mask] - np.median(self.data[_mask])) ** 2)
                weights[weights < 0] = 0.

            return least_squares(residuals, x0, args=(depth[_mask], self.data[_mask]),
                                 kwargs={'target_function': trend_function, 'weight': weights},
                                 loss=loss, verbose=verbosity_level)

        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    this_depth_mask = msks.create_mask(depth, '<', depth[discrete_intervals[i]])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', depth[discrete_intervals[-1]])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [depth[discrete_intervals[i-1]], depth[discrete_intervals[i]-1]])

                combined_mask = msks.combine_masks([mask, this_depth_mask])
                res = do_the_fit(combined_mask)
                results.append(res.x)
                if verbose:
                    this_depth = np.linspace(depth[this_depth_mask][0], depth[this_depth_mask][-1], 10)
                    ax.plot(self.data[combined_mask], depth[combined_mask])
                    ax.plot(trend_function(this_depth, *res.x), this_depth, c='b')
        else:
            res = do_the_fit(mask)
            results.append(res.x)
            if verbose:
                this_depth = np.linspace(depth[0], depth[-1], 10)
                ax.plot(self.data[mask], depth[mask])
                ax.plot(trend_function(this_depth, *res.x), this_depth, c='b')

        if verbose:
            ax.set_ylim(ax.get_ylim()[::-1])
            title_txt = 'Well {}'.format(self.well)
            if mask_desc is not None:
                title_txt += ', using mask: {}'.format(mask_desc)
            ax.set_title(title_txt)
            ax.set_xlabel(self.name)

        return results

    def apply_trend_function(self,
                             depth,
                             trend_parameters,
                             trend_function=None,
                             discrete_intervals=None,
                             verbose=False
                             ):
        """
        Uses the result from calc_depth_trend() to calculate the trend over the whole well.
        Which is trivial when discrete_intervals is None

        :param depth:
            numpy.ndarray of length N
            The independent variable, could be MD, TVD, max burial depth, ...

        :param trend_parameters:
            list
            List of arrays with optimized parameters of the trend_function
            This list is the result from calc_depth_trend()

        :param trend_function:
            function
            Function to calculate the residual against
                residual = target_function(depth, *x) - self.data
            trend_function takes x as arguments, and depth as independent variable,
            E.G. for a linear target function:
            def trend_function(depth, a, b):
                return a*depth + b
            Default is the linear function

        :param discrete_intervals:
            list
            list of indexes (in the unmasked data) at which the trend are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        :param verbose:
            bool
            If True plots are created and more information is printed

        return
            np.ndarray of the trend, with same length as the original log
        """
        if trend_function is None:
            trend_function = linear_function
        fig, ax = None, None
        output = np.zeros(0)
        if verbose:
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(self.data, depth, lw=0.5, c='grey')
            if discrete_intervals is not None:
                for i in discrete_intervals:
                    ax.axhline(depth[i], 0, 1, ls='--')

        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise TypeError('Interval indexes must be provided as a list')
            if len(trend_parameters) != len(discrete_intervals) + 1:
                raise IOError('Number of trends must be one more than the number of interval boundaries')
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    this_depth_mask = msks.create_mask(depth, '<', depth[discrete_intervals[i]])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', depth[discrete_intervals[-1]])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [depth[discrete_intervals[i-1]], depth[discrete_intervals[i]-1]])
                output = np.append(output, trend_function(depth[this_depth_mask], *trend_parameters[i]))
        else:
            output = trend_function(depth, *trend_parameters[0])

        if verbose:
            ax.plot(output, depth)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_title(self.well)
            ax.set_xlabel(self.name)

        return output

    def return_dataframe(self):
        return DataFrame({'depth': self.depth.value, 'data': self.data})

    
def _data_sanity_checks(value):
    """
    Check if a given input is suitable to be used for LogCurve.data. Raises the
    corresponding exception if it is not, otherwise silently passes.
    """
    if not isinstance(value, np.ndarray):
        msg = "LogCurve.data must be a NumPy array."
        raise ValueError(msg)
    if value.ndim != 1:
        msg = ("NumPy array for LogCurve.data has bad shape ('%s'). Only 1-d "
               "arrays are allowed for initialization.") % str(value.shape)
        raise ValueError(msg)


def _depth_sanity_checks(value):
    if not isinstance(value, Param):
        raise ValueError('Depth data must be a Param object')
    if (value.name is not None) and (value.name.lower() not in allowed_depth_formats):
        raise ValueError('Depth name {} not valid. Must be either: {}'.format(
            value.name, ', '.join(allowed_depth_formats)))
    if value.name is None:
        value.name = 'MD'
    if (value.name.upper() == 'MD') and (value.unit is None):
        value.unit = 'm'

    return value


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


def _take_sampling_from(log_curve1: LogCurve2D, log_curve2: LogCurve2D, verbose=False):
    """
    Takes values from log_curve1 and interpolates them according to depth parameter (sampling) of log_curve2,
    and returns a new log_curve
    """
    if log_curve1.depth.name.lower() != log_curve2.depth.name.lower():
        raise ValueError('The two depth formats (names) are not the same: {} != {}'.format(
            log_curve1.depth.name, log_curve2.depth.name
        ))
    if log_curve1.depth.unit != log_curve2.depth.unit:
        raise ValueError('The two depth units are not the same: {} != {}'.format(
            log_curve1.depth.unit, log_curve2.depth.unit
        ))
    new_y = _interpolate(log_curve1.depth.value, log_curve1.data, log_curve2.depth.value)
    if verbose:
        plt.plot(log_curve1.depth.value, log_curve1.data, 'o', log_curve2.depth.value, new_y, '-')
        plt.show()
    new_log_curve = deepcopy(log_curve1)
    new_log_curve.header.modification_date = datetime.now().isoformat()
    new_log_curve.header.modification_history += \
        '\nValues from {} adapted to depth sampling of {}'.format(log_curve1.name, log_curve2.name)
    new_log_curve.data = new_y
    new_log_curve.depth = log_curve2.depth

    return new_log_curve


def _read_las(log_name, file_name, verbose=False, encoding='UTF8'):
    """
    Ineffecient and simple las reader that is used to populates a LogCurve2dNew object with data from
    one log (log_name)
    NOTE: no information about well and log type is used

    :param log_name:
    :param file_name:
    :param verbose:
    :param encoding:
    :return:
    """
    from blixt_utils.io.io import well_reader
    # from blixt_utils.io.io import (get_las_well_info, get_las_curve_info, get_las_header, get_las_names_units,
    #                                get_las_start_data_line as las_line)
    accepted_depth_keys = rud.rename_well_logs['depth']

    with open(file_name, "r", encoding=encoding) as f:
        lines = f.readlines()
    null_val, generated_keys, well_dict = well_reader(lines, file_format='las')

    # Find and extract data
    data_units = None
    data_key = None
    # log_names, log_units = get_las_names_units(file_name, encoding)
    for _key in generated_keys:
        if log_name.lower() == _key.lower():
            data_key = _key
    # for _key, _units in zip(log_names, log_units):
    #     if log_name.lower() == _key.lower():
    #         data_key = _key
    #         data_units = _units

    if data_key is None:
        warn_txt = 'The parameter {} was not found in {}, which contains: {}'.format(
            # log_name, os.path.basename(file_name), ', '.join(log_names)
            log_name, os.path.basename(file_name), ', '.join(generated_keys)
        )
        print_info(warn_txt, 'warning', logger)
    else:
        data_units = well_dict['curve'][data_key]['unit']
    if (data_units is None) or (data_units == ''):
        warn_txt = 'No valid data unit was found in {} for data key {}'.format(
            os.path.basename(file_name), data_key
        )
        print_info(warn_txt, 'warning', logger)

    # Find depth parameter
    depth_key = None
    coord_type = None
    coord_units = None
    for _key in generated_keys:
    # for _key, _units in zip(log_names, log_units):
        if _key.lower() in [_x.lower() for _x in accepted_depth_keys]:
            depth_key = _key
            coord_type = 'md'
            coord_units = well_dict['curve'][_key]['unit']
            # coord_units = _units
        elif _key.lower() == 'owt':
            depth_key = _key
            coord_type = 'owt'
            coord_units = well_dict['curve'][_key]['unit']
            # coord_units = _units
        elif _key.lower() == 'twt':
            depth_key = _key
            coord_type = 'twt'
            coord_units = well_dict['curve'][_key]['unit']
            # coord_units = _units
    if depth_key is None:
        warn_txt = 'No valid depth parameter was found in {}, which contains: {}'.format(
            # os.path.basename(file_name), ', '.join(log_names)
            os.path.basename(file_name), ', '.join(generated_keys)
        )
        print_info(warn_txt, 'warning', logger)
    if (coord_units is None) or (coord_units == ''):
        warn_txt = 'No valid depth unit was found in {} for depth coordinate {}'.format(
            os.path.basename(file_name), depth_key
        )
        print_info(warn_txt, 'warning', logger)

    return log_name.lower(), well_dict['data'][data_key], data_units, well_dict['data'][depth_key], \
        coord_units, coord_type
    # data_line = las_line(file_name, encoding)
    # data, units = read_file(file_name, 'space', data_line, log_names, log_units, encoding)
    # return log_name.lower(), data[data_key], data_units, data[depth_key], coord_units, coord_type


def fix_units_for_pint(unit):

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

    return unit


def translate_units_for_pint(unit):
    """
    BECAUSE OF THE PROBLEMS WITH PINT-XARRAY AND UNITS. WE WILL STOP USING XARRAY AND THIS FUNCTION
    Make units understandable for pint-xarray
    I think there is a bug in pint-xarray, which stops it from taking custom units into account
    To alleviate this, we do a simplistic translation here

    You need to add the following lines to: "C:\<PATH TO site-packages>\pint\default_en.txt"

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


def handle_coords(coords, coord_type='md', coord_units=None, verbose=False):
    """
    Based on the input, handle_coords returns a pint.Quantity in the accepted
    units

    if coords is a pint.Quantity, coord_units are ignored.

    if coords is not a pint.Quantity, and coord_units is None, an Error is raised

    :param coords:
        flt, int, np.ndarray or pint.Quantity
    :param coord_type:
        str
        'md', 'tvd', 'owt', or 'twt'
    :param coord_units:
        str
        Any unit handled by pint.
        Extensions to pint default units should be added here:
        blixt_rp/units_to_pint.txt
        (jetbrains://pycharm/navigate/reference?project=PycharmProjects&path=blixt_rp/blixt_rp/units_to_pint.txt)
    :param verbose:
    :return:
        pint.Quantity
    """
    if coord_type.lower() not in ['md', 'tvd', 'twt', 'owt']:
        raise IOError('Coordinate type must be either md, tvd, twt or owt. Not {}'.format(coord_type))

    if not isinstance(coords, pint.Quantity):
        if coord_units is None:
            raise IOError('Units must be specified')
        coord_units = fix_units_for_pint(coord_units)
        # Assign units to the data through pint.Quantity
        coords = Q_(coords, coord_units)

    # Convert coordinates to the accepted units of meter or millisecond
    if coord_type.lower() in ['md', 'tvd']:
        return coords.to('meter')
    else:
        return coords.to('millisecond')


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
    coord_type = coord_type.lower()
    if coord_type not in ['md', 'tvd', 'twt', 'owt']:
        raise IOError('Dimension type must be either md, tvd, twt or owt. Not {}'.format(coord_type))

    coord_units = translate_units_for_pint(coord_units)

    # automatically convert the coordinate units to meter or ms
    if coord_type in ['md', 'tvd']:
        if coord_units.lower() in ['f', 'ft', 'feet', 'foot']:
            # _coords = coords * ureg.foot  # attach units to the coordinates
            # _coords = _coords.to(ureg.meter)  # convert to meter
            _coords = Q_(coords, 'foot')  # attach units to the coordinates
            _coords = _coords.to('meter')  # convert to meter
            if verbose:
                info_txt = 'Converting coords from {} to meter'.format(coord_units)
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
            #_coords = coords * ureg.second  # attach units to the coordinates
            # _coords = _coords.to(ureg.millisecond)  # convert to milliseconds
            _coords = Q_(coords, 'second')  # attach units to the coordinates
            _coords = _coords.to('millisecond')  # convert to milliseconds
            if verbose:
                info_txt = 'Converting coords from {} to ms'.format(coord_units)
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
    > log_values = _create_data_array(name, data, data_units, coords, coord_units, coord_type)

    It automatically converts the coordinate units to meter or ms

    The data is accessed through
    > print(log_values.name)
    > log_values.data.magnitude  # returns the data without coordinates
    > print(log_values.data.units) # prints the units

    The coord_type of the result is seen using
    > print(list(log_values.coords.keys())[0])
    The units of the coordinates
    > print(log_values.coords[coord_type].units)
    And the coordinate (index) themselves  can be accessed by
    > log_values.coords[coord_type].data

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


def replace_data_array(
        logcurve_in: LogCurve2dNew,
        data: np.ndarray,
        coordinates: np.ndarray,
        overwrite: bool,
        info_txt,
        suffix=''
):
    new_d_arr_coords = xr.DataArray(coordinates, dims=logcurve_in.coord_type)
    new_d_arr = xr.DataArray(name=logcurve_in.name, data=data, coords={logcurve_in.coord_type: new_d_arr_coords})
    new_d_arr = new_d_arr.pint.quantify(
        {logcurve_in.name: logcurve_in.units, logcurve_in.coord_type: logcurve_in.coord_units}
    )

    output = None
    if overwrite:
        logcurve_in.header.modification_date = datetime.now().isoformat()
        logcurve_in.header.modification_history += '\n{}'.format(info_txt)
        logcurve_in.d_arr = new_d_arr
    else:
        output = logcurve_in.copy(suffix)
        output.d_arr = new_d_arr
        output.header.modification_date = datetime.now().isoformat()
    return output


def is_equivalent(first: pint.Unit, second: pint.Unit):
    """
    Test if two units are equivalent
    :param first:
    :param second:
    :return:
    """
    try:
        factor = ureg.convert(1, first, second)
    except pint.DimensionalityError:
        return False
    return isclose(factor, 1)


def return_2d_log_curves(n=1500):
    lc = LogCurve2D(
        name='test_data',
        data=np.linspace(6, 8, n) + np.random.random(n),
        depth=Param(np.linspace(1400., 2200., n)),
        style={'full_name': 'A test',
               'min': 1,
               'max': 10},
        header={'unit': 'X'}
    )
    lc2 = LogCurve2D(
        name='irregular data',
        data=np.linspace(2, 4, n) + np.random.random(n),
        depth=Param(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), name='md', unit='ft'),
        style={'full_name': 'Another test', 'min': 1.5, 'max': 4.5, 'unit': 's'}
    )
    return lc, lc2


def test_2d():
    n = 1500
    # fig, ax = plt.subplots()
    lc, lc2 = return_2d_log_curves(n)
    # TODO
    # Smoothing looks wrong:
    #lc.smooth(41, discrete_intervals=[int(n/3), int(2*n/3)], verbose=True)
    # lc.smooth(40, discrete_intervals=None, verbose=True)
    # lc.smooth(41, discrete_intervals=None, verbose=True)
    # mask = np.ma.masked_inside(lc.data, 6.5, 7.5).mask
    # lc.depth_plot(mask=mask, mask_desc='Inside 2.5 to 3.5', ax=ax)

    # fig2, ax2 = plt.subplots()

    lc2.smooth(40, discrete_intervals=[1800., 2200.], verbose=True)

    # mask = np.ma.masked_inside(lc2.depth.value, 1750, 2500).mask
    # lc2.depth_plot(mask=mask, mask_desc='Depth 1750 to 2500', ax=ax2)
    # print(lc.header)
    # lc2 = lc.copy()
    # print(lc2.header)
    # df = lc.return_dataframe()
    # print(df)
    plt.show()


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


def test_take_sampling():
    lc, lc2 = return_2d_log_curves()
    # lc2.depth_plot()
    lc2.convert_depth_to('m', verbose=False)
    lc3 = lc.take_sampling_from(lc2, verbose=True)
    # lc3.style['min'] = None
    # lc3.style['max'] = None
    # lc3.depth_plot()


if __name__ == '__main__':
    test_interpolate()
