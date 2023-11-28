# -*- coding: utf-8 -*-
"""
Module for handling LogCurve objects
:copyright:
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from datetime import datetime
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from blixt_rp.core.header import Header
from blixt_rp.core.param import Param
from blixt_utils.signal_analysis.signal_analysis import smooth as _smooth
from blixt_utils.misc.curve_fitting import residuals, linear_function
from blixt_rp.rp_utils.definitions import allowed_depth_formats
from blixt_utils.misc.convert_data import convert
import blixt_utils.misc.masks as msks
import blixt_utils.misc.templates as tmplts


class MdDataPair(DataFrame):
    """
    Testing this class which inherits the pandas DataFrame object.
    Unsure if it is necessary
    """
    def __init__(self,
                 name,
                 md,
                 data,
                 *args,
                 **kwargs):
        print(*args)
        super().__init__({'md': md, 'data': data}, **kwargs)
        # super().__init__(*args, **kwargs)
        self.name = name

    @property
    def _constructor(self):
        return MdDataPair


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
            raise IOError('start value must be given as a Param object with name, value, units & description')
        self.start = start

        if 'stop' not in list(header.keys()) or header['stop'] is None:
            header['stop'] = stop
        elif (stop.value is None) and ('stop' in list(header.keys())) and (header['stop'] is not None):
            stop = header['stop']
        if (stop is not None) and not isinstance(stop, Param):
            raise IOError('stop value must be given as a Param object with name, value, units & description')
        self.stop = stop

        if 'step' not in list(header.keys()) or header['step'] is None:
            header['step'] = step
        elif (step.value is None) and ('step' in list(header.keys())) and (header['step'] is not None):
            step = header['step']
        if (step is not None) and not isinstance(step, Param):
            raise IOError('step value must be given as a Param object with name, value, units & description')
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

        if isinstance(header, dict):
            self.header = Header(header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise IOError('Input header is neither dictionary nor Header')
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
                raise IOError('Interval indexes must be provided as a list')
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
                raise IOError('Interval indexes must be provided as a list')
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
                raise IOError('Interval indexes must be provided as a list')
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


class LogCurve2D(object):
    """
    Class handling logs of a well
    Each log is a pair of depth data and log data
    """
    
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

        if isinstance(header, dict):
            self.header = Header(header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise IOError('Input header is neither dictionary nor Header')

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
                raise IOError('Interval indexes must be provided as a list')

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
                raise IOError('Interval indexes must be provided as a list')
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
                raise IOError('Interval indexes must be provided as a list')
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


def test():
    lc = LogCurve(
        name='test_data',
        data=np.linspace(2, 4, 1500) + np.random.random(1500),
        start=Param(name='start', value=24, unit='m'),
        stop=Param(name='stop', value=3430, unit='m'),
        style={'full_name': 'A test',
               'min': 1,
               'max': 10},
        header={'unit': 'X'}
    )
    fig, ax = plt.subplots()
    mask = np.ma.masked_inside(lc.get_depth(), 1000.,1750.).mask
    lc.depth_plot(mask=mask, mask_desc='Inside 1000 - 1750 m MD', ax=ax)
    lc.smooth(window_len=50, overwrite=True)
    lc.depth_plot(ax=ax)
    fit_parameters = lc.calc_depth_trend(
        lc.get_depth(),
        down_weight_outliers=True,
        verbose=False
    )
    fitted_log = lc.apply_trend_function(lc.get_depth(), fit_parameters, verbose=False)
    ax.plot(fitted_log, lc.get_depth(), '--')
    return lc


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


def test_md_data_pair():
    n = 1500
    data = np.linspace(2, 4, n) + np.random.random(n)
    md = np.linspace(1100., 2350., n)
    # lc = MdDataPair({'md': md, 'data': data})
    lc = MdDataPair(name='Test', md=md, data=data)
    print(lc.md)


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
    test_2d()
