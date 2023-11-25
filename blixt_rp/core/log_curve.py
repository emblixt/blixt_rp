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

from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.signal_analysis.signal_analysis import smooth as _smooth
from blixt_utils.misc.curve_fitting import residuals, linear_function
import blixt_utils.misc.masks as msks


class Header(AttribDict):
    """
    A container for header information for a log curve object.
    A "Header" object may contain all header information necessary to describe a well log
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required
    """
    defaults = {
        'name': None,
        'well': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
        'orig_filename': None,  
        'modification_date': None,
        'modification_history': '',
        'log_type': None
        }

    def __init__(self, header=None):
        """
        """
        if header is None:
            header = {}
        super(Header, self).__init__(header)

    def __setitem__(self, key, value):
        """
        """
        # keys which shouldn't be modified
        if key in ['creation_date', 'modification_date', 'modification_history']:
            if key == 'modification_history':  # append instead of overwrite
                this_history = self.__getitem__(key)
                if this_history is None:
                    super(Header, self).__setitem__(key, value)
                else:
                    super(Header, self).__setitem__(key, '{}, {}'.format(this_history, value))
            pass
        else:

            # all other keys
            if isinstance(value, dict):
                super(Header, self).__setitem__(key, AttribDict(value))
            else:
                super(Header, self).__setitem__(key, value)

        super(Header, self).__setitem__('modification_date', datetime.now().isoformat())

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        #keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class LogCurve(object):
    """
    Class handling logs of a well
    """
    
    def __init__(self,
                 name=None,
                 block=None,
                 well=None,
                 data=np.array([]),
                 header=None):
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
        if 'unit' in list(header.keys()):
            self.unit = header['unit']
        else:
            self.unit = None

        if isinstance(header, dict):
            self.header = Header(header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise IOError('Input header is neither dictionary nor Header')
        #super(LogCurve, self).__setattr__('data', data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_log_type(self):
        log_type = None
        if self.header.log_type is not None:
            log_type = self.header.log_type
        return log_type

    log_type = property(get_log_type)

    def copy(self, suffix='copy'):
        copied_logcurve = deepcopy(self)
        copied_logcurve.name = self.name + '_' + suffix
        copied_logcurve.header.name = self.name + '_' + suffix
        copied_logcurve.header.modification_date = datetime.now().isoformat()
        copied_logcurve.header.modification_history += \
            '\nCopy of {}'.format(self.name)
        return copied_logcurve

    def smooth(self, window_len=None, method='median',
               discrete_intervals=None,
               verbose=False, overwrite=False, **kwargs):
        """
        :param discrete_intervals:
            list
            list of indexes at which the smoothened log are allowed discrete jumps.
            Typically the indexes of the boundaries between two intervals (formations) in a well.

        """
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
                            self.data[discrete_intervals[i-1]:discrete_intervals[i]],
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
                         mask_descr=None,
                         discrete_intervals=None,
                         down_weight_outliers=False,
                         down_weight_intervals=None,
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

        :param mask_descr:
            str
            Description of what cutoffs the mask is based on.

        :param discrete_intervals:
            list
            list of depth values at which the trend are allowed discrete jumps.
            Typically the boundaries between consequent intervals (formations) in a well.

        :param down_weight_outliers:
            bool
            If True, a weighting is calculated to minimize the impact of data far away from the median.

        :param down_weight_intervals:
            list
            # The indexes are for the unmasked data
            # List of lists, where each inner list is a 3 item list with [start_index, stop_index, weight]
            List of lists, where each inner list is a 3 item list with [start_depth, stop_depth, weight]
            where weight is a float between 0 and 1. 0 indicates the data within this interval is not taken into
            account at all. 1 is normal weight.

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
        if mask is None:
            mask = np.array(np.ones(len(self.data)), dtype=bool)  # All True values -> all data is included
        if (mask is not None) and (mask_descr is None):
            mask_descr = 'UNKNOWN'
        verbosity_level = 0
        fig, ax = None, None
        results = []
        if verbose:
            verbosity_level = 2
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(self.data, depth, lw=0.5, c='grey')
            if discrete_intervals is not None:
                for i in discrete_intervals:
                    ax.axhline(i, 0, 1, ls='--')

        def do_the_fit(_mask):
            weights = None
            # fig1, ax1 = plt.subplots(nrows=2)

            # Test if there are NaN values in input data, which needs to be masked out
            if np.any(np.isnan(self.data[_mask])):
                nan_mask = np.isnan(self.data)
                _mask = msks.combine_masks([~nan_mask, _mask])

            # TODO
            #  do a similar test for infinite data: if np.any(np.isinf(_data))

            if down_weight_outliers:
                weights = 1. - np.sqrt((self.data[_mask] - np.median(self.data[_mask])) ** 2)
                weights[weights < 0] = 0.
                # ax1[0].plot(weights)

            if down_weight_intervals is not None:
                if isinstance(down_weight_intervals, list):
                    if len(down_weight_intervals[0]) != 3:
                        raise IOError('down_weight_intervals must contain lists with 3 items')
                else:
                    raise IOError('down_weight_intervals must be a list')
                interval_weights = np.ones(len(self.data[_mask]))
                for int_weight in down_weight_intervals:
                    interval_weights[
                        np.argmin(np.sqrt((depth[_mask] - int_weight[0])**2)):\
                        np.argmin(np.sqrt((depth[_mask] - int_weight[1])**2))
                    ] = int_weight[2]
                # interval_weights = np.ones(len(self.data))
                if weights is None:
                    # weights = interval_weights[mask]
                    weights = interval_weights
                else:
                    weights = weights * interval_weights
                # ax1[1].plot(weights)

            return least_squares(residuals, x0, args=(depth[_mask], self.data[_mask]),
                                 kwargs={'target_function': trend_function, 'weight': weights},
                                 loss=loss, verbose=verbosity_level)

        if discrete_intervals is not None:
            if not isinstance(discrete_intervals, list):
                raise IOError('Interval indexes must be provided as a list')
            for i in range(len(discrete_intervals) + 1):  # always one more section than boundaries between them
                if i == 0:  # first section
                    this_depth_mask = msks.create_mask(depth, '<', discrete_intervals[i])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', discrete_intervals[-1])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [discrete_intervals[i-1], discrete_intervals[i]])

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
            if mask_descr is not None:
                title_txt += ', using mask: {}'.format(mask_descr)
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
                    this_depth_mask = msks.create_mask(depth, '<', discrete_intervals[i])
                elif len(discrete_intervals) == i:  # last section
                    this_depth_mask = msks.create_mask(depth, '>=', discrete_intervals[-1])
                else:  # all middle sections
                    this_depth_mask = msks.create_mask(
                        depth, '><', [discrete_intervals[i-1], discrete_intervals[i]])
                output = np.append(output, trend_function(depth[this_depth_mask], *trend_parameters[i]))
        else:
            output = trend_function(depth, *trend_parameters[0])

        if verbose:
            ax.plot(output, depth)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_title(self.well)
            ax.set_xlabel(self.name)

        return output

    def uniform_sampling_in_time(self,
                                 true_twt,
                                 time_step):
        t = np.arange(np.nanmin(true_twt), np.nanmax(true_twt), time_step)
        return t, np.interp(x=t, xp=true_twt, fp=self.data)

    #def get_log_name(self):
    #    log_name = None
    #    if self.header.name is not None:
    #        log_type = self.header.name
    #    return log_name
    #
    #name = property(get_log_name)
    
    
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