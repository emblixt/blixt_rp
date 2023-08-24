# -*- coding: utf-8 -*-
"""
"""
from datetime import datetime
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from blixt_utils.io.io import well_reader
from blixt_utils.io.io import well_excel_reader
from blixt_rp.core.log_curve_new import LogCurve
from blixt_utils.misc.convert_data import convert as cnvrt
import blixt_rp.rp_utils.definitions as ud
from blixt_utils.misc.param import Param
from blixt_rp.core.header import Header

# global variables
logger = logging.getLogger(__name__)


class WellBlock(object):
    """
    A  Block is a collection of logs (or other data, perhaps with non-regular sampling) which share the same
    depth information, e.g start, stop, step, or if non-regular sampling all data must be sampled at likewise
    depths.
    """

    def __init__(self,
                 name=None,
                 well=None,
                 logs=None,
                 masks=None,
                 orig_filename=None,
                 start=Param(name='', value=None),
                 stop=Param(name='', value=None),
                 step=Param(name='', value=None),
                 regular_sampling=True,
                 header=None):
        """
        :param logs:
            list
            list of LogCurve objects
        """
        self.supported_version = ud.supported_version
        self.name = name
        if self.name is None:
            self.name = ud.def_lb_name
        self.well = well
        self.masks = masks

        if header is None:
            header = {}
        if 'name' not in list(header.keys()) or header['name'] is None:
            header['name'] = name
        if 'well' not in list(header.keys()) or header['well'] is None:
            header['well'] = well
        if orig_filename is None:
            orig_filename = 'See individual LogCurves'
        if 'orig_filename' not in list(header.keys()) or header['orig_filename'] is None:
            header['orig_filename'] = orig_filename
        self.header = Header(header)

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

        self.logs = {}
        if isinstance(logs, list) and (len(logs) > 0) and not isinstance(logs[0], LogCurve):
            raise IOError('Input log dictionary must have LogCurves as values')
        elif isinstance(logs, list) and (len(logs) > 0):
            for i, log_curve in enumerate(logs):
                try:
                    self.add_log_curve(log_curve)
                except IOError as msg:
                    print(msg)
                    continue
        else:
            print(logs)

    def __str__(self):
        return "Supported LAS Version : {0}".format(self.supported_version)

    def __len__(self):
        try:
            return len(self.logs[self.log_names()[0]].data)  # all logs within a Block should have same length
        except:
            return 0

    def get_depth_unit(self):
        return self.start.unit

    def get_md(self):
        return self.logs['depth'].data

    def get_tvd(self, tvd_key=None):
        if tvd_key is None:
            tvd_key = 'tvd'

        if tvd_key not in self.log_names():
            warn_txt = 'No True Vertical Depth log in {}, using MD'.format(self.well)
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            tvd = self.get_md()
        else:
            tvd = self.logs[tvd_key].data
        return tvd

    def keys(self):
        return self.__dict__.keys()

    def log_types(self):
        return [log_curve.log_type for log_curve in list(self.logs.values())]

    def log_names(self):
        return [log_curve.name for log_curve in list(self.logs.values())]

    def get_logs_of_name(self, log_name):
        return [log_curve for log_curve in list(self.logs.values()) if log_curve.name == log_name]

    def get_logs_of_type(self, log_type):
        return [log_curve for log_curve in list(self.logs.values()) if log_curve.log_type == log_type]

    def add_log_curve(self, log_curve):
        """
        Adds the provided log_curve to the Block.
        :param log_curve:
            core.log_curve.LogCurve
        :return:
        """
        if not isinstance(log_curve, LogCurve):
            raise IOError('Only LogCurve objects are valid input')

        if len(self.logs) != 0 and (len(self) != len(log_curve)):
            raise IOError('LogCurve must have same length as the other curves in this Block')

        if log_curve.name is None:
            raise IOError('LogCurve must have a name')

        if len(self.logs) == 0:
            # No previous log curves, use the current log curve to set vital information about the logs
            self.start = log_curve.start
            self.stop = log_curve.stop
            self.step = log_curve.step
            self.regular_sampling = log_curve.regular_sampling

        # if log_curve.log_type is None:
        #     raise IOError('LogCurve must have a log type')

        # test if start, stop and step are the same. But only when there are logs already present
        if len(self.logs) != 0:
            for attr in ['start', 'stop', 'step']:
                if self.__getattribute__(attr).value != log_curve.__getattribute__(attr).value:
                    raise IOError('The geometry of LogCurve {} ({}={}) does not match this well block ({}={})'.format(
                        log_curve.name, attr, log_curve.__getattribute__(attr), attr, self.__getattribute__(attr)
                    ))

        log_curve.well = self.well
        log_curve.block = self.name

        self.logs[log_curve.name.lower()] = log_curve

    def add_log(self,
                data,
                name,
                log_type,
                start, stop, step,
                unit,
                style=None,
                header=None):
        """
        Creates a LogCurve object based on input information, and tries to add the log curve to this Block.

        :param data:
            np.ndarray
        :param name:
            str
        :param log_type:
            str
        :param start:
        :param stop:
        :param step:
            Param
            Parameters containing start, step and step, with units
        :param unit:
            str
        :param style:
            dict
            If present, it should contain a dictionary with keys typical for a template. See
            blixt_utils.misc.templates.necessary_keys
        :param header:
            dict
            Should at least contain the keywords 'unit' and 'desc'
        :return:
        """
        # modify header
        if header is None:
            header = {}
        if 'name' not in list(header.keys()):
            header['name'] = name
        if 'full_name' not in list(header.keys()):
            header['full_name'] = name
        if 'well' not in list(header.keys()):
            header['well'] = self.well
        if 'log_type' not in list(header.keys()):
            header['log_type'] = log_type
        if 'unit' not in list(header.keys()):
            header['unit'] = None
        if 'desc' not in list(header.keys()):
            header['desc'] = None

        log_curve = LogCurve(
            name=name,
            log_type=log_type,
            block=self.name,
            well=self.well,
            start=start,
            stop=stop,
            step=step,
            style=style,
            data=data,
            unit=unit,
            header=header)

        self.add_log_curve(log_curve)

    def twt_at_logstart(self, log_name, water_vel, repl_vel, water_depth, kb):
        """
        Calculates the two-way time [s] to the top of the log.

        Inspired by
        https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb

        :param log_name:
            string
        :param water_vel:
            float
            Sound velocity in water [m/s]
        :param repl_vel:
            float
            Sound velocity [m/s] in section between sea-floor and top of log
        water_depth
            float
            Water depth in meters.
        kb
            float
            Kelly bushing elevation in meters

        :return:
            float
            twt [s] to start of log
        """

        top_of_log = self.get_start(log_name=log_name)  # Start of log in meters MD
        repl_int = top_of_log - np.abs(kb) - np.abs(water_depth)  # Distance from sea-floor to start of log
        # water_twt = 2.0 * (np.abs(water_depth) + np.abs(kb)) / water_vel  # TODO could it be np.abs(water_depth + np.abs(kb)) / water_vel
        water_twt = 2.0 * np.abs(water_depth + np.abs(kb)) / water_vel
        repl_twt = 2.0 * repl_int / repl_vel

        # print('KB elevation: {} [m]'.format(kb))
        # print('Seafloor elevation: {} [m]'.format(water_depth))
        # print('Water time: {} [s]'.format(water_twt))
        # print('Top of Sonic log: {} [m]'.format(top_of_log))
        # print('Replacement interval: {} [m]'.format(repl_int))
        # print('Two-way replacement time: {} [s]'.format(repl_twt))
        # print('Top-of-log starting time: {} [s]'.format(repl_twt + water_twt))

        return water_twt + repl_twt

    def time_to_depth(self, log_start_twt, log_name, spike_threshold, repl_vel,
                      sonic=False, feet_unit=False, us_unit=False,
                      debug=False):
        """
        Calculates the twt as a function of md
        https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb

        :param log_start_twt:
            float
            Two-way time in seconds down to start of log
        :param log_name:
            str
            Name of log. slowness or velocity, used to calculate the integrated time
        :param repl_vel:
            float
            Sound velocity [m/s] in section between sea-floor and top of log, Also used to fill in NaNs in sonic or velocity
            log
        :param sonic:
            bool
            Set to true if input log is sonic or slowness
        :param feet_unit:
            bool
            Set to true if input log is using feet (e.g. "us/f")
        :param us_unit:
            bool
            Set to true if input log is using micro seconds and not seconds (e.g. "us/f" or "s/f"
        :return:
        """
        if log_name not in self.log_names():
            raise IOError('Log {} does not exist in well {}'.format(
                log_name, self.well
            ))

        # Replace NaN values of input log using repl_vel
        if feet_unit:
            success, repl_vel = cnvrt(repl_vel, 'm', 'ft')
        if sonic:
            repl = 1. / repl_vel
        else:
            repl = repl_vel
        if us_unit:
            repl = repl * 1e6
        nan_mask = np.ma.masked_invalid(self.logs[log_name].data).mask

        # Smooth and despiked version of vp
        smooth_log = self.logs[log_name].despike(spike_threshold)
        smooth_log[nan_mask] = repl

        if debug:
            fig, ax = plt.subplots()
            ax.plot(self.logs['depth'].data, smooth_log, 'r', lw=2)
            ax.plot(self.logs['depth'].data, self.logs[log_name].data, 'k', lw=0.5)
            # ax.plot(self.logs['depth'].data[13000:14500]/3.2804, smooth_log[13000:14500]*3.2804, 'r', lw=2)
            # ax.plot(self.logs['depth'].data[13000:14500]/3.2804, self.logs[log_name].data[13000:14500]*3.2804, 'k', lw=0.5)
            ax.legend(['Smooth and despiked', 'Original'])

        # Handle units
        if sonic:
            scaled_dt = self.get_step() * np.nan_to_num(smooth_log)
            if feet_unit:  # sonic is in feet units, step is always in meters
                scaled_dt = scaled_dt * 3.28084
        else:
            scaled_dt = self.get_step() * np.nan_to_num(1. / smooth_log)
            if feet_unit:  # velcity is in feet, step is always in meters
                scaled_dt = scaled_dt / 3.28084
        if us_unit:
            scaled_dt = scaled_dt * 1.e-6

        tcum = 2 * np.cumsum(scaled_dt)
        tdr = log_start_twt + tcum

        if debug:
            fig, ax = plt.subplots()
            ax.plot(self.logs['depth'].data, scaled_dt, 'b', lw=1)
            ax2 = ax
            ax2.plot(self.logs['depth'].data, tdr, 'k', lw=1)
            plt.show()

        return tdr

    def sonic_to_vel(self, vel_names=None):
        """
        Converts sonic to velocity

        :param vel_names:
            list
            Optional
            list of strings, with names of the resulting velocities
            ['vp', 'vs']
        :return:
        """
        from blixt_utils.misc.convert_data import convert as cnvrt

        if vel_names is None:
            vel_names = ['vp', 'vs']

        for ss, vv, vtype in zip(
                ['ac', 'acs'], vel_names, ['P velocity', 'S velocity']
        ):
            info_txt = ''
            if ss not in self.log_names():
                continue
            else:
                din = self.logs[ss].data
                success, dout = cnvrt(din, 'us/ft', 'm/s')

            self.add_log(
                dout,
                vv,
                vtype,
                header={
                    'unit': 'm/s',
                    'modification_history': '{} Calculated from {}'.format(info_txt, ss.upper()),
                    'orig_filename': self.logs['ac'].header.orig_filename
                }
            )

    def add_well_path(self, survey_points, survey_file=None, verbose=True):
        """
        adds (interpolates) the TVD (relative to KB) based on the input survey points for each MD value in this well

        :param survey_points:
            dict
            Dictionary with required keywords 'MD' and 'TVD'
            the associated items for 'MD' and 'TVD' keys are lists of measured depth and True vertical depth in meters
            relative to KB.
            If key 'INC' exists, it assumes it is the inclination in degrees
            Because there are so many flavors of how the survey points are stored in a file, you need to write specific
            readers for each files that spits out the result in a dictionary with 'MD' and 'TVD' keys

            The function read_wellpath() in the blixt_utils library tries to read many variants of survey data files

            if survey_points is None, the well is assumed vertical and MD is used as TVD

        :param survey_file:
            str
            Name of file survey points are calculated from.
            Used in history of objects
        :return:

        """
        if isinstance(survey_file, str):
            fname = survey_file
        else:
            fname = 'unknown file'

        md = self.logs['depth'].data

        # Calculate and write TVD to well
        if survey_points is None:
            new_tvd = self.get_tvd()
        else:
            new_tvd = interp1d(survey_points['MD'], survey_points['TVD'],
                               kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')(md)
        if verbose:
            fig, axes = plt.subplots(1, 2, figsize=(10, 8))
            if survey_points is not None:
                axes[0].plot(survey_points['MD'], survey_points['TVD'], '-or', lw=0)
            axes[0].plot(md, new_tvd)
            axes[0].set_xlabel('MD [m]')
            axes[0].set_ylabel('TVD [m]')
            axes[0].legend(['Survey points', 'Interpolated well data'])

        if survey_points is None:
            mod_history = 'Assuming well is vertical, TVD calculated from MD directly'
        else:
            mod_history = 'TVD calculated from {}'.format(fname)

        self.add_log(
            new_tvd,
            'tvd',
            'Depth',
            header={
                'unit': 'm',
                'desc': 'True vertical depth',
                'modification_history': mod_history})

        # Try the same for inclination
        if (survey_points is not None) and ('INC' in list(survey_points.keys())):
            new_inc = interp1d(survey_points['MD'], survey_points['INC'],
                               kind='linear',
                               bounds_error=False)(md)
            if verbose:
                axes[1].plot(survey_points['MD'], survey_points['INC'], '-or', lw=0)
                axes[1].plot(md, new_inc)
                axes[1].set_xlabel('MD [m]')
                axes[1].set_ylabel('Inclination [deg]')
                axes[1].legend(['Survey points', 'Interpolated well data'])
            self.add_log(
                new_inc,
                'inc',
                'Inclination',
                header={
                    'unit': 'deg',
                    'desc': 'Inclination',
                    'modification_history': 'calculated from {}'.format(fname)})

        if verbose:
            fig.suptitle('Well: {}'.format(self.well))
            plt.show()

    def add_twt(self, twt_points, twt_file=None, verbose=True):
        """
       adds (through interpolation) the two-way time (TWT) in seconds [s] to the well based on the input twt points

       If no twt_points (None) are provided, it will search for an existing One-way time log in the well, and convert those to
       TWT

       If the overwrite option is set to True, it will try to overwrite any existing TWT log with the interpolated
       results from twt_points

        :param twt_points:
            dict
            Dictionary with required keywords 'MD' and 'TWT'
            the associated items for 'MD' and 'TWT' keys are lists of measured depth [m] and two-way-time [s]
            NOTE: If TWT is negative, and increasingly negative with depth, this function changes its sign so
            that it is increasingly positive with depth

            The function read_petrel_checkshots() in the blixt_utils is useful to calculate the input twt_points data
            based on a checkshots file exported from Petrel

        :param twt_file:
            str
            Name of file the twt points are calculated from.
            Used in history of objects

        :param verbose:
            Bool
            If True, plots are generated to show the result

        :return:

        """
        _x = None
        _y = None
        _name = None

        if isinstance(twt_file, str):
            fname = twt_file
        else:
            fname = 'unknown file'

        md = self.logs['depth'].data

        if twt_points is None:
            # Use the first existing One-way time log to calculate a TWT log
            if 'One-way time' in self.log_types():
                log_curve = self.get_logs_of_type('One-way time')[0]
                new_twt = 2. * log_curve.data
                fname = log_curve.name
                _name = 'twt_from_owt'
                _x = md
                _y = 1000.0 * log_curve.data
            else:
                return None
        else:
            # Calculate TWT from input points
            sign = 1.0
            if sum(twt_points['TWT'][-10:]) < 0:
                sign = -1.0
            new_twt = interp1d(twt_points['MD'], sign * np.array(twt_points['TWT']),
                               kind='linear',
                               bounds_error=False,
                               fill_value='extrapolate')(md)
            _name = 'twt_from_interp'
            _x = twt_points['MD']
            _y = sign * 1000. * np.array(twt_points['TWT'])

        if verbose:
            fig, axes = plt.subplots(1, 1, figsize=(5, 8))
            axes.plot(_x, _y, '-or', lw=0)
            axes.plot(md, 1000. * new_twt)
            axes.set_xlabel('MD [m]')
            axes.set_ylabel('TWT [ms]')
            axes.legend(['TDR points', 'Interpolated well data'])

        self.add_log(
            new_twt,
            _name,
            'Two-way time',
            header={
                'unit': 's',
                'desc': 'Two-way time, positive downwards',
                'modification_history': 'calculated from {}'.format(fname)})

        if verbose:
            fig.suptitle('Well: {}'.format(self.well))
            plt.show()

    def calc_toc(self, log_table, r0=None, ac0=None, lom=None, mask_name=None, verbose=False):
        """
        Calculates the TOC using the functions from Passey et al. 1990 "A practical model for organic richness ...",
        which are implemented in rp_core.py
        The "trend" TOC is calculated using a linear trend in r and ac as baseline, so that the manual picking is not
        necessary

        Args:
           log_table:
                dict
                A dictionary specifying which resistivity and ac log to use. E.G.
                {'Resistivity: 'rdep', 'Sonic': 'ac'}
           r0:
                float
                Picked baseline value of the resistivity
           ac0:
                float
                Picked baseline value of the sonic
           lom:
                float
                Unitless number somewhere between 6 and 11 which describes the level of organic metamorphism units
                (Hood et al. 1975)
            mask_name:
                str
                Name of the mask to be used
           verbose:

        Returns:

        """
        from blixt_rp.rp_utils.calc_toc import calc_toc
        necessary_log_types = ['Resistivity', 'Sonic']
        for _key in necessary_log_types:
            if _key not in list(log_table.keys()):
                raise IOError('No log of type {} is specified'.format(_key))
            if log_table[_key] not in self.log_names():
                raise IOError('The specified log {} is not present in this well block: {} {}'.format(
                    log_table[_key], self.well, self.name
                ))

        # check if a mask named mask_name exists
        mask = None
        mask_desc = None
        if mask_name is not None:
            if self.masks is not None:
                if mask_name not in list(self.masks.keys()):
                    warn_txt = 'No mask applied. The desired mask {} does not exist in well {}'.format(
                        mask_name, self.well)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)
                else:
                    mask = self.masks[mask_name].data
                    mask_desc = self.masks[mask_name].header.desc
        toc_trend, toc_picked, dlr_trend, dlr_picked = calc_toc(self.logs[log_table['Resistivity']],
                                         self.logs[log_table['Sonic']],
                                         self.logs['depth'],
                                         r0, ac0, lom,
                                         mask=mask, mask_desc=mask_desc, verbose=verbose)
        self.add_log_curve(toc_trend)
        self.add_log_curve(toc_picked)
        self.add_log_curve(dlr_trend)
        self.add_log_curve(dlr_picked)

    def depth_plot(self,
                   log_names=None,
                   log_types=None,
                   mask=None,
                   mask_desc=None,
                   wis=None,
                   ax=None,
                   savefig=None,
                   **kwargs):
        """
        Plots log as a function of MD.

        :param log_names:
            list
            list of log names which we want to plot.
            If both log_names and log_types are specified, log_names takes precedence
        :param log_types:
            list
            list of log types we like to plot
            If both log_names and log_types are specified, log_names takes precedence
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
        """
        _savefig = False
        if savefig is not None:
            _savefig = True

        # set up plotting environment
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        else:
            fig = ax.get_figure()

        if (log_names is None) and (log_types is None):
            warn_txt = 'Either log_names or log_types must be specified'
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)

        list_of_log_curves =[]
        if log_names is not None:
            if not isinstance(log_names, list):
                raise IOError('log_names must be list of log names')
            for log_name in log_names:
                list_of_log_curves += self.get_logs_of_name(log_name)
        elif log_types is not None:
            if not isinstance(log_types, list):
                raise IOError('log_types must be list of log types')
            for log_type in log_types:
                list_of_log_curves += self.get_logs_of_type(log_type)
        if len(list_of_log_curves) == 0:
            raise IOError('No valid log_names {}, nor log_types {}, found'.format(log_names, log_types))

        logs_plotted = []
        lines_drawn = []
        for log_curve in list_of_log_curves:
            logs_plotted.append(log_curve.name)
            lines_drawn.append(log_curve.depth_plot(mask, mask_desc, wis, ax=ax, savefig=None, **kwargs))

        ax.legend(handles=lines_drawn, labels=logs_plotted)

        if _savefig:
            fig.savefig(savefig)
        else:
            plt.show()

    def read_log_data(self,
                      filename,
                      only_these_logs=None,
                      rename_well_logs=None,
                      sheet_name=None,
                      header_line=None,
                      well_key=None,
                      depth_key=None,
                      rename_logs_string=None,
                      depth_is_md=True
                      ):
        """
        Reads in a las file or excel file (filename) and adds the selected logs (listed in only_these_logs) to this
        well block.

        When reading from an excel file, the well name to which this well block belongs must be set to be sure we
        are reading the right data

        :param filename:
            str
            name of las file
        :param only_these_logs:
            dict
            dictionary of log names to load from the las file (keys), and corresponding log type as value
            if None, all are loaded
        :param rename_well_logs:
            dict
            E.G.
            {'depth': ['DEPT', 'MD']}
            where the key is the wanted well log name, and the value list is a list of well log names to translate from
        :param use_this_well_name:
            str
            Name we would like to use.
            Useful when different las file have different well names for the same well
        :param note:
            str
            String containing notes for the las file being read

        :param sheet_name:
            str
            Only needed when reading from excel file.
            Name of sheet to fetch the data from
        :param header_line:
            int
            Only needed when reading from excel file.
            (Pythonic) line number of the header lines (from which the column headers (keys) are taken from)
        :param well_key:
            str
            Only needed when reading from excel file.
            Column header from which the well name is taken
        :param depth_key:
            str
            Only needed when reading from excel file.
            Column header from which the depth information is taken
        :param rename_logs_string:
            str
            Only needed when reading from excel file.
            String containing data column header name and data output name in the same format as the 'Translate log names'
            item in the output from project_wells_new()
            e.g. 'TOC.Any->toc, LOM from equivalent Ro->lom'
            Where 'TOC.Any' and 'LOM from equivalent Ro' are the column headers under which the data is found.
            The function interpret_rename_string() must be able to interpret this string
        :param depth_is_md:
            bool
            Only needed when reading from excel file.
            If True, the depth information in the excel file is in MD.
            If False, the depth information is interpreted as TVD

        :return:
        """
        file_format = None
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "las":
            file_format = 'las'
        elif ext == 'txt':  # text files from Avseth & Lehocki in the  Spirit study
            file_format = 'RP well table'
        elif ext == 'xlsx':
            file_format = 'Excel sheet'
        else:
            raise IOError("File format '{}'. not supported!".format(ext))

        # make sure the only_these_logs is using small caps for all well logs
        if isinstance(only_these_logs, dict):
            for key in list(only_these_logs.keys()):
                only_these_logs[key.lower()] = only_these_logs.pop(key)

        # Make sure the only_these_logs is updated with the desired name changes defined in rename_well_logs
        if isinstance(only_these_logs, dict) and (isinstance(rename_well_logs, dict)):
            for okey in list(only_these_logs.keys()):
                for rkey in list(rename_well_logs.keys()):
                    if okey.lower() in [x.lower() for x in rename_well_logs[rkey]]:
                        only_these_logs[rkey.lower()] = only_these_logs.pop(okey)

        # Make sure only_these_logs is a dictionary
        if isinstance(only_these_logs, str):
            only_these_logs = {only_these_logs.lower(): None}

        if file_format == 'Excel sheet':
            if self.well is None:
                raise IOError('Well name must be given to block prior to reading an Excel sheet with data')

            null_val, generated_keys, well_dict = well_excel_reader(
                self.well,
                filename,
                sheet_name,
                header_line,
                well_key,
                depth_key,
                rename_logs_string,
                depth_is_md
            )
        else:
            with open(filename, "r", encoding='UTF8') as f:
                lines = f.readlines()
            null_val, generated_keys, well_dict = well_reader(lines, file_format=file_format)

            # Rename well logs
            if rename_well_logs is None:
                rename_well_logs = {'depth': ['Depth', 'DEPT', 'MD', 'DEPTH']}
            elif isinstance(rename_well_logs, dict) and ('depth' not in list(rename_well_logs.keys())):
                rename_well_logs['depth'] = ['Depth', 'DEPT', 'MD', 'DEPTH']

            for key in list(well_dict['curve'].keys()):
                well_dict['curve'][key]['orig_name'] = ''
                for rname, value in rename_well_logs.items():
                    if key.lower() in [x.lower() for x in value]:
                        well_dict['curve'][key]['orig_name'] = '{}'.format(', '.join(value))
                        info_txt = 'Renaming log from {} to {}'.format(key, rname)
                        # print('INFO: {}'.format(info_txt))
                        logger.info(info_txt)
                        well_dict['curve'][rname.lower()] = well_dict['curve'].pop(key)
                        well_dict['data'][rname.lower()] = well_dict['data'].pop(key)

        logger.debug('Reading {}'.format(filename))

        winf = well_dict['well_info']
        for key in generated_keys:
            self.add_log_curve(
                LogCurve(
                    key,
                    well_dict['data'][key],
                    Param(name='start', value=winf['strt']['value'], unit=winf['strt']['unit'], desc=winf['strt']['desc'],
                    Param(name='stop', value=winf['stop']['value'], unit=winf['stop']['unit'], desc=winf['stop']['desc'],
                    Param(name='step', value=winf['step']['value'], unit=winf['step']['unit'], desc=winf['step']['desc'],
                          XXXX
                          # TODO Continue here

                                      )
            )



def test():
    list_of_log_curves = [
        LogCurve(
            name='log1',
            data=np.linspace(1, 100),
            start=Param(name='start', value=2500., unit='m'),
            stop=Param(name='stop', value=3500., unit='m'),
            log_type='A'
        ),
        LogCurve(
            name='log2',
            data=np.linspace(2, 102),
            start=Param(name='start', value=2500., unit='m'),
            stop=Param(name='stop', value=3500., unit='m'),
            log_type='A'
        ),
        LogCurve(
            name='log3',
            data=np.linspace(3, 103),
            start=Param(name='start', value=2500., unit='m'),
            stop=Param(name='stop', value=3500., unit='m'),
            log_type='B'
        ),
        LogCurve(
            name='log4',
            data=np.linspace(4, 104),
            start=Param(name='start', value=2000., unit='m'),
            stop=Param(name='stop', value=3000., unit='m'),
            log_type='B'
        )
    ]
    wb = WellBlock(
        logs=list_of_log_curves
    )
    return wb
