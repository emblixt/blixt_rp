# -*- coding: utf-8 -*-
"""
Module for handling wells
The goal is to have one Well object, which contains "all" the well specific
information (with a minimum required set), and a log (or curve) object for each
log.

A well can only have one trajectory

It should be possible to save a Well object as a las or json file.
It should be possible to add and remove logs from a Well object
It should be possible to do fluid replacement etc. on logs
Use inspiration from RokDoc how to classify the different logs
Use the WellsAndLogs_template.xlsx to add the plotting style of the well
and of the different log types
Take inspiration from obspy and converter to create these objects
:copyright:
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import logging
import os, sys
import matplotlib.pyplot as plt
import pint
from scipy.constants import degree
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

from blixt_rp.core.log_curve_new import LogCurve

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from .. import ureg, Q_

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)

def check_depth_type(input_lc: LogCurve, depth_type: str = 'md') -> bool:
    """

    :param input_lc:
    :param depth_type:
        str
        'md', 'tvd', 'owt', or 'twt'

    :return:
        bool
    """
    return input_lc.depth_type == depth_type

class WellTrajectory:
    """
    Handles the trajectory of a well.
    The common, and necessary, dimension is measured depth; MD
    Other dimensions, like true vertical depth (TVD), inclination (INC), burial depth (BD), will follow
    """
    def __init__(self,
                 md: pint.Quantity,
                 tvd_kb:  LogCurve | None = None,
                 inc: LogCurve | None = None,
                 verbose: bool = False):
        """

        :param md:
            pint.Quantity
            An array of measured depth along the well with units
        :param tvd:
            LogCurve
            LogCurve object with true vertical depth relative to Kelly Bushing TVD data as a function of MD
            So in a vertical well tvd_kb is equal to MD
        :param inc:
            LogCurve
            The inclination of a well trajectory is typically defined as the angle between the wellbore and a
            vertical line (parallel to Earth's gravity) at a specific point along the well path. This angle is
            measured in degrees, with:
                0° meaning the well is vertical (straight down),
                90° meaning the well is horizontal (parallel to the surface),
                Angles greater than 90° indicating "drilling up" rather than down.

        """
        # from blixt_rp.core.log_curve_new import _interpolate
        self._md = md.to('meter')
        self.verbose = verbose

        # TVD data are quite safe to extrapolate
        self._tvd_kb = self.set_param(tvd_kb, 'meter', 'extrapolate')

        # Inclination can not be extrapolated
        self._inc = self.set_param(inc, 'degree', None)

    def set_param(self,
                  input_log: LogCurve | None,
                  units: str | None,
                  fill_value: str | None) -> pint.Quantity | None:
        """
        Interpolates the input log to match the MD of the WellTrajectory

        :param input_log:
            LogCurve
            LogCurve with data as a function of MD that should be attached as a property of the WellTrajectory
        :param units:
            str
            Valid Pint name of a unit we want the property to be given in
        :param fill_value:
            str
            Parameter sent further to _interpolate which decides how to handle data outside the bound of the
            input LogCurve
        :return:
        """
        from blixt_rp.core.log_curve_new import _interpolate

        if input_log is None:
            return None

        # Check what depth domain the input log data is given in
        if check_depth_type(input_log, 'md'):  # Input log must be given as a function of MD
            if not input_log.units == ureg(units):
                input_log.units = units
            new_x = self._md.magnitude
            old_x = input_log.depth.values
            old_y = input_log.values
            new_y = _interpolate(old_x, old_y, new_x, fill_value=fill_value)

            if self.verbose:
                fig, ax = plt.subplots()
                ax.plot(old_x, old_y, 'k--')
                ax.plot(new_x, new_y, 'r')
                plt.show()

            return Q_(new_y, units)

    @property
    def md(self):
        return self._md

    @property
    def tvd_kb(self):
        return self._tvd_kb

    @tvd_kb.setter
    def tvd_kb(self, value: LogCurve):
        # TVD data are quite safe to extrapolate
        self._tvd_kb = self.set_param(value, 'meter', 'extrapolate')

    @property
    def inc(self):
        return self._inc

    @inc.setter
    def inc(self, value: LogCurve):
        self._inc = self.set_param(value, 'degree', None)

    def burial_depth(self, kelly_busing: pint.Quantity, water_depth:pint.Quantity) -> pint.Quantity | None:
        """
        Returns the burial depth (vertical depth below mud line (sea floor)) calculated from the tvd_kb
        :param kelly_busing:
        :param water_depth:
        :return:
        """
        if self.tvd_kb is None:
            return None
        return self.tvd_kb - np.abs(kelly_busing) - np.abs(water_depth)


class Well(object):
    """
    Class handling a well, with LogCurve (from log_curve_new) objects for each curve of well log data.
    Main difference with earlier versions is that each LogCurve object contains its associated depth data, and units,
    so that they don't have to be regularly sampled, or with the same sampling.
    This also means that a LogCurve object can contain other data than typical log data, such as core samples.
    """
    from blixt_rp.core.core import Header, LogTable, Template
    from blixt_rp.core.log_curve_new import LogCurve

    def __init__(self,
                 header: Header | None = None,
                 logs: list | None = None,
                 style: Template | None = None
                 ):
        """

        :param header:
            Header
            dict type which contains
        :param logs:
        :param style:
            Template object or dict
        """
        from blixt_rp.core.core import Header, Template
        if header is None:
            self.header = Header({})
        elif isinstance(header, dict):
            self.header = Header(header=header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise TypeError('header must be either a dict or a Header, not {}'.format(type(header)))
        self.logs = logs
        if self.header.name is None:
            if 'well_info' in list(self.header.__dict__.keys()):
                self.header.name = self.header.well_info.well.value
        if style is None:
            style = Template()
        elif isinstance(style, dict):
            style = Template(**style)
        elif isinstance(style, Template):
            self._style = style

    @property
    def name(self):
        return self.header.name

    @name.setter
    def name(self, value):
        self.header.name = value
        self.header.well = value
        if self.logs is not None:
            for _lc in self.logs:
                _lc.well = value

    @property
    def get_log_names(self):
        return [_lc.name for _lc in self.logs]

    @property
    def get_log_types(self):
        return list(set([_lc.log_type for _lc in self.logs]))

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

    def get_log_curve(self, name):
        for _log in self.logs:
            if _log.name == name:
                return _log
        return None

    def get_logs_of_type(self, log_type):
        return [_lc for _lc in self.logs if _lc.log_type == log_type]

    def get_logs_of_depth_type(self, depth_type):
        # depth_type = 'md', 'tvd', 'owt', or 'twt'
        return [_lc for _lc in self.logs if _lc.depth.depth_type == depth_type]

    def add_log(self, log_curve: LogCurve, if_log_exists: str = 'overwrite'):
        """
        Adds a LogCurve object to the well
        :param log_curve:
        :param if_log_exists:
            str
            Describes what to do if the log exists from before
            'overwrite': Overwrite old log
            'ask': Ask to overwrite or ignore
            'ignore': new log is ignored if a log of same name exists from before
        :return:
        """
        log_name_list = self.get_log_names
        if self.logs is None:
            self.logs = [log_curve]
        elif log_curve.name in log_name_list:
            _index = log_name_list.index(log_curve.name)
            if if_log_exists == 'overwrite':
                self.logs[_index] = log_curve
            elif if_log_exists == 'ask':
                response = input('Log curve ({}) exists! Overwrite? ["No"]:'.format(log_curve.name)) or "No"
                if response != "No":
                    self.logs[_index] = log_curve
            elif if_log_exists == 'ignore':
                pass
            else:
                raise IOError("Unknown value ({}) of 'if_log_exists'".format(if_log_exists))
        else:
            self.logs.append(log_curve)

    def harmonize_logs(self):
        """
        Adjusts all logs to have the same length and sample rate
        :return:
        """
        _harmonized_logs = []
        _longest = None
        _longest_length = 0
        for _log in self.logs:
            if len(_log) > _longest_length:
                _longest = _log
                _longest_length = len(_log)
        for _log in self.logs:
            if _log.name == _longest.name:
                _harmonized_logs.append(_log)
                continue
            _harmonized_logs.append(
                _log.take_sampling_from(_longest, suffix=None))
        self.logs = _harmonized_logs

    def templates(self):
        """
        Returns a dictionary of Templates for each log in this well, as well as the
        Template of the well itself
        :return:
            dict
        """
        _templates = {}
        for _log in self.logs:
            _templates[_log.name] = _log.style
        _templates[self.name.upper()] = self.style
        return _templates

    def read_las(self, file_name: str, verbose: bool = False, encoding: str = 'UTF8',
                 log_table: LogTable | None = None, ignore_header: bool = False,
                 rename_logs: dict | None = None,
                 if_log_exists: str = 'overwrite',
                 template_file: str | None = None):
        """
        Uses the log_curve_new.py function 'read_las()' to read a las file

        :param file_name:
        :param verbose:
        :param encoding:
        :param log_table:
            LogTable
            Object which contains which log types, and associated and which log(s) to use for each log type
            When this is specified, we only load those logs that are listed
        :param ignore_header:
        :param rename_logs:
            dict
            E.G.
            {'depth': ['DEPT', 'MD']}
            where the key is the wanted well log name, and the value list is a list of well log names to translate from
        :param if_log_exists:
            str
            Describes what to do if the log exists from before
            'overwrite': Overwrite old log
            'ask': Ask to overwrite or ignore
            'ignore': new log is ignored if a log of same name exists from before
        :param template_file:
            str
            full filename of .xlsx file that contains templates of each log type.
            Of the format used by the project_table_new.xlsx file
            Kelly bushing, water depth and other information is also extracted from the file
        :return:
        """
        from blixt_rp.core.log_curve_new import read_las as _read_las
        from blixt_rp.core.core import Template, templates_from_table
        log_curves, well_dict = _read_las(file_name, verbose=verbose, encoding=encoding, log_table=log_table,
                                          template=template_file, rename_logs=rename_logs)

        if self.header.name is None:
            self.header.name = well_dict['well_info']['well']['value']
        if self.header.orig_filename is None:
            self.header.orig_filename = file_name

        for _key, _value in log_curves.items():
            _value.style.well = self.header.name

        if self.logs is None:
            self.logs = list(log_curves.values())
        else:
            for _val in list(log_curves.values()):
                self.add_log(_val, if_log_exists=if_log_exists)

        if not ignore_header:
            self.header = add_headers(self.header, well_dict, [], None)

        if template_file is not None:
            table = pd.read_excel(template_file, header=1, sheet_name='Well settings', engine='openpyxl')

            # set the style from the template file (project table)
            template_dict = templates_from_table(table, well_style=True)
            if self.name in list(template_dict.keys()):
                self.style = template_dict[self.name]

            # add extra info to the header
            for i, ans in enumerate(table['Given well name']):
                if not isinstance(ans, str):
                    continue
                if ans.upper() == self.name.upper():
                    self.header.kb = Q_(float(table['KB'][i]), 'm')
                    self.header.water_depth = Q_(float(table['Water depth'][i]), 'm')
                    self.header.note = str(table['Note'][i])
                    self.header.content = table['Content'][i]
                    self.header.discovery_in = table['Discovery in'][i]



    def read_general_ascii(self,
                           file_name: str,
                           separator: str,
                           data_begins_on_row: int,
                           var_names: int | list | None = None,
                           var_columns: list | None = None,
                           var_units: int | list | None = None,
                           var_types: list | None = None,
                           if_log_exists: str = 'overwrite',
                           verbose: bool = False,
                           encoding: str | None = 'UTF8'):
        from blixt_rp.core.log_curve_new import read_general_ascii as _read_general_ascii
        log_curves = _read_general_ascii(file_name, separator, data_begins_on_row, var_names, var_columns, var_units,
                                         var_types, verbose, encoding)
        if self.logs is None:
            self.logs = list(log_curves.values())
        else:
            for _val in list(log_curves.values()):
                self.add_log(_val, if_log_exists=if_log_exists)

    def write_las(self, file_name, overwrite=False):
        from blixt_utils.utils import print_info
        from datetime import datetime
        if os.path.isfile(file_name) and (not overwrite):
            warn_txt = 'File {} already exist. Write cancelled'.format(file_name)
            print_info(warn_txt, 'warning', logger)
            return

        out = (
            '#----------------------------------------------------------------------------\n'
            '~VERSION INFORMATION\n'
            'VERS.            2.0                  :CWLS LOG ASCII STANDARD -VERSION 2.0\n'
            'WRAP.            NO                   :ONE LINE PER DEPTH STEP\n'
            '#\n'
        )

        out += '# {}\n'.format(self.header['creation_info'])
        if 'note' in list(self.header.keys()):
            out += '# NOTE: {}\n'.format(self.header['note'])
        out += '# Written to las on: {}\n'.format(datetime.now().isoformat())
        out += '# Modified on: {}\n'.format(self.header['modification_date'])
        for _log in self.logs:
            if _log.header['modification_history'] is not None:
                out += '#  Modification: {}: {}\n'.format(_log.name,
                                                          _log.header['modification_history'].replace('\n', '\n#   '))

        # WELL INFO
        out += (
            '#--------------------------------------------------------------------\n'
            '~WELL INFORMATION\n'
            '#MNEM .UNIT      DATA                 :DESCRIPTION OF MNEMONIC\n'
            '#----------      ------------         -------------------------------\n'
        )
        if 'well_info' not in list(self.header.keys()):
            print_info('Well info is lacking in {}'.format(self.name), 'error', logger, 'IOError')
        for _key in list(self.header['well_info'].keys()):
            out += '{0: <7}.{1: <9}{2: <21}:{3:}\n'.format(
                _key.upper(),
                self.header['well_info'][_key].unit,
                str(self.header['well_info'][_key].value) if self.header['well_info'][_key].value is not None else '',
                self.header['well_info'][_key].desc.upper()
            )

        # CURVE INFO
        out += (
            '#\n'
            '# ----------------------------------------------------------------------------\n'
            '~CURVE INFORMATION\n'
            '# MNEM.UNIT                                         : CURVE DESCRIPTION\n'
            '# ----------                                        -------------------------------\n'
        )
        # NOTE, this new version of the Well object can contain logs with different depth sampling, which the las
        # format does not support. So we try with the first log curve, and take the depth from that
        ref_depth = self.logs[0].depth
        # TODO We might need to "harmonize" all log curves before writing to las file, to make sure the start and
        # end MD are shared for all logs
        i = 1
        out += '{0: <20}.{1: <33}: {2: <9}{3:}\n'.format(
            'DEPTH',
            '{:~}'.format(ref_depth.units),
            i,
            ''
        )
        for _lc in self.logs:
            if _lc.step() != ref_depth.step():
                print_info('Not using the same step, {} can not be added to .las file. Skipping', 'warning', logger)
                continue
            i += 1
            out += '{0: <20}.{1: <33}: {2: <9}{3:}\n'.format(
                _lc.name.upper(),
                '{:~}'.format(_lc.units),
                i,
                _lc.header.log_type + ', ' + _lc.header.note
            )
        out += (
            '#\n'
            '# ----------------------------------------------------------------------------\n'
            '~A                  '
        )

        # write data column headers
        for _lc in self.logs:
            if _lc.step() != ref_depth.step():
                continue
            out += '{0: <20}'.format(_lc.name.upper())
        out += '\n'

        # start writing data
        for i, md in enumerate(ref_depth.values):
            out += '{0: <20}'.format(md)
            for _lc in self.logs:
                if _lc.step() != ref_depth.step():
                    continue
                out += '{:<20.8f}'.format(
                    self.header['well_info']['null'].value if np.isnan(_lc.values[i]) else _lc.values[i]
                )
            out += '\n'

        with open(file_name, 'w+') as f:
            f.write(out)

    def dict(self, harmonize=True):
        _dict = {}
        if harmonize:
            self.harmonize_logs()
        for _log in self.logs:
            _dict[_log.name] = _log.values
        return _dict

    def data_source(self):
        """
        Returns a DataSource object based on the well content
        :return:
            DataSource
        """
        from blixt_rp.plotting.cross_plotter import DataSource
        return DataSource(
            name=self.name,
            data=self.dict(),
            templates=self.templates()
        )

    def calc_press_ref(self, rho_sea: Q_):
        """
        Calculates the reference pressure in MPa (pressure at mudline (seafloor)) based on the water depth and
        sea water density.

        :param rho_sea:
            float
            Density of sea water
        """

        if not 'water_depth' in list(self.header.keys()):
            return None

        if rho_sea is None:
            rho_sea = Q_(1.025,  'gram / cm^3')

        return (rho_sea * self.header.water_depth * Q_(9.81, 'meter / s^2')).to('MPa')

    def calc_temp_ref(self):
        """
        Calculates the reference temperature in degC (pressure at mudline (seafloor))
        """
        return Q_(4.0, 'degC')

    def get_tvd_log(self) -> LogCurve | None:
        from blixt_utils.utils import print_info
        tvd_logs = self.get_logs_of_type('TVD')
        if len(tvd_logs) == 0:
            warn_txt = 'No True Vertical Depth log in {}, using MD'.format(self.name)
            print_info(warn_txt, 'warning', logger)
            return None
        return tvd_logs[0]

    def get_md_log(self) -> LogCurve | None:
        """
        Creates a new LogCurve object with MD data, and MD depth, taken from the LogCurve that has the largest
        depth span
        :return:
        """
        logs_with_md = self.get_logs_of_depth_type('md')
        _last_range = Q_(-1000., 'm')
        selected_log = None
        for _lc in logs_with_md:
            _range = _lc.base - _lc.top
            # print(_lc.name, _range)
            if _range > _last_range:
                selected_log = _lc
                _last_range = _range
        if selected_log is None:
            return None

        return LogCurve(
            name='md',
            log_data=selected_log.depth.depth,
            depth=selected_log.depth,
            log_type = 'MD',
            well=selected_log.well,
            style=dict(full_name='Measured depth',
                       name='md',
                       units=str(selected_log.depth_units)),
            header=dict(name='md',
                        well=selected_log.well,
                        log_type='MD',
                        note='MD log taken from {}'.format(selected_log.name),
                        orig_filename=selected_log.header.orig_filename)
        )


def add_headers(_header, _well_info, _ignore_keys, _note):
    """
    Helper function that add keys to header.
    :param _header:
    :param _well_info:
    :param _ignore_keys:
    :param _note:
        str
        String with notes for the specific well
    :return:
        modified header
    """
    for _key in list(_well_info.keys()):
        if _key in _ignore_keys:
            continue
        _header.__setitem__(_key, _well_info[_key])
    if _note is not None:
        if not isinstance(_note, str):
            raise IOError('Notes has to be of string format, not {}'.format(type(_note)))
        if 'note' in list(_header.keys()):
            _note = '{}\n{}'.format(_header.note.value, _note)
        _header.__setitem__('note', _note)

    return _header


def test():
    las_file = os.path.dirname(__file__).replace('blixt_rp\\core', 'test_data\\Well A.las')
    print(las_file)
    w = Well()
    w.read_las(las_file)
    return w

#    w.read_las(las_file, only_these_logs=well_table[las_file]['logs'])
#
#    w.calc_mask({'test': ['>', 10], 'phie': ['><', [0.05, 0.15]]}, name=ud.def_msk_name)
#    msk = w.block[ud.def_lb_name].masks[ud.def_msk_name].data
#    fig1, fig2 = plt.figure(1), plt.figure(2)
#    w.depth_plot('P velocity', fig=fig1, mask=msk, show_masked=True)
#    print('Before mask: {}'.format(len(w.block[ud.def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.block[ud.def_lb_name].masks.keys()))))
#    w.apply_mask(ud.def_msk_name)
#    print('After mask: {}'.format(len(w.block[ud.def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.block[ud.def_lb_name].masks.keys()))))
#    w.depth_plot('P velocity', fig=fig2)
#    plt.show()
#    print(w.block[ud.def_lb_name].logs['phie'].header)


if __name__ == '__main__':
    test()
