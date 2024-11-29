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
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

from blixt_rp.core.header_new import Header
from blixt_rp.core.log_curve_new import LogCurve

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)


class Well(object):
    """
    Class handling a well, with LogCurve (from log_curve_new) objects for each curve of well log data.
    Main difference with earlier versions is that each LogCurve object contains its associated depth data, and units,
    so that they don't have to be regularly sampled, or with the same sampling.
    This also means that a LogCurve object can contain other data than typical log data, such as core samples.
    """
    def __init__(self,
                 header: Header | None = None,
                 logs: list | None = None
                 ):
        """

        :param header:
            Header
            dict type which contains
        :param logs:
        """
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

    def get_log_curve(self, name):
        for _log in self.logs:
            if _log.name == name:
                return _log
        return None

    def get_logs_of_type(self, log_type):
        return [_lc for _lc in self.logs if _lc.log_type == log_type]

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

    def read_las(self, file_name: str, verbose: bool = False, encoding: str = 'UTF8',
                 log_table: dict | None = None, inv_log_table: dict | None = None, ignore_header: bool = False,
                 if_log_exists: str = 'overwrite'):
        """
        Uses the log_curve_new.py function 'read_las()' to read a las file

        :param file_name:
        :param verbose:
        :param encoding:
        :param log_table:
            dict
            Dictionary of log type: log name as "key: value" pairs that specify which log to use for each log type
            When this is specified, we only load those logs that are listed among the log names in this dictionary
        :param ignore_header:
        :param if_log_exists:
            str
            Describes what to do if the log exists from before
            'overwrite': Overwrite old log
            'ask': Ask to overwrite or ignore
            'ignore': new log is ignored if a log of same name exists from before
        :return:
        """
        from blixt_rp.core.log_curve_new import read_las as _read_las
        log_curves, well_dict = _read_las(file_name, verbose=verbose, encoding=encoding, log_table=log_table,
                                          inv_log_table=inv_log_table)
        if self.logs is None:
            self.logs = list(log_curves.values())
        else:
            for _val in list(log_curves.values()):
                self.add_log(_val, if_log_exists=if_log_exists)

        if not ignore_header:
            self.header = add_headers(self.header, well_dict, [], None)

        if self.header.name is None:
            self.header.name = well_dict['well_info']['well']['value']

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
                           encoding: str = 'UTF8'):
        from blixt_rp.core.log_curve_new import read_general_ascii as _read_general_ascii
        log_curves = _read_general_ascii(file_name, separator, data_begins_on_row, var_names, var_columns, var_units,
                                         var_types, verbose, encoding)
        if self.logs is None:
            self.logs = list(log_curves.values())
        else:
            for _val in list(log_curves.values()):
                self.add_log(_val, if_log_exists=if_log_exists)


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
