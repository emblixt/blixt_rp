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

from blixt_utils.misc.templates import log_header_to_template as l2tmpl
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.utils import print_info, add_one
import blixt_utils.io.io as uio
from blixt_utils.io.io import well_reader
import blixt_utils.misc.masks as msks
from blixt_utils.utils import arrange_logging
from blixt_rp.rp_utils.harmonize_logs import harmonize_logs as fixlogs
from blixt_utils.plotting import crossplot as xp
from blixt_rp.core.minerals import MineralMix
import blixt_rp.rp.rp_core as rp
from blixt_utils.misc.convert_data import convert as cnvrt
import blixt_rp.rp_utils.definitions as ud
from blixt_rp.core.well import Block
from blixt_rp.core.log_curve import LogCurve
from blixt_rp.core.header import Header
from blixt_rp.core.header_new import Header as HeaderNew

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)


class WellNew(object):
    """
    Class handling a well, with LogCurve2dNew LogCurve objects for each curve of well log data.
    Main difference with earlier versions is that each LogCurve object contains its associated depth data, and units,
    so that they don't have to be regularly sampled, or with the same sampling.
    This also means that a LogCurve object can contain other data than typical log data, such as core samples.
    """
    def __init__(self,
                 header: HeaderNew | None = None,
                 logs: list | None = None
                 ):
        """

        :param header:
            HeaderNew
            dict type which contains
        :param logs:
        """
        if header is None:
            self.header = HeaderNew({})
        elif isinstance(header, dict):
            self.header = HeaderNew(header=header)
        elif isinstance(header, HeaderNew):
            self.header = header
        else:
            raise TypeError('header must be either a dict or a Header, not {}'.format(type(header)))
        self.logs = logs

    def read_las(self, file_name: str, verbose: bool = False, encoding: str = 'UTF8',
                 log_table: dict | None = None, ignore_header: bool = False):
        from blixt_rp.core.log_curve_new import read_las as _read_las
        log_curves, well_dict = _read_las(file_name, verbose=verbose, encoding=encoding, log_table=log_table)
        if self.logs is None:
            self.logs = list(log_curves.values())
        else:
            self.logs.append(list(log_curves.values()))

        if not ignore_header:
            self.header = add_headers(self.header, well_dict, [], None)



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
