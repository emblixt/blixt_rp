# -*- coding: utf-8 -*-
"""
Module for handling a Project, consisting of many wells
"""
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import re
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.misc.templates import log_header_to_template as l2tmpl
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
import blixt_utils.io.io as uio
import blixt_utils.misc.masks as msks
from blixt_utils.utils import arrange_logging, print_info
from blixt_utils.misc.convert_data import convert as cnvrt
import blixt_rp.rp_utils.definitions as ud
from blixt_utils.utils import isnan

logger = logging.getLogger(__name__)


class Project(object):
    def __init__(self,
                 load_from=None,
                 name=None,
                 wells: list | None = None,
                 working_dir=None,
                 project_table=None,
                 log_to_stdout=False,
                 ):
        """
        class that keeps central information about the current well project in memory.

        :param load_from:
            str
            full pathname of existing log file from previously set up project.
            The other input parameters, except log_to_stdout, are ignored
        :param name:
            str
            Name of the project
        :param wells:
            list
            List of Well objects
        :param working_dir:
            str
            folder name of the project
        :param project_table:
            str
            full pathname of .xlsx file that store information of which wells and logs, ... to use
        :param log_to_stdout:
            bool
            If True, the logging information is sent to standard output and not to file
        """

        logging_level = logging.INFO

        if wells is None:
            wells = []

        if load_from is not None:
            self.load_logfile(load_from)
        else:
            if name is None:
                if project_table is not None:
                    name = os.path.basename(project_table).split('.')[0]
                else:
                    warn_txt = 'Either the name, or the project table, must be provided'
                    print_info(warn_txt, 'warning', logger)
                    raise Warning(warn_txt)

            if (working_dir is None) or (not os.path.isdir(working_dir)):
                working_dir = os.path.dirname(os.path.realpath(__file__))
                dir_list = working_dir.split(os.path.sep)
                working_dir = os.path.sep.join(dir_list[:-2])

            logging_file = os.path.join(
                working_dir,
                '{}_log.txt'.format(name))
            arrange_logging(log_to_stdout, logging_file, logging_level)

            print_info('Project created / modified on: {}'.format(datetime.now().isoformat()), 'info', logger)
            self.name = name
            self.working_dir = working_dir
            self.logging_file = logging_file
            self.wells = wells

            if project_table is None:
                self.project_table = os.path.join(self.working_dir, 'excels', 'project_table.xlsx')
            elif not os.path.isfile(project_table):
                self.project_table = os.path.join(self.working_dir, project_table)
            else:
                self.project_table = project_table

            if not os.path.isfile(self.project_table):
                warn_txt = 'The provided project table {}, does not exist'.format(self.project_table)
                print_info(warn_txt, 'warning', logger)
                raise Warning(warn_txt)

    def __setattr__(self, key, value):
        """
        Catch changing attributes in the log file
        :param key:
        :param value:
        :return:
        """
        this_str = ''
        if key == 'name':
            this_str = 'Project name'
        elif key == 'working_dir':
            this_str = 'Working directory'
        elif key == 'project_table':
            this_str = 'Project table'
        elif key == 'tops_file':
            this_str = 'Tops are taken from'
        elif key == 'tops_type':
            this_str = 'Tops are of type'
        else:
            pass

        if this_str != '':
            print_info('{}: {}'.format(this_str, value), 'info', logger)

        super(Project, self).__setattr__(key, value)

    def __str__(self):
        keys = list(self.__dict__.keys())

        pattern = "%%%ds: %%s" % len(keys)

        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def __len__(self):
        return len(self.wells)

    @property
    def get_well_names(self):
        return [_well.name for _well in self.wells]

    def load_logfile(self, file_name):
        if not os.path.isfile(file_name):
            warn_txt = 'The provided log file {}, does not exist'.format(file_name)
            print_info(warn_txt, 'warning', logger)
            raise IOError(warn_txt)
        self.logging_file = file_name

        with open(file_name, 'r') as f:
            lines = f.readlines()

        name = working_dir = project_table = tops_file = tops_type = None
        for line in lines:
            if 'Project name' in line:
                name = line.split(': ')[-1].strip()
            elif 'Working directory:' in line:
                working_dir = line.split(': ')[-1].strip()
            elif 'Project table:' in line:
                project_table = line.split(': ')[-1].strip()
            elif 'Tops are taken from:' in line:
                tops_file = line.split(': ')[-1].strip()
            elif 'Tops are of type:' in line:
                tops_type = line.split(': ')[-1].strip()
            else:
                continue

        self.name = name
        self.working_dir = working_dir
        self.project_table = project_table
        self.tops_file = tops_file
        self.tops_type = tops_type

        arrange_logging(False, file_name, logging.INFO)

        print_info('Loaded project settings from: {}'.format(file_name), 'info', logger)

    def load_all_wells(self):
        """
        Load all logs and well data that are listed in the project table where "Use" == "Yes"
        :param self:
        :return:
        """
        result = uio.project_wells_new(self.project_table, self.working_dir)
        for _key in list(result.keys()):
            print('-', _key)
            print('  -', result[_key])
        return None

    def return_dict(self, wells: list | None = None, intervals: list | None = None, logs: list | None = None):
        """
        Returns a dictionary where the data is sorted in wells -> working intervals -> logs
        eg:
            {'well A':
                {'interval 1':
                    {'log A': LogCurve2dNew,
                    'log B': LogCurve2dNew,
                    ...},
                'interval 2':
                    {'log B': LogCurve2dNew,
                    'log C': LogCurve2dNew,
                    ...},
                ...},
            'well B':
                {
                ...}
            ...}


        :return:
        """
        pass