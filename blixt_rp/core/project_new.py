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

from blixt_rp.core.well_new import Well
from blixt_rp.core.core import LogTable

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
            self.templates = None

            if project_table is None:
                self.project_table = os.path.join(self.working_dir, 'excels', 'project_table_new.xlsx')
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

    def get_well(self, name: str):
        for _well in self.wells:
            if _well.name.lower() == name.lower():
                return _well
        return None

    def add_well(self, well: Well, if_well_exists: str = 'append', if_log_exists: str = 'overwrite'):
        """
        Adds a Well object to the Project
        :param well:
        :param if_well_exists:
            str
            Describes what to do if the well exists from before
            'append': app
            'overwrite': Overwrite old well
            'ask': Ask to overwrite or ignore
            'ignore': new well is ignored if a log of same name exists from before
        :param if_log_exists:
            str
            Describes what to do if the log exists from before
            'overwrite': Overwrite old log
            'ask': Ask to overwrite or ignore
            'ignore': new log is ignored if a log of same name exists from before
        :return:
        """
        well_name_list = self.get_well_names
        if self.wells is None:
            self.wells = [well]
        elif well.name in well_name_list:
            _index = well_name_list.index(well.name)
            if if_well_exists == 'append':
                for _log in well.logs:
                    self.wells[_index].add_log(_log, if_log_exists=if_log_exists)
                if well.header.note is not None:
                    self.wells[_index].header.note += well.header.note
                self.wells[_index].header.modification_history += "Appended logs: {}, using '{}'".format(
                    ', '.join([_l.name for _l in well.logs]), if_log_exists)
            elif if_well_exists == 'overwrite':
                self.wells[_index] = well
            elif if_well_exists == 'ask':
                response = input('Well ({}) exists! Overwrite? ["No"]:'.format(well.name)) or "No"
                if response != "No":
                    self.wells[_index] = well
            elif if_well_exists == 'ignore':
                pass
            else:
                raise IOError("Unknown value ({}) of 'if_well_exists'".format(if_well_exists))
        else:
            self.wells.append(well)

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

    def load_all_wells(self, if_well_exists: str = 'append', if_log_exists: str = 'overwrite',
                       log_table: LogTable | None = None, verbose: bool = False):
        """
        Load all logs and well data that are listed in the project table where "Use" == "Yes"
        :param self:
        :param if_well_exists:
            str
            Describes what to do if the well exists from before
            'append': app
            'overwrite': Overwrite old well
            'ask': Ask to overwrite or ignore
            'ignore': new well is ignored if a log of same name exists from before
        :param if_log_exists:
            str
            Describes what to do if the log exists from before
            'overwrite': Overwrite old log
            'ask': Ask to overwrite or ignore
            'ignore': new log is ignored if a log of same name exists from before
        :param log_table:
            LogTable
            If provided only the logs in LogTable will be loaded
        :param verbose:
        :return:
        """
        from blixt_rp.core.core import Template, LogTable

        result = uio.project_wells_new(self.project_table, self.working_dir, do_rename=True)

        for _key in list(result.keys()):  # _key is the name of the file to read
            print('Reading from file ', _key)
            translate_dict = None
            if 'Translate log names' in list(result[_key].keys()) and result[_key]['Translate log names'] is not None:
                translate_dict = uio.interpret_rename_string(result[_key]['Translate log names'])
                print(translate_dict)
            w = Well()
            if uio.filetype(_key) == 'las':
                if log_table is None:
                    # Create a LogTable based on the logs listed in 'result'
                    _log_table = LogTable()
                    _log_table.from_invert(result[_key]['logs'])
                else:
                    _log_table = log_table.keep(list(result[_key]['logs'].keys()))

                w.read_las(_key, log_table=_log_table, template_file=self.project_table, rename_logs=translate_dict, verbose=verbose)
                # Force name of well to be that given in the project_table, and not given by the las file
                w.name = result[_key]['Given well name']

            elif uio.filetype(_key) in ['txt', 'dat', 'ascii', 'asc', 'dev', 'cs']:
                var_names = list(result[_key]['logs'].keys())
                var_columns = [result[_key]['columns'][_var] for _var in var_names]
                var_units = [result[_key]['units'][_var] for _var in var_names]
                var_types = [result[_key]['logs'][_var] for _var in var_names]
                w.read_general_ascii(
                    file_name=_key,
                    separator=result[_key]['Separator'],
                    data_begins_on_row=result[_key]['Data begins on line'],
                    var_names=var_names,
                    var_columns=var_columns,
                    var_units=var_units,
                    var_types=var_types,
                    encoding=None,
                )
                w.name = result[_key]['Given well name']
                w.header.note = result[_key]['Note']
            else:
                print_info('File {} is of unknown format. Skipped'.format(_key), 'warning', logger)

            if w.logs is not None:
                self.add_well(w, if_well_exists=if_well_exists, if_log_exists=if_log_exists)

        self.load_all_templates()

    def load_all_templates(self):
        from blixt_rp.core.core import Template, templates_from_table
        _templates = {}
        if self.project_table is not None:
            table = pd.read_excel(self.project_table, header=1, sheet_name='Templates', engine='openpyxl')
            template_dict = templates_from_table(table)
            for i, _key in enumerate(list(template_dict.keys())):
                    _templates[_key] = Template(template_dict[_key])

            table = pd.read_excel(self.project_table, header=1, sheet_name='Well settings', engine='openpyxl')
            template_dict = templates_from_table(table, well_style=True)
            for i, _key in enumerate(list(template_dict.keys())):
                _templates[_key] = Template(template_dict[_key])
            self.templates = _templates

    def load_all_wis(self):
        from blixt_rp.core.core import Intervals
        wis = Intervals()
        if self.project_table is not None:
            wis.read_blixt_tops(self.project_table)
        return wis

    def dict(self, wells: list | None = None, intervals: list | None = None, logs: list | None = None):
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