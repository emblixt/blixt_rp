# -*- coding: utf-8 -*-
"""
Module for handling wells
The goal is to have one Well object, which contains "all" the well specific
information (with a minimum required set), and a log (or curve) object for each
log.
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
from datetime import datetime
import numpy as np
import logging
import re
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from utils.attribdict import AttribDict
from utils.utils import info
from utils.utils import nan_corrcoef
from utils.utils import log_header_to_template as l2tmpl
from utils.utils import log_table_in_smallcaps as small_log_table
import utils.io as uio
from utils.io import convert
import utils.masks as msks
from utils.utils import arrange_logging
from utils.harmonize_logs import harmonize_logs as fixlogs
from plotting.crossplot import crossplot as xp
from core.minerals import MineralMix
from core.log_curve import LogCurve
import rp.rp_core as rp
from utils.convert_data import convert as cnvrt

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)
def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name
def_water_depth_keys = ['gl', 'egl', 'water_depth']
def_kelly_bushing_keys = ['kb', 'ekb', 'apd', 'edf', 'eref']
def_sonic_units = ['us/f', 'us/ft', 'us/feet',
                   'usec/f', 'usec/ft', 'usec/feet',
                   'us/m', 'usec/m', 's/m']

renamewelllogs = {
    # depth, or MD
    'depth': ['Depth', 'DEPT', 'MD', 'DEPTH'],

    # Bulk density
    'den': ['RHOB', 'HRHOB', 'DEN', 'HDEN', 'RHO', 'CPI:RHOB', 'HRHO'],

    # Density correction
    'denc': ['HDRHO', 'DENC', 'DCOR'],

    # Sonic
    'ac': ['HDT', 'DT', 'CPI:DT', 'HAC', 'AC'],

    # Shear sonic
    'acs': ['ACS'],

    # Gamma ray
    'gr': ['HGR', 'GR', 'CPI:GR'],

    # Caliper
    'cali': ['HCALI', 'CALI', 'HCAL'],

    # Deep resistivity
    'rdep': ['HRD', 'RDEP', 'ILD', 'RD'],

    # Medium resistivity
    'rmed': ['HRM', 'RMED', 'RM'],

    # Shallow resistivity
    'rsha': ['RS', 'RSHA', 'HRS'],

    # Water saturation
    'sw': ['CPI:SW', 'SW'],

    # Effective porosity
    'phie': ['CPI:PHIE', 'PHIE'],

    # Neutron porosity
    'neu': ['HNPHI', 'NEU', 'CPI:NPHI', 'HPHI'],

    # Shale volume
    'vcl': ['VCLGR', 'VCL', 'CPI:VWCL', 'CPI:VCL']
}


class Project(object):
    def __init__(self,
                 load_from=None,
                 name=None,
                 working_dir=None,
                 project_table=None,
                 tops_file=None,
                 tops_type=None,
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
        :param working_dir:
            str
            folder name of the project
        :param project_table:
            str
            full pathname of .xlsx file that store information of which wells and logs, ... to use
        :param tops_file:
            str
            full pathname of .xlsx file that contains the tops used in this project
        :param tops_type:
            str
            'petrel', 'npd' or 'rokdoc' depending on which source the tops_file comes from
        :param log_to_stdout:
            bool
            If True, the logging information is sent to standard output and not to file
        """
        logging_level = logging.INFO

        if load_from is not None:
            self.load_logfile(load_from)
        else:
            if name is None:
                name = 'test'

            if (working_dir is None) or (not os.path.isdir(working_dir)):
                working_dir = os.path.dirname(os.path.realpath(__file__))
                working_dir = working_dir.rstrip('core')

            logging_file = os.path.join(
                working_dir,
                '{}_log.txt'.format(name))
            arrange_logging(log_to_stdout, logging_file, logging_level)

            logger.info('Project created / modified on: {}'.format(datetime.now().isoformat()))
            self.name = name
            self.working_dir = working_dir
            self.logging_file = logging_file


            if project_table is None:
                self.project_table = os.path.join(self.working_dir, 'excels', 'project_table.xlsx')
            else:
                self.project_table = project_table

            if not os.path.isfile(self.project_table):
                warn_txt = 'The provided project table {}, does not exist'.format(self.project_table)
                logger.warning(warn_txt)
                raise Warning(warn_txt)

            self.tops_file = tops_file
            self.tops_type = tops_type

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
        elif key == 'logging_file':
            this_str = 'Logging to'
        else:
            pass

        if this_str != '':
            logger.info('{}: {}'.format(this_str, value))

        super(Project, self).__setattr__(key, value)


    def keys(self):
        return self.__dict__.keys()

    def __str__(self):
        keys = list(self.__dict__.keys())

        pattern = "%%%ds: %%s" % len(keys)

        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def load_logfile(self, file_name):
        if not os.path.isfile(file_name):
            warn_txt = 'The provided log file {}, does not exist'.format(file_name)
            logger.warning(warn_txt)
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

        self.name = name; self.working_dir = working_dir; self. project_table = project_table
        self.tops_file = tops_file; self.tops_type = tops_type

        arrange_logging(False, file_name, logging.INFO)

        logger.info('Loaded project settings from: {}'.format(file_name))

    def check_wells(self, all=True):
        """
        Loops through the well names in the project table and checks if las files can be read,  if the requested
        logs are to be found in the las file, and if the well name is in a consistent format
        (E.G. "6507_3_1S" and NOT "6507/3-1 S")

        :return:
        """
        _well_table = uio.project_wells(self.project_table, self.working_dir, all=True)
        for lfile in list(_well_table.keys()):
            wname = _well_table[lfile]['Given well name']
            print('Checking {}: {}'.format(wname, os.path.split(lfile)[-1]))

            # Check if las file exists (Redundant test, as uio.project_wells() tests that too
            if not os.path.isfile(lfile):
                warn_txt = 'Las file: {} does not exist'
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                continue

            # Check if well logs exists
            for log_name in list(_well_table[lfile]['logs'].keys()):
                log_exists = False
                for line in uio.get_las_curve_info(lfile):
                    if log_name in line.lower():
                        log_exists = True
                if not log_exists:
                    warn_txt = 'Log {} does not exist in {}'.format(log_name, os.path.split(lfile)[-1])
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)

            # Check if well name is consistent
            if ('/' in wname) or ('-' in wname) or (' ' in wname):
                warn_txt = "Special signs, like '/', '-' or ' ', are not allowed in well name: {}".format(wname)
                print("WARNING: {}".format(warn_txt))
                logger.warning(warn_txt)


    def active_wells(self):
        active_wells = []
        well_table = uio.project_wells(self.project_table, self.working_dir)
        for _key, _value in well_table.items():
            active_wells.append(_value['Given well name'])

        # remove duplicates
        return list(set(active_wells))

    def load_all_wells(self, block_name=None, rename_logs=None):
        """
        :param rename_logs:
            dict
            Dictionary contain information about how well logs are renamed to achieve harmonized log names
            across the whole project.
            E.G. in las file for Well_A, the shale volume is called VCL, while in Well_E it is called VSH.
            To to rename the VSH log to VCL upon import (not in the las files) the rename_logs dict should be set to
                {'VCL': ['VSH']}

        :return:
        """
        if block_name is None:
            block_name = def_lb_name
        well_table = uio.project_wells(self.project_table, self.working_dir)
        if rename_logs is None:
            # Try reading the renaming out from the well table
            rename_logs = uio.get_rename_logs_dict(well_table)

        all_wells = {}
        last_wname = ''
        for i, lasfile in enumerate(well_table):
            wname = well_table[lasfile]['Given well name']
            print(i, wname, lasfile)
            if wname != last_wname:  # New well
                w = Well()
                w.read_well_table(well_table, i, block_name=block_name, rename_well_logs=rename_logs)
                all_wells[wname] = w
                last_wname = wname
            else:  # Existing well
                all_wells[wname].read_well_table(well_table, i,
                                                 block_name=block_name,
                                                 rename_well_logs=rename_logs,
                                                 use_this_well_name=wname)

        # rename well names so that the well name defined in the well_table is used "inside" each well
        for wname, well in all_wells.items():
            well.header.well.value = wname
            well.header.name.value = wname
            well.block[block_name].header.well = wname
            well.block[block_name].__setattr__('well', wname)
            for key in list(well.block[block_name].logs.keys()):
                well.block[block_name].logs[key].header.well = wname

        return all_wells

    def data_frame(self, block_name=None, rename_logs=None):
        import pandas as pd
        all_wells = self.load_all_wells(block_name=block_name, rename_logs=rename_logs)

        log_names = []
        for well_name in list(all_wells.keys()):
            for this_log_name in all_wells[well_name].log_names():
                log_names.append(this_log_name)

        log_names = list(set(log_names))
        return log_names


class Header(AttribDict):
    """
    Class for well header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a Well object.
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every variable import and export modules
    :param
        header: Dictionary containing meta information of a single
        Well object.
    """
    defaults = {
        'name': None,
        'creation_info': AttribDict({
            'value': info(),
            'unit': '',
            'desc': 'Creation info'}),
        'creation_date': AttribDict({
            'value': datetime.now().isoformat(),
            'unit': 'datetime',
            'desc': 'Creation date'})
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
        if key in ['creation_date', 'modification_date']:
            pass
        super(Header, self).__setitem__(
            'modification_date',
            AttribDict(
                {'value': datetime.now().isoformat(),
                 'unit': 'datetime', 'desc': 'Modification date'}))

        # all other keys
        if isinstance(value, dict):
            super(Header, self).__setitem__(key, AttribDict(value))
        elif key == 'well':
            super(Header, self).__setitem__(
                key, value)
        else:
            super(Header, self).__setitem__(
                key,
                AttribDict(
                    {'value': value,
                     'unit': '',
                     'desc': key}
                ))

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Header object.
        """
        # keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Well(object):
    """
    class handling wells, with Block objects that in its turn contain LogCurve objects.
    Other well related information is stored at suitable object level.
    The reading .las files is more or less copied from converter.py
        https://pypi.org/project/las-converter/
    """

    def __init__(self,
                 header=None,
                 block=None):
        if header is None:
            header = {}
        self.header = Header(header)
        if block is None:
            self.block = {}
        elif isinstance(block, Block):
            self.block[block.name] = block

    @property
    def meta(self):
        return self.header

    @meta.setter
    def meta(self, value):
        self.header = value

    @property
    def well(self):
        try:
            return self.header.well.value
        except:
            return None

    def get_logs_of_name(self, log_name):
        log_list = []
        for lblock in list(self.block.keys()):
            log_list = log_list + self.block[lblock].get_logs_of_name(log_name)
        return log_list

    def get_logs_of_type(self, log_type):
        log_list = []
        for lblock in list(self.block.keys()):
            log_list = log_list + self.block[lblock].get_logs_of_type(log_type)
        return log_list

    def log_names(self):
        ln_list = []
        for lblock in list(self.block.keys()):
            ln_list = ln_list + self.block[lblock].log_names()
        # remove any duplicates
        ln_list = list(set(ln_list))
        return ln_list

    def log_types(self):
        lt_list = []
        for lblock in list(self.block.keys()):
            lt_list = lt_list + self.block[lblock].log_types()
        # remove any duplicates
        lt_list = list(set(lt_list))
        return lt_list

    def get_step(self, block_name=None):
        """
        Tries to return the step length in meters
        :return:
        """
        if block_name is None:
            block_name = def_lb_name
        return self.block[block_name].get_step()

    def depth_unit(self, block_name=None):
        """
        Returns the units used for the depth measurement in Block 'block_name'.
        :return:
            str
            Name of unit used for depth
        """
        if block_name is None:
            block_name = def_lb_name
        return self.block[block_name].header.strt.unit.lower()

    def get_from_well_info(self, what, templates=None, block_name=None, search_keys=None):
        """
        Tries to return something (e.g. Kelly bushing or water depth) from well info, in unit meters.
        :param what:
            str
            'kb' for kelly bushing
            'water depth' for water depth
        :param templates:
            dict
            templates that can contain the desired information for wells
            templates = utils.io.project_tempplates(wp.project_table)
        :param search_keys:
            list
            List of strings that can be the key for the desired information in the well header
        :return:
            float
            value in meters
        """
        info_txt = 'Extract {} in well {}'.format(what, self.well)
        if search_keys is None:
            if what == 'kb':
                search_keys = def_kelly_bushing_keys
            elif what == 'water depth':
                search_keys = def_water_depth_keys
            else:
                search_keys = None

        # First try the templates
        if templates is not None:
            if (self.well in list(templates.keys())) and \
                    (what in list(templates[self.well].keys())) and \
                    (templates[self.well][what] is not None):
                info_txt += ' from templates'
                print('INFO: {}'.format(info_txt))
                logger.info(info_txt)
                return templates[self.well][what]

        # When above didn't work, try the well header
        start_unit = ''
        if search_keys is None:
            warn_txt = 'No keys, to search for {} in well info, is provided'.format(what)
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            return None

        for _key in list(self.header.keys()):
            if _key in search_keys:
                if self.header[_key].value is None:
                    continue
                this_result = self.header[_key].value
                info_txt += ' using key: {:}, with value {:} ({}) '.format(_key, this_result, type(this_result))
                if self.header[_key].unit.lower() == '':
                    # We assume it has the same unit as the Start, Stop, Step values, who's units are more often set
                    start_unit = self.depth_unit(block_name=block_name)
                    if start_unit == 'm':
                        info_txt += '[m].'
                        return_value = this_result
                    else:
                        # assume it is in feet
                        info_txt += '[feet].'
                        return_value = cnvrt(this_result, 'ft', 'm')
                elif self.header[_key].unit.lower() == 'm':
                    info_txt += '[m].'
                    return_value = this_result
                else:
                    # assume it is in feet
                    info_txt += '[feet].'
                    return_value = cnvrt(this_result, 'ft', 'm')

                print('INFO: {}'.format(info_txt))
                logger.info(info_txt)
                return return_value

        info_txt += ' failed. No matching keys in header.'
        print('WARNING: {}'.format(info_txt))
        logger.warning(info_txt)
        return 0.0

    def get_burial_depth(self, templates=None, block_name=None, tvd_key=None):
        if block_name is None:
            block_name = def_lb_name

        tvd = self.block[block_name].get_tvd(tvd_key=tvd_key)

        return tvd - np.abs(self.get_from_well_info('water depth', templates, block_name=block_name)) - \
               np.abs(self.get_from_well_info('kb', templates, block_name=block_name))

    def sonic_to_vel(self, block_name=None):
        if block_name is None:
            block_name = def_lb_name

        self.block[block_name].sonic_to_vel()

    def time_to_depth(self, log_name='vp', water_vel=None, repl_vel=None, water_depth=None, block_name=None,
                      sonic=None, feet_unit=None, us_unit=None,
                      spike_threshold=None, templates=None, debug=False):
        """
        Calculates the twt as a function of md
        https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb

        :param log_name:
            str
            Name of slowness or velocity log used to calculate the time-depth relation
        :param water_vel:
            float
            Sound velocity in water [m/s]
        :param repl_vel:
            float
            Sound velocity [m/s] in section between sea-floor and top of log, Also used to fill in NaNs in sonic or velocity
            log
        :param water_depth:
            float
            Water depth in meters.
            If not specified, tries to read from well header.
            if that fails, it tries to read it from the templates (if given)
            If that fails, uses 0 (zero) water depth
        :param block_name:
        :param sonic:
            bool
            Set to true if input log is sonic or slowness
            If None, the scripts tries to guess using the units of the input log
        :param feet_unit:
            bool
            Set to true if input log is using feet (e.g. "us/f")
            If None, the scripts tries to guess using the units of the input log
        :param us_unit:
            bool
            Set to true if input log is using micro seconds and not seconds (e.g. "us/f" or "s/f"
            If None, the scripts tries to guess using the units of the input log
        :return:
        """
        if block_name is None:
            block_name = def_lb_name

        tb = self.block[block_name]

        if log_name not in tb.log_names():
            warn_txt = 'Log {} does not exist in well {}'.format(log_name, self.well)
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            return None

        if water_vel is None:
            water_vel = 1480.
        if repl_vel is None:
            repl_vel = 1600.
        if water_depth is None:
            # Tries to extract the water depth from the well header
            water_depth = self.get_from_well_info('water depth', templates=templates)

        kb = self.get_from_well_info('kb', templates=templates)  # m
        log_start_twt = self.block[block_name].twt_at_logstart(log_name, water_vel, repl_vel, water_depth, kb)

        if sonic is None:
            # Determine if log is a Sonic log, or a velocity log
            if tb.logs[log_name].header.unit.lower() in def_sonic_units:
                sonic = True
            else:
                sonic = False

        if feet_unit is None:
            # Determine if units are in feet or not
            feet_unit = False
            for feet_test in ['f', 'ft', 'feet']:
                if feet_test in tb.logs[log_name].header.unit.lower():
                    feet_unit = True

        if spike_threshold is None:
            spike_threshold = 200.
            if feet_unit:
                spike_threshold = 10

        if us_unit is None:
            # Determine if units are in s or us
            if 'u' in tb.logs[log_name].header.unit.lower():
                us_unit = True
            else:
                us_unit = False

        print('Sonic? {},  feets? {},  usec? {}'.format(sonic, feet_unit, us_unit))

        tdr = self.block[block_name].time_to_depth(log_start_twt, log_name,
                                                   spike_threshold, repl_vel, sonic=sonic, feet_unit=feet_unit,
                                                   us_unit=us_unit, debug=debug)

        return tdr

    def calc_vrh_bounds(self, fluid_minerals, param='k', wis=None, method='Voigt', block_name=None):
        """
        Calculates the Voigt-Reuss-Hill bounds of parameter param, for the  fluid or mineral mixture defined in
        fluid_minerals, for the given Block.

        IMPORTANT
        If the fluids or minerals are defined using a "Calculation method", it is necessary to run XXX.calc_elastics()
        on them before

        :param fluid_minerals:
            core.minerals.MineralMix
            or
            core.fluid.fluids['xx'] where 'xx' is 'initial' or 'final'

            minerals = core.minerals.MineralMix()
            minerals.read_excel(working_project.project_table)
        :param param:
            str
            'k' for Bulk modulus
            'mu' for shear modulus
            'rho' for density
        :param wis:
            dict
            Dictionary of working intervals
            E.G.
            wis = utils.io.project_working_intervals(project_table.xlsx)
        :param method:
            str
            'Voigt' for the upper bound or Voigt average
            'Reuss' for the lower bound or Reuss average
            'Voigt-Reuss-Hill'  for the average of the two above
        :param block_name:
            str
            Name of the Block for which the bounds are calculated

        :return
            np.ndarray
            Voigt / Reuss / Hill bounds of parameter 'param'
        """
        # TODO
        # Subdivide this function into several subroutines, so that it can be used more easily elsewhere
        if block_name is None:
            block_name = def_lb_name
        obj = None
        fluid = False

        if not (isinstance(fluid_minerals, MineralMix) or isinstance(fluid_minerals, dict)):
            warn_txt = \
                'Input fluid_minerals must be a MineralMix or a subselection of a FluidMix object, not {}'.format(
                    type(fluid_minerals)
                )
            logger.warning(warn_txt)
            raise Warning(warn_txt)

        if isinstance(fluid_minerals, MineralMix):
            obj = fluid_minerals.minerals

        #if isinstance(fluid_minerals, FluidSet):
        if isinstance(fluid_minerals, dict):
            # Assume that one of the 'initial' or 'final' fluid sets have been chosen
            fluid = True
            obj = fluid_minerals

        # Test if this well is present among the minerals / fluids
        if self.well not in list(obj.keys()):
            warn_txt = 'Well {} not among the given fluid / mineral mixes'.format(self.well)
            logger.warning(warn_txt)
            raise Warning(warn_txt)

        # Use the fluid / minerals for this well only
        obj = obj[self.well]

        # test if the wanted working intervals exists in the defined working intervals
        if wis is not None:
            for wi in list(obj.keys()):
                if wi not in list(wis[self.well].keys()):
                    warn_txt = 'Interval {} not present in the given working intervals for {}'.format(
                        wi, self.well)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)


        if param not in ['k', 'mu', 'rho']:
            raise IOError('Bounds can only be calculated for k, mu and rho')

        if fluid:
            fm_type = 'Fluid'
        else:
            fm_type = 'Mineral'

        # Calculate the fluid / mineral average k, mu or rho for each working interval
        # Please note that the calculation is done over the entire length of the well!
        # The user have to apply these averages in the designated working interval later
        result = {}
        for wi, val in obj.items():
            complement = None
            this_fraction = None  # A volume fraction log
            this_component = None  #
            fractions = []
            components = []
            info_txt = 'Calculating {} bound of {} for well {} in interval {}\n'.format(
                method, param, self.well, wi
            )
            for m in list(val.keys()):
                info_txt += "  {}".format(m)
                info_txt += "    K: {}, Mu: {}, Rho {}\n".format(
                    val[m].k.value, val[m].mu.value, val[m].rho.value)
                info_txt += "    Volume fraction: {}\n".format(val[m].volume_fraction)
            logger.info(info_txt)
            print(info_txt)

            if len(list(val.keys())) > 2:
                warn_txt = 'The bounds calculation has only been tested for two-components mixtures\n'
                warn_txt += 'There is no guarantee that the total fraction does not exceed one'
                print('Warning {}'.format(warn_txt))
                logger.warning(warn_txt)

            for this_fm in list(val.keys()):  # loop over each fluid / mineral component
                print(' {} {}: {}, volume frac: {}'.format(fm_type, param, this_fm, val[this_fm].volume_fraction))
                tmp_frac = val[this_fm].volume_fraction
                if tmp_frac == 'complement':  # Calculated as 1. - the others
                    if complement is not None:
                        raise IOError('Only one complement log is allowed')
                    complement = this_fm
                    this_fraction = this_fm  # Insert mineral name for the complement mineral
                elif isinstance(tmp_frac, float):
                    this_fraction = tmp_frac
                else:
                    _name = tmp_frac.lower()
                    if _name not in list(self.block[block_name].logs.keys()):
                        warn_txt = 'The volume fraction {} is lacking in Block {} of well {}'.format(
                            _name, block_name, self.well
                        )
                        print(warn_txt)
                        logger.warning(warn_txt)
                        continue
                    this_fraction = self.block[block_name].logs[_name].data
                if param == 'k':
                    this_component = val[this_fm].k.value
                elif param == 'mu':
                    this_component = val[this_fm].mu.value
                elif param == 'rho':
                    this_component = val[this_fm].rho.value
                else:
                    this_component = None

                fractions.append(this_fraction)
                components.append(this_component)

            # Calculate the complement fraction only when there are more than one constituent
            if len(fractions) == len(components) > 1:
                if complement not in fractions:
                    raise IOError('No complement log given')
                compl = 1. - sum([f for f in fractions if not isinstance(f, str)])

                # insert the complement at the location of the complement mineral
                fractions[fractions.index(complement)] = compl

            tmp = rp.vrh_bounds(fractions, components)
            if method == 'Voigt':
                result[wi] = tmp[0]
            elif method == 'Reuss':
                result[wi] = tmp[1]
            else:
                result[wi] = tmp[2]
        return result

    def calc_mask(self,
                    cutoffs,
                    name=def_msk_name,
                    tops=None,
                    use_tops=None,
                    wis=None,
                    wi_name=None,
                    overwrite=True,
                    append=False,
                    log_type_input=True,
                    log_table=None
        ):
        """
        Based on the different cutoffs in the 'cutoffs' dictionary, each Block in well is masked accordingly.

        :param cutoffs:
            dict
            dictionary with log name as keys, and list with mask operator and limits as values
            E.G. {'depth': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
        :param name:
            str
            name of the mask
        :param tops:
            dict
            as returned from utils.io.read_tops() function
        :param use_tops:
            list
            List of top names inside the tops dictionary that will be used to mask the data
            NOTE: if the 'depth' parameter is already inside the cutoffs dictionary, this option will
            be ignored
        :param wis:
            dict
            working intervals, as defined in the "Working intervals" sheet of the project table, and
            loaded through:
            wp = Project()
            wis = utils.io.project_working_intervals(wp.project_table)

        :param wi_name:
            str
            name of working interval to mask, other intervals will be set to False in boolean mask
        :param overwrite:
            bool
            if True, any existing mask with given name will be overwritten
        :param append:
            bool
            if True, the new mask will be appended to any existing mask with given name
            append wins over overwrite
        :param log_type_input:
            bool
            if set to True, the keys in the cutoffs dictionary refer to log types, and not log names
        :param log_table:
            dict
            Dictionary of log type: log name key: value pairs to create mask on when log_type_input is True
            NOTE: If this is not set when log_type_input is True, the first log under each log type will be used.
            E.G.
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'rhob',
                   'Porosity': 'phie',
                   'Volume': 'vcl'}
        :return:
        """
        if log_table is not None:
            log_table = small_log_table(log_table)

        #
        # Helper functions
        #
        def rename_cutoffs(_cutoffs):
            # When the cutoffs are based on log types, and not the individual log names, we need to
            # associate each log type with it FIRST INSTANCE log name if log_table is not set
            _this_cutoffs = {}
            for _key in list(_cutoffs.keys()):
                if len(self.get_logs_of_type(_key)) < 1:
                    warn_txt = 'No logs of type {} in well {}'.format(_key, self.well)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)
                    continue
                if log_table is not None:
                    _this_cutoffs[log_table[_key]] = _cutoffs[_key]
                else:
                    _this_cutoffs[self.get_logs_of_type(_key)[0].name] = _cutoffs[_key]
            return _this_cutoffs

        def apply_wis(_cutoffs):
            if (wis is None) or (wi_name is None):
                return _cutoffs
            if self.well not in list(wis.keys()):
                warn_txt = 'Well: {} is not in the list of working intervals: {}'.format(
                    self.well,
                    ', '.join(list(wis.keys()))
                )
                logger.warning(warn_txt)
                print('WARNING: {}'.format(warn_txt))
                return _cutoffs
            else:
                # Append the depth mask from the selected working interval
                try:
                    _cutoffs['depth'] = ['><',
                                     [wis[self.well][wi_name.upper()][0],
                                      wis[self.well][wi_name.upper()][1]]
                                     ]
                except KeyError:
                    warn_txt = '{} not present in {}'.format(wi_name.upper(), self.well)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)
            return _cutoffs

        def apply_tops(_cutoffs):
            if self.well not in list(tops.keys()):
                warn_txt = 'Well: {} is not in the list of tops: {}'.format(
                    self.well,
                    ', '.join(list(tops.keys()))
                )
                logger.warning(warn_txt)
                print('WARNING: {}'.format(warn_txt))
            elif not all([t.upper() in list(tops[self.well].keys()) for t in use_tops]):
                warn_txt = 'The selected tops {} are not among the well tops {}'.format(
                    ', '.join(use_tops),
                    '. '.join(list(tops[self.well].keys()))
                )
                logger.warning(warn_txt)
                print('WARNING: {}'.format(warn_txt))
            else:
                # Append the depth mask from the tops file
                _cutoffs['depth'] = ['><',
                                    [tops[self.well][use_tops[0].upper()],
                                     tops[self.well][use_tops[1].upper()]]
                                    ]
            return _cutoffs

        def mask_string(_cutoffs):
            msk_str = ''
            for key in list(_cutoffs.keys()):
                msk_str += '{}: {} [{}]'.format(
                    key, _cutoffs[key][0], ', '.join([str(m) for m in _cutoffs[key][1]])) if \
                    isinstance(_cutoffs[key][1], list) else \
                    '{}: {} {}, '.format(
                    key, _cutoffs[key][0], _cutoffs[key][1])
            if wi_name is not None:
                msk_str += ' Working interval: {}'.format(wi_name)
            return msk_str

        #
        # Main functionality
        #
        if not isinstance(cutoffs, dict):
            raise IOError('Cutoffs must be specified as dict, not {}'.format(type(cutoffs)))

        # When the cutoffs are based on log types, and not the individual log names,
        if log_type_input:
            cutoffs = rename_cutoffs(cutoffs)

        # Test if there are tops / working intervals in the input, and use them to modify the
        # desired cutoffs
        if isinstance(use_tops, list) and (tops is not None):
            cutoffs = apply_tops(cutoffs)
        elif isinstance(wi_name, str) and (wis is not None):
            cutoffs = apply_wis(cutoffs)

        msk_str = mask_string(cutoffs)

        for lblock in list(self.block.keys()):
            masks = []
            if len(cutoffs) < 1:
                warn_txt = 'No logs selected to base the mask on'
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                #raise IOError(warn_txt)
                continue

            for lname in list(cutoffs.keys()):
                if lname.lower() not in list(self.block[lblock].logs.keys()):
                    warn_txt = 'Log {} to calculate mask from is not present in well {}'.format(lname, self.well)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)
                    continue
                else:
                    # calculate mask
                    masks.append(msks.create_mask(
                        self.block[lblock].logs[lname.lower()].data, cutoffs[lname][0], cutoffs[lname][1]
                    ))
            if len(masks) > 0:
                # combine all masks for this Block
                block_mask = msks.combine_masks(masks)
                if self.block[lblock].masks is None:
                    self.block[lblock].masks = {}
                if (name in list(self.block[lblock].masks.keys())) and (not overwrite) and (not append):
                    # create a new name for the mask
                    name = add_one(name)
                elif (name in list(self.block[lblock].masks.keys())) and append:
                    # read in old mask
                    old_mask = self.block[lblock].masks[name].data
                    old_desc = self.block[lblock].masks[name].header.desc
                    # modify the new
                    block_mask = msks.combine_masks([old_mask, block_mask])
                    msk_str = '{} AND {}'.format(msk_str, old_desc)

                # Create an object, similar to the logs object of a Block, that contain the masks
                self.block[lblock].masks[name] = LogCurve(
                    name=name,
                    well=self.well,
                    data=block_mask,
                    header={
                        'name': name,
                        'well': self.well,
                        'log_type': 'Mask',
                        'desc': msk_str
                    }
                )
                # Test for number of True values
                if np.sum(block_mask) < 1:
                    warn_txt = 'All values in well {}, block {}, are masked out using {}'.format(
                        self.well, lblock, msk_str)
                    print('WARNING: {}'.format(warn_txt))
                    logger.warning(warn_txt)
                else:
                    #print('{} True values in mask: {}'.format(np.sum(block_mask), msk_str))
                    pass
            else:
                continue


    def apply_mask(self,
            name=None):
        """
        Applies the named mask to the logs under each Block where the named mask exists, adds the masking description
        to the LogCurve header and deletes the named masks object under each Block.

        :param name:
            str
            name of the mask to apply.
            If if doesn't exist, nothing is done
        :return:
        """
        if name is not None:
            for lblock in list(self.block.keys()):
                if name in list(self.block[lblock].masks.keys()):
                    msk = self.block[lblock].masks[name].data
                    desc = self.block[lblock].masks[name].header.desc
                    for lname in list(self.block[lblock].logs.keys()):
                        self.block[lblock].logs[lname].data = self.block[lblock].logs[lname].data[msk]
                        self.block[lblock].logs[lname].header.modification_history = 'Mask: {}'.format(desc)
                    del(self.block[lblock].masks[name])

    def depth_plot(self,
                   log_type='P velocity',
                   log_name=None,
                   mask=None,
                   tops=None,
                   wis=None,
                   fig=None,
                   ax=None,
                   templates=None,
                   savefig=None,
                   **kwargs):
        """
        Plots selected log type as a function of MD.

        :param log_type:
            str
            Name of the log type we want to plot
        :param log_name:
            str
            overrides the log_type selection, and plots only this log
        :param mask:
            boolean numpy array of same length as xdata
        :param tops:
            dict
            as returned from utils.io.read_tops() function
        :param wis:
            dict
            dictionary of working intervals
        :param fig:
            matplotlib.figure.Figure object
        :param ax:
            matplotlib.axes._subplots.AxesSubplot object
        :param templates:
            dict
            templates dictionary as returned from utils.io.project_templates()
        :param savefig:
            str
            full path name of file to save plot to
        :param kwargs:
        :return:
        """
        _savefig = False
        if savefig is not None:
            _savefig = True

        # set up plotting environment
        if fig is None:
            if ax is None:
                fig = plt.figure(figsize=(8,10))
                ax = fig.subplots()
            else:
                _savefig = False
        elif ax is None:
            ax = fig.subplots()

        show_masked = kwargs.pop('show_masked', False)

        if log_name is not None:
            list_of_logs = self.get_logs_of_name(log_name)
            ttl = ''
        else:
            list_of_logs = self.get_logs_of_type(log_type)
            ttl = log_type

        # handle x template
        x_templ = None
        if (templates is not None) and (log_type in list(templates.keys())):
            x_templ = templates[log_type]

        # loop over all Blocks
        cnt = -1
        legends = []
        for logcurve in list_of_logs:
            cnt += 1
            if x_templ is None:
                x_templ = l2tmpl(logcurve.header)
            #print(cnt, logcurve.name, xp.cnames[cnt], mask)
            xdata = logcurve.data
            ydata = self.block[logcurve.block].logs['depth'].data
            legends.append(logcurve.name)
            xp.plot(
                xdata,
                ydata,
                cdata=xp.cnames[cnt],
                title='{}: {}'.format(self.well, ttl),
                xtempl=x_templ,
                ytempl=l2tmpl(self.block[logcurve.block].logs['depth'].header),
                mask=mask,
                show_masked=show_masked,
                fig=fig,
                ax=ax,
                pointsize=10,
                edge_color=False,
                grid=True
            )
        # Draw tops
        if tops is not None:
            wname = uio.fix_well_name(self.well)
            if wname not in list(tops.keys()):
                logger.warning('No tops in loaded tops for well {}'.format(wname))
            else:
                for top_name, top_md in tops[wname].items():
                    if (top_md < ax.get_ylim()[0]) or (top_md > ax.get_ylim()[-1]):
                        continue  # skip tops outside plotting range of md
                    ax.axhline(top_md, c='r', lw=0.5, label='_nolegend_')
                    ax.text(ax.get_xlim()[0], top_md, top_name, fontsize=FontProperties(size='smaller').get_size())
        elif wis is not None:
            wname = uio.fix_well_name(self.well)
            if wname not in list(wis.keys()):
                logger.warning('No working intervals in loaded valid for well {}'.format(wname))
            else:
                for top_name, top_md in wis[wname].items():
                    if (top_md[0] < ax.get_ylim()[0]) or (top_md[0] > ax.get_ylim()[-1]):
                        continue  # skip tops outside plotting range of md
                    ax.axhline(top_md[0], c='r', lw=0.5, label='_nolegend_')
                    ax.text(ax.get_xlim()[0], top_md[0], top_name, fontsize=FontProperties(size='smaller').get_size())
                    ax.axhline(top_md[1], c='r', lw=0.5, label='_nolegend_')

        ax.set_ylim(ax.get_ylim()[::-1])
        this_legend = ax.legend(
            legends,
            prop=FontProperties(size='smaller'),
            scatterpoints = 1,
            markerscale=2,
            loc=1
        )
        if _savefig:
            fig.savefig(savefig)


    def read_well_table(self, well_table, index, block_name=None,
                        rename_well_logs=None, use_this_well_name=None):
        """
        Takes the well_table and reads in the well defined by the index number.

        :param well_table:
            dict
            E.G.
            > from well_project import Project
            > wp = Project()
            > import utils.io as uio
            > well_table = uio.project_wells(wp.project_table, wp.working_dir)
        :param index:
            int
            index of well in well_table to read
        :param rename_well_logs:
            dict
            E.G.
            {'depth': ['DEPT', 'MD']}
            where the key is the wanted well log name, and the value list is a list of well log names to translate from
        :param use_this_well_name:
            str
            Name we would like to use.
            Useful when different las file have different wells for the same well
        :return:
        """
        if block_name is None:
            block_name = def_lb_name
        lfile = list(well_table.keys())[index]
        if 'Note' in list(well_table[lfile].keys()):
            note = well_table[lfile]['Note']
        else:
            note = None
        self.read_las(lfile,
                      only_these_logs=well_table[lfile]['logs'],
                      block_name=block_name,
                      rename_well_logs=rename_well_logs,
                      use_this_well_name=use_this_well_name,
                      note=note)

        # rename well names so that the well name defined in the well_table is used "inside" each well
        wname = well_table[lfile]['Given well name']
        self.header.well.value = wname
        self.header.name.value = wname
        self.block[block_name].header.well = wname
        for key in list(self.block[block_name].logs.keys()):
            self.block[block_name].logs[key].header.well = wname


    def read_las(
            self,
            filename,
            block_name=None,
            only_these_logs=None,
            rename_well_logs=None,
            use_this_well_name=None,
            note=None
    ):
        """
        Reads in a las file (filename) and adds the selected logs (listed in only_these_logs) to the
        Block specified by block_name.

        :param filename:
            str
            name of las file
        :param block_name:
            str
            Name of Block where the logs should be added to
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
            Useful when different las file have different wells for the same well
        :param note:
            str
            String containing notes for the las file being read
        :return:
        """
        if block_name is None:
            block_name = def_lb_name

        # make sure the only_these_logs is using small caps for all well logs
        if (only_these_logs is not None) and isinstance(only_these_logs, dict):
            for key in list(only_these_logs.keys()):
                only_these_logs[key.lower()] = only_these_logs.pop(key)

        # Make sure the only_these_logs is updated with the desired name changes defined in rename_well_logs
        if isinstance(only_these_logs, dict) and (isinstance(rename_well_logs, dict)):
            for okey in list(only_these_logs.keys()):
                for rkey in list(rename_well_logs.keys()):
                    if okey.lower() in [x.lower() for x in rename_well_logs[rkey]]:
                        only_these_logs[rkey.lower()] = only_these_logs.pop(okey)

        def add_headers(well_info, ignore_keys, _note):
            """
            Helper function that add keys to header.
            :param well_info: 
            :param ignore_keys:
            :param _note:
                str
                String with notes for the specific well
            :return: 
            """
            for key in list(well_info.keys()):
                if key in ignore_keys:
                    continue
                self.header.__setitem__(key, well_info[key])
            if _note is not None:
                if not isinstance(_note, str):
                    raise IOError('Notes has to be of string format, not {}'.format(type(_note)))
                if 'note' in list(self.header.keys()):
                    _note = '{}\n{}'.format(self.header.note.value, _note)
                self.header.__setitem__('note', _note)


        def add_logs_to_block(_well_dict, _block_name, _only_these_logs, _filename):
            """
            Helper function that add logs to the given block name.

            :param _well_dict:
            :param _only_these_logs:
            :return:
            """

            # Make sure depth is always read in
            if isinstance(_only_these_logs, dict) and ('depth' not in list(_only_these_logs.keys())):
                _only_these_logs['depth'] = 'Depth'

            exists = False
            same = True
            # Test if Block already exists
            if _block_name in list(self.block.keys()):
                exists = True
                #print('Length of existing data: {}'.format(len(self.block[_block_name].logs['depth'].data)))
                # Test if Block has the same header
                for key in ['strt', 'stop', 'step']:
                    if _well_dict['well_info'][key]['value'] != self.block[_block_name].header[key].value:
                        #print('{} in new versus existing log block {}: {}'.format(
                        #    key,
                        #    _well_dict['well_info'][key]['value'],
                        #    self.block[_block_name].header[key].value))
                        same = False

            if exists and not same:
                ## Create a new Block, with a new name, and warn the user
                #new_block_name = add_one(_block_name)
                #logger.warning(
                #    'Block {} existed and was different from imported las file, new Block {} was created'.format(
                #        _block_name, new_block_name
                #    ))
                #_block_name = new_block_name
                #info_txt = 'Start modifying the logs in las file to fit the existing Block'
                #print(info_txt)
                #logger.info(info_txt)
                #print(' Length of existing data in well: {}'.format(
                #    len(self.block[_block_name])
                #))
                #print(' Length before fixing: {}'.format(len(_well_dict['data']['depth'])))
                fixlogs(
                    _well_dict,
                    self.block[_block_name].header['strt'].value,
                    self.block[_block_name].header['stop'].value,
                    self.block[_block_name].header['step'].value,
                    len(self.block[_block_name])
                )
                #print(' Length after fixing: {}'.format(len(_well_dict['data']['depth'])))
                # Remove the 'depth' log, so we are not using two
                xx = _well_dict['data'].pop('depth')
                xx = _well_dict['curve'].pop('depth')
                same = True

            # Create a dictionary of the logs we are reading in
            these_logs = {}
            if isinstance(_only_these_logs, dict):
                # add only the selected logs
                for _key in list(_only_these_logs.keys()):
                    if _key in list(_well_dict['curve'].keys()):
                        this_header = _well_dict['curve'][_key]
                        this_header.update({'log_type': only_these_logs[_key]})
                        logger.debug('Adding log {}'.format(_key))
                        these_logs[_key] = LogCurve(
                            name=_key,
                            block=_block_name,
                            well=_well_dict['well_info']['well']['value'],
                            data=np.array(_well_dict['data'][_key]),
                            header=this_header
                        )
                        these_logs[_key].header.orig_filename = filename
                    else:
                        logger.warning("Log '{}' in {} is missing\n  [{}]".format(
                                _key,
                                _well_dict['well_info']['well']['value'],
                                ', '.join(list(_well_dict['curve'].keys()))
                                #', '.join(list(_only_these_logs.keys()))
                                )
                        )
            elif _only_these_logs is None:
                # add all logs
                for _key in list(_well_dict['curve'].keys()):
                    logger.debug('Adding log {}'.format(_key))
                    these_logs[_key] = LogCurve(
                        name=_key,
                        block=_block_name,
                        well=_well_dict['well_info']['well']['value'],
                        data=np.array(_well_dict['data'][_key]),
                        header=_well_dict['curve'][_key]
                    )
                    these_logs[_key].header.orig_filename = filename
                    these_logs[_key].header.name = _key
            else:
                logger.warning('No logs added to {}'.format(_well_dict['well_info']['well']['value']))

            # Test if Block already exists
            if exists and same:
                self.block[_block_name].logs.update(these_logs)
                self.block[_block_name].header.orig_filename.value = '{}, {}'.format(
                    self.block[_block_name].header.orig_filename.value, _filename
                )
            else:
                self.block[_block_name] = Block(
                    name=_block_name,
                    well=_well_dict['well_info']['well']['value'],
                    logs=these_logs,
                    orig_filename=_filename,
                    header={
                        key: _well_dict['well_info'][key] for key in ['strt', 'stop', 'step']
                    }
                )
                self.block[_block_name].header.name = _block_name

        # Make sure only_these_logs is a dictionary
        if isinstance(only_these_logs, str):
            only_these_logs = {only_these_logs.lower(): None}

        null_val, generated_keys, well_dict = _read_las(filename)
        # Rename well logs
        if rename_well_logs is None:
            rename_well_logs = {'depth': ['Depth', 'DEPT', 'MD', 'DEPTH']}
        elif isinstance(rename_well_logs, dict) and ('depth' not in list(rename_well_logs.keys())):
            rename_well_logs['depth'] = ['Depth', 'DEPT', 'MD', 'DEPTH']

        for key in list(well_dict['curve'].keys()):
            for rname, value in rename_well_logs.items():
                if key.lower() in [x.lower() for x in value]:
                    info_txt = 'Renaming log from {} to {}'.format(key, rname)
                    #print('INFO: {}'.format(info_txt))
                    logger.info(info_txt)
                    well_dict['curve'][rname.lower()] = well_dict['curve'].pop(key)
                    well_dict['data'][rname.lower()] = well_dict['data'].pop(key)

        logger.debug('Reading {}'.format(filename))
        if well_dict['version']['vers']['value'] not in supported_version:
            raise Exception("Version {} not supported!".format(
                well_dict['version']['vers']['value']))

        # Rename well
        if use_this_well_name is not None:
            well_dict['well_info']['well']['value'] = uio.fix_well_name(use_this_well_name)

        # Test if this well has a header, and that we are loading from the same well
        if 'well' not in list(self.header.keys()):  # current well object is empty
            # Add all well specific headers to well header
            add_headers(well_dict['well_info'], ['strt', 'stop', 'step'], note)
            self.header.name = {
                'value': well_dict['well_info']['well']['value'],
                'unit': '',
                'desc': 'Well name'}
            # Add logs to a Block
            add_logs_to_block(well_dict, block_name, only_these_logs, filename)

        else:  # There is a well loaded already
            if well_dict['well_info']['well']['value'] == self.header.well.value:
                add_well = True
            else:
                message = 'Trying to add a well with different name ({}) to existing well object ({})'.format(
                    well_dict['well_info']['well']['value'],
                    self.header.well.value
                )
                logger.warning(message)
                print(message)
                question = 'Do you want to add well {} to existing well {}? Yes or No'.format(
                    well_dict['well_info']['well']['value'],
                    self.header.well.value
                )
                ans = input(question)
                logger.info('{}: {}'.format(question, ans))
                if 'y' in ans.lower():
                    add_well = True
                else:
                    add_well = False

            if add_well:
                # Add all well specific headers to well header, except well name
                add_headers(well_dict['well_info'], ['strt', 'stop', 'step', 'well'], note)
                # add to existing LogBloc
                add_logs_to_block(well_dict, block_name, only_these_logs, filename)

    def keys(self):
        return self.__dict__.keys()


class Block(object):

    """
    A  Block is a collection of logs (or other data, perhaps with non-uniform sampling) which share the same
    depth information, e.g start, stop, step
    """

    def __init__(self,
                 name=None,
                 well=None,
                 logs=None,
                 masks=None,
                 orig_filename=None,
                 header=None):
        self.supported_version = supported_version
        self.name = name
        self.well = well
        self.masks = masks

        if header is None:
            header = {}
        if 'name' not in list(header.keys()) or header['name'] is None:
            header['name'] = name
        if 'well' not in list(header.keys()) or header['well'] is None:
            header['well'] = well
        if 'orig_filename' not in list(header.keys()) or header['orig_filename'] is None:
            header['orig_filename'] = orig_filename
        self.header = Header(header)

        if logs is None:
            logs = {}
        self.logs = logs

    def __str__(self):
        return "Supported LAS Version : {0}".format(self.supported_version)

    def __len__(self):
        try:
            return len(self.logs[self.log_names()[0]].data)  # all logs within a Block should have same length
        except:
            return 0

    def get_depth_unit(self):
        return self.header.strt.unit.lower()

    def get_start(self, log_name=None):
        """
        The start value of the log can differ from from whats specified in the header
        log_name
            str
            if log_name specified, return the depth to where that log starts
        :return:
            float
            Start depth of log in meters
        """
        if log_name is not None:
            # first check that the log exists
            if log_name not in self.log_names():
                raise IOError('Log {} does not exist in well {}'.format(
                    log_name, self.well
                ))
            # mask out nans
            data = self.logs[log_name].data
            msk = np.ma.masked_invalid(data).mask
            # Get first value depth value where data is not a nan
            start = self.logs['depth'].data[~msk][0]
        else:
            start = np.nanmin(self.logs['depth'].data)
        if self.get_depth_unit() != 'm':
           # Assume it is in feet
            return cnvrt(start, 'ft', 'm')
        else:
            return start

    start = property(get_start)

    def get_stop(self):
        stop = None
        if self.header.stop.value is not None:
            stop = self.header.stop.value
        return stop

    stop = property(get_stop)

    def get_step(self):
        """
        Tries to return the step length in meters
        :return:
        """
        step = None
        if self.header.step.value is not None:
            step = self.header.step.value
            if 'f' in self.header.step.unit:  # step length is in feet
                step = cnvrt(step, 'ft', 'm')
        return step

    step = property(get_step)

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
        The user has to check that it has the correct depth information
        :param log_curve:
            core.log_curve.LogCurve
        :return:
        """
        if not isinstance(log_curve, LogCurve):
            raise IOError('Only LogCurve objects are valid input')

        if len(self) != len(log_curve):
            raise IOError('LogCurve must have same length as the other curves in this Block')

        if log_curve.name is None:
            raise IOError('LogCurve must have a name')

        if log_curve.log_type is None:
            raise IOError('LogCurve must have a log type')

        log_curve.well = self.well
        log_curve.block = self.name

        self.logs[log_curve.name.lower()] = log_curve

    def add_log(self, data, name, log_type, header=None):
        """
        Creates a LogCurve object based on input information, and tries to add the log curve to this Block.

        :param data:
            np.ndarray
        :param name:
            str
        :param log_type:
            str
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
            block=self.name,
            well=self.well,
            data=data,
            header=header)

        self.add_log_curve(log_curve)

    def twt_at_logstart(self, log_name, water_vel, repl_vel, water_depth, kb):
        """
        Calculates the two-way time [s] to the top of the log.

        Inspired by
        https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb

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
        #water_twt = 2.0 * (np.abs(water_depth) + np.abs(kb)) / water_vel  # TODO could it be np.abs(water_depth + np.abs(kb)) / water_vel
        water_twt = 2.0 * np.abs(water_depth + np.abs(kb)) / water_vel
        repl_twt = 2.0 * repl_int / repl_vel

        #print('KB elevation: {} [m]'.format(kb))
        #print('Seafloor elevation: {} [m]'.format(water_depth))
        #print('Water time: {} [s]'.format(water_twt))
        #print('Top of Sonic log: {} [m]'.format(top_of_log))
        #print('Replacement interval: {} [m]'.format(repl_int))
        #print('Two-way replacement time: {} [s]'.format(repl_twt))
        #print('Top-of-log starting time: {} [s]'.format(repl_twt + water_twt))

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
            repl_vel = cnvrt(repl_vel, 'm', 'ft')
        if sonic:
            repl = 1./repl_vel
        else:
            repl = repl_vel
        if us_unit:
            repl = repl*1e6
        nan_mask = np.ma.masked_invalid(self.logs[log_name].data).mask

        # Smooth and despiked version of vp
        smooth_log = self.logs[log_name].despike(spike_threshold)
        smooth_log[nan_mask] = repl

        if debug:
            fig, ax = plt.subplots()
            ax.plot(self.logs['depth'].data, smooth_log, 'r', lw=2)
            ax.plot(self.logs['depth'].data, self.logs[log_name].data, 'k', lw=0.5)
            #ax.plot(self.logs['depth'].data[13000:14500]/3.2804, smooth_log[13000:14500]*3.2804, 'r', lw=2)
            #ax.plot(self.logs['depth'].data[13000:14500]/3.2804, self.logs[log_name].data[13000:14500]*3.2804, 'k', lw=0.5)
            ax.legend(['Smooth and despiked', 'Original'])

        # Handle units
        if sonic:
            scaled_dt = self.get_step() * np.nan_to_num(smooth_log)
            if feet_unit:  #  sonic is in feet units, step is always in meters
                scaled_dt = scaled_dt * 3.28084
        else:
            scaled_dt = self.get_step() * np.nan_to_num(1./smooth_log)
            if feet_unit:  # velcity is in feet, step is always in meters
                scaled_dt = scaled_dt / 3.28084
        if us_unit:
            scaled_dt = scaled_dt * 1.e-6



        tcum = 2 * np.cumsum(scaled_dt)
        tdr = log_start_twt + tcum

        if debug:
            fig, ax = plt.subplots()
            ax.plot(self.logs['depth'].data, scaled_dt, 'b', lw=1)
            ax2 = ax.twinx()
            ax2.plot(self.logs['depth'].data, tdr, 'k', lw=1)
            plt.show()

        return tdr

    def sonic_to_vel(self):
        """
        Converts sonic to velocity
        :return:
        """
        from utils.convert_data import convert

        for ss, vv, vtype in zip(
                ['ac', 'acs'], ['vp', 'vs'], ['P velocity', 'S velocity']
        ):
            info_txt = ''
            if ss not in self.log_names():
                continue
            else:
                din = self.logs[ss].data
                dout = convert(din, 'us/ft', 'm/s')

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


def _read_las(file):
    """Convert file and Return `self`. """
    file_format = None
    ext = file.rsplit(".", 1)[-1].lower()

    if ext == "las":
        file_format = 'las'

    elif ext == 'txt':  # text files from Avseth & Lehocki in the  Spirit study
        file_format = 'RP well table'

    else:
        raise Exception("File format '{}'. not supported!".format(ext))

    with open(file, "r") as f:
        lines = f.readlines()
    return convert(lines, file_format=file_format)  # read all lines from data


def add_one(instring):
    trailing_nr = re.findall(r"\d+", instring)
    if len(trailing_nr) > 0:
        new_trail = str(int(trailing_nr[-1]) + 1)
        instring = instring.replace(trailing_nr[-1], new_trail)
    else:
        instring = instring + ' 1'
    return instring


def test():
    import utils.io as uio
    wp = Project(name='MyProject', log_to_stdout=True)
    
    print(wp.data_frame)

#    well_table = uio.project_wells(wp.project_table, wp.working_dir)
#    w = Well()
#    las_file = list(well_table.keys())[0]
#    logs = list(well_table[las_file]['logs'].keys())
#    print(logs)

#    w.read_las(las_file, only_these_logs=well_table[las_file]['logs'])
#
#    w.calc_mask({'test': ['>', 10], 'phie': ['><', [0.05, 0.15]]}, name=def_msk_name)
#    msk = w.block[def_lb_name].masks[def_msk_name].data
#    fig1, fig2 = plt.figure(1), plt.figure(2)
#    w.depth_plot('P velocity', fig=fig1, mask=msk, show_masked=True)
#    print('Before mask: {}'.format(len(w.block[def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.block[def_lb_name].masks.keys()))))
#    w.apply_mask(def_msk_name)
#    print('After mask: {}'.format(len(w.block[def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.block[def_lb_name].masks.keys()))))
#    w.depth_plot('P velocity', fig=fig2)
#    plt.show()
#    print(w.block[def_lb_name].logs['phie'].header)


if __name__ == '__main__':
    test()
