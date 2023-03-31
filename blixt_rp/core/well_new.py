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
import numpy as np
import pandas as pd
import logging
import re
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties

from blixt_utils.misc.templates import log_header_to_template as l2tmpl
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
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

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)


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
            block_name = ud.def_lb_name
        return self.block[block_name].get_step()

    def depth_unit(self, block_name=None):
        """
        Returns the units used for the depth measurement in Block 'block_name'.
        :return:
            str
            Name of unit used for depth
        """
        if block_name is None:
            block_name = ud.def_lb_name
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
            templates = rp_utils.io.project_tempplates(wp.project_table)
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
                search_keys = ud.def_kelly_bushing_keys
            elif what == 'water depth':
                search_keys = ud.def_water_depth_keys
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
                        success, return_value = cnvrt(this_result, 'ft', 'm')
                elif self.header[_key].unit.lower() == 'm':
                    info_txt += '[m].'
                    return_value = this_result
                else:
                    # assume it is in feet
                    info_txt += '[feet].'
                    success, return_value = cnvrt(this_result, 'ft', 'm')

                print('INFO: {}'.format(info_txt))
                logger.info(info_txt)
                return return_value

        info_txt += ' failed. No matching keys in header.'
        print('WARNING: {}'.format(info_txt))
        logger.warning(info_txt)
        return 0.0

    def get_burial_depth(self, templates=None, block_name=None, tvd_key=None):
        if block_name is None:
            block_name = ud.def_lb_name

        tvd = self.block[block_name].get_tvd(tvd_key=tvd_key)

        return tvd - np.abs(self.get_from_well_info('water depth', templates, block_name=block_name)) - \
               np.abs(self.get_from_well_info('kb', templates, block_name=block_name))

    def sonic_to_vel(self, block_name=None):
        if block_name is None:
            block_name = ud.def_lb_name

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
            block_name = ud.def_lb_name

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
            if tb.logs[log_name].header.unit.lower() in ud.def_sonic_units:
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
            wis = rp_utils.io.project_working_intervals(project_table.xlsx)
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
            block_name = ud.def_lb_name
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

        # if isinstance(fluid_minerals, FluidSet):
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
            print('INFO: {}'.format(info_txt))

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
                        print('WARNING: {}'.format(warn_txt))
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

    def add_well_path(self, project_table, verbose=True):
        """
        calculates (interpolates) the TVD (relative to KB) based on the input survey points for each MD value in this well

        :param project_table:
            str
            filename of project excel file

        :param verbose:
            bool
            If True, QC plots are created
        :return:
        """
        survey_points, survey_points_info = uio.read_checkshot_or_wellpath(project_table, self.well, "Well paths")
        for lblock in list(self.block.keys()):
            self.block[lblock].add_well_path(survey_points, survey_points_info['filename'], verbose)

    def add_twt(self, project_table, verbose=True):
        """
        calculates (interpolates) the Two way time based on the input time-depth relations specified in the 
        project table.
        If no checkshots are given for a specific well, it will try to use an existing OWT log to calculate the
        TWT, and when no OWT log exists, nothing is done. Living in the hope that the well already have a TWT log:-)

        :param project_table:
            str
            filename of project excel file

        :param verbose:
            bool
            If True, QC plots are created
        :return:
        """
        checkshots, checkshot_info = uio.read_checkshot_or_wellpath(project_table, self.well, "Checkshots")

        if checkshots is None:
            _twt_points = None
            _filename = None
        else:
            _twt_points = {
                'MD': checkshots['MD'],
                'TWT': np.array(checkshots['TWT'])
            }
            _filename = checkshot_info['filename']

        for lblock in list(self.block.keys()):
            self.block[lblock].add_twt(
                _twt_points,
                twt_file=_filename,
                verbose=verbose)

    def calc_mask(self,
                  cutoffs,
                  name=ud.def_msk_name,
                  tops=None,
                  use_tops=None,
                  wis=None,
                  wi_name=None,
                  overwrite=True,
                  append=None,
                  log_type_input=True,
                  log_table=None
                  ):
        """
        Based on the different cutoffs in the 'cutoffs' dictionary, each Block in well is masked accordingly.
        In the resulting mask, a False value indicates that the data is masked out

        :param cutoffs:
            dict
            dictionary with log name as keys, and list with mask operator and limits as values
            E.G. {'depth': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
        :param name:
            str
            name of the mask
        :param tops:
            dict
            as returned from rp_utils.io.read_tops() function
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
            wis = rp_utils.io.project_working_intervals(wp.project_table)

        :param wi_name:
            str
            name of working interval to mask, other intervals will be set to False in boolean mask
        :param overwrite:
            bool
            if True, any existing mask with given name will be overwritten
        :param append:
            str
            if equal to 'AND' or 'OR, the new mask will be appended to any existing mask with given name
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
        from blixt_utils.utils import mask_string
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
                    if _key not in log_table:
                        warn_txt = 'Log table does not contain the log types the cutoffs are based on: {}'.format(_key)
                        logger.warning(warn_txt)
                        raise IOError(warn_txt)
                    _this_cutoffs[log_table[_key]] = _cutoffs[_key]
                else:
                    warn_txt = 'Mask in {} is based on log types, but no log table is specified'.format(self.well)
                    print(warn_txt)
                    logger.warning(warn_txt)
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

        msk_str = mask_string(cutoffs, wi_name)

        for lblock in list(self.block.keys()):
            masks = []
            if len(cutoffs) == 0:  # no cutoffs, mask is all true
                self.block[lblock].masks[name] = LogCurve(
                    name=name,
                    well=self.well,
                    data=np.array(np.ones(len(self.block[lblock])), dtype=bool),
                    header={
                        'name': name,
                        'well': self.well,
                        'log_type': 'Mask',
                        'desc': 'All true mask for empty cutoffs'
                    }
                )
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
                if (name in list(self.block[lblock].masks.keys())) and (not overwrite) and (not isinstance(append, str)):
                    # create a new name for the mask
                    name = add_one(name)
                elif (name in list(self.block[lblock].masks.keys())) and isinstance(append, str):
                    # read in old mask
                    if append not in ['AND', 'OR']:
                        raise IOError("Parameter 'append' must be either 'AND' or 'OR'")
                    old_mask = self.block[lblock].masks[name].data
                    old_desc = self.block[lblock].masks[name].header.desc
                    # modify the new
                    block_mask = msks.combine_masks([old_mask, block_mask], combine_operator=append)
                    msk_str = '{} {} {}'.format(msk_str, append, old_desc)

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
                    # print('{} True values in mask: {}'.format(np.sum(block_mask), msk_str))
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
                    del (self.block[lblock].masks[name])

    def depth_plot(self,
                   log_type='P velocity',
                   log_name=None,
                   mask=None,
                   tops=None,
                   wis=None,
                   #fig=None,
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
            as returned from rp_utils.io.read_tops() function
        :param wis:
            dict
            dictionary of working intervals
        :param fig:
            matplotlib.figure.Figure object
        :param ax:
            matplotlib.axes._subplots.AxesSubplot object
        :param templates:
            dict
            templates dictionary as returned from rp_utils.io.project_templates()
        :param savefig:
            str
            full path name of file to save plot to
        :param kwargs:
         y_log_name: str, default value 'depth'. Set it to 'twt' to plot against time
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

        #if fig is None:
        #    if ax is None:
        #        fig = plt.figure(figsize=(8, 10))
        #        ax = fig.subplots()
        #    else:
        #        _savefig = False
        #elif ax is None:
        #    ax = fig.subplots()

        y_log_name = kwargs.pop('y_log_name', 'depth')
        show_masked = kwargs.pop('show_masked', False)

        if y_log_name not in self.log_names():
            info_txt = 'No log named {} in {}, plotting data against depth instead'.format(y_log_name, self.well)
            print('WARNING: {}'.format(info_txt))
            logger.warning(info_txt)
            y_log_name = 'depth'

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
            # print(cnt, logcurve.name, xp.cnames[cnt], mask)
            xdata = logcurve.data
            ydata = self.block[logcurve.block].logs[y_log_name].data
            legends.append(logcurve.name)
            xp.plot(
                xdata,
                ydata,
                cdata=xp.cnames[cnt],
                title='{}: {}'.format(self.well, ttl),
                xtempl=x_templ,
                ytempl=l2tmpl(self.block[logcurve.block].logs[y_log_name].header),
                mask=mask,
                show_masked=show_masked,
                #fig=fig,
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
            scatterpoints=1,
            markerscale=2,
            loc=1
        )
        if _savefig:
            fig.savefig(savefig)

    def read_well_table(self, well_table, index, block_name=None,
                        rename_well_logs=None, use_this_well_name=None, templates=None):
        """
        Takes the well_table and reads in the well defined by the index number.

        :param well_table:
            dict
            E.G.
            > from well_project import Project
            > wp = Project()
            > import rp_utils.io as uio
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
        :param templates:
            dict
            Template dictionary as returned from blixt_utils.io.io.project_templates('project_table.xlsx')
        :return:
        """
        if block_name is None:
            block_name = ud.def_lb_name
        lfile = list(well_table.keys())[index]
        if 'Note' in list(well_table[lfile].keys()):
            note = well_table[lfile]['Note']
        else:
            note = None
        self.read_las(lfile,
                      only_these_logs=well_table[lfile]['logs'],
                      block_name=block_name,
                      rename_well_logs=rename_well_logs[lfile],
                      use_this_well_name=use_this_well_name,
                      note=note,
                      templates=templates)

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
            note=None,
            templates=None
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
            Useful when different las file have different well names for the same well
        :param note:
            str
            String containing notes for the las file being read
        :param templates: 
            dict
            Template dictionary as returned from blixt_utils.io.io.project_templates('project_table.xlsx')
            If this is provided, read_las() tries to compare the units in the las file, with the units defined
            in the Template dictionary, for the specific log type, and convert data accordingly
        :return:
        """
        if block_name is None:
            block_name = ud.def_lb_name

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

        # Make sure only_these_logs is a dictionary
        if isinstance(only_these_logs, str):
            only_these_logs = {only_these_logs.lower(): None}

        null_val, generated_keys, well_dict = _read_data(filename)
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
        if well_dict['version']['vers']['value'] not in supported_version:
            raise Exception("Version {} not supported!".format(
                well_dict['version']['vers']['value']))

        # Rename well
        if use_this_well_name is not None:
            well_dict['well_info']['well']['value'] = uio.fix_well_name(use_this_well_name)

        # Test if this well has a header, and that we are loading from the same well
        if 'well' not in list(self.header.keys()):  # current well object is empty
            # Add all well specific headers to well header
            self.header = add_headers(self.header, well_dict['well_info'], ['strt', 'stop', 'step'], note)
            self.header.name = {
                'value': well_dict['well_info']['well']['value'],
                'unit': '',
                'desc': 'Well name'}
            # Add logs to a Block
            # _add_logs_to_block(well_dict, block_name, only_these_logs, filename, templates)
            self.block = add_logs_to_block(self.block, well_dict, block_name, only_these_logs, filename, templates)

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
                self.header = add_headers(self.header, well_dict['well_info'], ['strt', 'stop', 'step'], note)
                # add to existing LogBloc
                # _add_logs_to_block(well_dict, block_name, only_these_logs, filename, templates)
                self.block = add_logs_to_block(self.block, well_dict, block_name, only_these_logs, filename, templates)

    def read_log_data(
            self,
            filename,
            block_name=None,
            only_these_logs=None,
            rename_well_logs=None,
            use_this_well_name=None,
            note=None,
            templates=None
    ):
        """
        A more general version of above 'read_las'
        Initially, it was written to support reading excel sheets with well log data, typically core results and
        geochemistry

        Args:
            filename:
                str
                full path name of file with data
            block_name:
                str
                Name Block that should store the data
            only_these_logs:
                dict
                Dictionary with log name: log type as key: value pairs. E.G.
                {ac': 'Sonic', 'acs': 'Shear sonic', 'rdep': 'Resistivity', 'rmed': 'Resistivity'}
            rename_well_logs:
                dict
                E.G.
                {'depth': ['DEPT', 'MD']}
                where the key is the wanted well log name, and the value list is a list of well log names to translate from
            use_this_well_name:
                str
                Name we would like to use.
                Useful when different las file have different wells for the same well
            note:
                str
                String containing notes for the las file being read
            :param templates:
                dict
                Template dictionary as returned from blixt_utils.io.io.project_templates('project_table.xlsx')
                If this is provided, read_las() tries to compare the units in the las file, with the units defined
                in the Template dictionary, for the specific log type, and convert data accordingly.
                If only_these_logs is None, no unit conversion can take place

        Returns:
        """
        if block_name is None:
            block_name = ud.def_lb_name

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

        # Make sure only_these_logs is a dictionary
        if isinstance(only_these_logs, str):
            only_these_logs = {only_these_logs.lower(): None}

        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == 'xlsx':
            # reads an excel sheet and extracts all data.
            # A column header named 'Well', and a column header named 'MD' is necessary for it to work

            null_val = 'NaN'
            accepted_sections = {"version", "well_info", "parameter", "curve", "data"}
            well_dict = {xx: {} for xx in accepted_sections}
            well_dict['version'] = {'vers': {'value': 'xlsx', 'unit': '', 'desc': 'Read from excel table of well data'}}

            # TODO
            # If data is stored on another sheet than the first, we need to add support for this!
            table = pd.read_excel(filename, header=0, engine='openpyxl')
            generated_keys = [xx.lower() for xx in list(table.keys())]
            for key in generated_keys:
                well_dict['curve'][key] = {'api_code': None, 'unit': None, 'desc' : None}
                well_dict['data'][key] = []  # initate empty lists of data
            # collect data from excel sheet for current well
            for i in range(len(table)):
                this_well_name = uio.fix_well_name(table.iloc[i, 0])
                if this_well_name != self.well:
                    continue
                for j, key in enumerate(generated_keys):
                    well_dict['data'][key].append(table.iloc[i, j])

            # Create / update the well_info part based on the imported data
            md = None
            try:
                md = np.array(well_dict['data']['md'])
            except:
                raise IOError('Excel sheet {} lacks information about MD'.format(filename))
            if md is not None:
                well_dict['well_info']['strt'] = {'value': md.min(), 'unit': 'm', 'desc': 'First reference value'}
                well_dict['well_info']['stop'] = {'value': md.max(), 'unit': 'm', 'desc': 'Last reference value'}
                well_dict['well_info']['step'] = {'value': None, 'unit': '', 'desc': 'Data is not regularly sampled'}
                well_dict['well_info']['null'] = {'value': null_val, 'unit': '', 'desc': 'Missing value'}
                well_dict['well_info']['well'] = {'value': self.well, 'unit': '', 'desc': 'Well name'}

            # return null_val, generated_keys, well_dict
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

            # Rename well
            if use_this_well_name is not None:
                well_dict['well_info']['well']['value'] = uio.fix_well_name(use_this_well_name)

            # Test if this well has a header, and that we are loading from the same well
            if 'well' not in list(self.header.keys()):  # current well object is empty
                # Add all well specific headers to well header
                self.header = add_headers(self.header, well_dict['well_info'], ['strt', 'stop', 'step'], note)
                self.header.name = {
                    'value': well_dict['well_info']['well']['value'],
                    'unit': '',
                    'desc': 'Well name'}
                # Add logs to a Block
                self.block = add_logs_to_block(self.block, well_dict, block_name, only_these_logs, filename)

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
                    self.header = add_headers(self.header, well_dict['well_info'], ['strt', 'stop', 'step', 'well'], note)
                    # add to existing LogBloc
                    self.block = add_logs_to_block(self.block, well_dict, block_name, only_these_logs, filename)

        else:
            # TODO
            # Make read_log_data() more flexible so that it can handle standard las files too.
            error_txt = 'read_log_data() only supports excel sheet data yet'
            logger.warning(error_txt)
            raise NotImplementedError(error_txt)

    def keys(self):
        return self.__dict__.keys()


def _read_data(file):
    """Convert file and Return `self`. """
    file_format = None
    ext = file.rsplit(".", 1)[-1].lower()

    if ext == "las":
        file_format = 'las'

    elif ext == 'txt':  # text files from Avseth & Lehocki in the  Spirit study
        file_format = 'RP well table'

    elif ext == 'xlsx':
        file_format = 'Excel sheet'

    else:
        raise Exception("File format '{}'. not supported!".format(ext))

    if (file_format == 'las') or (file_format == 'RP well table'):
        with open(file, "r", encoding='UTF8') as f:
            lines = f.readlines()
        return well_reader(lines, file_format=file_format)  # read all lines from data


def add_one(instring):
    trailing_nr = re.findall(r"\d+", instring)
    if len(trailing_nr) > 0:
        new_trail = str(int(trailing_nr[-1]) + 1)
        instring = instring.replace(trailing_nr[-1], new_trail)
    else:
        instring = instring + ' 1'
    return instring


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


def add_logs_to_block(_block, _well_dict, _block_name, _only_these_logs, _filename, _templates):
    """
    Helper function that add logs to the given block name.

    :param _block:
        Block object
    :param _well_dict:
    :param _block_name:
    :param _only_these_logs:
    :param _filename:
    :param _templates:
        dict
        Template dictionary as returned from blixt_utils.io.io.project_templates('project_table.xlsx')
        When provided, the template is used to convert the input data from the given unit to the units specified
        in the templates.

    :return:
        modified block

    """

    # Make sure depth is always read in
    if isinstance(_only_these_logs, dict) and ('depth' not in list(_only_these_logs.keys())):
        _only_these_logs['depth'] = 'Depth'

    exists = False
    same = True
    # Test if Block already exists
    if _block_name in list(_block.keys()):
        exists = True
        # Test if Block has the same header
        for _key in ['strt', 'stop', 'step']:
            if _well_dict['well_info'][_key]['value'] != _block[_block_name].header[_key].value:
                same = False

    if exists and not same:
        fixlogs(
            _well_dict,
            _block[_block_name].header['strt'].value,
            _block[_block_name].header['stop'].value,
            _block[_block_name].header['step'].value,
            len(_block[_block_name])
        )
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
                logger.debug('Adding log {}'.format(_key))
                _data = np.array(_well_dict['data'][_key])
                this_header = _well_dict['curve'][_key]
                this_header.update({'log_type': _only_these_logs[_key]})
                # Try to convert the units of the data according to the units given in _templates
                if _templates is not None:
                    _success, _data = cnvrt(
                        _data,
                        _well_dict['curve'][_key]['unit'],
                        _templates[_only_these_logs[_key]]['unit'])
                    if _success:
                        logger.debug('Units converted from {} to {}'.format(
                            _well_dict['curve'][_key]['unit'],
                            _templates[_only_these_logs[_key]]['unit']))
                        this_header.update({'unit': _templates[_only_these_logs[_key]]['unit']})
                these_logs[_key] = LogCurve(
                    name=_key,
                    block=_block_name,
                    well=_well_dict['well_info']['well']['value'],
                    data=np.array(_well_dict['data'][_key]),
                    header=this_header
                )
                these_logs[_key].header.orig_filename = _filename
            else:
                logger.warning("Log '{}' in {} is missing\n  [{}]".format(
                    _key,
                    _well_dict['well_info']['well']['value'],
                    ', '.join(list(_well_dict['curve'].keys()))
                    # ', '.join(list(_only_these_logs.keys()))
                )
                )
    elif _only_these_logs is None:
        # add all logs
        for _key in list(_well_dict['curve'].keys()):
            logger.debug('Adding log {}'.format(_key))
            _data = np.array(_well_dict['data'][_key])
            # Try to convert the units of the data according to the units given in _templates
            if _templates is not None:
                logger.warning(
                    'No unit conversion can be done. Lack information about which log type each log belongs to' +
                    ' Need the only_these_logs dictionary for that')
            these_logs[_key] = LogCurve(
                name=_key,
                block=_block_name,
                well=_well_dict['well_info']['well']['value'],
                data=np.array(_well_dict['data'][_key]),
                header=_well_dict['curve'][_key]
            )
            these_logs[_key].header.orig_filename = _filename
            these_logs[_key].header.name = _key
    else:
        logger.warning('No logs added to {}'.format(_well_dict['well_info']['well']['value']))

    # Test if Block already exists
    if exists and same:
        _block[_block_name].logs.update(these_logs)
        _block[_block_name].header.orig_filename.value = '{}, {}'.format(
            _block[_block_name].header.orig_filename.value, _filename
        )
    else:
        _block[_block_name] = Block(
            name=_block_name,
            well=_well_dict['well_info']['well']['value'],
            logs=these_logs,
            orig_filename=_filename,
            header={
                key: _well_dict['well_info'][key] for key in ['strt', 'stop', 'step']
            }
        )
        _block[_block_name].header.name = _block_name

    return _block


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
