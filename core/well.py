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
import utils.io as uio
from utils.io import convert
import utils.masks as msks
from utils.utils import arrange_logging
from plotting import crossplot as xp
from core.minerals import MineralSet
from core.log_curve import LogCurve
import rp.rp_core as rp
from utils.harmonize_logs import harmonize_logs as fixlogs

# global variables
supported_version = {2.0, 3.0}
logger = logging.getLogger(__name__)
def_lb_name = 'LogBlock'  # default LogBlock name
def_msk_name = 'Mask'  # default mask name

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
                 tops_type='petrel',
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
                warn_text = 'The provided project table {}, does not exist'.format(self.project_table)
                logger.warning(warn_text)
                raise Warning(warn_text)

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
            warn_text = 'The provided log file {}, does not exist'.format(file_name)
            logger.warning(warn_text)
            raise IOError(warn_text)
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

    def check_well_names(self):
        """
        Loops through the well names in the project table and checks if they are in
        a consistent format (E.G. "6507_3_1S" and NOT "6507/3-1 S"), and if they corresponds to the well name
        in the given las files.

        :return:
        """

        _well_table = uio.project_wells(self.project_table, self.working_dir)
        for lfile in list(_well_table.keys()):
            wname = _well_table[lfile]['Given well name']
            if ('/' in wname) or ('-' in wname) or (' ' in wname):
                warn_txt = "Special signs, like '/', '-' or ' ', are not allowed in well name: {}".format(wname)
                print("WARNING: {}".format(warn_txt))
                logger.warning(warn_txt)
            for line in uio.get_las_well_info(lfile):
                if re.search("[.]{1}", line) is None:
                    continue
                if re.search("[ ]{1}", line) is None:
                    continue
                if re.search("[:]{1}", line) is None:
                    continue
                mnem_end = re.search("[.]{1}", line).end()
                unit_end = mnem_end + re.search("[ ]{1}", line[mnem_end:]).end()
                colon_end = unit_end + re.search("[:]{1}", line[unit_end:]).start()
                # divide line
                mnem = line[:mnem_end - 1].strip()
                data = line[unit_end:colon_end].strip()
                if mnem == 'WELL':
                    if uio.fix_well_name(data) != wname:
                        warn_txt = 'Well name in las file ({}) does not correspond to well name in project table ({})'.format(
                            uio.fix_well_name(data), wname)
                        print("WARNING: {}".format(warn_txt))
                        logger.warning(warn_txt)

    def load_all_wells(self, rename_well_logs=None):
        """

        :param rename_well_logs:
            dict
            E.G.
            {'depth': ['DEPT', 'MD']}
            where the key is the wanted well log name, and the value list is a list of well log names to translate from
        :return:
        """
        well_table = uio.project_wells(self.project_table, self.working_dir)
        all_wells = {}
        last_wname = ''
        for i, lasfile in enumerate(well_table):
            wname = well_table[lasfile]['Given well name']
            print(i, wname, lasfile)
            if wname != last_wname:  # New well
                w = Well()
                w.read_well_table(well_table, i, block_name=def_lb_name, rename_well_logs=rename_well_logs)
                all_wells[wname] = w
                last_wname = wname
            else:  # Existing well
                all_wells[wname].read_well_table(well_table, i, block_name=def_lb_name, rename_well_logs=rename_well_logs)

        return all_wells

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
    class handling wells, with LogBlock objects that in its turn contain LogCurve objects.
    Other well related information is stored at suitable object level.
    The reading .las files is more or less copied from converter.py
        https://pypi.org/project/las-converter/
    """

    def __init__(self,
                 header=None,
                 log_blocks=None):
        if header is None:
            header = {}
        self.header = Header(header)
        if log_blocks is None:
            self.log_blocks = {}

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
        for lblock in list(self.log_blocks.keys()):
            log_list = log_list + self.log_blocks[lblock].get_logs_of_name(log_name)
        return log_list

    def get_logs_of_type(self, log_type):
        log_list = []
        for lblock in list(self.log_blocks.keys()):
            log_list = log_list + self.log_blocks[lblock].get_logs_of_type(log_type)
        return log_list

    def log_names(self):
        ln_list = []
        for lblock in list(self.log_blocks.keys()):
            ln_list = ln_list + self.log_blocks[lblock].log_names()
        # remove any duplicates
        ln_list = list(set(ln_list))
        return ln_list

    def log_types(self):
        lt_list = []
        for lblock in list(self.log_blocks.keys()):
            lt_list = lt_list + self.log_blocks[lblock].log_types()
        # remove any duplicates
        lt_list = list(set(lt_list))
        return lt_list

    def calc_vrh_bounds(self, fluid_minerals, param='k', method='Voigt', log_block=None):
        """
        Calculates the Voigt-Reuss-Hill bounds of parameter param, for the  fluid or mineral mixture defined in
        fluid_minerals, for the given LogBlock.

        :param fluid_minerals:
            core.minerals.MineralSet
            or
            core.fluid.fluids['xx'] where 'xx' is 'initial' or 'fluid'

            minerals = core.minerals.MineralSet()
            minerals.read_excel(working_project.project_table)
        :param param:
            str
            'k' for Bulk modulus
            'mu' for shear modulus
            'rho' for density
        :param method:
            str
            'Voigt' for the upper bound or Voigt average
            'Reuss' for the lower bound or Reuss average
            'Voigt-Reuss-Hill'  for the average of the two above
        :param log_block:
            str
            Name of the LogBlock for which the bounds are calculated

        :return
            np.ndarray
            Bounds of parameter 'param'
        """
        if log_block is None:
            log_block = def_lb_name

        fluid = False
        if not (isinstance(fluid_minerals, MineralSet) or isinstance(fluid_minerals, dict)):
            warn_txt = 'Input fluid_minerals must be a MineralSet or a subselection of a FluidSet object'
            logger.warning(warn_txt)
            raise Warning(warn_txt)

        if isinstance(fluid_minerals, MineralSet):
            obj = fluid_minerals.minerals

        #if isinstance(fluid_minerals, FluidSet):
        if isinstance(fluid_minerals, dict):
            # Assume that one of the 'initial' or 'final' fluid sets have been chosen
            fluid = True
            obj = fluid_minerals

        if param not in ['k', 'mu', 'rho']:
            raise IOError('Bounds can only be calculated for k, mu and rho')

        if len(list(obj.keys())) > 2:
            warn_txt = 'The bounds calculation has only been tested for two-components mixtures'
            print('Warning {}'.format(warn_txt))
            logger.warning(warn_txt)

        complement = None
        this_fraction = None  # A volume fraction log
        this_component = None  #
        fractions = []
        components = []
        for this_fm in list(obj.keys()):
            print(' Mineral: {}, volume frac: {}'.format(this_fm, obj[this_fm].volume_fraction))
            tmp_frac = obj[this_fm].volume_fraction
            if tmp_frac == 'complement':  # Calculated as 1. - the others
                if complement is not None:
                    raise IOError('Only one complement log is allowed')
                complement = this_fm
                this_fraction = this_fm  # Insert mineral name for the complement mineral
            elif isinstance(tmp_frac, float):
                this_fraction = tmp_frac
            else:
                _name = tmp_frac.lower()
                if _name not in list(self.log_blocks[log_block].logs.keys()):
                    warn_txt = 'The volume fraction {} is lacking in LogBlock {} of well {}'.format(
                        _name, log_block, self.well
                    )
                    print(warn_txt)
                    logger.warning(warn_txt)
                    continue
                this_fraction = self.log_blocks[log_block].logs[_name].data
            this_component = obj[this_fm].__getattribute__(param).value
            fractions.append(this_fraction)
            components.append(this_component)

        # Calculate the complement fraction only when there are more than one constituent
        if len(fractions) == len(components) > 1:
            if complement not in fractions:
                raise IOError('No complement log given')
            compl = 1. - sum([f for f in fractions if not isinstance(f, str)])

            # insert the complement at the location of the complement mineral
            fractions[fractions.index(complement)] = compl

        #return fractions, components
        tmp = rp.vrh_bounds(fractions, components)
        if method == 'Voigt':
            return tmp[0]
        elif method == 'Reuss':
            return tmp[1]
        else:
            return tmp[2]

    def calc_mask(self,
                    cutoffs,
                    name=def_msk_name,
                    tops=None,
                    use_tops=None,
                    overwrite=True,
                    append=False,
                    log_type_input=False
    ):
        """
        Based on the different cutoffs in the 'cutoffs' dictionary, each LogBlock in well is masked accordingly.

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
        :param tops:
            dict
            as returned from utils.io.read_tops() function

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
        :return:
        """
        if not isinstance(cutoffs, dict):
            raise IOError('Cutoffs must be specified as dict, not {}'.format(type(cutoffs)))

        if log_type_input:
            # When the cutoffs are based on log types, and not the individual log names, we need to create a
            # copy of the cutoffs that use the first log name under each log type, eg.
            this_cutoffs = {}
            for key in list(cutoffs.keys()):
                if len(self.get_logs_of_type(key)) < 1:
                    print('XXXXX')
                this_cutoffs[self.get_logs_of_type(key)[0].name] = cutoffs[key]
            cutoffs = this_cutoffs
            #print('FROM CALC_MASK', cutoffs)

        if isinstance(use_tops, list) and (tops is not None):
            if self.well not in list(tops.keys()):
                warn_text = 'Well: {} is not in the list of tops: {}'.format(
                    self.well,
                    ', '.join(list(tops.keys()))
                )
                logger.warning(warn_text)
                print('WARNING: {}'.format(warn_text))
            elif not all([t.upper() in list(tops[self.well].keys()) for t in use_tops]):
                warn_text = 'The selected tops {} are not among the well tops {}'.format(
                    ', '.join(use_tops),
                    '. '.join(list(tops[self.well].keys()))
                )
                logger.warning(warn_text)
                print('WARNING: {}'.format(warn_text))
            else:
                # Append the depth mask from the tops file
                cutoffs['depth'] = ['><',
                     [
                         tops[self.well][use_tops[0].upper()],
                         tops[self.well][use_tops[1].upper()]
                     ]
                 ]

        msk_str = ''
        for key in list(cutoffs.keys()):
            msk_str += '{}: {} [{}]'.format(
                key, cutoffs[key][0], ', '.join([str(m) for m in cutoffs[key][1]])) if \
                isinstance(cutoffs[key][1], list) else \
                '{}: {} {}, '.format(
                key, cutoffs[key][0], cutoffs[key][1])
        print('XXXX')
        print(msk_str)

        for lblock in list(self.log_blocks.keys()):
            masks = []
            for lname in list(self.log_blocks[lblock].logs.keys()):
                if lname not in list(cutoffs.keys()):
                    continue
                else:
                    # calculate mask
                    masks.append(msks.create_mask(
                        self.log_blocks[lblock].logs[lname].data, cutoffs[lname][0], cutoffs[lname][1]
                    ))
            if len(masks) > 0:
                # combine all masks for this LogBlock
                block_mask = msks.combine_masks(masks)
                if self.log_blocks[lblock].masks is None:
                    self.log_blocks[lblock].masks = {}
                if (name in list(self.log_blocks[lblock].masks.keys())) and (not overwrite) and (not append):
                    # create a new name for the mask
                    name = add_one(name)
                elif (name in list(self.log_blocks[lblock].masks.keys())) and append:
                    # read in old mask
                    old_mask = self.log_blocks[lblock].masks[name].data
                    old_desc = self.log_blocks[lblock].masks[name].header.desc
                    # modify the new
                    block_mask = msks.combine_masks([old_mask, block_mask])
                    msk_str = '{} AND {}'.format(msk_str, old_desc)

                # Create an object, similar to the logs object of a LogBlock, that contain the masks
                self.log_blocks[lblock].masks[name] = LogCurve(
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
            else:
                continue


    def apply_mask(self,
            name=None):
        """
        Applies the named mask to the logs under each LogBlock where the named mask exists, adds the masking description
        to the LogCurve header and deletes the named masks object under each LogBlock.

        :param name:
            str
            name of the mask to apply.
            If if doesn't exist, nothing is done
        :return:
        """
        if name is not None:
            for lblock in list(self.log_blocks.keys()):
                if name in list(self.log_blocks[lblock].masks.keys()):
                    msk = self.log_blocks[lblock].masks[name].data
                    desc = self.log_blocks[lblock].masks[name].header.desc
                    for lname in list(self.log_blocks[lblock].logs.keys()):
                        self.log_blocks[lblock].logs[lname].data = self.log_blocks[lblock].logs[lname].data[msk]
                        self.log_blocks[lblock].logs[lname].header.modification_history = 'Mask: {}'.format(desc)
                    del(self.log_blocks[lblock].masks[name])

    def depth_plot(self,
                   log_type='P velocity',
                   log_name=None,
                   mask=None,
                   tops=None,
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

        # set up plotting environment
        if fig is None:
            fig = plt.figure(figsize=(8,10))
        if ax is None:
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

        # loop over all LogBlocks
        cnt = -1
        legends = []
        for logcurve in list_of_logs:
            cnt += 1
            if x_templ is None:
                x_templ = l2tmpl(logcurve.header)
            #print(cnt, logcurve.name, xp.cnames[cnt], mask)
            xdata = logcurve.data
            ydata = self.log_blocks[logcurve.log_block].logs['depth'].data
            legends.append(logcurve.name)
            xp.plot(
                xdata,
                ydata,
                cdata=xp.cnames[cnt],
                title='{}: {}'.format(self.well, ttl),
                xtempl=x_templ,
                ytempl=l2tmpl(self.log_blocks[logcurve.log_block].logs['depth'].header),
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

        ax.set_ylim(ax.get_ylim()[::-1])
        this_legend = ax.legend(
            legends,
            prop=FontProperties(size='smaller'),
            scatterpoints = 1,
            markerscale=2,
            loc=1
        )
        if savefig is not None:
            fig.savefig(savefig)


    def read_well_table(self, well_table, index, block_name=None, rename_well_logs=None):
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
        :return:
        """
        lfile = list(well_table.keys())[index]
        if 'Note' in list(well_table[lfile].keys()):
            note = well_table[lfile]['Note']
        else:
            note = None
        self.read_las(lfile,
                      only_these_logs=well_table[lfile]['logs'],
                      block_name=block_name,
                      rename_well_logs=rename_well_logs,
                      note=note)

    def read_las(
            self,
            filename,
            block_name=None,
            only_these_logs=None,
            rename_well_logs=None,
            note=None
    ):
        """
        Reads in a las file (filename) and adds the selected logs (listed in only_these_logs) to the
        LogBlock specified by block_name.

        :param filename:
            str
            name of las file
        :param block_name:
            str
            Name of LogBlock where the logs should be added to
        :param only_these_logs:
            dict
            dictionary of log names to load from the las file (keys), and corresponding log type as value
            if None, all are loaded
        :param rename_well_logs:
            dict
            E.G.
            {'depth': ['DEPT', 'MD']}
            where the key is the wanted well log name, and the value list is a list of well log names to translate from
        :param note:
            str
            String containing notes for the las file being read
        :return:
        """
        if block_name is None:
            block_name = def_lb_name

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


        def add_logs_to_logblock(_well_dict, _block_name, _only_these_logs):
            """
            Helper function that add logs to the given block name.

            :param _well_dict:
            :param _only_these_logs:
            :return:
            """
            # TODO
            # add las filename to LogBlock header

            # Make sure depth is always read in
            if isinstance(_only_these_logs, dict) and ('depth' not in list(_only_these_logs.keys())):
                _only_these_logs['depth'] = 'Depth'

            exists = False
            same = True
            # Test if LogBlock already exists
            if _block_name in list(self.log_blocks.keys()):
                exists = True
                #print('Length of existing data: {}'.format(len(self.log_blocks[_block_name].logs['depth'].data)))
                # Test if LogBlock has the same header
                for key in ['strt', 'stop', 'step']:
                    if _well_dict['well_info'][key]['value'] != self.log_blocks[_block_name].header[key].value:
                        #print('{} in new versus existing log block {}: {}'.format(
                        #    key,
                        #    _well_dict['well_info'][key]['value'],
                        #    self.log_blocks[_block_name].header[key].value))
                        same = False

            if exists and not same:
                ## Create a new LogBlock, with a new name, and warn the user
                #new_block_name = add_one(_block_name)
                #logger.warning(
                #    'LogBlock {} existed and was different from imported las file, new LogBlock {} was created'.format(
                #        _block_name, new_block_name
                #    ))
                #_block_name = new_block_name
                #info_txt = 'Start modifying the logs in las file to fit the existing LogBlock'
                #print(info_txt)
                #logger.info(info_txt)
                #print(' Length of existing data in well: {}'.format(
                #    len(self.log_blocks[_block_name])
                #))
                #print(' Length before fixing: {}'.format(len(_well_dict['data']['depth'])))
                fixlogs(
                    _well_dict,
                    self.log_blocks[_block_name].header['strt'].value,
                    self.log_blocks[_block_name].header['stop'].value,
                    self.log_blocks[_block_name].header['step'].value,
                    len(self.log_blocks[_block_name])
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
                            log_block=_block_name,
                            well=_well_dict['well_info']['well']['value'],
                            data=np.array(_well_dict['data'][_key]),
                            header=this_header
                        )
                        these_logs[_key].header.orig_filename = filename
                        # TODO
                        # test and try why the header.name = _key isn't necessary here!
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
                        log_block=_block_name,
                        well=_well_dict['well_info']['well']['value'],
                        data=np.array(_well_dict['data'][_key]),
                        header=_well_dict['curve'][_key]
                    )
                    these_logs[_key].header.orig_filename = filename
                    these_logs[_key].header.name = _key
            else:
                logger.warning('No logs added to {}'.format(_well_dict['well_info']['well']['value']))

            # Test if LogBlock already exists
            if exists and same:
                self.log_blocks[_block_name].logs.update(these_logs)
            else:
                self.log_blocks[_block_name] = LogBlock(
                    name=_block_name,
                    well=_well_dict['well_info']['well']['value'],
                    logs=these_logs,
                    header={
                        key: _well_dict['well_info'][key] for key in ['strt', 'stop', 'step']
                    }
                )
                self.log_blocks[_block_name].header.name = _block_name

        # Make sure only_these_logs is a dictionary
        if isinstance(only_these_logs, str):
            only_these_logs = {only_these_logs: None}

        null_val, generated_keys, well_dict = _read_las(filename, rename_well_logs=rename_well_logs)
        logger.debug('Reading {}'.format(filename))
        if well_dict['version']['vers']['value'] not in supported_version:
            raise Exception("Version {} not supported!".format(
                well_dict['version']['vers']['value']))

        # Test if this well has a header, and that we are loading from the same well
        if 'well' not in list(self.header.keys()):  # current well object is empty
            # Add all well specific headers to well header
            add_headers(well_dict['well_info'], ['strt', 'stop', 'step'], note)
            self.header.name = {
                'value': well_dict['well_info']['well']['value'],
                'unit': '',
                'desc': 'Well name'}
            # Add logs to a LogBlock
            add_logs_to_logblock(well_dict, block_name, only_these_logs)

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
                if 'y' in ans.lower:
                    add_well = True
                else:
                    add_well = False

            # TODO make sure that adding a well with different name actually will work
            if add_well:
                # Add all well specific headers to well header, except well name
                add_headers(well_dict['well_info'], ['strt', 'stop', 'step', 'well'], note)
                # add to existing LogBloc
                add_logs_to_logblock(well_dict, block_name, only_these_logs)

    def keys(self):
        return self.__dict__.keys()


class LogBlock(object):
    """
    A log block is a collection of logs which share the same depth information, i.e 
    start, stop, step
    """

    def __init__(self,
                 name=None,
                 well=None,
                 logs=None,
                 masks=None,
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
        self.header = Header(header)

        if logs is None:
            logs = {}
        self.logs = logs

    def __str__(self):
        return "Supported LAS Version : {0}".format(self.supported_version)

    def __len__(self):
        try:
            return len(self.logs[self.log_names()[0]].data)  # all logs within a LogBlock should have same length
        except:
            return 0

    def get_start(self):
        start = None
        if self.header.strt.value is not None:
            start = self.header.strt.value
        return start

    start = property(get_start)

    def get_stop(self):
        stop = None
        if self.header.stop.value is not None:
            stop = self.header.stop.value
        return stop

    stop = property(get_stop)

    def get_step(self):
        step = None
        if self.header.step.value is not None:
            step = self.header.step.value
        return step

    step = property(get_step)

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
        Adds the provided log_curve to the LogBlock.
        The user has to check that it has the correct depth information
        :param log_curve:
            core.log_curve.LogCurve
        :return:
        """
        if not isinstance(log_curve, LogCurve):
            raise IOError('Only LogCurve objects are valid input')

        if len(self) != len(log_curve):
            raise IOError('LogCurve must have same length as the other curves in this LogBlock')

        if log_curve.name is None:
            raise IOError('LogCurve must have a name')

        if log_curve.log_type is None:
            raise IOError('LogCurve must have a log type')

        log_curve.well = self.well
        log_curve.log_block = self.name

        self.logs[log_curve.name.lower()] = log_curve

    def add_log(self, data, name, log_type, header=None):
        """
        Creates a LogCurve object based on input information, and tries to add the log curve to this LogBlock.

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

        header['name'] = name
        header['well'] = self.well
        header['log_type'] = log_type
        if 'unit' not in list(header.keys()):
            header['unit'] = None
        if 'desc' not in list(header.keys()):
            header['desc'] = None

        log_curve = LogCurve(
            name=name,
            log_block=self.name,
            well=self.well,
            data=data,
            header=header)

        self.add_log_curve(log_curve)

def _read_las(file, rename_well_logs=None):
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
    return convert(lines, file_format=file_format, rename_well_logs=rename_well_logs)  # read all lines from data


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
    well_table = uio.project_wells(wp.project_table, wp.working_dir)
    w = Well()
    las_file = list(well_table.keys())[0]
    logs = list(well_table[las_file]['logs'].keys())
    print(logs)
#    w.read_las(las_file, only_these_logs=well_table[las_file]['logs'])
#
#    w.calc_mask({'test': ['>', 10], 'phie': ['><', [0.05, 0.15]]}, name=def_msk_name)
#    msk = w.log_blocks[def_lb_name].masks[def_msk_name].data
#    fig1, fig2 = plt.figure(1), plt.figure(2)
#    w.depth_plot('P velocity', fig=fig1, mask=msk, show_masked=True)
#    print('Before mask: {}'.format(len(w.log_blocks[def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.log_blocks[def_lb_name].masks.keys()))))
#    w.apply_mask(def_msk_name)
#    print('After mask: {}'.format(len(w.log_blocks[def_lb_name].logs['phie'].data)))
#    print('Masks: {}'.format(', '.join(list(w.log_blocks[def_lb_name].masks.keys()))))
#    w.depth_plot('P velocity', fig=fig2)
#    plt.show()
#    print(w.log_blocks[def_lb_name].logs['phie'].header)


if __name__ == '__main__':
    test()
