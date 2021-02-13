# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:27:45 2020

Module for handling rockphysics mineral models

A mineral is defined by its:
    Name: str
    k, Bulk moduli [GPa]: float
    mu, Shear moduli [GPa]: float
    rho, Density, [g/cm3]: float

:copyright: Erik Mårten Blixt (marten.blixt@gmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)       

@author: mblixt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass

from blixt_utils.misc.attribdict import AttribDict
from rp_utils.version import info
from blixt_utils.utils import isnan
import blixt_utils.io.io as uio
import rp.rp_core as rp
import rp_utils.definitions as ud

# data class decorator explained here:
# https://realpython.com/python-data-classes/
@dataclass
class MineralData:
    name: str
    value: float
    unit: str
    desc: str


class Header(AttribDict):
    """
    Class for mineral set header information
    A ``Hedaer`` object may contain all header information (also known as meta
    data) of a MineralSet object. 
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There can be various default attributes which are
    required 
    
    :param
        header: Dictionary containing meta information of a single
        Mineral set object. 
        Possible keywords are
        summarized in the following `Default Attributes`_ section.
    
    .. rubric:: _`Default Attributes`
    
    """
    defaults = {
        'name': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
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
        super(Header, self).__setitem__('modification_date', datetime.now().isoformat())

        # all other keys
        if isinstance(value, dict):
            super(Header, self).__setitem__(key, AttribDict(value))
        else:
            super(Header, self).__setitem__(key, value)

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


class Mineral(object):
    """
    Small class handling a single mineral
    """
    def __init__(self,
                 calculation_method=None,  # 'Interval average' or 'User specified'
                 k=None,  # Bulk moduli [GPa] given by a MineralData object
                 mu=None,  # Shear moduli [GPa] given by a MineralData object
                 rho=None,  # Density [g/cm3] given by a MineralData object
                 name='Quartz',  # str
                 volume_fraction=None,  # str 'complement', volume log name, or float
                 cutoffs=None,   # cutoffs used when calculating the average mineral properties within an interval
                 status=None,
                 header=None):
        
        if header is None:
            header = {}
        if name is not None:
            header['name'] = name

        self.header = Header(header)
        
        # Case input data is given as a float or integer, create a MineralData 
        # object with default units and description
        for this_name, param, unit_str, desc_str, def_val in zip(
                ['k', 'mu', 'rho', 'calculation_method', 'cutoffs', 'status'],
                [k, mu, rho, calculation_method, cutoffs, status],
                ['GPa', 'GPa', 'g/cm3', 'str', 'str', 'str'],
                ['Bulk moduli', 'Shear moduli', 'Density', '', '', 'Status'],
                [36.6, 45., 2.65, 'User specified', None, None]):
            if (param is None) or (isnan(param)):
                param = def_val
            if isinstance(param, int):
                param = float(param)
            if isinstance(param, float):
                param = MineralData(this_name, param, unit_str, desc_str)
            
            super(Mineral, self).__setattr__(this_name, param)

        super(Mineral, self).__setattr__('name', name)
        super(Mineral, self).__setattr__('volume_fraction', volume_fraction)

    
    def __str__(self):
        keys = list(self.__dict__.keys())
        
        pattern = "%%%ds: %%s" % len(keys)
        
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def keys(self):
        """
        Return dict key list of content
        """
        return self.__dict__.keys()

    def calc_k(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def calc_mu(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def calc_rho(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def vp(self):
        """
        Calculates Vp for the mineral
        :return:
            MineralData object with Vp
        """
        #TODO
        # Add check of units!
        vp = rp.v_p(self.k.value, self.mu.value, self.rho.value)
        return MineralData('vp', vp, 'km/s', 'P-wave velocity')

    def plot(self, phic=0.4):
        fig, ax = plt.subplots()
        style = {'lw': 0.5, 'c': 'k'}
        phi = np.arange(0., 1, 0.02)
        for k in [0.01, 0.05, 0.1, 0.3, 0.5]:
            plt.plot(phi, 1./(1. + phi/k), **style)

        plt.plot(phi, 1. -phi, **style)
        plt.plot(phi, 1. -phi/phic, **style)
        plt.grid(True)

        ax.set_ylim(0., 1.05)


class MineralMix(object):
    """
    Class handling a set of minerals
    """
    
    def __init__(self,
                 name=None,
                 minerals=None,
                 header=None):
        """
        :param minerals:
            dict
            a Dictionary of Mineral objects
        """
        self.header = Header(header)

        if minerals is None:
            minerals = {}
        self.minerals = minerals

        if name is None:
            name = 'MyMinerals'
        self.name = name

    def __str__(self):
        out = 'Mineral mixture: {}\n'.format(self.name)
        for w in list(self.minerals.keys()):
            out += " - Well {}\n".format(w)
            for wi in list(self.minerals[w].keys()):
                out += "  + Working interval {}\n".format(wi)
                for m in list(self.minerals[w][wi].keys()):
                    out += "    {}\n".format(m)
                    out += "      K: {}, Mu: {}, Rho {}\n".format(self.minerals[w][wi][m].k.value,
                                                                self.minerals[w][wi][m].mu.value,
                                                                self.minerals[w][wi][m].rho.value)
                    out += "      " \
                           "Volume fraction: {}\n".format(self.minerals[w][wi][m].volume_fraction)
        return out

    def print_all_minerals(self):
        out = ''
        for w in list(self.minerals.keys()):
            for wi in list(self.minerals[w].keys()):
                out += self.print_minerals(w, wi)
        return out

    def print_minerals(self, well_name, wi_name):
        out = 'Mineral mixture: {}, {}, {}\n'.format(well_name, wi_name, self.name)
        this_min = self.minerals[well_name][wi_name]
        for m in list(this_min.keys()):
            out += "    {}\n".format(m)
            out += "      K: {}, Mu: {}, Rho {}\n".format(this_min[m].k.value,
                                                          this_min[m].mu.value,
                                                          this_min[m].rho.value)
            out += "      " \
                       "Calculation method: {}, cutoff: {}\n".format(
                this_min[m].calculation_method, this_min[m].cutoffs)
            out += "      Status: {}\n".format(this_min[m].status)
            out += "      " \
                   "Volume fraction: {}\n".format(this_min[m].volume_fraction)
        return out

    def read_excel(self, filename,
                   min_sheet='Minerals', min_header=1,
                   mix_sheet='Mineral mixtures', mix_header=1):
        """ 
        Read the mineral mixtures from the project table
        """
        # first read in all minerals
        all_minerals = {}
        min_table = pd.read_excel(filename, sheet_name=min_sheet, header=min_header)
        for i, name in enumerate(min_table['Name']):
            if isnan(name):
                continue
            this_min = Mineral(
                min_table['Calculation method'][i] if 'Calculation method' in list(min_table.keys()) else 'User specified',
                float(min_table['Bulk moduli [GPa]'][i]),
                float(min_table['Shear moduli [GPa]'][i]),
                float(min_table['Density [g/cm3]'][i]),
                name.lower(),
                cutoffs=min_table['Cutoffs'][i] if 'Cutoffs' in list(min_table.keys()) else None,
                status='from excel'
            )
            all_minerals[name.lower()] = this_min

        # Then read in the mineral mixtures
        min_mixes = {}
        mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header)
        for i, name in enumerate(mix_table['Mineral name']):
            vf = mix_table['Volume fraction'][i]
            if isnan(vf):
                continue  # Avoid minerals where the volume fraction is not set
            this_well = mix_table['Well name'][i].upper()
            this_wi = mix_table['Interval name'][i].upper()
            this_mineral = deepcopy(all_minerals[name.lower()])
            this_mineral.volume_fraction = float(vf) if (not isinstance(vf, str)) else vf.lower()
            if this_well in list(min_mixes.keys()):
                if this_wi in list(min_mixes[this_well].keys()):
                    min_mixes[this_well][this_wi][name.lower()] = this_mineral
                else:
                    min_mixes[this_well][this_wi] = {name.lower(): this_mineral}
            else:
                min_mixes[this_well] = {this_wi: {name.lower(): this_mineral}}

        self.minerals = min_mixes
        self.header['orig_file'] = filename

    def calc_elastics(self, wells, log_table, wis, block_name=None, calculation_method=None, debug=False):
        """
        Calculates k, mu, and rho for each well and working interval they are defined in, AND where the
        mineral calculation method is set to interval average. The other minerals are left untouched
        :param wells:
            dict
            {well name: core.wells.Well} key: value pairs
        :param log_table:
            dict
            Dictionary of {log type: log name} key: value pairs which decides which logs to use to calculate the averages
            E.G.
                log_table = {
                   'P velocity': 'vp',
                   'S velocity': 'vs',
                   'Density': 'den'}
        :param wis:
            dict
            dictionary of working intervals,
            e.g. wis = rp_utils.io.project_working_intervals(project_table)
        :param calculation_method:
            str
            Name of the calculation method.
        :param debug:
            bool
            if True, generate verbose information and create some plots
        :return:
        """
        if block_name is None:
            block_name = ud.def_lb_name
        if calculation_method is None:
            calculation_method = 'Interval average'
        if calculation_method != 'Interval average':
            raise NotImplementedError('{} calculation method not implemented yet')
        lstyle = {'lw': 0.5, 'color': 'k', 'alpha': 0.2}

        # Keep track of number of minerals the elastic properties should be calculated from
        figs = {}
        axes = {}

        warn_txt = None
        for well_name, well in wells.items():
            # test if this well is listed in the mineral mixture
            if well_name not in list(self.minerals.keys()):
                warn_txt = 'Well {} not present in the given mineral mix {}'.format(well_name, self.name)
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            # test if this will is listed in the working intervals
            if well_name not in list(wis.keys()):
                warn_txt = 'Well {} not present among the working intervals'.format(well_name)
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            # test if the necessary logs exists in this well
            necess_log_types = ['P velocity', 'S velocity', 'Density']
            if not all([log_table[xx] in well.log_names() for xx in necess_log_types]):
                warn_txt = 'The necessary logs ({}) are missing in well {}: {}'.format(
                    ', '.join([log_table[xx] for xx in necess_log_types]), well_name, ', '.join(well.log_names())
                )
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            figs[well_name] = {}
            axes[well_name] = {}
            # count number of minerals we should plot and create figures and axes
            _n = 0  # number of minerals
            if debug:
                for jj, wi_name in enumerate(list(self.minerals[well_name].keys())):
                    for kk, mineral in enumerate(list(self.minerals[well_name][wi_name].keys())):
                        if self.minerals[well_name][wi_name][mineral].calculation_method == calculation_method:
                            _n += 1
                    figs[well_name][wi_name] = plt.figure(figsize=(12, 4*_n))
                    axes[well_name][wi_name] = figs[well_name][wi_name].subplots(nrows=_n, ncols=3)

            _n = 0  # number of minerals
            for jj, wi_name in enumerate(list(self.minerals[well_name].keys())):
                for kk, mineral in enumerate(list(self.minerals[well_name][wi_name].keys())):
                    if self.minerals[well_name][wi_name][mineral].calculation_method == calculation_method:
                        _n += 1
                        # Start calculating the interval averages
                        if debug:
                            print('Start calculating the interval average of k, mu and rho for '
                                  '{} in interval {} in well {}, using the cutoff: {}'.format(
                                    mineral, wi_name, well_name, self.minerals[well_name][wi_name][mineral].cutoffs)
                            )
                        vp = well.block[block_name].logs[log_table['P velocity']].data
                        vs = well.block[block_name].logs[log_table['S velocity']].data
                        rho = well.block[block_name].logs[log_table['Density']].data
                        cutoffs = uio.interpret_cutoffs_string(self.minerals[well_name][wi_name][mineral].cutoffs)
                        well.calc_mask(cutoffs, 'this mask', wis=wis, wi_name=wi_name, log_type_input=False)
                        mask = well.block[block_name].masks['this mask'].data

                        this_rho = np.nanmedian(rho[mask])
                        this_mu = np.nanmedian(rho[mask] * vs[mask]**2 * 1.E-6)  # GPa
                        this_k = np.nanmedian(rho[mask] * (vp[mask]**2 - (4./3.) * vs[mask]**2) * 1.E-6)
                        if debug:
                            print(this_k, this_mu, this_rho)
                            this_depth = well.block[block_name].logs['depth'].data[mask]
                            # TODO the axes indices will only work when plotting one mineral (_n = 1)
                            axes[well_name][wi_name][0].plot(rho[mask], this_depth)
                            axes[well_name][wi_name][0].set_xlabel('Density [{}]'.format(
                                well.block[block_name].logs[log_table['Density']].header.unit))
                            axes[well_name][wi_name][0].axvline(this_rho, **lstyle)
                            axes[well_name][wi_name][1].plot(vp[mask], this_depth)
                            axes[well_name][wi_name][1].plot(vs[mask], this_depth)
                            axes[well_name][wi_name][1].set_xlabel('Vp & Vs [m/s]')
                            axes[well_name][wi_name][1].legend(['Vp', 'Vs'])
                            axes[well_name][wi_name][2].plot(rho[mask]*vs[mask]**2 * 1.E-6, this_depth)
                            axes[well_name][wi_name][2].plot(rho[mask] * (vp[mask]**2 - (4./3.)*vs[mask]**2) * 1.E-6,
                                                             this_depth)
                            axes[well_name][wi_name][2].set_xlabel('$\mu$ & K [GPa]')
                            axes[well_name][wi_name][2].legend(['$\mu$', 'K'])
                            axes[well_name][wi_name][2].axvline(this_mu, **lstyle)
                            axes[well_name][wi_name][2].axvline(this_k, **lstyle)
                            for an, ax in enumerate(axes[well_name][wi_name]):
                                if an == 0:
                                    ax.set_ylabel('MD [m]')
                                ax.set_ylim(ax.get_ylim()[::-1])
                            figs[well_name][wi_name].suptitle('{}, {}, {}'.format(well_name, wi_name, mineral))

                        # store the calculated elastic properties in the minerals object
                        if not np.isnan(this_k):
                            self.minerals[well_name][wi_name][mineral].k.value = this_k
                        if not np.isnan(this_mu):
                            self.minerals[well_name][wi_name][mineral].mu.value = this_mu
                        if not np.isnan(this_rho):
                            self.minerals[well_name][wi_name][mineral].rho.value = this_rho
                        # update the status of the mineral
                        self.minerals[well_name][wi_name][mineral].status = 'calculated from interval average'
                    else:
                        continue

