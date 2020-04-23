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
from decorator import decorator
import inspect
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass

from utils.attribdict import AttribDict
from utils.utils import info, isnan
import rp.rp_core as rp

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
                 k=None,  # Bulk moduli [GPa] given by a MineralData object
                 mu=None,  # Shear moduli [GPa] given by a MineralData object
                 rho=None,  # Density [g/cm3] given by a MineralData object
                 name='Quartz',  # str
                 volume_fraction='complement',  # str 'complement' or volume log name
                 header=None):
        
        if header is None:
            header = {}
        if name is not None:
            header['name'] = name

        self.header = Header(header)
        
        # Case input data is given as a float or integer, create a MineralData 
        # object with default units and description
        for this_name, param, unit_str, desc_str, def_val in zip(
                ['k', 'mu', 'rho'], 
                [k, mu, rho],
                ['GPa', 'GPa', 'g/cm3'],
                ['Bulk moduli', 'Shear moduli', 'Density'],
                [36.6, 45., 2.65]):
            if param is None:
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

    def calc_k(self, dummy_tvd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_tvd:
        :return:
        """
        return object.__getattribute__(self, 'k')

    def calc_mu(self, dummy_tvd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_tvd:
        :return:
        """
        return object.__getattribute__(self, 'mu')

    def calc_rho(self, dummy_tvd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_tvd:
        :return:
        """
        return object.__getattribute__(self, 'rho')

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

    def print_minerals(self, well_name, wi_name):
        out = 'Mineral mixture: {}, {}, {}\n'.format(well_name, wi_name, self.name)
        for m in list(self.minerals[well_name][wi_name].keys()):
            out += "    {}\n".format(m)
            out += "      K: {}, Mu: {}, Rho {}\n".format(self.minerals[well_name][wi_name][m].k.value,
                                                          self.minerals[well_name][wi_name][m].mu.value,
                                                          self.minerals[well_name][wi_name][m].rho.value)
            out += "      " \
                   "Volume fraction: {}\n".format(self.minerals[well_name][wi_name][m].volume_fraction)
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
                float(min_table['Bulk moduli [GPa]'][i]),
                float(min_table['Shear moduli [GPa]'][i]),
                float(min_table['Density [g/cm3]'][i]),
                name.lower(),
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
