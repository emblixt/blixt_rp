# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:31:28 2020
Module for handling rockphysics fluid models
:copyright: Erik Mårten Blixt (marten.blixt@gmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)       
@author: mblixt
"""
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import rp.rp_core as rp
from rp.rp_core import Param
from utils.attribdict import AttribDict
from utils.utils import info, isnan

# data class decorator explained here:
# https://realpython.com/python-data-classes/

logger = logging.getLogger(__name__)


class Header(AttribDict):
    """
    Class for fluid header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a Fluid object. 
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every variable import and export modules
    
    :param
        header: Dictionary containing meta information of a single
        Fluid object. 
        Possible keywords are
        summarized in the following `Default Attributes`_ section.
    
    .. rubric:: _`Default Attributes`
    
    """
    defaults = {
        'name': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat()
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
        Return better readable string representation of Header object.
        """
        #keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Fluid(object):
    def __init__(self,
            calculation_method=None,  # or Batzle and Wang',  # or 'User specified'
            k=None,  # Bulk modulus in GPa
            mu=None,  # Shear modulus in GPa
            rho=None,  # Density in g/cm3
            temp_gradient=None,
            temp_ref=None,
            pressure_gradient=None,
            pressure_ref=None,
            salinity=None,
            gor=None,
            oil_api=None,
            gas_gravity=None,
            gas_mixing=None, 
            brie_exponent=None,
            fluid_type=None,
            name='Default',
            volume_fraction=None,
            status=None,
            header=None):

        if header is None:
            header = {}
        if name is not None:
            header['name'] = name

        self.header = Header(header)
        # initiating class variables to None to avoid warnings
        self.calculation_method = None
        self.k = None
        self.mu = None
        self.rho = None
        self.temp_gradient = None
        self.temp_ref = None
        self.pressure_gradient = None
        self.pressure_ref = None
        self.salinity = None
        self.gor = None
        self.oil_api = None
        self.gas_gravity = None
        self.gas_mixing = None
        self.brie_exponent = None
        self.fluid_type = None
        self.name = None
        self.volume_fraction = None

        # Case input data is given as a float or integer, create a Param
        # object with default units and description
        for this_name, param, unit_str, desc_str, def_val in zip(
                ['k', 'mu', 'rho', 'calculation_method', 'temp_gradient', 'temp_ref',
                #['calculation_method', 'temp_gradient', 'temp_ref',
                 'pressure_gradient', 'pressure_ref', 'salinity', 'gor', 
                 'oil_api', 'gas_gravity', 'gas_mixing', 'brie_exponent', 'status'],
                [k, mu, rho, calculation_method, temp_gradient, temp_ref,
                #[calculation_method, temp_gradient, temp_ref,
                 pressure_gradient, pressure_ref, salinity, gor, 
                 oil_api, gas_gravity, gas_mixing, brie_exponent, status],
                ['GPa', 'GPa', 'g/cm3', '', 'C/m', 'C',
                #['', 'C/m', 'C',
                 'MPa/m', 'MPa', 'ppm', '',
                 'API', '', '', '', 'str'],
                ['Bulk moduli', 'Shear moduli', 'Density', '', '', '',
                #['', '', '',
                 '', '', '', 'Gas/Oil ratio',
                 '', '', 'Wood or Brie', '', 'Status'],
                [0.9, 0.0, 0.8, 'User specified', 0.03, 10.,
                #['User specified', 0.03, 10.,
                   0.0107, 0., 70000., 1., 
                   30., 0.6, 'Wood', 2., None]):
            if (param is None) or isnan(param):
                param = def_val
                #print('param {} is None'.format(this_name))
            if isinstance(param, int):
                param = float(param)
                #print('param {} is integer'.format(this_name))
            if isinstance(param, float) or isinstance(param, str):
                #print('param {} is float or string'.format(this_name))
                param = Param(this_name, param, unit_str, desc_str)
            # TODO
            # catch the cases where input data is neither None, int,  float or str
            # TODO
            # The handling of parameters that are strings are different for fluids (Param) and minerals (strings)
            
            super(Fluid, self).__setattr__(this_name, param)
        
        super(Fluid, self).__setattr__('name', name)
        super(Fluid, self).__setattr__('fluid_type', fluid_type)
        super(Fluid, self).__setattr__('volume_fraction', volume_fraction)

#    def __getattribute__(self, item):
#        if object.__getattribute__(self, 'calculation_method ') == 'Batzle and Wang':
#            print('Batzle and Wang')#
#        #    if (item == 'k') or (item == 'mu') or (item == 'rho'):
#        #        warn_txt = 'Calculation of fluid properties not yet implemented'
#        #        print('WARNING: {}'.format(warn_txt))
#        #        logger.warning(warn_txt)
#        #        return None
#        #else:
#        #    return object.__getattribute__(self, item)
#        return object.__getattribute__(self, item)

    def __str__(self):
        keys = list(self.__dict__.keys())
        r_keys = []
        for k in keys:
            if isinstance(self.__dict__[k],Param) and isnan(self.__dict__[k].value):
                r_keys.append(k)
        for k in r_keys:
            keys.remove(k)
        
        pattern = "%%%ds: %%s" % len(keys)
        
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def print_fluid(self, tvd=None):
        out = '  {}\n'.format(self.name)
        out += '      K: {}, Mu: {}, Rho {}\n'.format(
            self.calc_k(tvd).value, self.calc_mu(tvd).value, self.calc_rho(tvd).value)
        out += '      Volume fraction: {}\n'.format(self.volume_fraction)
        out += '      Calculation method: {}\n'.format(self.calculation_method.value)
        return out

    def keys(self):
        return self.__dict__.keys()

    def calc_k(self, tvd):
        if self.calculation_method.value == 'Batzle and Wang':
            #print('calc_k: {}, TVD: {}'.format(self.name, tvd))
            if tvd is None:
                warn_txt = 'No TVD value given for the fluid calculation'
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                return Param(name='',
                             value=np.nan,
                             unit='',
                             desc='')
            _s = self.salinity
            _p = self.pressure_ref.value + self.pressure_gradient.value * tvd
            _t = self.temp_ref.value + self.temp_gradient.value * tvd
            if self.fluid_type == 'brine':
                rho_b = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                return Param(name='k_b',
                             value=v_p_b**2 * rho_b * 1.E-6,
                             unit='GPa',
                             desc='Brine bulk modulus'
                )
            elif self.fluid_type == 'oil':
                k_o, rho_o = rp.k_and_rho_o(
                    self.oil_api,
                    self.gas_gravity,
                    self.gor,
                    _p,
                    _t
                )
                return k_o
            elif self.fluid_type == 'gas':
                k_g, rho_g = rp.k_and_rho_g(self.gas_gravity, _p, _t)
                return k_g
            else:
                raise NotImplementedError('Bulk modulus not possible to calculate for {}'.format(self.fluid_type))
        else:
            return object.__getattribute__(self, 'k')

    def calc_mu(self, tvd):
        #print('calc_mu: {}, TVD: {}'.format(self.name, tvd))
        if self.calculation_method.value == 'Batzle and Wang':
            if tvd is None:
                warn_txt = 'No TVD value given for the fluid calculation'
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                return Param(name='',
                             value=np.nan,
                             unit='',
                             desc='')
            if self.fluid_type == 'brine':
                return Param(name='mu_b',
                             value=np.nan,
                             unit='GPa',
                             desc='Brine shear modulus'
                             )
            elif self.fluid_type == 'oil':
                return Param(name='mu_o',
                             value=np.nan,
                             unit='GPa',
                             desc='Oil shear modulus'
                             )
            elif self.fluid_type == 'gas':
                return Param(name='mu_g',
                             value=np.nan,
                             unit='GPa',
                             desc='Gas shear modulus'
                             )
            else:
                raise NotImplementedError('Shear modulus not possible to calculate for {}'.format(self.fluid_type))
        else:
            return object.__getattribute__(self, 'mu')

    def calc_rho(self, tvd):
        if self.calculation_method.value == 'Batzle and Wang':
            #print('calc_rho: {}, TVD: {}'.format(self.name, tvd))
            if tvd is None:
                warn_txt = 'No TVD value given for the fluid calculation'
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                return Param(name='',
                             value=np.nan,
                             unit='',
                             desc='')
            _s = self.salinity
            _p = self.pressure_ref.value + self.pressure_gradient.value * tvd
            _t = self.temp_ref.value + self.temp_gradient.value * tvd
            if self.fluid_type == 'brine':
                rho_b = rp.rho_b(_s, _p,  _t)
                return rho_b
            elif self.fluid_type == 'oil':
                k_o, rho_o = rp.k_and_rho_o(
                    self.oil_api,
                    self.gas_gravity,
                    self.gor,
                    _p,
                    _t
                )
                return rho_o
            elif self.fluid_type == 'gas':
                k_g, rho_g = rp.k_and_rho_g(self.gas_gravity, _p, _t)
                return rho_g
            else:
                raise NotImplementedError('Bulk modulus not possible to calculate for {}'.format(self.fluid_type))
        else:
            return object.__getattribute__(self, 'rho')


class FluidMix(object):
    """
    Dictionary containing the initial, and final, fluids for given wells and intervals
    {substition order:
        {well name:
            {working interval:
                {fluid name: Fluid()}}}}
    """
    def __init__(self,
                 name=None,
                 fluids=None, 
                 header=None):
        """
        :param fluids:
            dict
            Dictionary of Param objects
        """
        self.header = Header(header)
        if fluids is None:
            fluids = {}
        self.fluids = fluids

        if name is None:
            name = 'MyFluids'
        self.name = name

    def print_all_fluids(self, tvd=None):
        out = ''
        for key in ['initial', 'final']:
            for w in list(self.fluids[key].keys()):
                for wi in list(self.fluids[key][w].keys()):
                    out += self.print_fluids(key, w, wi, tvd)
        return out

    def print_fluids(self, subst, well_name, wi_name, tvd=None):
        out = 'Fluid mixture: {}, {}, {}, {}'.format(subst, well_name, wi_name, self.name)
        if tvd is not None:
            out += ' at TVD: {} m\n'.format(tvd)
        else:
            out += '\n'
        for m in list(self.fluids[subst][well_name][wi_name].keys()):
            out += self.fluids[subst][well_name][wi_name][m].print_fluid(tvd)
        return out


    def read_excel(self, filename,
                   fluid_sheet='Fluids', fluid_header=1,
                   mix_sheet='Fluid mixtures', mix_header=1):

        # First read in all fluids defined in the project table
        all_fluids = {}
        fluids_table = pd.read_excel(filename,
                                     sheet_name=fluid_sheet, header=fluid_header)

        for i, name in enumerate(fluids_table['Name']):
            if isnan(name):
                continue  # Avoid empty lines
            #if fluids_table['Calculation method'][i] == 'Batzle and Wang':
            #    warn_txt = 'Calculation of fluid properties is still not implemented, please use constant values'
            #    print('WARNING {}'.format(warn_txt))
            #    logger.warning(warn_txt)
            this_fluid = Fluid(
                    fluids_table['Calculation method'][i],
                    float(fluids_table['Bulk moduli [GPa]'][i]),
                    float(fluids_table['Shear moduli [GPa]'][i]),
                    float(fluids_table['Density [g/cm3]'][i]),
                    float(fluids_table['T gradient [deg C/m]'][i]),
                    float(fluids_table['T ref [C]'][i]),
                    float(fluids_table['P gradient [MPa/m]'][i]),
                    float(fluids_table['P ref [MPa]'][i]),
                    float(fluids_table['Salinity [ppm]'][i]),
                    float(fluids_table['GOR'][i]),
                    float(fluids_table['Oil API'][i]),
                    float(fluids_table['Gas gravity'][i]),
                    fluids_table['Gas mixing'][i],
                    float(fluids_table['Brie exponent'][i]),
                    # Fluid type is now defined in the fluid mixture sheet
                    #fluid_type=None if isnan(fluids_table['Fluid type'][i]) else fluids_table['Fluid type'][i].lower(),
                    name=name.lower(),
                    status='from excel'
            )
            all_fluids[name.lower()] = this_fluid

        # Then read in the fluid mixes
        fluids_mixes = {
            'initial': {},
            'final': {}
        }
        mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header)
        for i, name in enumerate(mix_table['Fluid name']):
            if mix_table['Use'][i] != 'Yes':
                continue
            vf = mix_table['Volume fraction'][i]
            ftype = mix_table['Fluid type'][i]
            this_name = '{}_{}'.format(name.lower(), '' if isnan(ftype) else ftype.lower())
            if isnan(vf):
                continue  # Avoid fluids where the volume fraction is not set
            this_subst = mix_table['Substitution order'][i].lower()
            this_well = mix_table['Well name'][i].upper()
            this_wi = mix_table['Interval name'][i].upper()
            this_fluid = deepcopy(all_fluids[name.lower()])
            this_fluid.volume_fraction = \
                float(vf) if (not isinstance(vf, str)) else vf.lower()
            this_fluid.fluid_type = None if isnan(ftype) else ftype.lower()
            this_fluid.name = this_name
            this_fluid.header.name = this_name



            # iterate down in this complex dictionary
            # {this_subst:                          # 1'st level: initial or final
            #       {this_well:                     # 2'nd level: well
            #           {this_wi:                   # 3'd level: working interval
            #               {fluid_name: Fluid()    # 4'th level: fluid
            #       }}}}
            # 2'nd level
            if this_well in list(fluids_mixes[this_subst].keys()):
                # 3'rd level
                if this_wi in list(fluids_mixes[this_subst][this_well].keys()):
                    # 4'th level
                    fluids_mixes[this_subst][this_well][this_wi][this_name] = this_fluid
                else:
                    fluids_mixes[this_subst][this_well][this_wi] = {this_name: this_fluid}
            else:
                fluids_mixes[this_subst][this_well] = {this_wi: {this_name: this_fluid}}

        self.fluids = fluids_mixes
        self.header['orig_file'] = filename


def test_fluidsub():
    from importlib import reload
    import matplotlib.pyplot as plt
    import rp.rp_core as rp
    reload(rp)
    from core.well import Well

    w = Well()
    # Create a well table without using the excel sheet
    well_table = {'../test_data/Well A.las':
        {'Given well name': 'WELL_A',
         'logs':
             {'vp_dry': 'P velocity',
              'vp_so08': 'P velocity',
              'vp_sg08': 'P velocity',
              'vs_dry': 'S velocity',
              'vs_so08': 'S velocity',
              'vs_sg08': 'S velocity',
              'rho_dry': 'Density',
              'rho_so08': 'Density',
              'rho_sg08': 'Density',
              'phie': 'Porosity',
              'vcl': 'Volume'},
         'Note': ''}}
    w.read_well_table(well_table, 0)
    w.calc_mask({'vcl': ['<', 0.4], 'phie': ['>', 0.1]}, name='sand')
    mask = w.block['Logs'].masks['sand'].data
    vp = w.block['Logs'].logs['vp_dry'].data[mask]
    vs = w.block['Logs'].logs['vs_dry'].data[mask]
    rho = w.block['Logs'].logs['rho_dry'].data[mask]
    por = w.block['Logs'].logs['phie'].data[mask]

    test = 'constants'  #'array'
    # Test with constant Vsh and constant Sw
    v_sh = 0.2
    s_w = 0.2
    if test == 'array':  # test with arrays of v_sh and s_w
        v_sh = w.block['Logs'].logs['vcl'].data[mask]
        # create a mock-up water saturation
        s_w = 0.2 + v_sh
        s_w[s_w > 1.0] = 1.0

    # Mineral models, K & MU in GPa, rho in g/cm3
    rho_qz = 2.6;   k_qz = 37;  mu_qz = 44
    rho_sh = 2.8;   k_sh = 15;  mu_sh = 5

    # Mineral shear and bulk modulus using Voigt-Reuss-Hill average for given Vsh
    k0 = rp.vrh_bounds([v_sh, (1.-v_sh)], [k_sh, k_qz])[-1]  # GPa
    mu0 = rp.vrh_bounds([v_sh, (1.-v_sh)], [mu_sh, mu_qz])[-1]  # GPa
    rho0 = v_sh*rho_sh + (1. - v_sh)*rho_qz  # g/cm3

    # Brine model, rho in g/cm3, K in GPa
    rho_b = 1.1;   k_b = 2.8

    # Hydrocarbon model, K in GPa
    fluid = 'oil'
    rho_o = 0.8;    k_o = 0.9
    rho_g = 0.2;    k_g = 0.06
    (k_hc, rho_hc) = (k_g, rho_g) if fluid == 'gas' else (k_o, rho_o)

    # intitial fluid
    rho_f1 = rho_b; k_f1 = k_b

    # After fluid substitution
    rho_f2 = s_w*rho_b + (1.-s_w)*rho_hc
    k_f2 = rp.vrh_bounds([s_w, (1.-s_w)], [k_b, k_hc])[1]  # Reuss uniform fluid mix

    v_p_2, v_s_2, rho_2, k_2 = rp.gassmann_vel(vp, vs, rho, k_f1, rho_f1, k_f2, rho_f2, k0, por)

    plt.plot(vp, label='dry')
    plt.plot(w.block['Logs'].logs['vp_so08'].data[mask], label='RD oil')
    plt.plot(v_p_2, label='my oil')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_fluidsub()
