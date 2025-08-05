# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:31:28 2020
Module for handling rockphysics fluid models
This version uses the pint library instead of selfmade Param

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
import unittest
import os, sys
from bokeh.models import ColumnDataSource
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.param import Param
import blixt_rp.rp.rp_core as rp
import blixt_rp.core.well as cw
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info

from .. import ureg, Q_

logger = logging.getLogger(__name__)

def_fluid_vals = dict(
    k=Q_(2, 'GPa'),
    mu=Q_(2, 'GPa'),
    rho=Q_(2, 'gram / cm^3'),
    calculation_method='Batzle and Wang',
    temp_gradient=Q_(0.03, 'degC/m'),
    temp_ref=Q_(4.0, 'degC'),
    pressure_gradient=Q_(0.0107, 'MPa / m'),
    pressure_ref=Q_(0, 'MPa'),
    salinity=Q_(70000.0, 'ppm'),
    gor=Q_(1., ''),
    oil_api=Q_(30., ''),
    gas_gravity=Q_(0.6, ''),
    gas_mixing='Brie',
    brie_exponent=Q_(2., ''),
    status=None
)


def read_all_fluids_from_excel(filename, fluid_sheet='Fluids', fluid_header=1) -> dict:
    # First read in all fluids defined in the project table
    all_fluids = {}
    fluids_table = pd.read_excel(filename,
                                 sheet_name=fluid_sheet, header=fluid_header, engine='openpyxl')

    for i, name in enumerate(fluids_table['Name']):
        if isnan(name):
            continue  # Avoid empty lines
        this_fluid = Fluid(
            calculation_method='User specified' if isnan(fluids_table['Calculation method'][i]) else fluids_table['Calculation method'][i],
            k=float(fluids_table['Bulk moduli [GPa]'][i]),
            mu=float(fluids_table['Shear moduli [GPa]'][i]),
            rho=float(fluids_table['Density [g/cm3]'][i]),
            temp_gradient=float(fluids_table['T gradient [deg C/m]'][i]),
            temp_ref=float(fluids_table['T ref [C]'][i]),
            pressure_gradient=float(fluids_table['P gradient [MPa/m]'][i]),
            pressure_ref=float(fluids_table['P ref [MPa]'][i]),
            salinity=float(fluids_table['Salinity [ppm]'][i]),
            gor=float(fluids_table['GOR'][i]),
            oil_api=float(fluids_table['Oil API'][i]),
            gas_gravity=float(fluids_table['Gas gravity'][i]),
            gas_mixing=fluids_table['Gas mixing'][i],
            brie_exponent=float(fluids_table['Brie exponent'][i]),
            name=name.lower(),
            status='from excel'
        )
        all_fluids[name.lower()] = this_fluid

    return all_fluids


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
                 calculation_method: None | str = None,  # or Batzle and Wang',  # or 'User specified'
                 k: None | Q_ = None,  # Bulk modulus in GPa
                 mu: None | Q_ = None,  # Shear modulus in GPa
                 rho: None | Q_ = None,  # Density in g/cm3
                 temp_gradient: None | Q_ = None,
                 temp_ref: None | Q_ = None,  # at seafloor
                 pressure_gradient: None | Q_ = None,
                 pressure_ref: None | Q_ = None,  # at seafloor
                 salinity: None | Q_ = None,
                 gor: None | Q_ = None,
                 oil_api: None | Q_ = None,
                 gas_gravity: None | Q_ = None,
                 gas_mixing: None | str = None,
                 brie_exponent: None | Q_ = None,
                 fluid_type: None | Q_ = None,
                 name: None | str = 'Default',
                 volume_fraction: None | str | Q_ = None,
                 status: None | str = None,
                 header: None | dict = None):

        if header is None:
            header = {}
        if name is not None:
            header['name'] = name

        self.header = Header(header)
        self.calculation_method = calculation_method
        self.k = k
        self.mu = mu
        self.rho = rho
        self.temp_gradient = temp_gradient
        self.temp_ref = temp_ref
        self.pressure_gradient = pressure_gradient
        self.pressure_ref = pressure_ref
        self.salinity = salinity
        self.gor = gor
        self.oil_api = oil_api
        self.gas_gravity = gas_gravity
        self.gas_mixing = gas_mixing
        self.brie_exponent = brie_exponent
        self.fluid_type = fluid_type.lower() if fluid_type is not None else None
        self.name = name
        self.volume_fraction = volume_fraction

    def __str__(self):
        keys = list(self.__dict__.keys())

        pattern = "%%%ds: %%s" % len(keys)

        head = [ pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def print_fluid(self, verbose=False):
        out = '  {}\n'.format(self.name)
        if verbose:
            out = str(self)
        else:
            out += '      K: {}, Mu: {}, Rho {}\n'.format(
                self.k, self.mu, self.rho)
            out += '      Calculation method: {}\n'.format(self.calculation_method)
            out += '      Status: {}\n'.format(self.status)
            out += '      Volume fraction: {}\n'.format(self.volume_fraction)
        return out

    def keys(self):
        return self.__dict__.keys()

    def calc_k(self, bd: Q_):
        """
        Calculates the fluid bulk modulus at given burial depth
        :param bd:
        :return:
        """
        if self.calculation_method == 'Batzle and Wang':
            print('calc_k: {}, Fluid: {}, Burial depth: {}'.format(self.name, self.fluid_type, bd))

            _s = self.salinity.to('ppm').magnitude
            _p = self.pressure_ref + self.pressure_gradient * bd
            _p = _p.to('MPa').magnitude
            _t = self.temp_ref + self.temp_gradient * bd
            _t = _t.to('degC').magnitude
            if self.fluid_type == 'brine':
                rho_b = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                _this_k = v_p_b**2 * rho_b * 1.E-6
            elif self.fluid_type == 'oil':
                _this_k, rho_o = rp.k_and_rho_o(
                    self.oil_api.magnitude,
                    self.gas_gravity.magnitude,
                    self.gor.magnitude,
                    _p,
                    _t
                )
            elif self.fluid_type == 'gas':
                _this_k, rho_g = rp.k_and_rho_g(self.gas_gravity.magnitude, _p, _t)
            else:
                raise NotImplementedError('Bulk modulus not possible to calculate for {}'.format(self.fluid_type))
            this_k = Q_(_this_k, 'GPa')
            self.k = this_k
        else:
            # No calculation done
            pass

    # TODO Continue here
    def calc_mu(self, bd):
        #print('calc_mu: {}, Burial depth: {}'.format(self.name, bd))
        if self.calculation_method.value == 'Batzle and Wang':
            if bd is None:
                warn_txt = 'No Burial depth value given for the fluid calculation. ' \
                           'Batzle and Wang not possible to calculate'
                print_info(warn_txt, 'warning', logger)
            if self.fluid_type == 'brine':
                this_mu = Param(name='mu_b',
                             value=np.nan,
                             unit='GPa',
                             desc='Brine shear modulus'
                             )
            elif self.fluid_type == 'oil':
                this_mu = Param(name='mu_o',
                             value=np.nan,
                             unit='GPa',
                             desc='Oil shear modulus'
                             )
            elif self.fluid_type == 'gas':
                this_mu = Param(name='mu_g',
                             value=np.nan,
                             unit='GPa',
                             desc='Gas shear modulus'
                             )
            else:
                raise NotImplementedError('Shear modulus not possible to calculate for {}'.format(self.fluid_type))
            self.mu = this_mu
        else:
            # No calculation done
            pass

    def calc_rho(self, bd):
        if self.calculation_method.value == 'Batzle and Wang':
            #print('calc_rho: {}, Burial depth: {}'.format(self.name, bd))
            if bd is None:
                warn_txt = 'No Burial depth value given for the fluid calculation. ' \
                           'Batzle and Wang not possible to calculate'
                print_info(warn_txt, 'warning', logger)

            _s = self.salinity
            _p = self.pressure_ref.value + self.pressure_gradient.value * bd
            _t = self.temp_ref.value + self.temp_gradient.value * bd
            if self.fluid_type == 'brine':
                this_rho = rp.rho_b(_s, _p,  _t)
            elif self.fluid_type == 'oil':
                k_o, this_rho = rp.k_and_rho_o(
                    self.oil_api,
                    self.gas_gravity,
                    self.gor,
                    _p,
                    _t
                )
            elif self.fluid_type == 'gas':
                k_g, this_rho = rp.k_and_rho_g(self.gas_gravity, _p, _t)
            else:
                raise NotImplementedError('Bulk modulus not possible to calculate for {}'.format(self.fluid_type))
            self.rho = this_rho
        else:
            # No calculation done
            pass


class FluidsTable:
    """
    Returns a table and bokeh CDS for the provided fluids
    """
    def __init__(self,
                 fluids: None | list = None,
                 width: None | int = None):
        """

        :param fluids:
            list of Fluid objects
        :param width:
            int
        """
        if width is None:
                width = 700
        self.width = width
        if fluids is None:
            fluids = []
        self.fluids = fluids
        self.keys = ['name', 'k', 'mu', 'rho', 'bd', 'calculation_method',
                     'temp_gradient', 'temp_ref',
                     'pressure_gradient', 'pressure_ref',
                     'salinity', 'gor', 'oil_api', 'gas_gravity', 'gas_mixing', 'brie_exponent', 'fluid_type']
    @property
    def source(self) -> ColumnDataSource:
        _dict = {_x:[] for _x in self.keys}
        for _fluid in self.fluids:
            for _key in self.keys:
                if _key == 'name':
                    _dict['name'].append(_fluid.header.name)
                elif _key in ['bd', 'fluid_type']:
                    _dict[_key].append(None)
                else:
                    _dict[_key].append(_fluid.__dict__[_key].value)
        return ColumnDataSource(_dict)

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        _calc_methods = ['Batzle and Wang', 'User specified']
        _fluid_types = ['-', 'Brine', 'Oil', 'Gas']
        table_columns = [
            TableColumn(field='name', title='Name'),
            TableColumn(field='fluid_type', title='Fluid type',
                        editor=SelectEditor(options=_fluid_types)),
            TableColumn(field='k', title='K [GPa]'),
            TableColumn(field='mu', title='G [GPa]'),
            TableColumn(field='rho', title='Den. [g/cm3]'),
            TableColumn(field='bd', title='Burial depth [m]'),
            TableColumn(field='calculation_method', title='Calc. meth.',
                        editor=SelectEditor(options=_calc_methods)),
            TableColumn(field='temp_ref', title='T. ref. [deg C]'),
            TableColumn(field='temp_gradient', title='T. grad. [deg C/m]'),
            TableColumn(field='pressure_ref', title='P. ref. [MPa]'),
            TableColumn(field='pressure_gradient', title='P. grad. [MPa/m]'),
            TableColumn(field='salinity', title='Salinity [ppm]'),
            TableColumn(field='gor', title='GOR'),
            TableColumn(field='oil_api', title='Oil API'),
            TableColumn(field='gas_gravity', title='Gas gravity')
        ]
        return table_columns

    def draw(self,
             source: ColumnDataSource):
        from bokeh.models import DataTable, Button, CheckboxGroup

        def add_row_function():
            new_data = dict(source.data)
            for _key in list(new_data.keys()):
                new_data[_key].append(None)
            source.data = new_data

        def delete_row_function():
            selected_index = source.selected.indices
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(source.data['name'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(source.data[_x][_i])
            source.selected.indices = []
            source.data = new_data

        def update_table_function():
            new_data = dict(source.data)
            _fluids = []
            for _i in range(len(new_data['name'])):
                this_fluid =  Fluid(
                    calculation_method=new_data['calculation_method'][_i],
                    k=new_data['k'][_i],
                    mu=new_data['mu'][_i],
                    rho=new_data['rho'][_i],
                    temp_gradient=new_data['temp_gradient'][_i],
                    temp_ref=new_data['temp_ref'][_i],
                    pressure_gradient=new_data['pressure_gradient'][_i],
                    pressure_ref=new_data['pressure_ref'][_i],
                    salinity=new_data['salinity'][_i],
                    gor=new_data['gor'][_i],
                    oil_api=new_data['oil_api'][_i],
                    gas_gravity=new_data['gas_gravity'][_i],
                    fluid_type=new_data['fluid_type'][_i].lower(),
                    name=new_data['name'][_i],
                    )
                _fluids.append(this_fluid)
                if new_data['calculation_method'][_i] == 'Batzle and Wang' and (new_data['bd'][_i] is not None) and (new_data['fluid_type'][_i] in ['Brine', 'Oil', 'Gas']):
                    print('XXX1', new_data['name'][_i], new_data['fluid_type'][_i], new_data['bd'][_i])
                    print('XXX2', this_fluid.name, this_fluid.fluid_type)
                    # TODO Can come to this point
                    this_fluid.calc_k(new_data['bd'][_i])
                    this_fluid.calc_mu(new_data['bd'][_i])
                    this_fluid.calc_rho(new_data['bd'][_i])
                    print('YYY', this_fluid.fluid_type.value)
                    new_data['k'][_i] = this_fluid.k
                    new_data['mu'][_i] = this_fluid.mu
                    new_data['rho'][_i] = this_fluid.rho
                    print('YYY', this_fluid.k, this_fluid.mu, this_fluid.rho)
                    # TODO But not here!

            self.fluids = _fluids
            source.data = new_data

        dt = DataTable(
            source=source,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            index_position = -1,
            index_header = 'index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_function)

        return dt, add_row, delete_row, update_table


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

    def print_all_fluids(self, verbose=False):
        out = ''
        for key in ['initial', 'final']:
            for w in list(self.fluids[key].keys()):
                for wi in list(self.fluids[key][w].keys()):
                    out += self.print_fluids(key, w, wi, verbose)
        return out

    def print_fluids(self, subst, well_name, wi_name, verbose=False):
        out = 'Fluid mixture: {}, {}, {}, {}\n'.format(subst, well_name, wi_name, self.name)
        for m in list(self.fluids[subst][well_name][wi_name].keys()):
            out += self.fluids[subst][well_name][wi_name][m].print_fluid(verbose)
        return out

    def read_excel(self, filename,
                   fluid_sheet='Fluids', fluid_header=1,
                   mix_sheet='Fluid mixtures', mix_header=1):

        # First read in all fluids defined in the project table
        all_fluids = read_all_fluids_from_excel(filename, fluid_sheet, fluid_header)

        # Then read in the fluid mixes
        fluids_mixes = {
            'initial': {},
            'final': {}
        }
        mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header, engine='openpyxl')
        for i, name in enumerate(mix_table['Fluid name']):
            if mix_table['Use'][i] != 'Yes':
                continue
            # TODO
            # Add the tag to this_name, then a fluid mixture can contain several fluid substitution cases
            # print(mix_table['Tag'][i])

            # TODO
            vf = mix_table['Volume fraction'][i]
            ftype = mix_table['Fluid type'][i]
            # Need to pair fluid name with fluid type to get unique fluid names in fluid mixture
            this_name = '{}_{}'.format(name.lower(), '' if isnan(ftype) else ftype.lower())
            #this_name = name.lower()
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

    def calc_press_ref(self, wells, templates=None, rho_sea=None, block_name=None, debug=False):
        """
        Calculates the reference pressure in MPa (pressure at mudline (seafloor)) based on the water depth and
        sea water density.
        The pressure reference value is only calculated if the existing value is set to zero
        :param wells:
            dict
            dictionary of {well name: core.wells.Well} key: value pairs
        :param templates:
            dict
            templates that can contain the sea water depth for wells
            templates = rp_utils.io.project_templates(wp.project_table)
        :param rho_sea:
            float
            Density of sea water in g/cm3
        :param debug:
        :return:
        """
        if block_name is None:
            block_name = cw.def_lb_name
        if rho_sea is None:
            rho_sea = 1.025  # g/cm3

        # iterate over all fluids in this fluid mixture
        for subst_ordr in list(self.fluids.keys()):
            for this_well in list(self.fluids[subst_ordr].keys()):
                if this_well not in list(wells.keys()):
                    warn_txt = 'Pressure reference not calculated for {}'.format(this_well)
                    print_info(warn_txt, 'warning', logger)
                    continue
                for wi_name in list(self.fluids[subst_ordr][this_well].keys()):
                    for fluid in list(self.fluids[subst_ordr][this_well][wi_name].keys()):
                        if self.fluids[subst_ordr][this_well][wi_name][fluid].pressure_ref.value == 0.0:
                            # try to extract water depth
                            water_depth = wells[this_well].get_from_well_info('water depth', templates=templates,
                                                                              block_name=block_name)

                            self.fluids[subst_ordr][this_well][wi_name][fluid].pressure_ref.value = \
                                rho_sea * abs(water_depth) * 9.81 * 1.E-3   # MPa

    def calc_elastics(self, wells, wis, templates=None, block_name=None, debug=False):
        """
        Calculates k, mu, and rho for all fluids for each well and working interval they are defined in, and where the
        calculation method is not 'User specified'
        :param wells:
            dict
            {well name: core.wells.Well} key: value pairs
        :param wis:
            dict
            dictionary of working intervals,
            e.g. wis = rp_utils.io.project_working_intervals(project_table)
        :param templates:
            dict
            templates that can contain well information such as kelly bushing and sea water depth
            templates = rp_utils.io.project_tempplates(wp.project_table)
        :param debug:
            bool
            if True, generate verbose information and create some plots
        :return:
        """
        if block_name is None:
            block_name = cw.def_lb_name

        for key in ['initial', 'final']:
            # loop over all wells
            for w in list(self.fluids[key].keys()):
                if w not in list(wells.keys()):
                    warn_txt = 'Well {} not present among the input wells'.format(w)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # test if this will is listed in the working intervals
                if w not in list(wis.keys()):
                    warn_txt = 'Well {} not present among the working intervals'.format(w)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # Extract the measured and burial depth for this well
                bd = wells[w].get_burial_depth(block_name=block_name, templates=templates)
                md = wells[w].block[block_name].get_md()

                # loop over all working intervals
                for wi in list(self.fluids[key][w].keys()):
                    if wi not in list(wis[w].keys()):
                        warn_txt = 'Interval {} not present among the working intervals'.format(wi)
                        print_info(warn_txt, 'warning', logger)
                        continue
                    # Extract the mean burial depth for this working interval
                    wi_md = np.mean(wis[w][wi])
                    wi_md_i = np.nanargmin((md - wi_md)**2)
                    wi_bd = bd[wi_md_i]

                    if debug:
                        print('{}, Well: {}, interval: {}, burial depth: {:.2f}'.format(key, w, wi, wi_bd))

                    # iterate over all fluids
                    info_txt = ''
                    for f in list(self.fluids[key][w][wi].keys()):
                        this_fluid = self.fluids[key][w][wi][f]
                        info_txt += ' Fluid: {}, '.format(f)
                        if this_fluid.calculation_method.value == 'User specified':
                            info_txt += 'user specified. Skipped.'
                            continue
                        if this_fluid.status.value == 'from excel':
                            # start calculating fluid properties
                            info_txt += "has status 'from excel', "
                            if this_fluid.calculation_method.value == 'Batzle and Wang':
                                info_txt += "and will be calculated using 'Batzle and Wang'. "
                                if debug:
                                    print(info_txt)
                                # Start Batzle and Wang calculation
                                this_fluid.calc_k(wi_bd)
                                this_fluid.calc_mu(wi_bd)
                                this_fluid.calc_rho(wi_bd)
                                this_fluid.status.value = \
                                    'calculated using Batzle and Wang at burial depth {:.2f}'.format(wi_bd)
                        else:
                            if debug:
                                info_txt += 'has already been calculated'
                                print(info_txt)
                            continue


class TestCases(unittest.TestCase):
    def test_data(self):
        import os
        working_dir = 'C:\\Users\\marte\\PycharmProjects\\blixt_rp'
        # working_dir = 'C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp'

        from blixt_rp.core.well import Project
        wp = Project(
            name='MyProject',
            working_dir=working_dir,
            project_table=os.path.join(working_dir, 'excels\\project_table_new.xlsx')
        )

        myfluids = FluidMix()
        myfluids.read_excel(wp.project_table)
        all_fluids = read_all_fluids_from_excel(wp.project_table)
        return all_fluids, myfluids

    def test_fluid(self):
        default = def_fluid_vals
        f1 = Fluid(**default)
        print(f1)
        f1.fluid_type = 'brine'
        f1.calc_k(Q_(1000, 'm'))
        print('Brine: ', f1.k)
        f1.fluid_type = 'oil'
        f1.calc_k(Q_(1000, 'm'))
        print('Oil: ', f1.k)
        f1.fluid_type = 'gas'
        f1.calc_k(Q_(1000, 'm'))
        print('Gas: ', f1.k)


    def test_fluid_table(self):
        from bokeh.io import output_file
        from bokeh.plotting import show, row, column
        # output_file('C:\\Users\\emb\\Downloads\\plot.html')
        output_file('C:\\Users\\marte\\Downloads\\plot.html')

        all_fluids, fluid_mix = self.test_data()
        ft = FluidsTable(fluids=list(all_fluids.values()))
        source = ft.source
        # for _key in list(source.data.keys()):
        #     print(_key, source.data[_key])

        table, add_row, delete_row, update = ft.draw(source)
        # show(column(table, row(add_row, delete_row, update)))
        return table, add_row, delete_row, update

    def test_fluidsub():
        # TODO
        # This needs to be updated to the new well model
        from importlib import reload
        import matplotlib.pyplot as plt
        import blixt_rp.rp.rp_core as rp
        reload(rp)
        from blixt_rp.core.well import Well

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
        mask = w.block['Logs'].masks['sand'].values
        vp = w.block['Logs'].logs['vp_dry'].values[mask]
        vs = w.block['Logs'].logs['vs_dry'].values[mask]
        rho = w.block['Logs'].logs['rho_dry'].values[mask]
        por = w.block['Logs'].logs['phie'].values[mask]

        test = 'constants'  #'array'
        # Test with constant Vsh and constant Sw
        v_sh = 0.2
        s_w = 0.2
        if test == 'array':  # test with arrays of v_sh and s_w
            v_sh = w.block['Logs'].logs['vcl'].values[mask]
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
        plt.plot(w.block['Logs'].logs['vp_so08'].values[mask], label='RD oil')
        plt.plot(v_p_2, label='my oil')
        plt.legend()
        plt.show()


