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

from bokeh.io import output_file
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
from blixt_rp.core.core import Intervals

from .. import ureg, Q_

logger = logging.getLogger(__name__)

def_fluid_vals = dict(
    k=Q_(2, 'GPa'),
    mu=Q_(2, 'GPa'),
    rho=Q_(2, 'gram / cm^3'),
    calculation_method='Batzle and Wang',
    temp_gradient=Q_(0.03, 'degC/m'),
    # temp_ref=Q_(4.0, 'degC'),
    pressure_gradient=Q_(0.0107, 'MPa / m'),
    # pressure_ref=Q_(0, 'MPa'),
    salinity=Q_(70000.0, 'ppm'),
    gor=Q_(1., ''),
    oil_api=Q_(30., ''),
    gas_gravity=Q_(0.6, ''),
    gas_mixing='Brie',
    brie_exponent=Q_(2., ''),
    status=None
)


def read_all_fluids_from_excel(filename, fluid_sheet: str = 'Fluids', fluid_header: int = 1) -> dict:
    # First read in all fluids defined in the project table
    all_fluids = {}
    fluids_table = pd.read_excel(filename,
                                 sheet_name=fluid_sheet, header=fluid_header, engine='openpyxl')

    for i, name in enumerate(fluids_table['Name']):
        if isnan(name):
            continue  # Avoid empty lines
        this_fluid = Fluid(
            calculation_method='User specified' if isnan(fluids_table['Calculation method'][i]) else fluids_table['Calculation method'][i],
            k=Q_(float(fluids_table['Bulk moduli [GPa]'][i]), 'GPa'),
            mu=Q_(float(fluids_table['Shear moduli [GPa]'][i]), 'GPa'),
            rho=Q_(float(fluids_table['Density [g/cm3]'][i]), 'gram / cm^3'),
            temp_gradient=Q_(float(fluids_table['T gradient [deg C/m]'][i]), 'degC / m'),
            # temp_ref=Q_(float(fluids_table['T ref [C]'][i]), 'degC'),
            pressure_gradient=Q_(float(fluids_table['P gradient [MPa/m]'][i]), 'MPa / m'),
            # pressure_ref=Q_(float(fluids_table['P ref [MPa]'][i]), 'MPa'),
            salinity=Q_(float(fluids_table['Salinity [ppm]'][i]), 'ppm'),
            gor=Q_(float(fluids_table['GOR'][i]), ''),
            oil_api=Q_(float(fluids_table['Oil API'][i]), ''),
            gas_gravity=Q_(float(fluids_table['Gas gravity'][i]), ''),
            gas_mixing=fluids_table['Gas mixing'][i],
            brie_exponent=Q_(float(fluids_table['Brie exponent'][i]), ''),
            name=name.lower(),
            status='from excel'
        )
        all_fluids[name.lower()] = this_fluid

    return all_fluids

def read_fluidmixes_from_excel(filename,
                               fluid_sheet: str = 'Fluids', fluid_header: int = 1,
                               mix_sheet: str = 'Fluid mixtures', mix_header: int = 1):
    """

    :param filename:
    :param fluid_sheet:
    :param fluid_header:
    :param mix_sheet:
    :param mix_header:
    :return:
      Returns a fluid mixture dictionary
      {this_subst:                          # 1'st level: 'initial' or 'final'
            {this_well:                     # 2'nd level: well name
                {this_wi:                   # 3'd level: working interval name
                    {fluid_name: Fluid(),   # 4'th level: fluid object
                    fluid_type: str,       # 4'th level: Fluid type: Brine, Oil, Gas
                    volume_fraction: str | float,    # 4'th level: <Name of saturation log>, 'complement',
                                                     # or saturation as a float
                    tag: str,              # 4'th level: Tag that can be used when exporting logs
            }}}}
    """

    # First read in all fluids defined in the project table
    all_fluids = read_all_fluids_from_excel(filename, fluid_sheet, fluid_header)

    # Then read in the fluid mixes
    fluid_mixes = {
        'initial': {},
        'final': {}
    }
    mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header, engine='openpyxl')
    for i, name in enumerate(mix_table['Fluid name']):
        if mix_table['Use'][i] != 'Yes':
            continue
        vf = mix_table['Volume fraction'][i]
        ftype = mix_table['Fluid type'][i]
        # Need to pair fluid name with fluid type to get unique fluid names in fluid mixture
        this_name = '{}_{}'.format(name.lower(), '' if isnan(ftype) else ftype.lower())
        if isnan(vf):
            continue  # Avoid fluids where the volume fraction is not set
        this_subst = mix_table['Substitution order'][i].lower()
        this_well = mix_table['Well name'][i].upper()
        this_wi = mix_table['Interval name'][i].upper()
        this_fluid = deepcopy(all_fluids[name.lower()])
        this_fluid.name = this_name
        try:
            this_tag = mix_table['Tag'][i]
        except KeyError:
            this_tag = None

        # 2'nd level
        if this_well in list(fluid_mixes[this_subst].keys()):
            # 3'rd level
            if this_wi in list(fluid_mixes[this_subst][this_well].keys()):
                # 4'th level
                fluid_mixes[this_subst][this_well][this_wi][this_name] = this_fluid
                fluid_mixes[this_subst][this_well][this_wi]['fluid_type'] = ftype
                fluid_mixes[this_subst][this_well][this_wi]['volume_fraction'] = vf
                fluid_mixes[this_subst][this_well][this_wi]['tag'] = this_tag
            else:
                fluid_mixes[this_subst][this_well][this_wi] = {this_name: this_fluid,
                                                                'fluid_type': ftype,
                                                                'volume_fraction': vf,
                                                                'tag': this_tag}
        else:
            fluid_mixes[this_subst][this_well] = {this_wi: {this_name: this_fluid,
                                                             'fluid_type': ftype,
                                                             'volume_fraction': vf,
                                                             'tag': this_tag}}

    return fluid_mixes


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
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Fluid(object):
    def __init__(self,
                 calculation_method: str | None = None,  # or Batzle and Wang',  # or 'User specified'
                 k: Q_ | None = None,  # Bulk modulus in GPa
                 mu: Q_ | None = None,  # Shear modulus in GPa
                 rho: Q_ | None = None,  # Density in g/cm3
                 temp_gradient: Q_ | None = None,
                 pressure_gradient: Q_ | None = None,
                 salinity: Q_ | None = None,
                 gor: Q_ | None = None,
                 oil_api: Q_ | None = None,
                 gas_gravity: Q_ | None = None,
                 gas_mixing: str | None = None,
                 brie_exponent: Q_ | None = None,
                 name: str | None = 'Default',
                 status: str | None = None,
                 header: dict | None = None):

        if header is None:
            header = {}
        if name is not None:
            header['name'] = name
        elif 'name' not in list(header.keys()):
             header['name'] = None

        self.header = Header(header)
        self.calculation_method = calculation_method
        self._k = k
        self._mu = mu
        self._rho = rho
        self._bd = None  # Burial depth
        self.temp_gradient = temp_gradient
        self.pressure_gradient = pressure_gradient
        self.salinity = salinity
        self.gor = gor
        self.oil_api = oil_api
        self.gas_gravity = gas_gravity
        self.gas_mixing = gas_mixing
        self.brie_exponent = brie_exponent
        self._name = header['name']
        self.status = status

    def __str__(self):
        keys = list(self.__dict__.keys())

        pattern = "%%%ds: %%s" % len(keys)

        head = [ pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.header['name'] = value

    def print_fluid(self, verbose=False):
        out = '  {}\n'.format(self.name)
        if verbose:
            out = str(self)
        else:
            out += '      K: {}, Mu: {}, Rho {}\n'.format(
                self.k, self.mu, self.rho)
            out += '      Calculation method: {}\n'.format(self.calculation_method)
            out += '      Status: {}\n'.format(self.status)
        return out

    def keys(self):
        return self.__dict__.keys()

    @property
    def k(self):
        return self._k

    @property
    def mu(self):
        return self._mu

    @property
    def rho(self):
        return self._rho

    @property
    def bd(self):
        return self._bd

    def calc_elastics(self, fluid_type: str, burial_depth: Q_):
        if self.calculation_method == 'Batzle and Wang':
            _s = self.salinity.to('ppm').magnitude
            # _p = self.pressure_ref + self.pressure_gradient * burial_depth
            # We ignore the pressure reference now. Will be calculated more exactly in FluidMix
            _p = Q_(0.0, 'MPa') + self.pressure_gradient * burial_depth
            _p = _p.to('MPa').magnitude
            # _t = self.temp_ref + self.temp_gradient * burial_depth
            _t = Q_(4.0, 'degC') + self.temp_gradient * burial_depth
            _t = _t.to('degC').magnitude
            if fluid_type.lower() == 'brine':
                _this_rho = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                _this_k = v_p_b**2 * _this_rho * 1.E-6
            elif fluid_type.lower() == 'oil':
                _this_k, _this_rho = rp.k_and_rho_o(
                    self.oil_api.magnitude,
                    self.gas_gravity.magnitude,
                    self.gor.magnitude,
                    _p,
                    _t
                )
            elif fluid_type.lower() == 'gas':
                _this_k, _this_rho = rp.k_and_rho_g(self.gas_gravity.magnitude, _p, _t)
            else:
                raise NotImplementedError('Elastic properties not possible to calculate for {}'.format(fluid_type))
            self._bd = burial_depth
            self._k = Q_(_this_k, 'GPa')
            self._mu = Q_(np.nan, 'GPa')
            self._rho = Q_(_this_rho, 'GPa')


class FluidsTable:
    """
    Returns a table and bokeh CDS for the provided fluids
    """
    def __init__(self,
                 fluids: list | None = None,
                 width: int | None = None,
                 height: int | None = None):
        """

        :param fluids:
            list of Fluid objects
        :param width:
            int
        """
        if width is None:
            width = 700
        self.width = width
        if height is None:
                height = 135
        self.height = height
        if fluids is None:
            fluids = []
        self.fluids = fluids
        self.keys = ['name', 'k', 'mu', 'rho', 'calculation_method',
                     'temp_gradient', 'pressure_gradient',
                     'salinity', 'gor', 'oil_api', 'gas_gravity', 'gas_mixing', 'brie_exponent']
    @property
    def cds(self) -> ColumnDataSource:
        _dict = {_x:[] for _x in self.keys}
        for _fluid in self.fluids:
            for _key in self.keys:
                if _key == 'name':
                    _dict['name'].append(_fluid.header.name)
                elif _key in ['bd']:
                    _dict[_key].append(None)
                elif _key in ['k', 'mu', 'rho']:
                    _dict[_key].append(_fluid.__dict__['_{}'.format(_key)].magnitude)
                elif _key in ['gas_mixing', 'calculation_method']:
                    _dict[_key].append(_fluid.__dict__[_key])
                else:
                    _dict[_key].append(_fluid.__dict__[_key].magnitude)
        return ColumnDataSource(_dict)

    @property
    def fluid_names(self):
        _fluid_names = []
        for _fluid in self.fluids:
            _this_name = _fluid.name
            if _this_name not in _fluid_names:
                _fluid_names.append(_this_name)
        return _fluid_names

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        _calc_methods = ['Batzle and Wang', 'User specified']
        _fluid_types = ['-', 'Brine', 'Oil', 'Gas']
        table_columns = [
            TableColumn(field='name', title='Name'),
            # TableColumn(field='fluid_type', title='Fluid type',
            #             editor=SelectEditor(options=_fluid_types)),
            TableColumn(field='k', title='K [GPa]'),
            TableColumn(field='mu', title='G [GPa]'),
            TableColumn(field='rho', title='Den. [g/cm3]'),
            # TableColumn(field='bd', title='Burial depth [m]'),
            TableColumn(field='calculation_method', title='Calc. meth.',
                        editor=SelectEditor(options=_calc_methods)),
            # TableColumn(field='temp_ref', title='T. ref. [deg C]'),
            TableColumn(field='temp_gradient', title='T. grad. [deg C/m]'),
            # TableColumn(field='pressure_ref', title='P. ref. [MPa]'),
            TableColumn(field='pressure_gradient', title='P. grad. [MPa/m]'),
            TableColumn(field='salinity', title='Salinity [ppm]'),
            TableColumn(field='gor', title='GOR'),
            TableColumn(field='oil_api', title='Oil API'),
            TableColumn(field='gas_gravity', title='Gas gravity')
        ]
        return table_columns

    def draw(self,
             cds: ColumnDataSource):
        from bokeh.models import DataTable, Button, CheckboxGroup

        def add_row_function():
            new_data = dict(cds.data)
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append('default')
                else:
                    new_data[_key].append(def_fluid_vals[_key])
            cds.data = new_data

        def delete_row_function():
            selected_index = cds.selected.indices
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(cds.data['name'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        def update_table_function():
            new_data = dict(cds.data)
            _fluids = []
            for _i in range(len(new_data['name'])):
                # print(' - Iteration: {} of {}'.format( _i, len(new_data['name'])))
                this_fluid = Fluid(
                    calculation_method=new_data['calculation_method'][_i],
                    k=Q_(new_data['k'][_i], 'GPa'),
                    mu=Q_(new_data['mu'][_i], 'GPa'),
                    rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                    temp_gradient=Q_(new_data['temp_gradient'][_i], 'degC / meter'),
                    # temp_ref=Q_(new_data['temp_ref'][_i], 'degC'),
                    pressure_gradient=Q_(new_data['pressure_gradient'][_i], 'MPa / meter'),
                    # pressure_ref=Q_(new_data['pressure_ref'][_i], 'MPa'),
                    salinity=Q_(new_data['salinity'][_i], 'ppm'),
                    gor=Q_(new_data['gor'][_i], ''),
                    oil_api=Q_(new_data['oil_api'][_i], ''),
                    gas_gravity=Q_(new_data['gas_gravity'][_i], ''),
                    name=new_data['name'][_i])
                _fluids.append(this_fluid)

            self.fluids = _fluids

            print(self.fluid_names)
            cds.data = new_data

        dt = DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
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
                 fluid_mixes=None,
                 header=None):
        """
        :param fluid_mixes:
            dict
            Dictionary of fluid mixtures as returned from read_fluidmixes_from_excel()
        """
        if header is None:
            header = {}
        if name is not None:
            header['name'] = name
        elif 'name' not in list(header.keys()):
            header['name'] = None

        self.header = Header(header)

        if fluid_mixes is None:
            fluid_mixes = {}
        self.fluid_mixes = fluid_mixes

        self._name = header['name']

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.header['name'] = value


    def print_all_fluids(self, verbose=False):
        out = ''
        for key in ['initial', 'final']:
            for w in list(self.fluid_mixes[key].keys()):
                for wi in list(self.fluid_mixes[key][w].keys()):
                    out += self.print_fluids(key, w, wi, verbose)
        return out

    def print_fluids(self, subst, well_name, wi_name, verbose=False):
        out = 'Fluid mixture: {}, {}, {}, {}\n'.format(subst, well_name, wi_name, self.name)
        for m in list(self.fluid_mixes[subst][well_name][wi_name].keys()):
            if m in ['fluid_type', 'volume_fraction', 'tag']:
                out += '\n{}: {}'.format(m, self.fluid_mixes[subst][well_name][wi_name][m])
            else:
                out += self.fluid_mixes[subst][well_name][wi_name][m].print_fluid(verbose)
        return out + '\n'

    def read_excel(self, filename,
                   fluid_sheet: str = 'Fluids', fluid_header: int = 1,
                   mix_sheet: str = 'Fluid mixtures', mix_header: int = 1):

        self.fluid_mixes = read_fluidmixes_from_excel(filename, fluid_sheet, fluid_header, mix_sheet, mix_header)
        self.header['orig_file'] = filename

    def calc_elastics(self, wells: dict, wis: Intervals, debug=False):
        """
        Calculates k, mu, and rho for all fluids for each well and working interval they are defined in, and where the
        calculation method is not 'User specified'
        :param wells:
            dict
            {well name: core.wells.Well} key: value pairs
        :param wis:
            Intervals
            blixt_rp.core.core Interval object which contain multiple working intervals for multiple wells
            e.g. wis = Intervals()
        :param debug:
            bool
            if True, generate verbose information and create some plots
        :return:
        """
        # 1'st level: Initial and Final fluids
        for key in ['initial', 'final']:
            # 2'nd level: loop over all wells
            for w in list(self.fluid_mixes[key].keys()):
                if w not in list(wells.keys()):
                    warn_txt = 'Well {} not present among the input wells'.format(w)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # test if this will is listed in the working intervals
                if w not in wis.well_names():
                    warn_txt = 'Well {} not present among the working intervals'.format(w)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # Extract the measured and burial depth for this well
                bd = wells[w].get_burial_depth(block_name=block_name, templates=templates)
                md = wells[w].block[block_name].get_md()

                # 3'rd level: loop over all working intervals
                for wi in list(self.fluid_mixes[key][w].keys()):
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
                    for f in list(self.fluid_mixes[key][w][wi].keys()):
                        this_fluid = self.fluid_mixes[key][w][wi][f]
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
    def test_read_fluid_mixes(self):
        # project_table = 'C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx'
        project_table = 'C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx'
        my_fluid_mixes = read_fluidmixes_from_excel(project_table)
        fmix = FluidMix(fluid_mixes=my_fluid_mixes)
        print(fmix.print_all_fluids())


    def test_data(self):
        import os

        from blixt_rp.core.well import Project
        wp = Project(name='MyProject')

        # myfluids = FluidMix()
        # myfluids.read_excel(wp.project_table)
        myfluids = read_fluidmixes_from_excel(wp.project_table)
        all_fluids = read_all_fluids_from_excel(wp.project_table)
        return all_fluids, myfluids

    def test_fluid(self):
        default = def_fluid_vals
        f1 = Fluid(**default)
        print(f1)
        for ft in ['Brine', 'Oil', 'Gas']:
            f1.calc_elastics(ft, Q_(1000., 'm'))
            print('{}: '.format(ft), f1.k, f1.mu, f1.rho)


    def test_fluid_table(self):
        from bokeh.io import output_file
        from bokeh.plotting import show, row, column
        # output_file('C:\\Users\\emb\\Downloads\\plot.html')
        output_file('C:\\Users\\marte\\Downloads\\plot.html')

        all_fluids, fluid_mix = self.test_data()
        ft = FluidsTable(fluids=list(all_fluids.values()))
        cds = ft.cds
        # for _key in list(cds.data.keys()):
        #     print(_key, cds.data[_key])
        table, add_row, delete_row, update = ft.draw(cds)
        show(column(table, row(add_row, delete_row, update)))
        # return table, add_row, delete_row, update

    def test_fluidsub(self):
        import matplotlib.pyplot as plt
        from blixt_rp.core.core import LogTable, Cutoffs, CutoffRule
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.log_curve_new import Depth, LogCurve
        from blixt_rp.plotting.cross_plotter import CrossPlotter

        file_dir = str(os.path.dirname(__file__).replace(
            'blixt_rp\\core',
            ''))
        project_table = os.path.join(file_dir, 'excels\\project_table_new.xlsx')

        las_file =  os.path.join(file_dir, 'test_data/Well A.las')
        output_file('C:\\Users\\emb\\Downloads\\plot.html')

        logs = {'vp_dry': 'P velocity', 'vp_so08': 'P velocity', 'vp_sg08': 'P velocity',
                'vs_dry': 'S velocity', 'vs_so08': 'S velocity', 'vs_sg08': 'S velocity',
                'rho_dry': 'Density', 'rho_so08': 'Density', 'rho_sg08': 'Density',
                'phie': 'Porosity', 'vcl': 'Volume', 'dept': 'MD'}
        lt = LogTable()
        lt.from_invert(logs)

        w1 = Well()
        w1.read_las(las_file, log_table=lt, template_file=project_table)

        # Calculate and apply this mask:
        # w.calc_mask({'vcl': ['<', 0.4], 'phie': ['>', 0.1]}, name='sand')  # calc_mask only works for the old Well
        cr1 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cr2 = CutoffRule('phie', '>', Q_(0.1, ''))
        ct = Cutoffs([cr1, cr2])

        w_dict = w1.dict(cutoffs=ct, use_cutoffs=True)
        # w_dict = w1.dict(cutoffs=ct, use_cutoffs=False)  # Now it creates and an extra parameter 'mask'

        test = 'constants'  #'array'
        # Test with constant Vsh and constant Sw
        v_sh = 0.2
        s_w = 0.2
        if test == 'array':  # test with arrays of v_sh and s_w
            # v_sh = w.block['Logs'].logs['vcl'].values[mask]
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

        w2 = Well()
        w2.name = 'Fluid Sub'
        v_p_2, v_s_2, rho_2, k_2 = rp.gassmann_vel(
            w_dict['vp_dry'].magnitude,
            w_dict['vs_dry'].magnitude,
            w_dict['rho_dry'].magnitude,
            k_f1, rho_f1, k_f2, rho_f2, k0,
            w_dict['phie'].magnitude)
        # Add fluid subst. logs to well
        depth = Depth(w_dict['dept'])
        lc_vp = LogCurve('vp_hc', Q_(v_p_2, 'm/s'), depth, log_type='P velocity', style=deepcopy(w1.get_log_curve('vp_dry').style))
        w2.add_log(lc_vp)
        lc_vs = LogCurve('vs_hc', Q_(v_s_2, 'm/s'), depth, log_type='S velocity', style=deepcopy(w1.get_log_curve('vs_dry').style))
        w2.add_log(lc_vs)
        lc_rho = LogCurve('rho_hc', Q_(rho_2, 'g/cm**3'), depth, log_type='Density', style=deepcopy(w1.get_log_curve('rho_dry').style))
        w2.add_log(lc_rho)
        lc_depth = LogCurve('dept', Q_(depth.magnitude, depth.units), depth, log_type='MD')
        w2.add_log(lc_depth)
        print(w2.get_log_names)


        xp = CrossPlotter({w.name: w.data_source() for w in [w1, w2]})
        cds = xp.cds
        xp.show_plot(cds, out_file='C:/Users/emb/Downloads/plot.html')

