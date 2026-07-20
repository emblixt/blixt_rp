# -*- coding: utf-8 -*-
"""
Created on Sun 19th July 2026
Module for handling rockphysics fluid and mineral models

#
# The coding are inspired from CoPilot suggestions
#

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
from typing import Literal, List, Any
from pathlib import Path
from dataclasses import dataclass

from bokeh.io import output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import column, row
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp import Q_
import blixt_rp.rp.rp_core as rp
from blixt_rp.core.well_new import Well
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info
from blixt_rp.core.core import Interval, Intervals


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeFraction:
    """
    A VolumeFraction object reflects how the volume fraction is defined in a single row of the 'FluidMixture' sheet
    of the project_table Excel file.
    For project-table definitions like fluids, minerals and substitutions, "frozen=True" is generally better because these
    objects represent configuration rather than something that changes during calculations.
    """
    mode: Literal["constant", "log", "complement"]
    value: float | str | None
    excel_value: float | str | None

@dataclass()
class FluidSpec:
    """
    Represents one row of the "Fluids" sheet in the project_table Excel file, formerly known as the "mother fluid"
    """
    name: str
    calculation_method: Literal["batzle and wang", "user specified"]
    bulk_gpa: float | None = None
    shear_gpa: float | None = 0.0
    density_gcc: float | None = None
    params: dict[str, Any] | None = None
    bd: Q_ | None = None

    def __str__(self, indent=4):
        """
        Return better readable string representation of the FluidSpec object.
        """
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        out = ''
        indent = ' ' * indent
        for _key, _value in zip(keys, values):
            out += '{}{}: {}\n'.format(indent, _key, _value)
        return out

    def calc_elastics(self, fluid_type: str, burial_depth: Q_):
        if self.calculation_method.lower() == 'batzle and wang':
            _this_density = None
            _this_bulk = None
            _s = self.params['Salinity [ppm]']
            _p = self.params['P ref [MPa]'] + self.params['P gradient [MPa/m]'] * burial_depth.to('meter').magnitude
            _t = self.params['T ref [C]'] + self.params['T gradient [deg C/m]'] * burial_depth.to('meter').magnitude
            if fluid_type.lower() == 'brine':
                _this_density = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                _this_bulk = v_p_b**2 * _this_density * 1.E-6
                calc_success = True
            elif fluid_type.lower() == 'oil':
                _this_bulk, _this_density = rp.k_and_rho_o(
                    self.params['Oil API'],
                    self.params['Gas gravity'],
                    self.params['GOR'],
                    _p,
                    _t
                )
                _this_bulk = _this_bulk.value
                _this_density = _this_density.value
                calc_success = True
            elif fluid_type.lower() == 'gas':
                _this_bulk, _this_density = rp.k_and_rho_g(self.params['Gas gravity'], _p, _t)
                _this_bulk = _this_bulk.value
                _this_density = _this_density.value
                calc_success = True
            else:
                calc_success = False
                print_info('No calculation done for fluid type {}'.format(fluid_type),
                           'warning', logger)
            if calc_success:
                self.bd = burial_depth
                self.bulk_gpa = _this_bulk
                self.shear_gpa = 0.0
                self.density_gcc = _this_density


@dataclass(frozen=True)
class MixtureComponent:
    """
    One object per row in the 'FluidMixture' sheet of the project_table Excel file.
    A combination of MixtureComponent's are used to specify one SubstitutionCase, and represents one row in the
    'FluidMixture' sheet of the project_table Excel file.
    "frozen=True" is generally better because this object shouldn't change during calculation
    """
    mother_fluid_name: str
    fluid_type: Literal["gas", "oil", "brine", "user specified"]
    volume: VolumeFraction
    fluid: FluidSpec

    def __str__(self, indent=4):
        """
        Return better readable string representation of the FluidSpec object.
        """
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        out = ''
        indent = ' ' * indent
        for _key, _value in zip(keys, values):
            out += '{}{}: {}\n'.format(indent, _key, _value)
        return out

@dataclass()
class SubstitutionCase:
    """
    One substitution case object per well, interval and tag
    """
    well: str
    interval: str
    tag: str
    initial: List[MixtureComponent]
    final: List[MixtureComponent]
    initial_bulk_gpa: float | None = None
    initial_shear_gpa: float | None = 0.0
    initial_density_gcc: float | None = None
    final_bulk_gpa: float | None = None
    final_shear_gpa: float | None = 0.0
    final_density_gcc: float | None = None

    def calc_elastics(self, burial_depth: Q_, logs: dict[str, np.ndarray], verbose: bool = False):
        """
        Calculates the elastic properties of the Voigt-Reuss-Hill averaged fluid for the initial and final fluids
        for this specific SubstitutionCase

        :param burial_depth:
            pint.Quantity
            The burial depth representative for this SubstitutionCase
        :param logs:
            dict
            Dictionary with log values that represents the volume fraction for a fluid (e.g. SW = water saturation)

        :return:
        """
        if len(self.initial) == 0 or len(self.final) == 0:
            raise IOError('Need a list of initial and final MixtureComponents to calculate the elastics')
        n = None
        last_n = 0
        for i, _key in enumerate(list(logs.keys())):
            if i == 0:
                last_n = len(logs[_key])
            if last_n != len(logs[_key]):
                raise IOError('Each log must have the same length ({}). Not {}'.format(last_n, len(logs[_key])))
            last_n = len(logs[_key])
        n = last_n

        # Start with the initial MixtureComponents
        vol_fractions = resolve_volume_fractions(self.initial, logs, n)
        _f = []
        _bulk = []
        _shear = []
        _density = []
        for _x in self.initial:
            _key = f'{_x.mother_fluid_name}:User specified' if _x.fluid_type == 'User specified' else f'{_x.mother_fluid_name}:{_x.fluid_type}'
            _f.append(vol_fractions[_key])
            _x.fluid.calc_elastics(_x.fluid_type, burial_depth)
            _bulk.append([_x.fluid.bulk_gpa] * n)
            _shear.append([_x.fluid.shear_gpa] * n)
            _density.append([_x.fluid.density_gcc] * n)
        # Bulk modulus
        self.initial_bulk_gpa = rp.vrh_bounds(_f, _bulk)[2]
        # Shear modulus
        self.initial_shear_gpa = rp.vrh_bounds(_f, _shear)[2]
        # Density
        self.initial_density_gcc = rp.vrh_bounds(_f, _density)[0]
        if verbose:
            print('initial_bulk_gpa: {}'.format(self.initial_bulk_gpa))
            print('initial_shear_gpa: {}'.format(self.initial_shear_gpa))
            print('initial_density_gcc: {}'.format(self.initial_density_gcc))

        # Then take the final fluids
        vol_fractions = resolve_volume_fractions(self.final, logs, n)
        _f = []
        _bulk = []
        _shear = []
        _density = []
        for _x in self.final:
            _key = f'{_x.mother_fluid_name}:User specified' if _x.fluid_type == 'User specified' else f'{_x.mother_fluid_name}:{_x.fluid_type}'
            _f.append(vol_fractions[_key])
            _x.fluid.calc_elastics(_x.fluid_type, burial_depth)
            _bulk.append([_x.fluid.bulk_gpa] * n)
            _shear.append([_x.fluid.shear_gpa] * n)
            _density.append([_x.fluid.density_gcc] * n)
        # Bulk modulus
        self.final_bulk_gpa = rp.vrh_bounds(_f, _bulk)[2]
        # Shear modulus
        self.final_shear_gpa = rp.vrh_bounds(_f, _shear)[2]
        # Density
        self.final_density_gcc = rp.vrh_bounds(_f, _density)[0]
        if verbose:
            print('final_bulk_gpa: {}'.format(self.final_bulk_gpa))
            print('final_shear_gpa: {}'.format(self.final_shear_gpa))
            print('final_density_gcc: {}'.format(self.final_density_gcc))


def read_sheet_table(xlsx: str | Path, sheet_name: str, required_columns: set[str]) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name=sheet_name, header=None, engine='openpyxl')
    header_row = None
    for i, row in raw.iterrows():
        vals = {str(v).strip() for v in row.dropna().tolist()}
        if required_columns.issubset(vals):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f'Could not find header row in sheet {sheet_name!r}')
    df = pd.read_excel(xlsx, sheet_name=sheet_name, header=header_row, engine='openpyxl')
    df = df.dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_str(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def parse_volume_fraction(x: Any) -> VolumeFraction:
    s = clean_str(x)
    if s is None:
        raise ValueError('Missing Volume fraction')
    if s.lower() == 'complement':
        return VolumeFraction('complement', None, x)
    # Accept comma decimal input from Excel/users, e.g. '0,2'
    try:
        val = float(s.replace(',', '.'))
        if not 0 <= val <= 1:
            raise ValueError(f'Constant volume fraction outside [0, 1]: {x!r}')
        return VolumeFraction('constant', val, x)
    except ValueError:
        # Not numeric: interpret as well-log name, e.g. SW
        return VolumeFraction('log', s, x)

def as_float_or_none(x: Any) -> float | None:
    if pd.isna(x):
        return None
    return float(x)

def load_fluid_specs(xlsx: str | Path) -> dict[str, FluidSpec]:
    fluids = read_sheet_table(xlsx, 'Fluids', {'Name', 'Calculation method'})
    fluids = fluids[fluids['Name'].notna()].copy()
    if fluids['Name'].duplicated().any():
        dups = fluids.loc[fluids['Name'].duplicated(), 'Name'].tolist()
        raise ValueError(f'Duplicate fluid names in Fluids sheet: {dups}')
    out = {}
    for _, r in fluids.iterrows():
        name = clean_str(r['Name'])
        method = clean_str(r['Calculation method']) or ''
        if method.lower() not in ['batzle and wang', 'user specified']:
            raise IOError('Fluid calculation method must be either "Batzle and Wang" or "User specified". Not: {}'.format(
                method
            ))
        params = {c: None if pd.isna(r[c]) else r[c] for c in fluids.columns if c not in {'Name'}}
        shear = as_float_or_none(r.get('Shear moduli [GPa]'))
        # For pore fluids, missing shear modulus is normally treated as zero.
        if shear is None and method.lower() == 'user specified':
            shear = 0.0
        out[name] = FluidSpec(
            name=name,
            calculation_method=method.lower(),
            bulk_gpa=as_float_or_none(r.get('Bulk moduli [GPa]')),
            shear_gpa=shear,
            density_gcc=as_float_or_none(r.get('Density [g/cm3]')),
            params=params,
        )
    return out

def load_mineral_specs(xlsx: str | Path) -> dict[str, FluidSpec]:
    minerals = read_sheet_table(xlsx, 'Minerals', {'Name', 'Calculation method'})
    minerals = minerals[minerals['Name'].notna()].copy()
    if minerals['Name'].duplicated().any():
        dups = minerals.loc[minerals['Name'].duplicated(), 'Name'].tolist()
        raise ValueError(f'Duplicate fluid names in Fluids sheet: {dups}')
    out = {}
    for _, r in minerals.iterrows():
        name = clean_str(r['Name'])
        method = clean_str(r['Calculation method']) or ''
        if method.lower() not in ['interval average', '']:
            raise IOError('Mineral calculation method must be either "Interval average" or empty. Not: {}'.format(
                method
            ))
        params = {c: None if pd.isna(r[c]) else r[c] for c in minerals.columns if c not in {'Name'}}

        # out[name] = MineralSpec(
        #     name=name,
        #     calculation_method=method.lower(),
        #     bulk_gpa=as_float_or_none(r.get('Bulk moduli [GPa]')),
        #     shear_gpa=as_float_or_none(r.get('Shear moduli [GPa]')),
        #     density_gcc=as_float_or_none(r.get('Density [g/cm3]')),
        #     params=params,
        # )
        print(name, method.lower(), params)
    return out


def component_from_row(r: pd.Series, fluids: dict[str, FluidSpec]) -> MixtureComponent:
    fluid_name = clean_str(r['Fluid name'])
    if fluid_name not in fluids:
        raise ValueError(f'Fluid mixture references unknown fluid {fluid_name!r}')
    return MixtureComponent(
        mother_fluid_name=fluid_name,
        fluid_type=clean_str(r['Fluid type']) or '',
        volume=parse_volume_fraction(r['Volume fraction']),
        fluid=fluids[fluid_name],
    )

def validate_component_set(components: list[MixtureComponent], label: str) -> None:
    n_comp = sum(c.volume.mode == 'complement' for c in components)
    # for c in components:
    #     print('-x-')
    #     print(c.volume.mode)
    #     print(c.volume.value)
    if n_comp > 1:
        raise ValueError(f'{label}: more than one complement fraction')
    const_sum = sum(c.volume.value for c in components if c.volume.mode == 'constant')
    if const_sum > 1 + 1e-9:
        raise ValueError(f'{label}: constant fractions sum to {const_sum:.3f} > 1')
    if n_comp == 0 and all(c.volume.mode == 'constant' for c in components):
        if abs(const_sum - 1) > 1e-6:
            raise ValueError(f'{label}: constant fractions sum to {const_sum:.3f}, expected 1')

def load_substitution_cases(xlsx: str | Path) -> list[SubstitutionCase]:
    """
    By including "Well name", "Interval name", "Tag" in the groupby method of a panda table we can run several
    substitution scenarios for the same well and interval
    :param xlsx:
    :return:
    """
    fluids = load_fluid_specs(xlsx)
    mix = read_sheet_table(xlsx, 'Fluid mixtures', {'Use', 'Substitution order', 'Well name', 'Interval name', 'Fluid name', 'Fluid type', 'Volume fraction'})
    mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
    mix['Substitution order'] = mix['Substitution order'].map(lambda x: (clean_str(x) or '').title())
    bad_orders = sorted(set(mix['Substitution order']) - {'Initial', 'Final'})
    if bad_orders:
        raise ValueError(f'Unsupported substitution order values: {bad_orders}')
    for col in ['Well name', 'Interval name']:
        if mix[col].isna().any():
            raise ValueError(f'Missing {col} in enabled Fluid mixtures rows')
    if 'Tag' not in mix.columns:
        mix['Tag'] = None
    cases: list[SubstitutionCase] = []
    group_cols = ['Well name', 'Interval name', 'Tag']
    for key, g in mix.groupby(group_cols, dropna=False, sort=False):
        well, interval, tag = key
        well, interval, tag = clean_str(well), clean_str(interval), clean_str(tag)
        initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
        final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
        if not initial or not final:
            raise ValueError(f'{well}/{interval}/{tag}: need both Initial and Final mixture rows')
        validate_component_set(initial, f'{well}/{interval}/{tag}/Initial')
        validate_component_set(final, f'{well}/{interval}/{tag}/Final')
        cases.append(SubstitutionCase(well=well, interval=interval, tag=tag, initial=initial, final=final))
    return cases

def resolve_volume_fractions(components: list[MixtureComponent], logs: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """
    Calculates the volume fractions of each of the fluids listed in components

    :param components:
        list
        List of MixtureComponent's
    :param logs:
        dict
        Dictionary with log values that represents the volume fraction for a fluid (e.g. SW = water saturation)
    :param n:
        int
        Length of the input log and/or output
    :return:
        dict
        Dictionary with the volume fraction for each fluid in the SubstitionCase which is made up from the input list
        of MixtureComponent's
    """
    known_sum = np.zeros(n, dtype=float)
    out: dict[str, np.ndarray] = {}
    complement_name = None
    for c in components:
        key = f'{c.mother_fluid_name}:User specified' if c.fluid_type == 'User specified' else f'{c.mother_fluid_name}:{c.fluid_type}'
        if c.volume.mode == 'constant':
            arr = np.full(n, float(c.volume.value))
            out[key] = arr
            known_sum += arr
        elif c.volume.mode == 'log':
            log_name = str(c.volume.value)
            if log_name not in logs:
                raise KeyError(f'Missing log curve {log_name!r}')
            arr = np.asarray(logs[log_name], dtype=float)
            if arr.shape[0] != n:
                raise ValueError(f'Log {log_name!r} length {arr.shape[0]} != expected {n}')
            out[key] = arr
            known_sum += arr
        else:
            complement_name = key
    if complement_name is not None:
        out[complement_name] = 1.0 - known_sum
    if np.nanmin(np.vstack(list(out.values()))) < -1e-6:
        raise ValueError('One or more resolved volume fractions are negative')
    total = np.sum(np.vstack(list(out.values())), axis=0)
    if not np.allclose(total, 1.0, atol=1e-6, equal_nan=True):
        raise ValueError(f'Resolved volume fractions do not sum to 1; range={np.nanmin(total):.3f}-{np.nanmax(total):.3f}')
    return out

class TestCases(unittest.TestCase):
    def test_print_fluid_spec(self):
        fs = FluidSpec(
            name='AnyThing',
            calculation_method='user specified',
            bulk_gpa=30,
            shear_gpa=16,
            density_gcc=3
        )
        print(fs)

    def test_load_fluid_spec(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        for _key, _value in list(load_fluid_specs(project_table).items()):
            for _fluid in ['brine', 'oil', 'gas']:
                _value.calc_elastics(_fluid, Q_(3000., 'm'))
                print(_key, _fluid , _value)

    def test_read_sheet_table(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        print(mix)


    def test_component_from_row(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        fluids = load_fluid_specs(project_table)
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name', 'Tag']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
            final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
            print('Initial:\n', '\n'.join([str(_i) for _i in initial]))
            print('Final:\n', '\n'.join([str(_f) for _f in final]))

    def test_validate_component_set(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        fluids = load_fluid_specs(project_table)
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name', 'Tag']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
            final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
            print('Initial is valid?:\n', validate_component_set(initial, 'TEST'))
            print('Final is valid?:\n', validate_component_set(final, 'TEST FINAL'))


    def test_load_substitution_cases(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        cases = load_substitution_cases(project_table)
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            print('  Initial:', [(x.mother_fluid_name, x.fluid_type, x.volume.mode, x.volume.value) for x in c.initial])
            print('  Final:  ', [(x.mother_fluid_name, x.fluid_type, x.volume.mode, x.volume.value) for x in c.final])

    def test_resolve_volume_fraction(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        cases = load_substitution_cases(project_table)
        n = 10
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            _out = resolve_volume_fractions(c.initial, {'SW': np.full(n, 0.33)}, n)
            print('  Initial: ', _out)
            _out = resolve_volume_fractions(c.final, {'SW': np.full(n, 0.33)}, n)
            print('  Final: ', _out)

    def test_calc_elastics_for_substitution_cases(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        cases = load_substitution_cases(project_table)
        n = 10
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            c.calc_elastics(Q_(3000., 'meter'), {'SW': np.full(n, 0.4)}, verbose=True)

    def test_calc_elastics_for_substitution_cases_detailed(self):
        fluid_params = {'Bulk moduli [GPa]': None, 'Shear moduli [GPa]': None, 'Density [g/cm3]': None,
                        'Calculation method': 'Batzle and Wang',
                        'T gradient [deg C/m]': 0.03, 'T ref [C]': 5.0,
                        'P gradient [MPa/m]': 0.0107, 'P ref [MPa]': 0.0,
                        'Salinity [ppm]': 70000.0, 'GOR': 1.0, 'Oil API': 30.0,
                        'Gas gravity': 0.6, 'Gas mixing': 'Brie', 'Brie exponent': 2.0}
        fs = FluidSpec(
            name='MyFluid',
            calculation_method='batzle and wang',
            params=fluid_params
        )
        mc1 = MixtureComponent(
            mother_fluid_name='MyFluid',
            fluid_type='brine',
            volume=VolumeFraction(
                mode='constant',
                value=1,
                excel_value=1
            ),
            fluid=fs
        )
        mc2 = MixtureComponent(
            mother_fluid_name='MyFluid',
            fluid_type='brine',
            volume=VolumeFraction(
                mode='log',
                value='SW',
                excel_value='SW'
            ),
            fluid=fs
        )
        mc3 = MixtureComponent(
            mother_fluid_name='MyFluid',
            fluid_type='gas',
            volume=VolumeFraction(
                mode='complement',
                value=None,
                excel_value='complement'
            ),
            fluid=fs
        )
        sc = SubstitutionCase(
            well='Well 1',
            interval='Cook FM',
            tag='GAS',
            initial=[mc1],
            final=[mc2, mc3]
        )

        fs.calc_elastics('brine', Q_(2000., 'm'))
        brine_bulk = fs.bulk_gpa
        brine_density = fs.density_gcc

        fs.calc_elastics('gas', Q_(2000., 'm'))
        gas_bulk = fs.bulk_gpa
        gas_density = fs.density_gcc

        # The water saturation goes from 0 to 1
        sc.calc_elastics(Q_(2000., 'm'), {'SW': np.linspace(0., 1., 10)}, verbose=False)

        # print(brine_bulk, brine_density)
        # print(gas_bulk, gas_density)
        # print(sc.final_bulk_gpa)
        # print(sc.final_density_gcc)

        # The last sample of the VRH averaged value of the final bulk should equal the brine bulk moduli
        self.assertAlmostEqual(sc.final_bulk_gpa[-1], brine_bulk)

        # The last sample of the VRH averaged value of the final density should equal the brine density
        self.assertAlmostEqual(sc.final_density_gcc[-1], brine_density)

        # The first sample of the VRH averaged value of the final bulk should equal the gas bulk moduli
        self.assertAlmostEqual(sc.final_bulk_gpa[0], gas_bulk)

        # The first sample of the VRH averaged value of the final density should equal the gas density
        self.assertAlmostEqual(sc.final_density_gcc[0], gas_density)

