# -*- coding: utf-8 -*-
"""
Original Created on Tue Oct  1 15:25:40 2019
New version started at Thu Apr 30 08:30 2026
    Implement Pint Quantities to handle units

@author: mblixt
"""
import unittest

import numpy as np
import logging
import sys
import os
from typing import Callable

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\rp', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))

import blixt_rp.rp_utils.definitions as ud
import blixt_rp.rp.rp_core as rp
from blixt_rp import Q_
from blixt_rp.core.param import Param
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.utils import print_info

logger = logging.getLogger(__name__)

# Default rock physics template constants
_rpt_kwargs = {
    'vsh': Q_(0.2, ''),         # Volume of shale, fraction
    'phi_c': Q_(0.4, ''),       # critical porosity, fraction
    'c_n': 8,                   # Coordination number in the contact cement model
    'p_conf': Q_(45, 'MPa'),    # Confining pressure in MPa (effective pressure?)
    'smcf': 0.3,                # shear modulus correction factor  (1=dry pack with perfect adhesion,
                                #   0=dry frictionless pack)
    'rho_hc': Q_(0.2, 'G/cc'),  # HC density  g/cm3
    'k_hc': Q_(0.06, 'GPa'),    # HC bulk modulus  GPa
    'rho_b': Q_(1.02, 'G/cc'),  # Brine density  g/cm3
    'k_b': Q_(2.8, 'GPa'),      # Brine bulk modulus  GPa
    'rho_qz': Q_(2.6, 'G/cc'),  # Density [g/cm3] of quartz
    'k_qz': Q_(37, 'GPa'),      # Bulk modulus GPa of quartz
    'mu_qz': Q_(45, 'GPa'),     # Shear modulus of quartz GPa
    'rho_sh': Q_(2.6, 'G/cc'),  # Density [g/cm3] of shale
    'k_sh': Q_(20, 'GPa'),      # Bulk modulus GPa of shale
    'mu_sh': Q_(10, 'GPa'),     # Shear modulus of shale GPa
    'apc': Q_(10, 'percent'),   # absolut percentage cement in percent
    'ntg': Q_(1., ''),          # Net to Gross, use the shale properties above in the non-Net
    'scheme': 2                 # scheme for calculating cement deposition
                                #   1=cement deposited at grain contacts
                                #   2=uniform layer around grains (default)
}

def rpt_wrapper(
        t: Q_,
        rpt: Callable,
        constants: list,
        rpt_keywords: dict
):
    """

    From plt_rpt() in plot_rp.py

    Utility function that takes a RPT (rock physics template) as input, evaluates it along one independent variable
    (t, eg. Porosity) for a specific set of rpt constants (constants, eg. list of water saturation values), to produce
    a set of Vp, Vs, Rho values (one set per item in constants) which can be drawn in a plot to illustrate the RPT

    :param t:
        Pint Quantity with an np.array of length N
        t values used to draw the rockphysics template, preferably less than about 10 items long for creating
        nice plots
    :param rpt:
        function
        Rock physics template function of t
        Should take a second argument which is used to parameterize the function
        e.g.
        def rpt(t, c, **rpt_keywords):
            return c*t + rpt_keywords.pop('zero_crossing', 0)
    :param constants:
        list
        list of length M of constants used to parametrize the rpt function
    :param rpt_keywords:
        dict
        Dictionary with keywords passed on to the rpt function

    :return:
        Vp, Vs, Rho
        Each is a list of Pint Quantity arrays. Each item in the list corresponds to a constant in the 'constants' input
    """

def rpt_phi_sw(phi: Q_, sw: Q_, **kwargs):
    """
    Useful rock physics template that can return elastic properties Vp, Vs & Rho for many
    different rock physics functions (constant cement, contact cement, stiff sand, soft sand)
    from the input variables phi and sw (porosity and water saturation)

    """
    model = kwargs.pop('model', 'stiff')
    phi_c = kwargs.pop('phi_c', _rpt_kwargs['phi_c'])     # critical porosity
    p_conf = kwargs.pop('p_conf', _rpt_kwargs['p_conf'])  # Confining pressure in MPa (effective pressure?)
    smcf = kwargs.pop('smcf', _rpt_kwargs['smcf'])        # shear modulus correction factor  (1=dry pack with perfect adhesion, 0=dry frictionless pack)
    rho_b = kwargs.pop('rho_b', _rpt_kwargs['rho_b'])     # Brine density  g/cm3
    k_b = kwargs.pop('k_b', _rpt_kwargs['k_b'])           # Brine bulk modulus  GPa
    rho_hc = kwargs.pop('rho_hc', _rpt_kwargs['rho_hc'])  # HC density
    k_hc = kwargs.pop('k_hc', _rpt_kwargs['k_hc'])        # HC bulk modulus
    rho_min = kwargs.pop('rho_min', None)                 # Density [g/cm3] of mineral mix
    k_min = kwargs.pop('k_min', None)                     # Bulk modulus GPa of mineral mix
    mu_min = kwargs.pop('mu_min', None)                   # Shear modulus GPa of mineral mix
    rho_qz = kwargs.pop('rho_qz', _rpt_kwargs['rho_qz'])  # Density [g/cm3] of quartz
    k_qz = kwargs.pop('k_qz', _rpt_kwargs['k_qz'])        # Bulk modulus GPa of quartz
    mu_qz = kwargs.pop('mu_qz', _rpt_kwargs['mu_qz'])     # Shear modulus of quartz
    vsh = kwargs.pop('vsh', _rpt_kwargs['vsh'])           # Volume fraction of shale
    rho_sh = kwargs.pop('rho_sh', _rpt_kwargs['rho_sh'])  # Density [g/cm3] of shale
    k_sh = kwargs.pop('k_sh', _rpt_kwargs['k_sh'])        # Bulk modulus GPa of shale
    mu_sh = kwargs.pop('mu_sh', _rpt_kwargs['mu_sh'])     # Shear modulus of shale
    ntg = kwargs.pop('ntg', _rpt_kwargs['ntg'])           # Net to Gross, use the shale properties above in the non-Net
    c_n = kwargs.pop('c_n', _rpt_kwargs['c_n'])           # Coordination number in the contact cement model
    k_cem = kwargs.pop('k_cem', _rpt_kwargs['k_qz'])      # Cement bulk modulus in GPa (quartz)
    mu_cem = kwargs.pop('mu_cem', _rpt_kwargs['mu_qz'])   # Cement shear modulus in GPa (quartz)
    scheme = kwargs.pop('scheme', _rpt_kwargs['scheme'])  # scheme for calculating cement deposition
    apc = kwargs.pop('apc', _rpt_kwargs['apc'])           # absolute percent cement (in %)
    calc_phi_eff = kwargs.pop('calc_phi_eff', False)      # set this to True to calculate effective porosity and use it

    # rp_core is not handling Pint Quantities correctly, so we need to take their magnitude
    # TODO DO THIS FOR ALL VARIABLES
    _phi = phi.to('dimensionless').magnitude
    _sw = sw.to('dimensionless').magnitude
    _vsh = vsh.to('dimensionless').magnitude
    _ntg = ntg.to('dimensionless').magnitude
    _phi_c = phi_c.to('dimensionless').magnitude
    _p_conf = p_conf.to('MPa').magnitude
    _k_b= k_b.to('GPa').magnitude
    _rho_b = rho_b.to('G/cc').magnitude
    _k_hc= k_hc.to('GPa').magnitude
    _rho_hc = rho_hc.to('G/cc').magnitude
    _k_qz = k_qz.to('GPa').magnitude
    _mu_qz = mu_qz.to('GPa').magnitude
    _rho_qz = rho_qz.to('G/cc').magnitude
    _k_cem = k_cem.to('GPa').magnitude
    _mu_cem = mu_cem.to('GPa').magnitude
    _k_sh = k_sh.to('GPa').magnitude
    _mu_sh = mu_sh.to('GPa').magnitude
    _rho_sh = rho_sh.to('G/cc').magnitude
    _apc = apc.to('percent').magnitude

    # Check if we need to calculate the average mineral properties, or if they are given
    if k_min is None:
        _k_min = rp.vrh_bounds( [_vsh, 1 - _vsh], [_k_sh, _k_qz] )[2]  # Mineral bulk modulus
    else:
        _k_min = k_min.to('GPa').magnitude

    if mu_min is None:
        _mu_min = rp.vrh_bounds( [_vsh, 1 - _vsh], [_mu_sh, _mu_qz] )[2]  # Mineral shear modulus
    else:
        _mu_min = mu_min.to('GPa').magnitude

    if rho_min is None:
        _rho_min = rp.vrh_bounds( [_vsh, 1 - _vsh], [_rho_sh, _rho_qz] )[0]  # Density of minerals
    else:
        _rho_min = rho_min.to('G/cc').magnitude

    if calc_phi_eff:
        #phi = _phi - vsh * ((rho_min - rho_sh) / (rho_min - rho_b))  # eq. 2 in https://link.springer.com/article/10.1007/s13202-020-01073-2#Fig8
        _phi = _phi * (1. - _vsh)  # eq. 1.4 in https://www.informit.com/articles/article.aspx?p=1626870&seqNum=2
    else:
        _phi = 1. * _phi

    # Apply the rock physics model to modify the mineral properties
    if model == 'stiff':
        _k_dry, _mu_dry = rp.stiffsand(
            _k_min,
            _mu_min,
            _phi,
            _phi_c,
            c_n,
            _p_conf,
            smcf)
    elif model == 'contactcement':
        _k_dry, _mu_dry = rp.contactcement(
            _k_min,
            _mu_min,
            _phi,
            _phi_c,
            c_n,
            _k_cem,
            _mu_cem,
            scheme)
    elif model == 'constantcement':
        _k_dry, _mu_dry = rp.constantcement(
            _k_min,
            _mu_min,
            _phi,
            _phi_c,
            _apc,
            c_n,
            _k_cem,
            _mu_cem,
            scheme,
            smcf=smcf)
    else:
        model = 'soft'
        _k_dry, _mu_dry = rp.softsand(
            _k_min,
            _mu_min,
            _phi,
            _phi_c,
            c_n,
            _p_conf,
            smcf)
    info_txt = 'Using {} rock physics template'.format(model)
    print_info(info_txt, 'info', logger)

    # Calculate the final fluid properties for the given water saturation
    print(sw, k_b, k_hc)
    _k_f2 = rp.vrh_bounds([_sw, 1.-_sw], [_k_b, _k_hc])[1]  # K_f
    _rho_f2 = rp.vrh_bounds([_sw, 1.-_sw],  [_rho_b, _rho_hc])[0]  #RHO_f

    # Use Gassman to calculate the final elastic properties
    _vp_2, _vs_2, _rho_2, _k_2 = rp.vels(_k_dry, _mu_dry, _k_min, _rho_min, _k_f2, _rho_f2, _phi)
    # # Give them units
    # vp_2 = Q_(vp_2, 'km/s')
    # vs_2 = Q_(vs_2, 'km/s')
    # rho_2 = Q_(rho_2, 'G/cc')
    # k_2 = Q_(k_2, 'GPa')

    # if ntg is not 1, the final values are modified with the harmonic mean of the Net to Gross properties
    if ntg != 1:
        _vp_2 = rp.vrh_bounds([_ntg, 1.-_ntg], [_vp_2, 1000. * rp.v_p(_k_sh, _mu_sh, _rho_sh)])[1]
        _vs_2 = rp.vrh_bounds([_ntg, 1.-_ntg], [_vs_2, 1000. * rp.v_s(_mu_sh, _rho_sh)])[1]
        _rho_2 = rp.vrh_bounds([_ntg, 1.-_ntg], [_rho_2, _rho_sh])[1]
        _k_2 = rp.vrh_bounds([_ntg, 1.-_ntg], [_k_2, _k_sh])[1]

    return _vp_2, _vs_2, _rho_2


def return_rpt_keywords(prev_rpt_kwargs=None, modify_keys=None, verbose=False):
    """
    Returns a default set of rock physics template keywords if prev_rpt_kwargs is set to None,
    The keywords can be modified using the dict modify_keys
    """
    if prev_rpt_kwargs is None:
        rpt_kwargs = _rpt_kwargs
    else:
        rpt_kwargs = prev_rpt_kwargs

    # modify selected keyword values
    if modify_keys is not None and isinstance(modify_keys, dict):
        for _key, _value in modify_keys.items():
            rpt_kwargs[_key] = _value

    # Determine mineral properties
    rpt_kwargs['k_min'] = Q_(
        rp.vrh_bounds(
            [rpt_kwargs['vsh'].to('dimensionless').magnitude, 1 - rpt_kwargs['vsh'].to('dimensionless').magnitude],
            [rpt_kwargs['k_sh'].to('GPa').magnitude, rpt_kwargs['k_qz'].to('GPa').magnitude]
        )[2], 'GPa')  # Mineral bulk modulus
    if verbose:
        print('k_min: {}'.format(rpt_kwargs['k_min']))
    rpt_kwargs['mu_min'] = Q_(
        rp.vrh_bounds(
            [rpt_kwargs['vsh'].to('dimensionless').magnitude, 1 - rpt_kwargs['vsh'].to('dimensionless').magnitude],
            [rpt_kwargs['mu_sh'].to('GPa').magnitude, rpt_kwargs['mu_qz'].to('GPa').magnitude]
        )[2], 'GPa')  # Mineral shear modulus
    if verbose:
        print('mu_min: {}'.format(rpt_kwargs['mu_min']))
    rpt_kwargs['rho_min'] = Q_(
        rp.vrh_bounds(
            [rpt_kwargs['vsh'].to('dimensionless').magnitude, 1 - rpt_kwargs['vsh'].to('dimensionless').magnitude],
            [rpt_kwargs['rho_sh'].to('G/cc').magnitude, rpt_kwargs['rho_qz'].to('G/cc').magnitude]
        )[0], 'G/cc')  # Density of minerals

    return rpt_kwargs

class TestCases(unittest.TestCase):
    def test_rpt_phi_sw(self):
        phi = Q_(0.2, ''); sw = Q_(0.1, '')
        print(rpt_phi_sw(phi, sw, **return_rpt_keywords()))
        phi = Q_(np.linspace(0.05, 0.3, 9), '')
        print(rpt_phi_sw(phi, sw, **return_rpt_keywords()))
        sw = Q_(np.linspace(0.1, 0.9, 9 ), '')
        print(rpt_phi_sw(phi, sw, **return_rpt_keywords()))

