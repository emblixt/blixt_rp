# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:25:40 2019

@author: mblixt
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
# from dataclasses import dataclass
from copy import deepcopy

import bruges.rockphysics.rockphysicsmodels as brr

import blixt_rp.rp_utils.definitions as ud
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_rp.core.param import Param

logger = logging.getLogger(__name__)


def test_value(val, unit):
    """
    Test the input value val if it is a Param instance, and if not, create one with unit unit
    :param val:
        Input parameter to be tested
    :param unit:
        str
    :return:
        Param
        the input parameter val, or a newly created Param object with unit unit
    """
    try:
        test = val.value
    except:
        val = Param(name='DEFAULT', value=val, unit=unit, desc='')
       # warn_str = 'Input parameter needs to be a Param instance with units. Default unit {} is now used'.format(
       #     unit
       #)
       #logger.warning(warn_str)

    if (val.unit != unit):
        raise NotImplementedError('Unit conversion not yet implemented')

    return val


def step(x_1, x_2, along_wiggle=False):
    """
    Returns the averaged difference from layer 1 (top) to layer 2 (bottom)
    of variable x.
    E.G. Delta X / X_avg according to eq. 4.6 in Avseth et. al 2011

    For the 'along_wiggle' case we calculate the difference in acoustic impedence along  one track without any layer,
    and the '_2' input variables are not used
    """
    if along_wiggle and isinstance(x_1, np.ndarray):
        return 2.*(x_1[1:] - x_1[:-1]) / (x_1[1:] + x_1[:-1])
    else:
        return 2 * (x_2 - x_1) / (x_2 + x_1)


def sqrd_avg(x):
    """
    Common term occuring in the reflection coefficient, see eq. 4.6 in Avseth et. al 2011
    :param x:
    :return:
    """
    return 0.25 * (x[1:] + x[:-1])**2


def poissons_ratio(vp, vs):
    return 0.5 * (vp ** 2 - 2 * vs ** 2) / (vp ** 2 - vs ** 2)


def intercept(vp_1, vp_2, rho_1, rho_2, along_wiggle=None):
    """
    Returns the AVO intercept, or normal incidence reflectivity
    From eq. 2 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.

    For the 'along_wiggle' case we calculate the difference in acoustic impedence along  one track without any layer,
    and the '_2' input variables are not used
    """
    if along_wiggle is None: along_wiggle = False
    if along_wiggle and isinstance(vp_1, np.ndarray) and isinstance(rho_1, np.ndarray):
        return (vp_1[1:]*rho_1[1:] - vp_1[:-1]*rho_1[:-1])/(vp_1[1:]*rho_1[1:] + vp_1[:-1]*rho_1[:-1])
    else:
        return 0.5 * (step(vp_1, vp_2) + step(rho_1, rho_2))


def gradient(vp_1, vp_2, vs_1, vs_2, rho_1, rho_2, along_wiggle=None):
    """
    Returns the AVO gradient
    From eq. 3 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.

    For the 'along_wiggle' case we calculate the difference in acoustic impedence along  one track without any layer,
    and the '_2' input variables are not used
    """
    if along_wiggle is None: along_wiggle = False
    if along_wiggle and isinstance(vp_1, np.ndarray) and isinstance(vs_1, np.ndarray) and isinstance(rho_1, np.ndarray):
        out = 0.5 * step(vp_1, None,  along_wiggle=along_wiggle) - 2.*(sqrd_avg(vs_1)/sqrd_avg(vp_1)) * \
            (step(rho_1, None, along_wiggle=along_wiggle) + 2.*step(vs_1, None, along_wiggle=along_wiggle))
        return out

    else:
        return 0.5 * step(vp_1, vp_2) - 2. * ((vs_1 + vs_2) / (vp_1 + vp_2)) ** 2 * \
               (2. * step(vs_1, vs_2) + step(rho_1, rho_2))


def reflectivity(vp_1, vp_2, vs_1, vs_2, rho_1, rho_2, version='WigginsAkiRich', eei=False, along_wiggle=None):
    """
    returns function which returns the reflectivity as function of theta
    theta is in degrees
    When 'along_wiggle' is True, the return array will be one item shorter than the input because of how step(),
    intercept() and gradient() works.
    To alleviate this, we add an extra item at the end of the returned array
    :param version:
        str
        'WigginsAkiRich' uses the formulation of Aki Richards by Wiggins et al. 1984 according to Hampson Russell
        'ShueyAkiRich' uses Shuey 1985's formulation of Aki Richards according to Avseth 2011 (eq. 4.7)
        'AkiRich' Aki Richards formulation according to Avseth 2011 (eq. 4.6)
    :param eei:
        bool
        If true the function returns the reflectivity as a function of EEI rotation angle chi (-90 - 90 deg) instead
        it follows the equation described in:
        "Application of extended elastic impedance in seismic geomechanics"
        Javad Sharifi, Naser Hafezi Moghaddas, Gholam Reza Lashkaripour, Abdolrahim Javaherian, and Marzieh Mirzakhanian
        GEOPHYSICS 2019 84:3, R429-R446
    """
    def func(_):
        raise NotImplementedError
    if along_wiggle is None: along_wiggle = False
    if version is None: version = 'WigginsAkiRich'
    if (version == 'WigginsAkiRich') or (version == 'ShueyAkiRich') or eei:
        a = intercept(vp_1, vp_2, rho_1, rho_2, along_wiggle=along_wiggle)
        b = gradient(vp_1, vp_2, vs_1, vs_2, rho_1, rho_2, along_wiggle=along_wiggle)
        c = 0.5 * step(vp_1, vp_2, along_wiggle=along_wiggle)
        if eei:
            def func(chi):
                result = a * np.cos(chi * np.pi / 180.) + b * np.sin(chi * np.pi / 180.)
                if along_wiggle:
                    result = np.append(result, np.ones(1) * result[-1])
                return result
            return func

        if version == 'WigginsAkiRich':
            def func(theta):
                result = a + b * (np.sin(theta * np.pi / 180.)) ** 2 + \
                       c * ((np.tan(theta * np.pi / 180.)) ** 2 * (np.sin(theta * np.pi / 180.)) ** 2)
                if along_wiggle:
                    result = np.append(result, np.ones(1) * result[-1])
                return result

        elif version == 'ShueyAkiRich':
            def func(theta):
                result = a + b * (np.sin(theta * np.pi / 180.)) ** 2 + \
                       c * ((np.tan(theta * np.pi / 180.)) ** 2 - (np.sin(theta * np.pi / 180.)) ** 2)
                if along_wiggle:
                    result = np.append(result, np.ones(1) * result[-1])
                return result
        return func
    elif version == 'AkiRich':
        a = (vs_1 + vs_2) ** 2 / vp_1 ** 2  # = 4 * p**2 * vs**2/sin(theta)**2 in eq. 4.6

        def func(theta):
            result = 0.5 * (1. - a * (np.sin(theta * np.pi / 180.)) ** 2) * step(rho_1, rho_2) + \
                   0.5 * step(vp_1, vp_2) / (np.cos(theta * np.pi / 180.)) ** 2 - \
                   a * (np.sin(theta * np.pi / 180.)) ** 2 * step(vs_1, vs_2)
            if along_wiggle:
                result = np.append(result, np.ones(1) * result[-1])
            return result
        return func
    else:
        raise NotImplementedError


def eei(vp, vs, rho, vp0=None, vs0=None, rho0=None, k=None):
    """
    Calculates the Extended Elastic Impedance according to eq. 19 in Whitcombe et al. 2002
    Returns EEI as a function of chi, where chi is an angle in degrees

    :param vp, vs & rho
        float or float arrays of same length
    :param k:
        float
        Constant in the EEI equation, typically set to the average of (vs/vp)**2
    """
    if k is None:
        k = 0.25
    if vp0 is None:
        vp0 = np.nanmean(vp)
    if vs0 is None:
        vs0 = np.nanmean(vs)
    if rho0 is None:
        rho0 = np.nanmean(rho)

    def func(chi):
        cos_chi = np.cos(chi * np.pi / 180.)
        sin_chi = np.sin(chi * np.pi / 180.)
        p = cos_chi + sin_chi
        q = - 8. * k * sin_chi
        r = cos_chi - 4. * sin_chi
        return vp0 * rho0 * (
            np.power(vp / vp0, p) *
            np.power(vs / vs0, q) *
            np.power(rho / rho0, r)
        )
    return func


def v_p(k, mu, rho):
    return np.sqrt(
        (k + 4. * mu / 3.) / rho)


def v_s(mu, rho):
    return np.sqrt(mu / rho)


def k_from_v(v_p, v_s, rho):
    return rho * (v_p**2 - 4. * v_s**2 / 3.)


def mu_from_v(v_s, rho):
    return rho * v_s**2


def han_castagna(v_p, method=None):
    """
    Calculates Vs based on the Han (1986) and Castagna et al (1993) regressions.

    :param v_p:
        Param
        P velocity [m/s]
    :param method
        str
        'Han' or 'Castagna' or 'mudrock'
        default is 'Han'
    :return:
        Param
        Estimated shear velocity
    """
    v_p = test_value(v_p, 'm/s')

    if method == 'Castagna':
        v_s = 0.8042*v_p.value - 855.9
    elif method == 'mudrock':
        v_s = 0.8621*v_p.value - 1172.4
    else:
        v_s = 0.7936*v_p.value - 786.8

    return Param(
        name='v_s',
        value=v_s,
        unit='m/s',
        desc='Shear velocity'
    )


def greenberg_castagna(v_p, f, mono_mins, gc_coeffs=None):
    """
    Estimates Vs in a L-component multimineral brine saturated rock based on p. 246 in Rock physics handbook (Mavko et al. 1999)
    which is based on Greenberg & Castagna (1992).

    :param v_p:
        Param
        P velocity [m/s]
        float, or ndarray of floats
    :param f:
        list
        list of Volumetric fractions for each monomineralic constituent
        The items in f could either be all constants, or all arrays of same length (as vp)
        sum(f) must equal one
    :param mono_mins:
        list
        List of names of each constituent in the multimineral rock
        Length of mono_mins and f must be the same

    :return:
        vs
    """
    if gc_coeffs is None:
        gc_coeffs = {
            'sandstone': [0., 0.80416, -0.85588],
            'limestone': [-0.05508, 1.01677, -1.03049],
            'dolomite': [0., 0.58321, -0.07775],
            'shale': [0., 0.76969, -0.86735]
        }

    # make sure names of each mineral is in lowercase
    for key in list(gc_coeffs.keys()):
        gc_coeffs[key.lower()] = gc_coeffs.pop(key)

    v_p = test_value(v_p, 'm/s')

    def calc_vs(_vp, _coeffs):
        # convert to km/s
        _vp = _vp/1000.
        return _coeffs[0]*_vp**2 + _coeffs[1]*_vp + _coeffs[2]

    m = [calc_vs(v_p.value, gc_coeffs[this_mono.lower()]) for this_mono in mono_mins]

    return Param(
        name='v_s',
        value=vrh_bounds(f, m)[2]*1000.,
        unit='m/s',
        desc='Vs estimated from Greenberg Castagna')


def k_and_rho_o(oil_gravity, gas_gravity, g_o_ratio, p, t):
    """
    Calculates the bulk modulus and density of oil using Batzle & Wang 1992.

    :param oil_gravity:
        Param
        API gravity, measures how heavy or light a petroleum liquid is compared to water:
        API > 10, it is lighter and floats on water;
        API < 10, it is heavier and sinks.
    :param gas_gravity:
        Param
        Gas gravity is the molar mass of the gas divided by the molar mass of air
        It ranges from 0.55 for dry sweet gas to approximately 1.5 for wet, sour gas.
        Default value is 0.65
    :param g_o_ratio:
        Param
        The gas/oil ratio (GOR) is the ratio of the volume of gas that comes out of solution,
        to the volume of oil at standard conditions.
    :param p:
        Param
        Hydrostatic pore pressure [MPa]
    :param t:
        Param
        Temperature [deg C]
    :return:
     Oil bulk modulus [GPa], Oil density [g/cm3]
    """
    oil_gravity = test_value(oil_gravity, 'API')
    gas_gravity = test_value(gas_gravity, '')
    g_o_ratio = test_value(g_o_ratio, '')
    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    ov = oil_gravity.value
    gv = gas_gravity.value
    gorv = g_o_ratio.value
    pv = p.value
    tv = t.value

    rho0 = 141.5/(ov + 131.5)

    if gorv < 0.01:  # Dead oil
        rop = rho0 + (0.00277 * pv - 1.71e-7 * pv**3) * (rho0 - 1.15)**2 + 3.49e-4 * pv
        rho_o = rop / (0.972 + 3.81e-4 * (tv + 17.78)**1.175)
    else:
        B0 = 0.972 + 0.00038 * (2.4 * gorv * np.sqrt(gv / rho0) + tv + 17.8)**1.175
        roog = (rho0 + 0.0012 * gv * gorv) / B0
        rho_o = roog + (0.00277*pv - 1.71e-7*pv**3) * (roog - 1.15)**2 + 3.49e-4*pv
        rho0 = rho0/(B0*(1.0 + 0.001*gorv))

    v_p_o = 2096*np.sqrt(rho0/(2.6-rho0))-3.7*tv+4.64*pv+0.0115*(4.12*np.sqrt(1.08/rho0-1)-1)*tv*pv
    v_p_o = v_p_o/1000
    k_o = v_p_o*v_p_o*rho_o

    return Param(name='k_o',
                 value=k_o,
                 unit='GPa',
                 desc='Oil bulk modulus'), \
           Param(name='rho_o',
                 value=rho_o,
                 unit='g/cm3',
                 desc='Oil density')


def k_and_rho_g(gas_gravity, p, t):
    """
    Calculates the bulk modulus and density of gas using Batzle & Wang 1992.

    :param gas_gravity:
        Param
        Gas gravity is the molar mass of the gas divided by the molar mass of air
        It ranges from 0.55 for dry sweet gas to approximately 1.5 for wet, sour gas.
        Default value is 0.65
    :param p:
        Param
        Hydrostatic pore pressure [MPa]
    :param t:
        Param
        Temperature [deg C]
    :return:
     Gas bulk modulus [GPa], Gas density [g/cm3]
    """
    gas_gravity = test_value(gas_gravity, '')
    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    gv = gas_gravity.value
    pv = p.value
    tv = t.value

    r0 = 8.31441  # Ideal gas constant

    # Gas density
    Pr = pv/(4.892 - 0.4048*gv)
    Tr = (tv + 273.15)/(94.72 + 170.75*gv)
    E = 0.109*(3.85 - Tr)**2 * np.exp(-1.0*(0.45+(8.*(0.56 - 1./Tr)**2))*(Pr**1.2/Tr))
    Z = (0.03+0.00527*(3.5-Tr)**3)*Pr + (0.642*Tr-0.007*Tr**4-0.52) + E
    rho_g = 28.8*gv*pv/(Z*r0*(tv+273.15))

    # Gas bulk modulus
    gamma = 0.85 + 5.6/(Pr+2) + 27.1/(Pr+3.5)**2 - 8.7*np.exp(-0.65*(Pr+1))
    f = E*1.2*(-(0.45+8.*(0.56-1./Tr)**2)*Pr**0.2/Tr)+(0.03+0.00527*(3.5-Tr)**3)
    k_g = pv*gamma/(1-Pr/Z*f)/1000.

    return Param(name='k_g',
                value=k_g,
                unit='GPa',
                desc='Gas bulk modulus'), \
          Param(name='rho_g',
                value=rho_g,
                unit='g/cm3',
                desc='Gas density')


def rho_w(p, t):
    """
    Calculates the density of water as a function of pressure and temperature according to eq. 27a in Batzle & Wang 1992.

    :param p:
        Param
        pressure in MPa
    :param t:
        Param
        temperature in deg C
    :return:
        Param
        Density in g/cm3
    """

    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    tv = t.value
    pv = p.value

    _rho_w = 1. + 10**(-6) * (-80.*tv - 3.3*tv**2 + 0.00175*tv**3 + 489.*pv - 2.*tv*pv + 0.016*pv*tv**2 - (1.3*10**(-5))*pv*tv**3 -
                              0.333*pv**2 - 0.002*tv*pv**2)
    return Param(
        name='rho_w',
        value=_rho_w,
        unit='g/cm3',
        desc='Water density'
    )


def rho_b(s, p, t):
    """
    Calculates the density of brine as a function of salinity, pressure and temperature according to eq. 27b
    in Batzle & Wang 1992.

    :param s:
        Param
        Salinity in ppm
    :param p:
        Param
        pressure in MPa
    :param t:
        Param
        temperature in deg C
    :return:
        Param
        Density in g/cm3
    """

    s = test_value(s, 'ppm')
    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    if (p.unit != 'MPa') or (t.unit != 'C') or (s.unit != 'ppm'):
        raise NotImplementedError('Unit conversion not yet implemented')

    sv = s.value/1E6
    tv = t.value
    pv = p.value

    _rho_b = rho_w(p, t).value + sv * ( 0.668+0.44*sv + 1E-6*(300.*pv - 2400.*pv*sv + tv*(
            80. + 3.*tv - 3300.*sv - 13.*pv + 47.*pv*sv)) )

    return Param(
        name='rho_b',
        value=_rho_b,
        unit='g/cm3',
        desc='Brine density'
    )


def v_p_w(p, t):
    """
    Calculates the velocity in water as a function of pressure and temperature according to eq. 28 in Batzle & Wang 1992
    :param p:
        Param
        pressure in MPa
    :param t:
        Param
        temperature in deg C
    :return:
        Param
        Velocity in m/s
    """

    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    # Matrix for water properties calculation according to Table 1 in Batzle & Wang 1992
    mwp = [
        [1402.850, 1.524, 3.437e-3, -1.197e-5],
        [4.871, -0.0111, 1.739e-4, -1.628e-6],
        [-0.047830, 2.747e-4, -2.135e-6, 1.237e-8],
        [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10 ],
        [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13]
    ]

    pv = p.value
    tv = t.value

    _v_p_w = sum(
        [mwp[i][j] * tv**i * pv**j for i in range(5) for j in range(4)]
    )

    return Param(
        name='v_p_w',
        value=_v_p_w,
        unit='m/s',
        desc='Sound velocity in water'
    )


def v_p_b(s, p, t, giib=None):
    """
    Calculates the velocity in brine as a function of salinity, pressure and temperature according to eq. 29 & 31 in
    Batzle & Wang 1992.

    :param s:
        Param
        Salinity in ppm
    :param p:
        Param
        pressure in MPa
    :param t:
        Param
        temperature in deg C
    :param giib:
        float
        Gas index in brine
    :return:
        Param
        Velocity in m/s
    """
    if giib is None: giib = 0.

    s = test_value(s, 'ppm')
    p = test_value(p, 'MPa')
    t = test_value(t, 'C')

    if (p.unit != 'MPa') or (t.unit != 'C') or (s.unit != 'ppm'):
        raise NotImplementedError('Unit conversion not yet implemented')

    sv = s.value/1E6
    tv = t.value
    pv = p.value

    # Gas water ratio
    gwrmax = 10.**(np.log10(0.712 * pv * (abs(tv - 76.71))**1.5 + 3676. * pv**0.64) - 4 -
                   7.786 * sv * (tv+17.78)**(-0.306))
    gwr = gwrmax * giib

    vpb0 = v_p_w(p, t).value + sv*(1170. - 9.6*tv + 0.055*tv**2 - 8.5e-5*tv**3 + 2.6*pv -
                                   0.0029*tv*pv-0.0476*pv**2) + sv**1.5*(
            780. - 10.*pv + 0.16*pv**2) - 1820.*sv**2
    _v_p_b = vpb0 / (np.sqrt(1. + 0.0494*gwr))

    return Param(
        name='v_p_b',
        value=_v_p_b,
        unit='m/s',
        desc='Sound velocity in brine'
    )


def vrh_bounds(f, m):
    """
    Simple Voigt-Reuss-Hill bounds for N-components mixture

    :param f:
        list
        list of Volumetric fractions for each fluid/mineral
        The items in f could either be all constants, or all arrays of same length
        sum(f) must equal one
    :param m:
        list
        List of Elastic modulus (K or mu) of each fluid/mineral
    :return:
        list
        list of float's or list of arrays
        [M_Voigt,  # upper bound or Voigt average
         M_Reuss,  # lower bound or Reuss average
         M_VRH]  # Voigt-Reuss-Hill average
    """
    if (not isinstance(f, list)) or (not isinstance(m, list)):
        raise IOError('Input must be given as lists')

    # lists must be of equal length
    if len(f) != len(m):
        raise IOError('Input lists must be of equal length')

    # Test if input volumetric fractions are constants or arrays
    if isinstance(f[0], float):  # testing the list of floats
        if abs(sum(f) - 1.0) > 0.02:
            raise IOError('Sum of volume fractions must equal one 1')
    else:  # testing the list of arrays
        if (abs(sum(f) - 1.0) > 0.02).any():
            raise IOError('Sum of volume fractions must equal one 1')

    #M_Voigt = f * m1 + (1. - f) * m2
    M_Voigt = sum([x*y for x, y in zip(f, m)])
    #M_Reuss = 1. / (f / m1 + (1. - f) / m2)
    M_Reuss = 1. / sum([x/y for x, y in zip(f, m)])
    M_VRH = (M_Voigt + M_Reuss) / 2.
    return [M_Voigt, M_Reuss, M_VRH]


def por_from_mass_balance(rho, rho_m, rho_f):
    """
    Calculates porosity from mass balance equation.
    :param rho:
        Saturated rock density
    :param rho_m:
        Density of rock matrix
    :param rho_f:
        Fluid density
    :return:
        Porosity
    """
    return (rho_m - rho)/(rho_m - rho_f)


def critpor(k0, g0, phi, phi_c=None):
    '''
    Critical porosity, Nur et al. (1991, 1995)
    written by aadm (2015) from Rock Physics Handbook, p.353
    Eg. 7.1.2 and 7.1.3 in Rock Physics Handbook, 3rd ed. 2020

    INPUT
    k0, g0: mineral bulk & shear modulus in GPa
    phi: porosity
    phi_c: critical porosity (default 0.4)

    :returns
        dry rock bulk and shear moduli in GPa
    '''
    if phi_c is None: phi_c = 0.4
    k_dry = k0 * (1 - phi / phi_c)
    g_dry = g0 * (1 - phi / phi_c)
    return k_dry, g_dry


def hertz_mindlin(k0, mu0, phi_c=None, c_n=None, p_conf=None, smcf=None):
    """
    Hertz-Mindlin model
    The elastic moduli of a dry well-sorted end member
    Rock Physics Handbook 3rd ed. 2020, p.344
    Eq. 2.3 and 2.4 in Avseth 2011

    :param k0:
        Param
        Mineral bulk modulus in GPa
    :param mu0:
        Param
        Mineral shear modulus in GPa
    :param phi_c:
        float
        critical porosity
    :param c_n:
        float
       coordination number (default 8.6), average number of contacts per grain
    :param p_conf:
        Param
        Confining pressure, default 10/1E3 GPa
    :param smcf:
        float
        shear modulus correction factor
        1=dry pack with perfect adhesion
        0=dry frictionless pack
    :returns
        Effective bulk and shear modulus in GPa
    """
    if phi_c is None: phi_c = 0.4
    if c_n is None: c_n = 8.6
    if smcf is None: smcf = 1
    if p_conf is None:
        p_conf = Param(
            name='Pressure',
            value=10./1e3,
            unit='GPa',
            desc='Confining pressure'
        )
    k0 = test_value(k0, 'GPa')
    mu0 = test_value(mu0, 'GPa')
    p_conf = test_value(p_conf, 'GPa')

    k0 = k0.value
    mu0 = mu0.value
    p_conf = p_conf.value

    pr0 = (3*k0-2*mu0)/(6*k0+2*mu0)  # poisson's ratio of mineral mixture
    k_HM = (p_conf * (c_n ** 2 * (1 - phi_c) ** 2 * mu0 ** 2) / (18 * np.pi ** 2 * (1 - pr0) ** 2)) ** (1 / 3)
    mu_HM = ((2+3*smcf-pr0*(1+3*smcf))/(5*(2-pr0))) * (p_conf * (3 * c_n ** 2 * (1 - phi_c) ** 2 * mu0 ** 2) / (2 * np.pi ** 2 * (1 - pr0) ** 2)) ** (1 / 3)
    return k_HM, mu_HM


def softsand(k0, g0, phi, phi_c=None, c_n=None, p_conf=None, smcf=None):
    '''
    Soft-sand (uncemented) model
    or
    Friable sand
    written by aadm (2015) from Rock Physics Handbook, p.258
    page 359 in Rock Physics Handbook (2020)

    INPUT
    k0, g0: mineral bulk & shear modulus in GPa
    phi: porosity
    phi_c: critical porosity (default 0.4)
    c_n: coordination number (default 8.6)
    p_conf: confining pressure in MPa (default 10)
    smcf: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack

    :returns
        Effective bulk and shear modulus in GPa
    '''
    if phi_c is None: phi_c = 0.4
    if c_n is None: c_n = 8.6
    if p_conf is None: p_conf = 10
    if smcf is None: smcf = 1
    k_hm, g_hm = hertz_mindlin(k0, g0, phi_c, c_n, p_conf / 1.E3, smcf)
    k_eff = -4 / 3 * g_hm + (((phi / phi_c) / (k_hm + 4 / 3 * g_hm)) + ((1 - phi / phi_c) / (k0 + 4 / 3 * g_hm))) ** -1
    tmp = g_hm / 6 * ((9 * k_hm + 8 * g_hm) / (k_hm + 2 * g_hm))
    g_eff = -tmp + ((phi / phi_c) / (g_hm + tmp) + ((1 - phi / phi_c) / (g0 + tmp))) ** -1
    return k_eff, g_eff


def stiffsand(k0, g0, phi, phi_c=None, c_n=None, p_conf=None, smcf=None):
    '''
    Stiff-sand model
    written by aadm (2015) from Rock Physics Handbook, p.260
    page 361 in Rock Physics Handbook (2020)

    INPUT
    k0, g0: mineral bulk & shear modulus in GPa
    phi: porosity
    phi_c: critical porosity (default 0.4)
    c_n: coordination number (default 8.6)
    p_conf: confining pressure in MPa (default 10)
    smcf: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack

    :returns
        Effective bulk and shear modulus in GPa
    '''
    if phi_c is None: phi_c = 0.4
    if c_n is None: c_n = 8.6
    if p_conf is None: p_conf = 10
    if smcf is None: smcf = 1
    k_hm, g_hm = hertz_mindlin(k0, g0, phi_c, c_n, p_conf / 1.E3, smcf)
    print('K_HM: {}, G_HM: {}'.format(k_hm, g_hm))
    k_eff = -4 / 3 * g0 + (((phi / phi_c) / (k_hm + 4 / 3 * g0)) + ((1 - phi / phi_c) / (k0 + 4 / 3 * g0))) ** -1
    tmp = g0 / 6 * ((9 * k0 + 8 * g0) / (k0 + 2 * g0))
    g_eff = -tmp + ((phi / phi_c) / (g_hm + tmp) + ((1 - phi / phi_c) / (g0 + tmp))) ** -1
    return k_eff, g_eff


def contactcement(k0, g0, phi, phi_c=None, c_n=None, k_c=None, g_c=None, scheme=None):
    '''
    Contact cement (cemented sand) model, Dvorkin-Nur (1996)
    written by aadm (2015) from Rock Physics Handbook, p.354

    INPUT
    k0, g0: mineral bulk & shear modulus in GPa
    phi: porosity
    phi_c: critical porosity (default 0.4)
    c_n: coordination nnumber (default 8.6)
    k_c, g_c: cement bulk & shear modulus in GPa
            (default 37, 45 i.e. quartz)
    scheme: 1=cement deposited at grain contacts
            2=uniform layer around grains (default)

    :returns
        Effective bulk and shear modulus in GPa
    '''
    if phi_c is None: phi_c = 0.4
    if c_n is None: c_n = 8.6
    if k_c is None: k_c = 37
    if g_c is None: g_c = 45  # eq. 5.5.109

    if scheme is None: scheme = 2

    p_r0 = (3 * k0 - 2 * g0) / (6 * k0 + 2 * g0)
    p_rc = (3 * k_c - 2 * g_c) / (6 * k_c + 2 * g_c)
    if scheme == 1:  # scheme 1: cement deposited at grain contacts
        alpha = ((phi_c - phi) / (3 * c_n * (1 - phi_c))) ** (1 / 4)
    else:  # scheme 2: cement evenly deposited on grain surface
        alpha = ((2 * (phi_c - phi)) / (3 * (1 - phi_c))) ** (1 / 2)
    lambda_n = (2 * g_c * (1 - p_r0) * (1 - p_rc)) / (np.pi * g0 * (1 - 2 * p_rc))
    n1 = -0.024153 * lambda_n ** -1.3646
    n2 = 0.20405 * lambda_n ** -0.89008
    n3 = 0.00024649 * lambda_n ** -1.9864
    s_n = n1 * alpha ** 2 + n2 * alpha + n3
    lambda_t = g_c / (np.pi * g0)
    t1 = -10 ** -2 * (2.26 * p_r0 ** 2 + 2.07 * p_r0 + 2.3) * lambda_t ** (0.079 * p_r0 ** 2 + 0.1754 * p_r0 - 1.342)
    t2 = (0.0573 * p_r0 ** 2 + 0.0937 * p_r0 + 0.202) * lambda_t ** (0.0274 * p_r0 ** 2 + 0.0529 * p_r0 - 0.8765)
    t3 = 10 ** -4 * (9.654 * p_r0 ** 2 + 4.945 * p_r0 + 3.1) * lambda_t ** (0.01867 * p_r0 ** 2 + 0.4011 * p_r0 - 1.8186)
    s_t = t1 * alpha ** 2 + t2 * alpha + t3
    m_c = k_c + (4 / 3) * g_c  # rho_cement * v_p_cement**2, eq. 5.5.108
    k_eff = 1 / 6 * c_n * (1 - phi_c) * m_c * s_n  # eq. 5.5.106
    #k_eff = 1 / 6 * c_n * (1 - phi_c) * (k_c + (4 / 3) * g_c) * s_n
    g_eff = 3 / 5 * k_eff + 3 / 20 * c_n * (1 - phi_c) * g_c * s_t  # eq. 5.5.107
    return k_eff, g_eff


def constantcement(k0, g0, phi, phi_c=None, apc=None, c_n=None, k_c=None, g_c=None, scheme=None, smcf=None):
    """
    Constant cement model, Avseth et al. 2000 as implemented by agile geoscience in the
    bruges library

    smcf: shear modulus correction factor
        similar to what rokdoc has implemented
        smcf = 1 -> no correction
        smcf = 0 -> zero shear modulus

    """
    if smcf is None:
        smcf = 1.
    if apc is None:
        apc = 10.
    if phi_c is None:
        phi_c = 0.4
    if c_n is None: c_n = 8.6
    if k_c is None: k_c = 37
    if g_c is None: g_c = 45
    if scheme is None: scheme = 2

    phi_cem = phi_c - apc / 100.
    k_eff, mu_eff = brr.constant_cement(k0, g0, phi, phi_cem=phi_cem, phi_c=phi_c, Cn=c_n, Kc=k_c, Gc=g_c, scheme=scheme)
    return k_eff, smcf * mu_eff


def constantcement_hidden(k0, g0, phi, phi_c=None, apc=None, c_n=None, k_c=None, g_c=None, scheme=None):
    """
    Constant cement model, Avseth et al. 2000.
    written by emblixt 2019 from Geophysics Today.
    With a minor change so that we're dealing with percentage of cementation


    It should be noted that the 'Absolute Percent Cement' (apc) represents the percentage of
    cement within the whole volume, not a percentage relative to the critical porosity.
    Therefore the well-sorted end member phi_b = phi_c - (percent_cement/100).

    NOTE that phi should be smaller than phib = phi_c - apc/100. (Avseth 2005, p. 58)

    INPUT
    k0, g0: mineral bulk & shear modulus in GPa
    phi: porosity
    phi_c: critical porosity (default 0.4)
    apc: absolute percent cement (in %)
    c_n: coordination number (default 8.6)
    k_c, g_c: cement bulk & shear modulus in GPa
            (default 37, 45 i.e. quartz)
    scheme: 1=cement deposited at grain contacts
            2=uniform layer around grains (default)
    """
    if apc is None: apc = 10.
    if phi_c is None: phi_c = 0.4
    if c_n is None: c_n = 8.6
    if k_c is None: k_c = 37
    if g_c is None: g_c = 45
    if scheme is None: scheme = 2

    def make_array(x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            return x
        else:
            return np.array([x])


    phib = phi_c - apc / 100.
    if max(make_array(phi)) > min(make_array(phib)):
        print('WARNING, phi, {:.2f}, should be smaller than the well sorted end-member porosity, phib, {:.2f}'.format(
            max(make_array(phi)), min(make_array(phib))))
    k_b, g_b = contactcement(k0, g0, phib, phi_c, c_n, k_c, g_c, scheme)
    z = g_b * ((9. * k_b + 8. * g_b) / (k_b + 2. * g_b)) / 6.
    t1 = 4. * g_b / 3.
    k_eff = ((phi / phib) / (k_b + t1) + (1. - phi / phib) / (k0 + t1)) ** (-1) - t1
    g_eff = ((phi / phib) / (g_b + z) + (1. - phi / phib) / (g0 + z)) ** (-1) - z
    return k_eff, g_eff


def delta_log_r(r, ac, r0=None, ac0=None, neal_morgan=False):
    """
    Calculates the Delta Log R value based on eq. 1 in Passey et al. 1990 "A practical model for organic richness ..."
    Args:
        r:
            Param
            Deep resistivity [Ohmm]
        ac:
            Param
            Acoustic sonic [us/ft]
        r0:
            Param
            Deep resistivity baseline [Ohmm], taken from a non-source, clay-rich, section of the well near the
            area of interest
        ac0:
            Param
            Acoustic sonic baseline [us/ft], taken from a non-source, clay-rich, section of the well
        neal_morgan:
            bool
            If True, it uses a different implementation of Delta Log R which is less sensitive to the baseline,
            BUT it requires you to plot the sonic on a 200 - 0 scale! AND that resistivity is plotted on a
            4 decades logarithmic plot
            In this case, r0 is not the baseline value of resistivity, but the left most limit of the resistivity in
            Delta Log R plot

    Returns:
        Delta log R [-]:
            Param
    """
    if r0 is None:
        r0 = Param(name='', value=1., unit='Ohmm')
    if ac0 is None:
        ac0 = Param(name='', value=100., unit='us/ft')

    r = test_value(r, 'Ohmm')
    ac = test_value(ac, 'us/ft')
    r0 = test_value(r0, 'Ohmm')
    ac0 = test_value(ac0, 'us/ft')
    if neal_morgan:
        return Param(
            name='DeltaLogR',
            value=np.log10(r.value / r0.value) + (ac.value - 200.)/200.,
            unit='-',
            desc='Delta log R according to Passey et al. 1990, using Neal Morgans modification'
        )
    else:
        return Param(
            name='DeltaLogR',
            value=np.log10(r.value / r0.value) + 0.02 * (ac.value - ac0.value),
            unit='-',
            desc='Delta log R according to Passey et al. 1990'
        )


def toc_from_delta_log_r(deltalogr, lom, a=None, b=None):
    """
    Calculates the TOC from Delta Log R and Level of organic metamorphism units (lom)
    based on eq. 2 in Passey et al. 1990 "A practical model for organic richness ..."
    Args:
        deltalogr:
            Param
            Delta log R [-] value estimated from Sonic and Resitivity logs using the method described in the
            Passey et al. article
        lom:
            float
            Unitless number somewhere between 6 and 11 which describes the level of organic metamorphism units
            (Hood et al. 1975)
        a:
            float
            Constant of the TOC equation
        b:
            float
            Constant of the TOC equation

    Returns:
        toc
            Total organic content
    """
    if a is None:
        a = 2.297
    if b is None:
        b = 0.1688

    dlr = test_value(deltalogr, '-')
    return Param(
        name='TOC',
        value=dlr.value * 10**(a - b * lom),
        unit='%',
        desc='TOC estimated from Delta log R according to Passey et al. 1990'
    )


def gassmann_vel(v_p_1, v_s_1, rho_1, k_f1, rho_f1, k_f2, rho_f2, k0, por):
    """
    Gassmann fluid substitution with velocity and density as input and output, following the
    recipe in chapter 1.31. of Avseth et. al 2011

    :param v_p_1:
        np.array
        Rock v_p with initial fluid [m/s]
    :param v_s_1:
        np.array
        Rock v_s with initial fluid [m/s]
    :param rho_1:
        np.array
        Rock density with initial fluid [g/cm3]
    :param k_f1 [k_f2]
        float
        Bulk modulus of fluid 1 [2]  in [GPa]
        k_f = (s_w/k_b + s_o/k_o + s_g/k_g)**(-1) for uniform (Reuss) fluid mix
        k_f = s_w*k_b + s_o*k_o + s_g*k_g for patchy (Voigt) fluid mix
    :param rho_f1 [rho_f2]
        float
        Density of fluid 1 [2] in [g/cm3]
        rho_f = s_w*rho_b + s_o*rho_o + s_g*rho_g
    :param k0:
        float
        Mineral bulk modulus [GPa]
    :param por:
        np.array
        Porosity
    """
    # TODO
    # Allow a mask to only do the fluid substitution where the mask is True

    # Avoid low porosity points
    if isinstance(por, np.ndarray):  # when input is an array
        por[por < 7E-3] = 7E-3

    # Extract the initial bulk and shear modulus from v_p_1, v_s_1 and rho_1
    mu_1 = rho_1 * v_s_1**2 * 1E-6  # GPa
    #k_1 = rho_1 * (v_p_1**2 - (4/3.)*v_s_1**2)*1E-6  # GPa
    k_1 = rho_1 * v_p_1**2 * 1E-6 - (4/3.)*mu_1  # GPa

    # Apply Gassmann's relation to transform the bulk modulus
    #a = k_1/(k0 - k_1) + (k_f2/(k0 - k_f2) - k_f1/(k0-k_f1))/por
    a = gassmann_a(k_1, k0, k_f1, k_f2, por)
    k_2 = k0*a / (1.+a)  # GPa

    # Correct the bulk density for the fluid change
    rho_2 = rho_1 + por*(rho_f2 - rho_f1)  # g/cm3

    # Leave the shear modulus unchanged
    mu_2 = mu_1  # GPa

    # Calculate the new velocities
    v_p_2 = np.sqrt((k_2+(4/3)*mu_2) / rho_2) * 1E3  # m/s
    v_s_2 = np.sqrt(mu_2 / rho_2) * 1E3  # m/s

    return v_p_2, v_s_2, rho_2, k_2


def gassmann_a(_k1, _k0, _k_f1, _k_f2, _por):
    """

    :param _k1:
        Initial saturated rock bulk modulus
    :param _k0:
        Mineral bulk modulus
    :param _k_f1:
        Initial fluid bulk modulus
    :param _k_f2:
        Final fluid bulk modulus
    :param _por:
        porosity
    :return:
    """
    return _k1/(_k0 - _k1) + (_k_f2/(_k0 - _k_f2) - _k_f1/(_k0-_k_f1))/_por


def vels(k_dry, g_dry, k_min, rho_min, k_f, rho_f, phi):
    '''
    Calculates velocities and densities of saturated rock via Gassmann equation, (C) aadm 2015

    INPUT
    k_dry,g_dry: dry rock bulk & shear modulus in GPa
    k_min, rho_min: mineral bulk modulus and density in GPa
    k_f, rho_f: fluid bulk modulus and density in GPa
    phi: porosity
    '''
    rho = rho_min * (1 - phi) + rho_f * phi
    k = k_dry + (1 - k_dry / k_min) ** 2 / ((phi / k_f) + ((1 - phi) / k_min) - (k_dry / k_min ** 2))
    vp = np.sqrt((k + 4. / 3 * g_dry) / rho) * 1e3
    vs = np.sqrt(g_dry / rho) * 1e3
    return vp, vs, rho, k


def linear_gassmann(_phi, _phi_r, _delta_k_fl):
    """
    Calculates the change in saturated rock bulk modulus, k_delta, as a
    function of porosity, _phi, and the difference in bulk modulus between initial and final
    fluids, _delta_k_fl, calculated at the 'intercept porosity', _phi_r
    See page 171 in Rock physics handbook, Mavko et al. 1998.
    :return:
    """
    delta_k = (_phi/(_phi_r**2)) * _delta_k_fl
    return delta_k


def linear_brine_elastics():
    pass


def run_fluid_sub(wells, log_table, mineral_mix, fluid_mix, cutoffs, working_intervals, tag, templates=None, block_name=None):
    """
    Run fluid substitution using the defined fluids and minerals given in mineral_mix and fluid_mix, and only for the
    wells where they have been defined.


    :param wells:
        dict
        dictionary of "well name": core.well.Well  key: value pairs
        E.G.
            from core.well import Project
            wp = Project( ... )
            wells = wp.load_all_wells()
    :param log_table:
        dict
        Dictionary of log type: log name key: value pairs that determines which logs to use in the substitution
        The Vp, Vs, Rho and Phi logs are necessary to calculate the fluid substitution
        E.G.
            log_table = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob',
               'Porosity': 'phie',
               'Volume': 'vcl'}
    :param mineral_mix:
        core.minerals.MineralMix
    :param fluid_mix:
        core.fluids.FluidMix
    :param cutoffs:
        dict
        Defines the properties of the well logs for which can be fluid substituted
        E.G.:
         {'Volume': ['<', 0.4], 'Porosity': ['>', 0.1]}
         means that the fluid substitution only can happen in sands where the
         volume of shale is less than 0.4, and porosity is greater than 0.1,
    :param working_intervals:
        dict
        E.G.
        import rp_utils.io as uio
        working_intervals = uio.project_working_intervals(wp.project_table)
    :param tag:
        str
        String to tag the resulting logs with
    :param templates:
        dict
        templates that can contain well information such as kelly bushing and sea water depth
        templates = rp_utils.io.project_tempplates(wp.project_table)
    :param block_name:
        str
        Name of the log block which should contain the logs to fluid substitute
    :return:
    """
    log_table = small_log_table(log_table)
    if block_name is None:
        block_name = ud.def_lb_name
    if tag is None:
        tag = ''
    elif tag[0] != '_':
        tag = '_{}'.format(tag)

    # rename variables to shorten lines
    wis = working_intervals
    lnd = log_table
    mm = mineral_mix
    fm = fluid_mix

    # Calculate the elastic properties of the fluids
    fm.calc_press_ref(wells, templates=templates, block_name=block_name)
    fm.calc_elastics(wells, wis, templates=templates, block_name=block_name)

    # and for the minerals
    mm.calc_elastics(wells, log_table, wis, block_name=block_name)

    # Loop over all wells
    for wname, well in wells.items():
        if wname not in list(fm.fluids['initial'].keys()):
            print('Well {} not listed in the fluid mixture, skipping'.format(wname))
            continue
        info_txt = 'Starting Gassmann fluid substitution on well {}'.format(wname)
        print('INFO: {}'.format(info_txt))
        logger.info(info_txt)

        # Extract log block
        lb = well.block[block_name]
        # test if necessary log types are present in the well
        skip_this_well = False
        for xx in ['Porosity', 'Density', 'P velocity', 'S velocity']:
            if xx not in lb.log_types():
                warn_txt = 'Log type {} not present in well {}'.format(xx, wname)
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                skip_this_well = True
        if skip_this_well:
            continue

        # Variables constant through fluid substitution:
        k0_dict = well.calc_vrh_bounds(mm, param='k', wis=wis, method='Voigt-Reuss-Hill', block_name=block_name)
        por = lb.logs[lnd['Porosity']].data

        # Initial fluids
        rho_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='rho', wis=wis, method='Voigt', block_name=block_name)
        k_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='k', wis=wis, method='Reuss', block_name=block_name)

        # Initial elastic logs as LogCurve objects
        v_p_1 = lb.logs[lnd['P velocity']]
        v_s_1 = lb.logs[lnd['S velocity']]
        rho_1 = lb.logs[lnd['Density']]

        # Final fluids
        rho_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='rho', wis=wis, method='Voigt', block_name=block_name)
        k_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='k', wis=wis, method='Reuss', block_name=block_name)

        # Run the fluid substitution separately in each interval
        for wi in list(rho_f1_dict.keys()):
            k_f1 = k_f1_dict[wi]
            rho_f1 = rho_f1_dict[wi]
            k_f2 = k_f2_dict[wi]
            rho_f2 = rho_f2_dict[wi]
            k0 = k0_dict[wi]

            # calculate the mask for the given cut-offs, and for the given working interval
            well.calc_mask(cutoffs, wis=wis, wi_name=wi, name='this_mask', log_table=log_table)
            mask = lb.masks['this_mask'].data

            # Do the fluid substitution itself
            _v_p_2, _v_s_2, _rho_2, _k_2 = gassmann_vel(
                v_p_1.data, v_s_1.data, rho_1.data, k_f1, rho_f1, k_f2, rho_f2, k0, por)

            # Add the fluid substituted results to the well
            for xx, yy in zip([v_p_1, v_s_1, rho_1], [_v_p_2, _v_s_2, _rho_2]):
                new_name = deepcopy(xx.name)
                new_name += '{}'.format(tag.lower())
                new_header = deepcopy(xx.header)
                new_header.name += '{}'.format(tag.lower())
                new_header.desc = 'Fluid substituted {}'.format(xx.name)
                mod_history = 'Calculated using Gassmann fluid substitution using following\n'
                mod_history += 'Mineral mixtures: {}\n'.format(mm.print_minerals(wname, wi))
                mod_history += 'Initial fluids: {}\n'.format(
                    fm.print_fluids('initial', wname, wi))
                mod_history += 'Final fluids: {}\n'.format(
                    fm.print_fluids('final', wname, wi))
                new_header.modification_history = mod_history
                new_data = deepcopy(xx.data)
                new_data[mask] = yy[mask]
                lb.add_log(new_data, new_name, xx.get_log_type(), new_header)


def test_mineral_existance(well_name, wi_name, mineral_mix, gc_coeffs):
    """
    Just keeping track of a useful code snippet
    :param well_name:
    :param wi_name:
    :param mineral_mix:
    :return:
    """
    # Test if minerals are present in given well, for the given working interval
    warn_txt = None
    if well_name not in list(mineral_mix.minerals.keys()):
        warn_txt = 'Well {} not present in the given mineral mix {}'.format(well_name, mineral_mix.name)
    elif wi_name not in list(mineral_mix.minerals[well_name].keys()):
        warn_txt = 'Working interval {} not present in the given mineral mix {}'.format(wi_name, mineral_mix.name)
    elif not any([xx.lower() in list(mineral_mix.minerals[well_name][wi_name].keys()) for xx in list(gc_coeffs)]):
        warn_txt = 'No minerals in {} ({}) present among the Greenberg Castagna coefficients'.format(
            mineral_mix.name, ','.join(list(mineral_mix.minerals[well_name][wi_name].keys()))
        )
    if warn_txt is not None:
        raise IOError(warn_txt)


def plot_model(model, k0, g0, _phi, rho, _ax, **kwargs):
    """
    Plots the selected model in the given axis
    :param model:
        One of the models defined in this file
    :param k0:
        float
        k0 mineral bulk modulus in GPa
    :param g0:
        float
        shear modulus in GPa
    :param _phi:
        np.ndarray
        porosity
    :param rho:
        float
        Density in g/cm3 for which the p velocity is calculated
    :param _ax:
        matplotlib.axes object
    :param kwargs:
        keyword arguments passed on to the model
    :return:
    """
    k_eff, g_eff = model(k0, g0, _phi, **kwargs)
    vp = v_p(k_eff,g_eff, rho)
    _ax.plot(_phi, vp)
    _ax.set_xlabel('Porosity')
    _ax.set_ylabel('P velocity [km/s]')
    _ax.legend(['k0: {:.1f}, mu0: {:.1f}, rho: {:.1f}'.format(k0, g0, rho)])


if __name__ == '__main__':
    # 'rho_b': 1.02, 'k_b': 2.8,
    # 'rho_qz': 2.6, 'k_qz': 37, 'mu_qz': 45,
    # 'rho_sh': 2.6, 'k_sh': 15, 'mu_sh': 6,
    fig, ax = plt.subplots()
    phi = np.linspace(0.1, 0.4, 6)
    plot_model(contactcement, 37, 45, phi, 2.6, ax)
    plot_model(constantcement, 37, 45, phi, 2.6, ax, apc=2)
    plot_model(hertz_mindlin, 37, 45, phi, 2.6, ax)
    plt.show()


