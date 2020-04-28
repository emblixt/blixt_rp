# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:25:40 2019

@author: mblixt
"""
import numpy as np
import logging
from dataclasses import dataclass
from copy import deepcopy


logger = logging.getLogger(__name__)


@dataclass
class Param:
    """
    Data class to hold the different parameters used in the rock physics functions
    """
    name: str
    value: float
    unit: str
    desc: str

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


def step(x_1, x_2):
    """
    Returns the averaged difference from layer 1 (top) to layer 2 (bottom)
    of variable x.
    """
    return 2 * (x_2 - x_1) / (x_2 + x_1)


def poissons_ratio(Vp, Vs):
    return 0.5 * (Vp ** 2 - 2 * Vs ** 2) / (Vp ** 2 - Vs ** 2)


def intercept(Vp_1, Vp_2, rho_1, rho_2):
    """
    Returns the AVO intercept, or normal incidence reflectivity
    From eq. 2 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.
    """
    return 0.5 * (step(Vp_1, Vp_2) + step(rho_1, rho_2))


def gradient(Vp_1, Vp_2, Vs_1, Vs_2, rho_1, rho_2):
    """
    Returns the AVO gradient
    From eq. 3 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.
    """
    return 0.5 * step(Vp_1, Vp_2) - 2. * ((Vs_1 + Vs_2) / (Vp_1 + Vp_2)) ** 2 * \
           (2. * step(Vs_1, Vs_2) + step(rho_1, rho_2))


def reflectivity(Vp_1, Vp_2, Vs_1, Vs_2, rho_1, rho_2, version='WigginsAkiRich'):
    """
    returns function which returns the reflectivity as function of theta
    theta is in degrees
    :param version:
        str
        'WigginsAkiRich' uses the formulation of Aki Richards by Wiggins et al. 1984 according to Hampson Russell
        'ShueyAkiRich' uses Shuey 1985's formulation of Aki Richards according to Avseth 2011 (eq. 4.7)
        'AkiRich' Aki Richards formulation according to Avseth 2011 (eq. 4.6)

    """
    if (version == 'WigginsAkiRich') or (version == 'ShueyAkiRich'):
        a = intercept(Vp_1, Vp_2, rho_1, rho_2)
        b = gradient(Vp_1, Vp_2, Vs_1, Vs_2, rho_1, rho_2)
        c = 0.5 * step(Vp_1, Vp_2)
        if version == 'WigginsAkiRich':
            def func(theta):
                return a + b * (np.sin(theta * np.pi / 180.)) ** 2 + \
                       c * ((np.tan(theta * np.pi / 180.)) ** 2 * (np.sin(theta * np.pi / 180.)) ** 2)
        elif version == 'ShueyAkiRich':
            def func(theta):
                return a + b * (np.sin(theta * np.pi / 180.)) ** 2 + \
                       c * ((np.tan(theta * np.pi / 180.)) ** 2 - (np.sin(theta * np.pi / 180.)) ** 2)
        return func
    elif version == 'AkiRich':
        a = (Vs_1 + Vs_2) ** 2 / Vp_1 ** 2  # = 4 * p**2 * Vs**2/sin(theta)**2 in eq. 4.6

        def func(theta):
            return 0.5 * (1. - a * (np.sin(theta * np.pi / 180.)) ** 2) * step(rho_1, rho_2) + \
                   0.5 * step(Vp_1, Vp_2) / (np.cos(theta * np.pi / 180.)) ** 2 - \
                   a * (np.sin(theta * np.pi / 180.)) ** 2 * step(Vs_1, Vs_2)

        return func
    else:
        raise NotImplementedError


def v_p(k, mu, rho):
    return np.sqrt(
        (k + 4. * mu / 3.) / rho)


def v_s(mu, rho):
    return np.sqrt(mu / rho)


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


def v_p_b(s, p, t, giib=0.):
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


def hertz_mindlin(k0, mu0, phic=0.4, cn=8.6, p=None, f=1):
    """
    Hertz-Mindlin model
    The elastic moduli of a dry well-sorted end member at critical porosity
    Rock Physics Handbook, p.246
    Eq. 2.3 and 2.4 in Avseth 2011

    :param k0:
        Param
        Mineral bulk modulus in GPa
    :param mu0:
        Param
        Mineral shear modulus in GPa
    :param phic:
        float
        critical porosity
    :param cn:
        float
       coordination number (default 8.6), average number of contacts per grain
    :param p:
        Param
        Confining pressure, default 10/1E3 GPa
    :param f:
        int
        shear modulus correction factor
        1=dry pack with perfect adhesion
        0=dry frictionless pack
    """
    if p is None:
        p = Param(
            name='Pressure',
            value=10./1e3,
            unit='GPa',
            desc='Confining pressure'
        )
    k0 = test_value(k0, 'GPa')
    mu0 = test_value(mu0, 'GPa')
    p = test_value(p, 'GPa')

    k0 = k0.value
    mu0 = mu0.value
    p = p.value

    pr0 = (3*k0-2*mu0)/(6*k0+2*mu0) # poisson's ratio of mineral mixture
    k_HM = (p*(cn**2*(1-phic)**2*mu0**2) / (18*np.pi**2*(1-pr0)**2))**(1/3)
    mu_HM = ((2+3*f-pr0*(1+3*f))/(5*(2-pr0))) * (p*(3*cn**2*(1-phic)**2*mu0**2)/(2*np.pi**2*(1-pr0)**2))**(1/3)
    return k_HM, mu_HM


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


def softsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
    Soft-sand (uncemented) model
    written by aadm (2015) from Rock Physics Handbook, p.258

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    '''
    K_HM, G_HM = hertz_mindlin(K0, G0, phic, Cn, P/1.E3, f)
    K_DRY =-4/3*G_HM + (((phi/phic)/(K_HM+4/3*G_HM)) + ((1-phi/phic)/(K0+4/3*G_HM)))**-1
    tmp   = G_HM/6*((9*K_HM+8*G_HM) / (K_HM+2*G_HM))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY


def stiffsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
    Stiff-sand model
    written by aadm (2015) from Rock Physics Handbook, p.260

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    '''
    K_HM, G_HM = hertz_mindlin(K0, G0, phic, Cn, P/1.E3, f)
    print('K_HM: {}, G_HM: {}'.format(K_HM, G_HM))
    K_DRY  = -4/3*G0 + (((phi/phic)/(K_HM+4/3*G0)) + ((1-phi/phic)/(K0+4/3*G0)))**-1
    tmp    = G0/6*((9*K0+8*G0) / (K0+2*G0))
    G_DRY  = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY


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
        Rock density with initial fluid [m/s]
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


def vels(K_DRY, G_DRY, K0, D0, Kf, Df, phi):
    '''
    Calculates velocities and densities of saturated rock via Gassmann equation, (C) aadm 2015

    INPUT
    K_DRY,G_DRY: dry rock bulk & shear modulus in GPa
    K0, D0: mineral bulk modulus and density in GPa
    Kf, Df: fluid bulk modulus and density in GPa
    phi: porosity
    '''
    rho  = D0*(1-phi)+Df*phi
    K    = K_DRY + (1-K_DRY/K0)**2 / ( (phi/Kf) + ((1-phi)/K0) - (K_DRY/K0**2) )
    vp   = np.sqrt((K+4./3*G_DRY)/rho)*1e3
    vs   = np.sqrt(G_DRY/rho)*1e3
    return vp, vs, rho, K


def run_fluid_sub(wells, logname_dict, mineral_mix, fluid_mix, cutoffs, working_intervals, tag, block_name='Logs'):
    """

    :param wells:
        dict
        dictionary of "well name": core.well.Well  key: value pairs
        E.G.
            from core.well import Project
            wp = Project( ... )
            wells = wp.load_all_wells()
    :param logname_dict:
        dict
        Dictionary of log type: log name key: value pairs to create statistics on
        The Vp, Vs, Rho and Phi logs are necessary for output to RokDoc compatible Sums & Average excel file
        E.G.
            logname_dict = {
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
        E.G. {'Volume': ['<', 0.4], 'Porosity': ['>', 0.1]}
    :param working_intervals:
        dict
        E.G.
        import utils.io as uio
        working_intervals = uio.project_working_intervals(wp.project_table)
    :param tag:
        str
        String to tag the resulting logs with
    :param block_name:
        str
        Name of the log block which should contain the logs to fluid substitute
    :return:
    """
    if tag is None:
        tag = ''
    elif tag[0] != '_':
        tag = '_{}'.format(tag)

    # rename variables to shorten lines
    wis = working_intervals
    lnd = logname_dict
    mm = mineral_mix
    fm = fluid_mix


    # Loop over all wells
    for wname, well in wells.items():
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
        k0_dict = well.calc_vrh_bounds(mm, param='k', wis=wis, method='Voigt-Reuss-Hill')
        por = lb.logs[lnd['Porosity']].data

        # Initial fluids
        rho_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='rho', wis=wis, method='Voigt')
        k_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='k', wis=wis, method='Reuss')

        # Initial elastic logs as LogCurve objects
        v_p_1 = lb.logs[lnd['P velocity']]
        v_s_1 = lb.logs[lnd['S velocity']]
        rho_1 = lb.logs[lnd['Density']]

        # Final fluids
        rho_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='rho', wis=wis, method='Voigt')
        k_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='k', wis=wis, method='Reuss')

        # Run the fluid substitution separately in each interval
        for wi in list(rho_f1_dict.keys()):
            print('WORKING INTERVAL (USING MD INSTEAD OF TVD) {}'.format(wi.upper()))
            # TODO Make this function extract the tvd, not the MD !!!
            this_tvd = np.mean(wis[wname][wi])

            k_f1 = k_f1_dict[wi]
            rho_f1 = rho_f1_dict[wi]
            k_f2 = k_f2_dict[wi]
            rho_f2 = rho_f2_dict[wi]
            k0 = k0_dict[wi]

            # calculate the mask for the given cut-offs, and for the given working interval
            well.calc_mask(cutoffs, wis=wis, wi_name=wi, name='this_mask')
            mask = lb.masks['this_mask'].data

            # Do the fluid substitution itself
            _v_p_2, _v_s_2, _rho_2, _k_2 = gassmann_vel(
                v_p_1.data, v_s_1.data, rho_1.data, k_f1, rho_f1, k_f2, rho_f2, k0, por)

            # Add the fluid substituted results to the well
            for xx, yy in zip([v_p_1, v_s_1, rho_1], [_v_p_2, _v_s_2, _rho_2]):
                new_name = deepcopy(xx.name)
                new_name += '_{}{}'.format(wi.lower().replace(' ','_'), tag.lower())
                new_header = deepcopy(xx.header)
                new_header.name += '_{}{}'.format(wi.lower().replace(' ','_'), tag.lower())
                new_header.desc = 'Fluid substituted {}'.format(xx.name)
                mod_history = 'Calculated using Gassmann fluid substitution using following\n'
                mod_history += 'Mineral mixtures: {}\n'.format(mm.print_minerals(wname, wi))
                mod_history += 'Initial fluids: {}\n'.format(
                    fm.print_fluids('initial', wname, wi, this_tvd))
                mod_history += 'Final fluids: {}\n'.format(
                    fm.print_fluids('final', wname, wi, this_tvd))
                new_header.modification_history = mod_history
                new_data = deepcopy(xx.data)
                new_data[mask] = yy[mask]
                lb.add_log(new_data, new_name, xx.get_log_type(), new_header)
