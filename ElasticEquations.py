# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:25:40 2019

@author: mblixt
"""
import numpy as np

def step(x_1, x_2):
    """
    Returns the averaged difference from layer 1 (top) to layer 2 (bottom)
    of variable x.
    """
    return 2 * (x_2 - x_1) / (x_2 + x_1)

def poissons_ratio(Vp, Vs):
    return 0.5 * (Vp**2 - 2 * Vs**2) / (Vp**2 - Vs**2)

def intercept(Vp_1, Vp_2, rho_1, rho_2):
    """
    Returns the AVO intercept, or normal incidence reflectivity
    From eq. 2 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.
    """
    return  0.5 * ( step(Vp_1, Vp_2) + step(rho_1, rho_2) )

def gradient(Vp_1, Vp_2, Vs_1, Vs_2, rho_1, rho_2):
    """
    Returns the AVO gradient
    From eq. 3 in Castagna et al. "Framework for AVO gradient and intercept interpretation"
    Geophysics 1998.
    """
    return 0.5 * step(Vp_1, Vp_2) - 2. * ((Vs_1 + Vs_2)/(Vp_1 + Vp_2))**2 *\
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
                return a + b*(np.sin(theta*np.pi/180.))**2 + \
               c*((np.tan(theta*np.pi/180.))**2 * (np.sin(theta*np.pi/180.))**2)
        elif version == 'ShueyAkiRich':
            def func(theta):
                return a + b*(np.sin(theta*np.pi/180.))**2 + \
                   c*((np.tan(theta*np.pi/180.))**2 - (np.sin(theta*np.pi/180.))**2)
        return func
    elif version == 'AkiRich':
        a = (Vs_1 + Vs_2)**2/Vp_1**2  # = 4 * p**2 * Vs**2/sin(theta)**2 in eq. 4.6
        def func(theta):
            return 0.5*(1. - a * (np.sin(theta*np.pi/180.))**2) * step(rho_1, rho_2) + \
                0.5*step(Vp_1, Vp_2)/(np.cos(theta*np.pi/180.))**2 - \
                a * (np.sin(theta*np.pi/180.))**2 * step(Vs_1, Vs_2)
        return func
    else:
        raise NotImplementedError

def v_p(K, mu, rho):
    return np.sqrt(
            (K + 4.* mu/3.) / rho)
    
def v_s(mu, rho):
    return np.sqrt(mu / rho)