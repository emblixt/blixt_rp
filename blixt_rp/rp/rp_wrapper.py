import numpy as np
import matplotlib.pyplot as plt
import logging

import blixt_rp.rp.rp_core as rp

logger = logging.getLogger(__name__)

def vp_vs_rho(phi, **kwargs):
    """
    Trying to replicate RokDoc Vp vs Porosity RPM Template
    """
    model = kwargs.pop('model', 'constant cement')
    phi_c = kwargs.pop('phi_c', 0.4)  # critical porosity
    c_n = kwargs.pop('c_n', 9)  # Coordination number in the contact cement model
    p_conf = kwargs.pop('p_conf', 30)  # Confining pressure in MPa. Used in stiff & soft sand model
    smcf = kwargs.pop('smcf', 1.0)  # Shear modulus correction factor. Used in stiff & soft sand model

    # cement parameters
    apc = kwargs.pop('apc', 2)  # absolute percent cement (in %)
    kc = kwargs.pop('kc', 36.6)  # Cement bulk modulus in GPa
    gc = kwargs.pop('gc', 45)  # Cement shear modulus in GPa

    # fluid parameters
    rho_b = kwargs.pop('rho_b', 1)  # Brine density  g/cm3
    k_b = kwargs.pop('k_b', 2.56)   # Brine bulk modulus  GPa
    rho_hc = kwargs.pop('rho_hc', 0.8)  # Oil density
    k_hc = kwargs.pop('k_hc', 1.152)  # Oil bulk modulus
    sw = kwargs.pop('sw', 1.)  # Water saturation

    # mineral parameters
    rho_min1 = kwargs.pop('rho_min', 2.65)  # Density [g/cm3] of main mineral, Quartz
    k_min1 = kwargs.pop('k_min', 36.6)  # Bulk modulus GPa of main mineral, Quartz
    mu_min1 = kwargs.pop('mu_min', 45.)  # Shear modulus of main mineral, Quartz
    rho_min2 = kwargs.pop('rho_min', 2.35)  # Density [g/cm3] of complement mineral, Shale
    k_min2 = kwargs.pop('k_min', 11.4)  # Bulk modulus GPa of complement mineral, Shale
    mu_min2 = kwargs.pop('mu_min', 3.)  # Shear modulus of complement mineral, Shale
    vsh = kwargs.pop('vsh', 0.)  # volume fraction of complement mineral
    mineral_mix_method = kwargs.pop('mineral_mix_method', 'default')

    # Calculate mineral mix properties
    k_min = None  # Mineral bulk modulus in GPa
    mu_min = None  # Mineral shear modulus in GPa
    rho_min = None  # Mineral density in g/cm3
    if mineral_mix_method == 'default':
        k_min = rp.vrh_bounds([1.0 - vsh, vsh], [k_min1, k_min2])[2]  # Mineral bulk modulus
        mu_min = rp.vrh_bounds([1.0 - vsh, vsh], [mu_min1, mu_min2])[2]  # Mineral shear modulus
        rho_min = rp.vrh_bounds([1.0 - vsh, vsh], [rho_min1, rho_min2])[0]  # Density of minerals
    elif mineral_mix_method == 'voigt-reuss-hill':
        k_min = rp.vrh_bounds([1.0 - vsh, vsh], [k_min1, k_min2])[2]  # Mineral bulk modulus
        mu_min = rp.vrh_bounds([1.0 - vsh, vsh], [mu_min1, mu_min2])[2]  # Mineral shear modulus
        rho_min = rp.vrh_bounds([1.0 - vsh, vsh], [rho_min1, rho_min2])[2]  # Density of minerals
    else:
        raise NotImplementedError('The {} mineral mixing method is not implemented'.format(mineral_mix_method))

    # Calculate effective modulus from rock physics function
    k_eff = None  # Effective bulk modulus in GPa
    mu_eff = None  # Effective shear modulus in GPa
    if model == 'constant cement':
        # correct porosity so that its not larger than critical porosity minus cement
        phi[phi > (phi_c - apc/100.)] = phi_c - apc/99.5
        k_eff, mu_eff = rp.constantcement(k_min, mu_min, phi, phi_c, apc, c_n, kc, gc)
    elif model == 'contact cement':
        k_eff, mu_eff = rp.contactcement(k_min, mu_min, phi, phi_c, c_n, kc, gc)
    elif model == 'stiff':
        k_eff, mu_eff = rp.stiffsand(k_min, mu_min, phi, phi_c, c_n, p_conf, smcf)
    elif model == 'soft':
        k_eff, mu_eff = rp.softsand(k_min, mu_min, phi, phi_c, c_n, p_conf, smcf)
    else:
        raise NotImplementedError('The {} RPM is not implemented'.format(model))

    # Calculate the final fluid properties for the given water saturation
    k_f2 = rp.vrh_bounds([sw, 1.-sw], [k_b, k_hc])[1]  # K_f
    rho_f2 = rp.vrh_bounds([sw, 1.-sw],  [rho_b, rho_hc])[0]  # RHO_f

    # Use Gassman to calculate the final elastic properties
    vp_2, vs_2, rho_2, k_2 = rp.vels(k_eff, mu_eff, k_min, rho_min, k_f2, rho_f2, phi)

    return vp_2, vs_2, rho_2


