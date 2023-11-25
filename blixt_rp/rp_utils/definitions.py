
supported_version = {2.0, 3.0}

allowed_depth_formats = ['md', 'tvd', 'owt', 'twt']  # allowed depth formats
def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name
def_water_depth_keys = ['gl', 'egl', 'water_depth', 'wdep']
def_kelly_bushing_keys = ['kb', 'ekb', 'apd', 'edf', 'eref']
def_sonic_units = ['us/f', 'us/ft', 'us/feet',
                   'usec/f', 'usec/ft', 'usec/feet',
                   'us/m', 'usec/m', 's/m']

rename_well_logs = {
    # depth, or MD
    'depth': ['Depth', 'DEPT', 'MD', 'DEPTH'],

    # Bulk density
    'den': ['RHOB', 'HRHOB', 'DEN', 'HDEN', 'RHO', 'CPI:RHOB', 'HRHO'],

    # Density correction
    'denc': ['HDRHO', 'DENC', 'DCOR'],

    # Sonic
    'ac': ['HDT', 'DT', 'CPI:DT', 'HAC', 'AC'],

    # Shear sonic
    'acs': ['ACS'],

    # Gamma ray
    'gr': ['HGR', 'GR', 'CPI:GR'],

    # Caliper
    'cali': ['HCALI', 'CALI', 'HCAL'],

    # Deep resistivity
    'rdep': ['HRD', 'RDEP', 'ILD', 'RD'],

    # Medium resistivity
    'rmed': ['HRM', 'RMED', 'RM'],

    # Shallow resistivity
    'rsha': ['RS', 'RSHA', 'HRS'],

    # Water saturation
    'sw': ['CPI:SW', 'SW'],

    # Effective porosity
    'phie': ['CPI:PHIE', 'PHIE'],

    # Neutron porosity
    'neu': ['HNPHI', 'NEU', 'CPI:NPHI', 'HPHI'],

    # Shale volume
    'vcl': ['VCLGR', 'VCL', 'CPI:VWCL', 'CPI:VCL']
}


def gr_cmap():
    """
    Creates a useful colormap (color map) for plotting gamma ray logs
    """
    from matplotlib.colors import ListedColormap
    import numpy as np
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 0 / 256, N)
    vals[:, 1] = np.linspace(1, 85 / 256, N)
    vals[:, 2] = np.linspace(0, 0 / 256, N)
    return ListedColormap(vals)
