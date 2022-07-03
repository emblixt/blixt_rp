import matplotlib.pyplot as plt
import numpy as np
import logging

# TODO remove this two lines, only used in testing
import sys
sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
from bruges.filters import ricker

logger = logging.getLogger(__name__)


def one_d_model(vp, vs, rho, layer_thicknesses, t0, dt):
    """
    Builds a 1D model of vp, vs, and rho, where the thickness of layer is specified by layer_thicknesses [s]
    All times are in seconds

    """
    if not(len(vp) == len(vs) == len(rho) == len(layer_thicknesses)):
        warn_txt = 'Input lists in one_d_model() are not of same length'
        print('WARNING: {}'.format(warn_txt))
        logger.warning(warn_txt)
        raise IOError

    # time goes from t0 to the last time given in end_times
    twt = np.arange(t0, t0 + sum(layer_thicknesses), dt)  # TWT in seconds

    vp_arr = np.zeros(len(twt))
    vs_arr = np.zeros(len(twt))
    rho_arr = np.zeros(len(twt))
    twt_boundaries = []
    for i in range(len(layer_thicknesses)):
        twt_boundaries.append(t0 + sum(layer_thicknesses[:i+1]))
    i = 0
    for _vp, _vs, _rho in zip(vp, vs, rho):
        if i == 0:
            vp_arr[twt <= twt_boundaries[i]] = _vp
            vs_arr[twt <= twt_boundaries[i]] = _vs
            rho_arr[twt <= twt_boundaries[i]] = _rho
        elif i < len(vp) - 1:
            vp_arr[(twt > twt_boundaries[i - 1]) & (twt <= twt_boundaries[i])] = _vp
            vs_arr[(twt > twt_boundaries[i - 1]) & (twt <= twt_boundaries[i])] = _vs
            rho_arr[(twt > twt_boundaries[i - 1]) & (twt <= twt_boundaries[i])] = _rho
        else:
            vp_arr[twt > twt_boundaries[i-1]] = _vp
            vs_arr[twt > twt_boundaries[i-1]] = _vs
            rho_arr[twt > twt_boundaries[i-1]] = _rho
        i += 1
    return twt, vp_arr, vs_arr, rho_arr


def wedge_modelling(vps, vss, rhos, up_to_thickness, wavelet,
                    time_step=None,
                    number_of_traces=None):
    """
    :param vps:
        list
        List of length 3 of Vp velocities in m/s
    :param vss:
        list
        List of length 3 of Vs velocities in m/s
    :param rhos:
        list
        List of length 3 of densities in g/cm3
    :param up_to_thickness:
        float
        Wedge model will go from 0 s TWT to up_to_thickness_s s TWT thickness
    :param wavelet:
        bruges.filters.wavelet functions
        or any function that takes duration, dt, and f as input and returns a wavelet
    :param time_step:
        float
        time step in s
    """

    if time_step is None:
        time_step = 0.001
    if number_of_traces is None:
        number_of_traces = 100

    fig, ax = plt.subplots()

    # loop over wedge thickness
    for i in range(number_of_traces):
        wedge_thickness = i * up_to_thickness / (number_of_traces - 1)
        twt, vp, vs, rho = one_d_model(vps, vss, rhos,
                                       [0.1, wedge_thickness, 0.1 + (up_to_thickness - wedge_thickness)], 2., time_step)

        print(wedge_thickness)
        print(twt.min(), twt.max())
        print(vp.min(), vp.max())
        ax.plot((i+1) + vp/1000., twt[::-1])

    plt.show()


if __name__ == '__main__':
    vps = [3500., 3600., 3500.]  # Vp [m/s] in the different layers
    vss = [1800., 2100., 1800.]
    rhos = [2.6, 2.3, 2.6]
    wedge_modelling(vps, vss, rhos, 0.1, None, number_of_traces=10)
