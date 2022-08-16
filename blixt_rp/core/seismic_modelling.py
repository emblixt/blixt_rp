import matplotlib.pyplot as plt
import numpy
import numpy as np
import logging

# TODO remove this two lines, only used in testing
import sys
#sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')
sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
import blixt_rp.plotting.plot_layered_model as model_plot
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


def wedge_modelling(vps: list, vss: list, rhos: list, up_to_thickness: float, incident_angle: float,
                    wavelet: numpy.ndarray,
                    time_step=None,
                    center_f=None,
                    duration=None,
                    t0=None,
                    buffer=None,
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
    :param incident_angle:
        float
        incident angle in degrees
    :param wavelet:
        numpy.ndarray
        Array representing a wavelet,
        E.G a bruges.filters wavelet, bruges.filters.ricker()
    :param time_step:
        float
        time step in s
    :param t0:
        float
        TWT time in seconds to where the model top is
    """

    if time_step is None:
        time_step = 0.001
    if center_f is None:
        center_f = 12.
    if duration is None:
        duration = 0.128
    if t0 is None:
        t0 = 2.
    if buffer is None:
        buffer = 0.1
    if number_of_traces is None:
        number_of_traces = 100

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(8, 8))
    # divide figure into a 3 by 3 grid
    spec = fig.add_gridspec(3,3)
    ax_thickness = fig.add_subplot(spec[0, 0:2])  # row 0, column 0 to 1
    ax_wedge = fig.add_subplot(spec[1:3, 0:2])  # row 1 to 2, column 0 to 1
    ax_model = fig.add_subplot(spec[1:3, 2])  # row 1 to 2, column 3

    top = []
    base = []
    wedge_thickness = []
    minimum_amp = []
    maximum_amp = []
    apparent_thickness = []
    # loop over wedge thickness
    for i in range(number_of_traces):
        this_wedge_thickness = i * up_to_thickness / (number_of_traces - 1)
        twt, vp, vs, rho = one_d_model(vps, vss, rhos,
                                       [buffer, this_wedge_thickness, buffer + (up_to_thickness - this_wedge_thickness)], t0, time_step)
        top.append(t0 + buffer)
        base.append(t0 + buffer + this_wedge_thickness)
        wedge_thickness.append(this_wedge_thickness)
        reflectivity = rp.reflectivity(vp, None, vs, None, rho, None, along_wiggle=True)
        wiggle = np.convolve(wavelet, np.nan_to_num(reflectivity(incident_angle)), mode='same')
        wiggle = np.append(wiggle, np.ones(1)*wiggle[-1])  # extend with one item

        minimum_amp.append(wiggle.min())
        time_at_minimum = twt[wiggle.argmin()]
        maximum_amp.append(wiggle.max())
        time_at_maximum = twt[wiggle.argmax()]
        apparent_thickness.append(np.abs(time_at_maximum - time_at_minimum))

        #print(this_wedge_thickness)
        #print(twt.min(), twt.max())
        #print(vp.min(), vp.max())
        #ax.plot((i+1) + vp/1000., twt[::-1])
        wiggle_plot(ax_wedge, twt, wiggle, i, scaling=10, fill_pos_style='pos-blue', fill_neg_style='neg-red')

    ax_wedge.plot(range(number_of_traces), top)
    ax_wedge.plot(range(number_of_traces), base)
    ax_wedge.set_ylim(ax_wedge.get_ylim()[::-1])

    ax_thickness.plot(range(number_of_traces), wedge_thickness)
    ax_thickness.plot(range(number_of_traces), apparent_thickness)

    ax_amplitude = ax_thickness.twinx()
    ax_amplitude.plot(range(number_of_traces), minimum_amp)
    ax_amplitude.plot(range(number_of_traces), maximum_amp)

    _, _ = model_plot.plot_model(ax_model, None, vps, vss, rhos,
                      [buffer, up_to_thickness, buffer], t0)
    extended_wavelet = np.zeros(len(twt))
    top_index = np.argmin(np.sqrt((twt - t0 - buffer)**2))
    extended_wavelet[top_index:(top_index+len(wavelet))] = wavelet
    ax_wavelet = ax_model.twiny()
    ax_wavelet.plot(extended_wavelet, twt)

    plt.show()


if __name__ == '__main__':
    _vps = [3500., 3600., 3500.]  # Vp [m/s] in the different layers
    _vss = [1800., 2100., 1800.]
    _rhos = [2.6, 2.3, 2.6]
    _time_step = 0.001  # s
    _center_f = 12.
    _duration = 0.128
    w = ricker(_duration, _time_step, _center_f)
    wedge_modelling(_vps, _vss, _rhos, 0.1, 15., w, time_step=_time_step, buffer=0.1, number_of_traces=10)
