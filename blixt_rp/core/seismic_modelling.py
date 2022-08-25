import matplotlib.pyplot as plt
import numpy
import numpy as np
import logging
from matplotlib.font_manager import FontProperties

# TODO remove this two lines, only used in testing
import sys
sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')
#sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')

from blixt_rp.core.models import Model, Layer
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
import blixt_rp.plotting.plot_layered_model as model_plot
from bruges.filters import ricker

logger = logging.getLogger(__name__)


def wedge_modelling(vps: list, vss: list, rhos: list, up_to_thickness: float, incident_angle: float,
                    wavelet: numpy.ndarray,
                    time_step=None,
                    t0=None,
                    buffer=None,
                    number_of_traces=None,
                    title=None):
    """
    Creates a wedge model of the given elastic parameters and plots the result
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
        TWT time in seconds at model top
    :param buffer:
        float
        Thickness in seconds of the buffer layer on top of the wedge
    :param number_of_traces:
        int
        Number of traces to plot in the wedge model
    :param title:
        str
        String to represent the title of the plot
    """

    if time_step is None:
        time_step = 0.001
    if t0 is None:
        t0 = 2.
    if buffer is None:
        buffer = 0.1
    if number_of_traces is None:
        number_of_traces = 100

    text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.7}}

    fig = plt.figure(figsize=(12, 8))
    # divide figure into a 3 by 6 grid
    spec = fig.add_gridspec(3, 6)
    ax_thickness = fig.add_subplot(spec[0, 0:4])  # row 0, column 0 to 4
    ax_wedge = fig.add_subplot(spec[1:3, 0:4])  # row 1 to 2, column 0 to 4
    ax_model = fig.add_subplot(spec[1:3, 4])  # row 1 to 2, column 5
    ax_wavelet = fig.add_subplot(spec[1:3, 5])  # row 1 to 2, column 6

    top = []
    base = []
    wedge_thickness = []
    minimum_amp = []
    maximum_amp = []
    apparent_thickness = []
    top_is_negative = False
    # create top layer, which is constant
    top_layer = Layer(vp=vps[0], vs=vss[0], rho=rhos[0], thickness=buffer)
    # loop over wedge thickness
    for i in range(number_of_traces):
        this_wedge_thickness = i * up_to_thickness / (number_of_traces - 1)
        wedge_layer = Layer(vp=vps[1], vs=vss[1], rho=rhos[1], thickness=this_wedge_thickness, target=True)
        base_layer = Layer(vp=vps[2], vs=vss[2], rho=rhos[2], thickness=buffer + (up_to_thickness - this_wedge_thickness))
        m = Model(layers=[top_layer, wedge_layer, base_layer])
        twt, vp, vs, rho = m.realize_model(time_step)
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
        if time_at_minimum < time_at_maximum:
            top_is_negative = True

        #print(this_wedge_thickness)
        #print(twt.min(), twt.max())
        #print(vp.min(), vp.max())
        #ax.plot((i+1) + vp/1000., twt[::-1])
        wiggle_plot(ax_wedge, twt, wiggle, i, scaling=20, fill_pos_style='pos-blue', fill_neg_style='neg-red')

    ax_wedge.plot(range(number_of_traces), top, 'r--')
    ax_wedge.plot(range(number_of_traces), base, 'b--')
    info_txt = '10ms = {:.1f}m'.format(0.01 * vps[1])
    print(t0 + 0.5 * (twt.max() - twt.min()))
    ax_wedge.text(0.8 * number_of_traces, t0 +  0.5 * (twt.max() - twt.min()), info_txt, **text_style)
    ax_wedge.set_ylim(ax_wedge.get_ylim()[::-1])
    labels = ax_wedge.get_xticks()
    new_labels = []
    for label in labels:
        new_labels.append('{:.0f}'.format(1000 * float(label) * up_to_thickness / float(labels[-2])))
    ax_wedge.set_xticklabels(new_labels)
    ax_wedge.set_xlabel('Wedge thickness [ms]')
    ax_wedge.set_ylabel('TWT [s]')

    ax_thickness.plot(range(number_of_traces), np.array(wedge_thickness) * 1000., label='True thickness')
    ax_thickness.plot(range(number_of_traces), np.array(apparent_thickness) * 1000., label='App. thickness')
    ax_thickness.set_xticklabels(new_labels)
    ax_thickness.set_ylabel('Wedge thickness [ms]')
    ax_thickness.legend(prop=FontProperties(size='smaller'), loc=4)

    ax_amplitude = ax_thickness.twinx()
    if top_is_negative:
        ax_amplitude.plot(range(number_of_traces), minimum_amp, 'r--', label='Trough amp.')
        ax_amplitude.plot(range(number_of_traces), maximum_amp, 'b--', label='Peak amp.')
    else:
        ax_amplitude.plot(range(number_of_traces), minimum_amp, 'b--', label='Trough amp.')
        ax_amplitude.plot(range(number_of_traces), maximum_amp, 'r--', label='Peak amp.')
    ax_amplitude.set_ylabel('Peak & trough amplitudes')
    ax_amplitude.legend(prop=FontProperties(size='smaller'), loc=7)

    m.plot(ax=ax_model)
    ax_model.set_ylim(ax_model.get_ylim()[::-1])
    ax_model.tick_params(labelleft=False)
    ax_model.set_title('Model')

    extended_wavelet = np.zeros(len(twt))
    top_index = int(0.5 * (len(extended_wavelet) - len(wavelet)))
    extended_wavelet[top_index:(top_index+len(wavelet))] = wavelet
    ax_wavelet.plot(extended_wavelet, twt)
    ax_wavelet.set_ylim(ax_wavelet.get_ylim()[::-1])
    ax_wavelet.tick_params(labelleft=False, labelbottom=False, labelright=True)
    ax_wavelet.grid(which='major', alpha=0.5)
    ax_wavelet.set_title('Wavelet')

    if title is not None:
        fig.suptitle(title)
    plt.show()


if __name__ == '__main__':
    _vps = [3500., 3600., 3500.]  # Vp [m/s] in the different layers
    _vss = [1800., 2100., 1800.]
    _rhos = [2.6, 2.3, 2.6]
    _time_step = 0.001  # s
    _center_f = 30.
    _duration = 0.128
    _up_to_thickness = 0.05
    _buffer = 0.05
    _incident_angle = 15.
    w = ricker(_duration, _time_step, _center_f)
    _title = 'Wedge model at {:.0f}$^\circ$ incidence, using a Ricker wavelet at {:.0f}Hz'.format(_incident_angle, _center_f)
    wedge_modelling(_vps, _vss, _rhos, _up_to_thickness, _incident_angle, w,
                    time_step=_time_step, buffer=_buffer, number_of_traces=30, title=_title)
