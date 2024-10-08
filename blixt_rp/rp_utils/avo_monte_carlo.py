# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:46:15 2019

Run Monte Carlo simulation of AVO for a
half space model with top "Layer 1" and base "Layer 2"

@author: mblixt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

import blixt_rp.rp.rp_core as rp
from blixt_utils.plotting import crossplot as myxp
from blixt_rp.plotting import plot_reflectivity as mypr

from blixt_rp.core.models import build_layered_model, laminar_model_analysis
import blixt_utils.misc.wavelets as bumw
from blixt_utils.utils import find_value

msymbols = np.array(['o', 's', 'v', '^', '<', '>', 'p', '*', 'h', 'H', '+', 'x',
                     'D', 'd', '|', '_', '.', '1', '2', '3', '4', '8'])
cnames = list(np.roll([str(u) for u in colors.cnames.keys()], 10))  # the first 10 elements have poor colors

# Global avo angle parameter, assuming 50 samples from 0 to 40 deg incidence angle
theta = np.linspace(0, 40, 50)


def straight_line(x, a, b):
    return a*x + b


def main(sums, intfs, fbase=None, templates=None, suffix=None,
         n_iter: int = 1000, thickness: tuple | None = None, wavelet: dict | None = None,
         extract_at: float = 2.0, extract_on: str = 'exact',
         model_type: str = 'half_space', verbose = False):
    """
    :param sums: 
        dict
        dictionary of sums and averages information containing the statistics from
        multiple layers
        
    :param intfs:
        list of lists
        Defines the interfaces to plot the elastic properties on
        Each item is a list of [toplayer_name, baselayer_name, color]
        where toplayer_name and baselayer_name must be a key in the sums dictionary
        E.G. 
        intfs = [['Knurr shales', 'Knurr sands', 'b'],
                 ['Knurr shales', 'Knurr oil sands', 'g'],
                 ['Knurr shales', 'Knurr gas sands', 'r']]
    :param fbase:
        str
        base of the file names is where the plots are saved
        if None, no plots are saved
    :param templates:
        dict
        templates dictionary as returned from rp_utils.io.project_templates()
    :param suffix:
        str
        Suffix added to output plots (png) to ease separating output from eachother
    """    

    if suffix is None:
        suffix = ''
    else:
        suffix = '_' + suffix

    fig1, ax1 = plt.subplots(figsize=(8,6))  # for plotting intercept vs gradient
    fig2, ax2 = plt.subplots(figsize=(8,6))  # for plotting reflectivity

    legends = ['{} on {}'.format(x[0], x[1]) for x in intfs]

    if (templates is not None) and ('Intercept' in list(templates.keys())):
        xmin = templates['Intercept']['min']
        xmax = templates['Intercept']['max']
    else:
        xmin = -0.75
        xmax = 0.75

    if (templates is not None) and ('Gradient' in list(templates.keys())):
        ymin = templates['Gradient']['min']
        ymax = templates['Gradient']['max']
    else:
        ymin = -0.75
        ymax = 0.75

    for interface in intfs:
        #  plot_one_half_space(sums, *interface, fig1, ax1, fig2, ax2, n_iter=1000)
        print('Running {} Monte Carlo for {} on {}'.format(model_type, interface[0], interface[1]))
        print(' Iterations: {}'.format(n_iter))
        print(' Thickness: {}'.format(thickness))
        if wavelet is not None:
            print(' Wavelet: {}'.format(wavelet['header']))
        print(' Extract at: {}'.format(extract_at))
        print(' Extract on: {}'.format(extract_on))
        result = execute_monte_carlo(sums, interface[1], interface[0], n_iter,
                                                          thickness=thickness, wavelet=wavelet,
                                                          extract_at=extract_at, extract_on=extract_on,
                                                          model_type=model_type, verbose=verbose)
        # print('XX1:', avos.shape, intercepts.shape, gradients.shape)  # XXX
        # print('XX2:', np.max(avos), np.max(intercepts), np.max(gradients))  # XXX

        _legend = '{} on {}'.format(interface[0], interface[1])
        plot_one_mc_result(result, _legend, interface[2], ax1, ax2)

    ax1.plot([0, 0], [ymin, ymax], 'k--', lw=0.5, label='_nolegend_')
    ax1.plot([xmin, xmax], [0, 0], 'k--', lw=0.5, label='_nolegend_')

    legend_labels = []
    for i, intf in enumerate(intfs):
        legend_labels.append(Line2D([0], [0], marker='o', color=intf[2], label=legends[i],
                                    markerfacecolor=intf[2], lw=0, markersize=10))
    this_legend = ax1.legend(
        handles=legend_labels,
        prop=FontProperties(size='smaller'),
        loc=1
    )

    legend_labels = []
    for i, intf in enumerate(intfs):
        legend_labels.append(Line2D([0], [0], color=intf[2], label=legends[i], lw=2))
    this_legend = ax2.legend(
        handles=legend_labels,
        # legends,
        prop=FontProperties(size='smaller'),
        loc=1
    )
    
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    
    ax1.set_xlabel('Intercept')
    ax1.set_ylabel('Gradient')
    ax2.set_xlabel('Incidence angle [$^{\circ}$]')
    ax2.set_ylabel('P-P Reflectivity')
    
    fig1.tight_layout()
    fig2.tight_layout()

    if fbase:
        fig1.savefig(os.path.join(
                fbase,
                '{}_IG{}.png'.format(
                        intfs[0][0].split(' ')[0], suffix)))
        fig2.savefig(os.path.join(
                fbase,
                '{}_refl{}.png'.format(
                        intfs[0][0].split(' ')[0], suffix)))
    else:
        plt.show()


def execute_monte_carlo(sums_avg: dict, target_name: str, bg_name: str, n_iter: int = 1000,
                        thickness: tuple | None = None, wavelet: dict | None = None,
                        extract_at: float = 2.0, extract_on: str = 'exact',
                        model_type: str = 'half_space', verbose=False) -> dict:
    """

    :param sums_avg:
        dict
        dictionary of sums and averages information containing the statistics from different intervals (formations, ...)
        collected over one or more wells.
        See read_sums_and_averages in blixt_utils.io.io
    :param target_name:
        str
        Name that identifies which member of the sums_avg dictionary we should use as target interval
    :param bg_name:
        str
        Name that identifies which member of the sums_avg dictionary we should use as background
    :param n_iter:
        Number of iterations
    :param thickness:
        tuple
        two tuple with mean thickness and std of thickness variation
        IGNORED when model_type is 'half_space'
    :param wavelet:
        dict
        dictionary with three keys:
            'wavelet': contains the wavelet amplitude
            'time': contains the time data [s]
            'header': a dictionary with info about the wavelet
        see blixt_utils.io.io.read_petrel_wavelet() for example
        IGNORED when model_type is 'half_space'
    :param extract_at:
        float
        IGNORED when model_type is 'half_space'
        TWT in seconds to where the amplitudes are to be extracted.
        Depending on 'extract_on' this depth can be slightly shifted to find the nearest min or max
    :param extract_on:
        str
        IGNORED when model_type is 'half_space'
        Used as input to find_value() to determine if the amplitudes extracted 'exact' at given 'extract_at' depth,
        or on the nearest min or max
    :param model_type:
        What kind of model we use to extract the AVO attributes
        'half_space'
        'layered_model'
    :param verbose:
        bool

    :return:
    """
    # elastics_from_stats calculates the normally distributed variables, with correlations, given
    # the mean, std and correlation, using a multivariate function
    vp_t, vs_t, rho_t = elastics_from_stats(sums_avg[target_name], n_iter)
    vp_bg, vs_bg, rho_bg = elastics_from_stats(sums_avg[bg_name], n_iter)
    avos = np.full((n_iter, 50), np.nan)

    if model_type == 'half_space':
        # calculate the reflectivity as a function of theta for all variations of the elastic properties
        for i, params in enumerate(zip(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)):
            avos[i, :] = rp.reflectivity(*params)(theta)

        intercepts = rp.intercept(vp_bg, vp_t, rho_bg, rho_t)
        gradients = rp.gradient(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)

        return {'amplitude': avos,
                'intercept': intercepts,
                'gradient': gradients}

    elif model_type == 'layered_model':
        thicknesses = np.random.normal(loc=thickness[0], scale=thickness[1], size=n_iter)
        nn = int(n_iter/2)
        intercepts = []
        gradients = []
        for i in range(n_iter):
            if verbose:
                _verbose = np.mod(i, nn) == 0
            else:
                _verbose = False
            target = {'vp': vp_t[i], 'vs': vs_t[i], 'rho': rho_t[i]}
            background = {'vp': vp_bg[i], 'vs': vs_bg[i], 'rho': rho_bg[i]}
            m = layered_model(thicknesses[i], target, background, wavelet, verbose=False)
            # _avo, _int, _grad = evaluate_layered_model(m, wavelet, extract_at, extract_on, _verbose)
            result = evaluate_layered_model(m, wavelet, extract_at, extract_on, _verbose)
            avos[i, :] = result['amplitude']
            intercepts.append(result['intercept'])
            gradients.append(result['gradient'])

        return {'amplitude': avos,
                'intercept': np.array(intercepts),
                'gradient': np.array(gradients),
                'thickness': thicknesses}

    else:
        raise NotImplementedError('Model type {} not implemented'.format(model_type))


def plot_one_mc_result(mc_results, legend, color, ax_ig, ax_refl):
    """
    A generalization of 'plot_one_half_space' which is meant to replace it.

    Takes the Monte Carlo results from one model, extracted at one depth, and plots the result

    :param mc_results:
        dict
        key = 'amplitude':
            np.ndarray
            size (n_iterations, n_theta)
            Contains the reflectivity (when using a half-space model) or AVO amplitude, extracted at one depth,
             as a function of theta for all iterations of the Monte Carlo simulation
        key = 'intercept':
            np.ndarray
            size (n_iterations)
            Contains the intercept values extracted at one depth for all iterations of the Monte Carlo simulation
        key = 'gradient':
            np.ndarray
            size (n_iterations)
            Contains the gradient values extracted at one depth for all iterations of the Monte Carlo simulation
    :param legend:
        str
        String that identifies this Monte Carlo simulation from others
    :param color:
        str
        Color that identifies this Monte Carlo simulation from others
    :param ax_ig:
        matplotlib Axes
        Axes that holds the Intercept vs Gradient plot
    :param ax_refl:
        matplotlib Axes
        Axes that plots the reflectivity, or amplitude, as a function of incidence angle theta
    :return:
    """
    intercepts = mc_results['intercept']
    gradients = mc_results['gradient']
    n_iter = intercepts.shape[0]

    mean_avo = np.mean(mc_results['amplitude'], 0)
    std_avo = np.std(mc_results['amplitude'], 0)

    # plot the mean avo curve together with the uncertainty
    mypr.plot(theta, mean_avo, c=color, yerror=std_avo,
              yerr_style='fill', ax=ax_refl)

    # plot all intercept and gradient values in a IxG plot
    if 'thickness' in list(mc_results.keys()):
        point_size = mc_results['thickness']
    else:
        point_size = None
    myxp.plot(
        intercepts,
        gradients,
        cdata=color,
        pdata = point_size,
        ax=ax_ig,
        edge_color=None,
        alpha=0.2
    )

    # Do AVO classification
    c1 = len(gradients[(intercepts > 0.) & (gradients > -4*intercepts) & (gradients < 0.)])
    c2p = len(gradients[(intercepts > 0.) & (gradients < -4*intercepts)])
    c2 = len(gradients[(intercepts > -0.02) & (intercepts < 0.) & (gradients < 0.)])
    c3 = len(gradients[(intercepts < -0.02) & (gradients < 0.)])
    c4 = len(gradients[(intercepts < 0.) & (gradients > 0.)])
    rest = len(gradients[(intercepts > 0.) & (gradients > 0.)])
    print('\n{}:'.format(legend))
    print(' Class I: {:.0f}% \n Class IIp: {:.0f}% \n Class II: {:.0f}% \n Class III: {:.0f}% \n Class IV: {:.0f}%'.format(
        100.*c1/n_iter, 100.*c2p/n_iter, 100.*c2/n_iter, 100.*c3/n_iter, 100.*c4/n_iter))
    print(' Rest: {:.0f}%'.format(100.*rest/n_iter))


def layered_model(target_thickness, target,  background,
                  wavelet, depth_to_target=2.0, verbose=False):
    """
    Creates one simple 3 layered model, with a target of thickness 'target_thickness' embedded in
    background

    :param target_thickness:
        float
        twt thickness in ms
    :param target:
        dict
        with keys: 'vp', 'vs', and 'rho', with the Vp [m/s], Vs [m/s] and Density [gr/cm3] as values
    :param background:
        dict
        with keys: 'vp', 'vs', and 'rho'
    :param wavelet:
            dict
            dictionary with three keys:
                'wavelet': contains the wavelet amplitude
                'time': contains the time data [s]
                'header': a dictionary with info about the wavelet
            see blixt_utils.io.io.read_petrel_wavelet() for example
    :param depth_to_target:
        float
        TWT value in seconds to top of target
    :param verbose:
    :return:
    """

    dt = wavelet['header']['Sample rate']  # should be given in seconds
    if dt > 0.1:
        print("WARNING: Sample rate most likely given in milliseconds. Please check")
    length = wavelet['time'][-1] - wavelet['time'][0]
    top_thickness = length / 2.

    model = build_layered_model(depth_to_target, top_thickness, target_thickness,
                                background, target, background, domain='TWT')

    if verbose:
        laminar_model_analysis(model, dt, wavelet)
        plt.show()

    return model


def evaluate_layered_model(model, wavelet, extract_at, extract_on='exact', verbose=False):
    """

    :param model:
    :param wavelet:
    :param extract_at:
        float
        TWT in seconds to where the amplitudes are to be extracted.
        Depending on 'extract_on' this depth can be slightly shifted to find the nearest min or max
    :param extract_on:
        str
        Used as input to find_value() to determine if the amplitudes extracted 'exact' at given 'extract_at' depth,
        or on the nearest min or max
    :param verbose:
    :return:
    """
    def signed_max_amplitude(_x, _center, _width):
        _data = _x[_center - _width:_center + _width]
        _max = np.max(_data)
        _min = np.min(_data)
        if np.abs(_max) > np.abs(_min):
            return _max
        else:
            return _min

    dt = wavelet['header']['Sample rate']  # should be given in seconds

    twt, layer_i, vp, vs, rho, this_z = model.realize_model(dt, voigt_reuss_hill=True)
    twt_index = np.argmin((twt - extract_at) ** 2)
    ref = rp.reflectivity(vp, None, vs, None, rho, None, along_wiggle=True)
    wiggles = [bumw.convolve_with_refl(wavelet['wavelet'], ref(_theta)) for _theta in theta]

    wiggle_value, twt_index = find_value(wiggles[0], twt_index, snap_to=extract_on)

    wiggle_values = np.array([this_wiggle[twt_index] for this_wiggle in wiggles])

    grad = rp.gradient(vp, None, vs, None, rho, None, along_wiggle=True)
    intercept = rp.intercept(vp, None,  rho, None,  along_wiggle=True)
    # print('XX3:', intercept[twt_index-2:twt_index+2], grad[twt_index-2:twt_index+2])
    # print('XX4:', signed_max_amplitude(intercept, twt_index, 2), signed_max_amplitude(grad, twt_index, 2))

    if verbose:
        laminar_model_analysis(model, dt, wavelet, extract_avo_at=(0., extract_at), extract_on=extract_on)
        plt.show()

    return {'amplitude': wiggle_values,
            'intercept': signed_max_amplitude(intercept, twt_index, 2),
            'gradient': signed_max_amplitude(grad, twt_index, 2)}


def elastics_from_stats(layer_stats, n_iter: int):
    """
    For the given statistical properties (mean, standard deviation and correlation coefficient) of the
    elastic variables (Vp, Vs and Rho) it returns normally
    distributed elastic properties Vp, Vs and Rho.
    :param layer_stats:
        dict
        where dict contains
            {'VpMean': XX,
             'VsMean': XX,
             'RhoMean': XX,
             'VpStdDev': XX,
             'VsStdDev': XX,
             'RhoStdDev': XX,
             'VpVsCorrCoef': XX,
             'VpRhoCorrCoef': XX,
             'VsRhoCorrCoef': XX}
    :param n_iter:
        int
        Number of iterations in the Monte Carlo simulation
    """
    # Take a look at the following:
    # https://www.linkedin.com/pulse/probabilistic-analysis-python-andre-cebastiant/?articleId=6465522109379633152
    # to investigate the effect of dependent variables

    vp_mean = layer_stats['VpMean']
    vs_mean = layer_stats['VsMean']
    rho_mean = layer_stats['RhoMean']
    vp_std = layer_stats['VpStdDev']
    vs_std = layer_stats['VsStdDev']
    rho_std = layer_stats['RhoStdDev']
    r_vp_vs = layer_stats['VpVsCorrCoef']
    r_vp_rho = layer_stats['VpRhoCorrCoef']
    r_vs_rho = layer_stats['VsRhoCorrCoef']

    # The multivariate variables that are correlated can be calculated using the
    # Covariance matrix

    covar = [[vp_std**2,                    vp_std*vs_std*r_vp_vs,        vp_std*rho_std*r_vp_rho],
            #
               [vp_std*vs_std*r_vp_vs,    vs_std**2,                        vs_std*rho_std*r_vs_rho],
            #
               [vp_std*rho_std*r_vp_rho,  vs_std*rho_std*r_vs_rho,      rho_std**2 ]]

    mvn = scipy.stats.multivariate_normal.rvs(
            [vp_mean, vs_mean, rho_mean], covar, n_iter)

    vp = mvn[:, 0].reshape((n_iter))
    vs = mvn[:, 1].reshape((n_iter))
    rho = mvn[:, 2].reshape((n_iter))

    return vp, vs, rho
