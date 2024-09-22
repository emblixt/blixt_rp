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

msymbols = np.array(['o','s','v','^','<','>','p','*','h','H','+','x','D','d','|','_','.','1','2','3','4','8'])
cnames = list(np.roll([str(u) for u in colors.cnames.keys()], 10))  # the first 10 elements have poor colors

# Global avo angle parameter, assuming 50 samples from 0 to 40 deg incidence angle
theta = np.linspace(0, 40, 50)


def straight_line(x, a, b):
    return a*x + b


def half_space_mc(sums, intfs, fbase=None, templates=None, suffix=None):
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
        avos, intercepts, gradients = execute_monte_carlo(sums, interface[1], interface[0], None, 1000,
                                                          model_type='half_space')
        _legend = '{} on {}'.format(interface[0], interface[1])
        plot_one_mc_result(avos, intercepts, gradients, _legend, interface[2], ax1, ax2)

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


def plot_one_half_space(sums, name1, name2, color, fig_ig, ax_ig, fig_refl, ax_refl, n_iter=1000):

    # elastics_from_stats calculates the normally distributed variables, with correlations, given
    # the mean, std and correlation, using a multivariate function
    vp1, vs1, rho1 = elastics_from_stats(sums[name1], n_iter)
    vp2, vs2, rho2 = elastics_from_stats(sums[name2], n_iter)

    refs = np.full((n_iter, 50), np.nan)
    # calculate the reflectivity as a function of theta for all variations of the elastic properties
    for i, params in enumerate(zip(vp1, vp2, vs1, vs2, rho1, rho2)):
        refs[i, :] = rp.reflectivity(*params)(theta)

    refl_stds = np.std(refs, 0)

    # Calculate the mean reflectivity curve
    mean_refl = rp.reflectivity(
        sums[name1]['VpMean'],
        sums[name2]['VpMean'],
        sums[name1]['VsMean'],
        sums[name2]['VsMean'],
        sums[name1]['RhoMean'],
        sums[name2]['RhoMean'],
    )

    # plot the mean reflectivity curve together with the uncertainty
    mypr.plot(theta, mean_refl(theta), c=color, yerror=refl_stds,
              yerr_style='fill', ax=ax_refl)

    intercept = rp.intercept(vp1, vp2, rho1, rho2)
    gradient = rp.gradient(vp1, vp2, vs1, vs2, rho1, rho2)

    #res = least_squares(
    #        mycf.residuals,
    #        [1.,1.],
    #        args=(intercept, gradient),
    #        kwargs={'target_function': straight_line}
    #)
    #print('{} on {}: WS = {:.4f}*I {:.4f} - G'.format(name1, name2, *res.x))
    #print(res.status)
    #print(res.message)
    #print(res.success)

    
    myxp.plot(
            intercept,
            gradient,
            cdata=color,
            ax=ax_ig,
            edge_color=None,
            alpha=0.2
            )
    #x_new = np.linspace(-0.75, 0.75, 50)
    #ax_ig.plot(x_new, straight_line(x_new, *res.x), c=color, label='_nolegend_')

    # Do AVO classification
    c1 = len(gradient[(intercept > 0.) & (gradient > -4*intercept) & (gradient < 0.)])
    c2p = len(gradient[(intercept > 0.) & (gradient < -4*intercept)])
    c2 = len(gradient[(intercept > -0.02) & (intercept < 0.) & (gradient < 0.)])
    c3 = len(gradient[(intercept < -0.02) & (gradient < 0.)])
    c4 = len(gradient[(intercept < 0.) & (gradient > 0.)])
    rest = len(gradient[(intercept > 0.) & (gradient > 0.)])
    print('\n{} on {}:'.format(name1, name2))
    print(' Class I: {:.0f}% \n Class IIp: {:.0f}% \n Class II: {:.0f}% \n Class III: {:.0f}% \n Class IV: {:.0f}%'.format(
            100.*c1/n_iter, 100.*c2p/n_iter, 100.*c2/n_iter, 100.*c3/n_iter, 100.*c4/n_iter))
    print(' Rest: {:.0f}%'.format(100.*rest/n_iter))


def execute_monte_carlo(sums_avg: dict, target_name:str, bg_name:str,
                        thickness: float | None, n_iter: int = 1000,
                        model_type: str = 'half_space') -> (np.ndarray, np.ndarray, np.ndarray):
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
    :param thickness:
        float
        Only used when model_type is NOT 'half_space'
    :param n_iter:
        Number of iterations
    :param model_type:
        What kind of model we use to extract the AVO attributes
        'half_space'
        'layered_model'

    :return:
    """
    # elastics_from_stats calculates the normally distributed variables, with correlations, given
    # the mean, std and correlation, using a multivariate function
    vp_t, vs_t, rho_t = elastics_from_stats(sums_avg[target_name], n_iter)
    vp_bg, vs_bg, rho_bg = elastics_from_stats(sums_avg[bg_name], n_iter)

    if model_type == 'half_space':
        refs = np.full((n_iter, 50), np.nan)
        # calculate the reflectivity as a function of theta for all variations of the elastic properties
        for i, params in enumerate(zip(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)):
            refs[i, :] = rp.reflectivity(*params)(theta)

        intercepts = rp.intercept(vp_bg, vp_t, rho_bg, rho_t)
        gradients = rp.gradient(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)

        return refs, intercepts, gradients
    else:
        raise NotImplementedError('Model type {} not implemented'.format(model_type))


def plot_one_mc_result(avos, intercepts, gradients, legend, color, ax_ig, ax_refl):
    """
    A generalization of 'plot_one_half_space' which is meant to replace it.

    Takes the Monte Carlo results from one model, extracted at one depth, and plots the result

    :param avos:
        np.ndarray
        size (n_iterations, n_theta)
        Contains the reflectivity (when using a half-space model) or AVO amplitude, extracted at one depth,
         as a function of theta for all iterations of the Monte Carlo simulation

    :param intercepts:
        np.ndarray
        size (n_iterations)
        Contains the intercept values extracted at one depth for all iterations of the Monte Carlo simulation
    :param gradients:
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
    n_iter = intercepts.shape[0]

    mean_avo = np.mean(avos, 0)
    std_avo = np.std(avos, 0)

    # plot the mean avo curve together with the uncertainty
    mypr.plot(theta, mean_avo, c=color, yerror=std_avo,
              yerr_style='fill', ax=ax_refl)

    # plot all intercept and gradient values in a IxG plot
    myxp.plot(
        intercepts,
        gradients,
        cdata=color,
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
                  wavelet, verbose=False):
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
    :param verbose:
    :return:
    """

    dt = wavelet['header']['Sample rate']  # should be given in seconds
    if dt > 0.1:
        print("WARNING: Sample rate most likely given in milliseconds. Please check")
    length = wavelet['time'][-1] - wavelet['time'][0]
    top_thickness = length / 2.

    model = build_layered_model(2., top_thickness, target_thickness,
                                background, target, background, domain='TWT')

    if verbose:
        laminar_model_analysis(model, dt, wavelet)
        plt.show()

    return model


def evaluate_layered_model(model, wavelet, extract_at, extract_on='exact', verbose=False):

    dt = wavelet['header']['Sample rate']  # should be given in seconds

    twt, layer_i, vp, vs, rho, this_z = model.realize_model(dt, voigt_reuss_hill=True)
    twt_index = np.argmin((twt - extract_at) ** 2)
    ref = rp.reflectivity(vp, None, vs, None, rho, None, along_wiggle=True)
    wiggle = bumw.convolve_with_refl(wavelet['wavelet'], ref(0))
    wiggle_value, twt_index = find_value(wiggle, twt_index, snap_to=extract_on)

    grad = rp.gradient(vp, None, vs, None, rho, None, along_wiggle=True)
    intercept = rp.intercept(vp, None,  rho, None,  along_wiggle=True)

    if verbose:
        laminar_model_analysis(model, dt, wavelet, extract_avo_at=(0., extract_at), extract_on=extract_on)
        plt.show()

    return intercept[twt_index], grad[twt_index]


def elastics_from_stats(layer_stats,  n_iter):
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

    vp = mvn[:,0].reshape((n_iter))
    vs = mvn[:,1].reshape((n_iter))
    rho = mvn[:,2].reshape((n_iter))

    return vp, vs, rho
