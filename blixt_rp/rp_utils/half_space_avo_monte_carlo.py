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

msymbols = np.array(['o','s','v','^','<','>','p','*','h','H','+','x','D','d','|','_','.','1','2','3','4','8'])
cnames = list(np.roll([str(u) for u in colors.cnames.keys()], 10))  # the first 10 elements have poor colors


def straight_line(x, a, b):
    return a*x + b


def plot_multi_interfaces(sums, intfs, fbase=None, templates=None, suffix=None):
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
    legend_labels = []
    for i, intf in enumerate(intfs):
        legend_labels.append(Line2D([0], [0], marker='o', color=intf[2], label=legends[i],
                          markerfacecolor=intf[2], lw=0, markersize=10))

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
        plot_one_interface(sums, *interface, fig1, ax1, fig2, ax2, n_samps=1000)

    ax1.plot([0, 0], [ymin, ymax], 'k--', lw=0.5, label='_nolegend_')
    ax1.plot([xmin, xmax], [0, 0], 'k--', lw=0.5, label='_nolegend_')

    this_legend = ax1.legend(
        handles=legend_labels,
        #legends,
        prop=FontProperties(size='smaller'),
        #scatterpoints = 1,
        #markerscale=1.5,
        loc=1
    )

    this_legend = ax2.legend(
        legends,
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


def plot_one_interface(sums, name1, name2, color, fig_ig, ax_ig, fig_refl, ax_refl, n_samps=1000):

    # elastics_from_stats calculates the normally distributed variables, with correlations, given
    # the mean, std and correlation, using a multivariate function
    vp1, vs1, rho1 = elastics_from_stats(sums[name1], n_samps)
    vp2, vs2, rho2 = elastics_from_stats(sums[name2], n_samps)

    # Calculate statistics of the reflection coefficient, assuming 50 samples from 0 to 40 deg incidence angle
    theta = np.linspace(0, 40, 50)
    refs = np.full((n_samps, 50), np.nan)
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
              yerr_style='fill', fig=fig_refl, ax=ax_refl)

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
            fig=fig_ig,
            ax=ax_ig,
            edge_color=None,
            alpha=0.2
            )
    #x_new = np.linspace(-0.75, 0.75, 50)
    #ax_ig.plot(x_new, straight_line(x_new, *res.x), c=color, label='_nolegend_')

    # Do AVO classification
    c1 = len(gradient[(intercept>0.) & (gradient>-4*intercept) & (gradient<0.)])
    c2p = len(gradient[(intercept>0.) & (gradient<-4*intercept)])
    c2 = len(gradient[(intercept>-0.02) & (intercept<0.) & (gradient<0.)])
    c3 = len(gradient[(intercept<-0.02) & (gradient<0.)])
    c4 = len(gradient[(intercept<0.) & (gradient>0.)])
    rest = len(gradient[(intercept>0.) & (gradient>0.)])
    print('Class I: {:.0f}% \nClass IIp: {:.0f}% \nClass II: {:.0f}% \nClass III: {:.0f}% \nClass IV: {:.0f}%'.format(
            100.*c1/n_samps, 100.*c2p/n_samps, 100.*c2/n_samps, 100.*c3/n_samps, 100.*c4/n_samps))
    print('Rest:  {:.0f}%'.format(100.*rest/n_samps))


def elastics_from_stats(layer_stats,  n_samps):
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
            [vp_mean, vs_mean, rho_mean], covar, n_samps)

    vp = mvn[:,0].reshape((n_samps))
    vs = mvn[:,1].reshape((n_samps))
    rho = mvn[:,2].reshape((n_samps))

    return vp, vs, rho