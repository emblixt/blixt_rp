"""
https://github.com/seg/tutorials-2017/blob/master/1706_Seismic_rock_physics/seismic_rock_physics.ipynb


ricker from bruges.wavelets
https://github.com/agile-geoscience/bruges/blob/master/bruges/filters/wavelets.py
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

import blixt_utils.misc.curve_fitting as mycf
from blixt_rp import rp as rp


def straight_line(x, a, b):
    return a*x + b


def twolayer(vp0, vs0, rho0, vp1, vs1, rho1, angels=None):
    #from bruges.reflection import shuey2
    #from bruges.filters import ricker
    from blixt_rp.rp.rp_core import reflectivity, intercept, gradient
    if angels is None:
        angels = [5, 15, 30]

    n_samples = 500
    interface = int(n_samples / 2)
    ang = np.arange(31)
    wavelet = ricker(.25, 0.001, 25)

    model_ip, model_vpvs, rc0, rc1, rc2 = (np.zeros(n_samples) for _ in range(5))
    model_z = np.arange(n_samples)
    model_ip[:interface] = vp0 * rho0
    model_ip[interface:] = vp1 * rho1
    model_vpvs[:interface] = np.true_divide(vp0, vs0)
    model_vpvs[interface:] = np.true_divide(vp1, vs1)

    #avo = shuey2(vp0, vs0, rho0, vp1, vs1, rho1, ang)
    _avo = reflectivity(vp0, vp1, vs0, vs1, rho0, rho1, version='ShueyAkiRich')
    avo = _avo(ang)
    rc0[interface] = avo[angels[0]]
    rc1[interface] = avo[angels[1]]
    rc2[interface] = avo[angels[2]]
    synt0 = np.convolve(rc0, wavelet, mode='same')
    synt1 = np.convolve(rc1, wavelet, mode='same')
    synt2 = np.convolve(rc2, wavelet, mode='same')
    clip = np.max(np.abs([synt0, synt1, synt2]))
    clip += clip * .2

    ic = intercept(vp0, vp1, rho0, rho1)
    gr = gradient(vp0, vp1, vs0, vs1, rho0, rho1)

    opz0 = {'color': 'b', 'linewidth': 4}
    opz1 = {'color': 'k', 'linewidth': 2}
    opz2 = {'linewidth': 0, 'alpha': 0.5}
    opz3 = {'color': 'tab:red', 'linewidth': 0, 'markersize': 8, 'marker': 'o'}
    opz4 = {'color': 'tab:blue', 'linewidth': 0, 'markersize': 8, 'marker': 'o'}

    f = plt.subplots(figsize=(12, 10))
    ax0 = plt.subplot2grid((2, 12), (0, 0), colspan=3)  # ip
    ax1 = plt.subplot2grid((2, 12), (0, 3), colspan=3)  # vp/vs
    ax2 = plt.subplot2grid((2, 12), (0, 6), colspan=2)  # synthetic @ 0 deg
    ax3 = plt.subplot2grid((2, 12), (0, 8), colspan=2)  # synthetic @ 30 deg
    ax35 = plt.subplot2grid((2, 12), (0, 10), colspan=2)  # synthetic @ 30 deg
    ax4 = plt.subplot2grid((2, 12), (1, 0), colspan=5)  # avo curve
    ax6 = plt.subplot2grid((2, 12), (1, 7), colspan=5)  # avo curve


    ax0.plot(model_ip, model_z, **opz0)
    ax0.set_xlabel('IP')
    ax0.locator_params(axis='x', nbins=2)

    ax1.plot(model_vpvs, model_z, **opz0)
    ax1.set_xlabel('VP/VS')
    ax1.locator_params(axis='x', nbins=2)

    ax2.plot(synt0, model_z, **opz1)
    ax2.plot(synt0[interface], model_z[interface], **opz3)
    ax2.fill_betweenx(model_z, 0, synt0, where=synt0 > 0, facecolor='black', **opz2)
    ax2.set_xlim(-clip, clip)
    ax2.set_xlabel('angle={:.0f}'.format(angels[0]))
    ax2.locator_params(axis='x', nbins=2)

    ax3.plot(synt1, model_z, **opz1)
    ax3.plot(synt1[interface], model_z[interface], **opz3)
    ax3.fill_betweenx(model_z, 0, synt1, where=synt1 > 0, facecolor='black', **opz2)
    ax3.set_xlim(-clip, clip)
    ax3.set_xlabel('angle={:.0f}'.format(angels[1]))
    ax3.locator_params(axis='x', nbins=2)

    ax35.plot(synt2, model_z, **opz1)
    ax35.plot(synt2[interface], model_z[interface], **opz3)
    ax35.fill_betweenx(model_z, 0, synt2, where=synt2 > 0, facecolor='black', **opz2)
    ax35.set_xlim(-clip, clip)
    ax35.set_xlabel('angle={:.0f}'.format(angels[2]))
    ax35.locator_params(axis='x', nbins=2)

    ax4.plot(ang, avo, **opz0)
    ax4.axhline(0, color='k', lw=2)
    ax4.set_xlabel('angle of incidence')
    ax4.tick_params(which='major', labelsize=8)

    ax5 = ax4.twinx()
    color = 'tab:red'
    ax5.plot(angels, [s[interface] for s in [synt0, synt1, synt2]], **opz3)
    ax5.set_ylabel('Seismic amplitude')
    ax5.tick_params(axis='y', labelcolor=color, labelsize=8)

    # Calculate intercept & gradient based on the three "angle stacks"1G
    res = least_squares(
        mycf.residuals,
        [1., 1.],
        args=(np.array(angels), np.array([s[interface] for s in [synt0, synt1, synt2]])),
        kwargs={'target_function': straight_line}
    )
    print('amp = {:.4f}*theta + {:.4f}'.format(*res.x))
    print(res.status)
    print(res.message)
    print(res.success)
    ax5.plot(ang, straight_line(ang, *res.x), c='tab:red')

    res2 = least_squares(
        mycf.residuals,
        [1., 1.],
        args=(np.sin(np.array(angels)*np.pi/180.)**2, np.array([s[interface] for s in [synt0, synt1, synt2]])),
        kwargs={'target_function': straight_line}
    )
    print('amp = {:.4f}*theta + {:.4f}'.format(*res2.x))
    print(res2.status)
    print(res2.message)
    print(res2.success)


    ax6.plot(ic, gr, **opz4)
    ax6.plot(*res2.x[::-1], **opz3)
    ax6.set_xlabel('Intercept')
    ax6.set_ylabel('Gradient')

    for aa in [ax0, ax1, ax2, ax3, ax35]:
        aa.set_ylim(350, 150)
        aa.tick_params(which='major', labelsize=8)
        aa.set_yticklabels([])

    plt.subplots_adjust(wspace=.8, left=0.05, right=0.95)


def ricker(duration, dt, f, return_t=False):
    """
    Also known as the mexican hat wavelet, models the function:

    .. math::
        A =  (1 - 2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}
    If you pass a 1D array of frequencies, you get a wavelet bank in return.
    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (often one of  0.001, 0.002,
            or 0.004).
        f (ndarray): Centre frequency of the wavelet in Hz. If a sequence is
            passed, you will get a 2D array in return, one row per frequency.
        return_t (bool): If True, then the function returns a tuple of
            wavelet, time-basis, where time is the range from -duration/2 to
            duration/2 in steps of dt.
    Returns:
        ndarray. Ricker wavelet(s) with centre frequency f sampled on t.
    .. plot::
        plt.plot(bruges.filters.ricker(.5, 0.002, 40))

    """
    f = np.asanyarray(f).reshape(-1, 1)
    t = np.arange(-duration / 2, duration / 2, dt)
    pft2 = (np.pi * f * t) ** 2
    w = np.squeeze((1 - (2 * pft2)) * np.exp(-pft2))

    if return_t:
        RickerWavelet = namedtuple('RickerWavelet', ['amplitude', 'time'])
        return RickerWavelet(w, t)
    else:
        return w


def test_synt():
    """
    https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb
    :return:
    """
    import blixt_utils.io.io as uio
    from blixt_rp import plotting as ppl
    from blixt_rp.core.well import Project
    from blixt_rp.core.well import Well
    import blixt_utils.misc.convert_data as ucd

    wp = Project()
    well_table = {os.path.join(wp.working_dir, 'test_data/L-30.las'):
                      {'Given well name': 'WELL_L',
                       'logs': {
                           'dt': 'Sonic',
                           'cald': 'Caliper',
                           'cals': 'Caliper',
                           'rhob': 'Density',
                           'grd': 'Gamma ray',
                           'grs': 'Gamma ray',
                           'ild': 'Resistivity',
                           'ilm': 'Resistivity',
                           'll8': 'Resistivity',
                           'nphils': 'Neutron density'},
                       'Note': 'Some notes for well A'}}
    wis = uio.project_working_intervals(wp.project_table)
    w = Well()
    w.read_well_table(
        well_table,
        0,
        block_name='Logs')

    depth = w.block['Logs'].logs['depth'].data / 3.28084  # feet to m
    rho_orig = w.block['Logs'].logs['rhob'].data * 1000.  # g/cm3 to kg/m3
    vp_orig = ucd.convert(w.block['Logs'].logs['dt'].data, 'us/ft', 'm/s')
    dt_orig = w.block['Logs'].logs['dt'].data * 3.2804  # convert usec/ft to usec/m

    #
    # Start of copying the notebook results:
    # https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb
    #
    def f2m(item_in_feet):
        "converts feet to meters"
        try:
            return item_in_feet / 3.28084
        except TypeError:
            return float(item_in_feet) / 3.28084

    kb = f2m(w.header.kb.value)
    water_depth = f2m(w.header.gl.value)  # has a negative value
    #top_of_log = f2m(w.block['Logs'].header.strt.value)
    top_of_log = f2m(1150.5000)  # The DT log starts at this value
    repl_int = top_of_log - kb + water_depth
    water_vel = 1480  # m/s
    EGL_time = 2.0 * np.abs(kb)/water_vel
    #water_twt = 2.0 * abs(water_depth + EGL_time) / water_vel # ORIG THIS SEEMS LIKE MIXING distance with time!
    water_twt = 2.0 * abs(water_depth + np.abs(kb)) / water_vel  # My version
    repl_vel = 1600.  # m/s
    repl_time = 2. * repl_int / repl_vel
    log_start_time = water_twt + repl_time

    print('KB elevation: {} [m]'.format(kb))
    print('Seafloor elevation: {} [m]'.format(water_depth))
    print('Ground level time above SRD: {} [s]'.format(EGL_time))
    print('Water time: {} [s]'.format(water_twt))
    print('Top of Sonic log: {} [m]'.format(top_of_log))
    print('Replacement interval: {} [m]'.format(repl_int))
    print('Two-way replacement time: {} [s]'.format(repl_time))
    print('Top-of-log starting time: {} [s]'.format(log_start_time))

    def tvdss(md):
        # Assumes a vertical well
        # md in meter
        return md - kb

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return rolled

    #plt.figure(figsize=(18, 4))
    #plt.plot(depth, rho_sm, 'b', depth, rho, 'k', alpha=0.5)

    def despike(curve, curve_sm, max_clip):
        spikes = np.where(curve - curve_sm > max_clip)[0]
        spukes = np.where(curve_sm - curve > max_clip)[0]
        out = np.copy(curve)
        out[spikes] = curve_sm[spikes] + max_clip  # Clip at the max allowed diff
        out[spukes] = curve_sm[spukes] - max_clip  # Clip at the min allowed diff
        return out

    window = 13  # the length of filter is 13 samples or ~ 2 metres

    # Density
    rho_sm = np.median(rolling_window(rho_orig, window), -1)
    rho_sm = np.pad(rho_sm, int(window / 2), mode='edge')
    rho = despike(rho_orig, rho_sm, max_clip=100)
    rho_test = w.block['Logs'].logs['rhob'].despike(0.1) * 1000.  # g/cm3 to kg/m3


    # dt
    dt_sm = np.median(rolling_window(dt_orig, window), -1)
    dt_sm = np.pad(dt_sm, int(window/2), mode='edge')
    dt = despike(dt_orig, dt_sm, max_clip=10)

    # My test of despiking the velocity directly
    vp_sm = np.median(rolling_window(vp_orig, window), -1)
    vp_sm = np.pad(vp_sm, int(window/2), mode='edge')
    vp = despike(vp_orig, vp_sm, max_clip=200)


    # Plot result of despiking
    start = 13000; end = 14500
    plot = True
    if plot:
        plt.figure(figsize=(18, 4))
        plt.plot(depth[start:end], rho_orig[start:end], 'b', lw = 3)
        #plt.plot(depth[start:end], rho_sm[start:end], 'b')
        plt.plot(depth[start:end], rho[start:end], 'r', lw=2)
        plt.plot(depth[start:end], rho_test[start:end], 'k--')
        plt.title('de-spiked density')

        plt.figure(figsize=(18, 4))
        plt.plot(depth[start:end], dt_orig[start:end], 'k')
        plt.plot(depth[start:end], dt_sm[start:end], 'b')
        plt.plot(depth[start:end], dt[start:end], 'r')
        plt.title('de-spiked sonic')

        #plt.figure(figsize=(18, 4))
        #plt.plot(depth[start:end], vp_orig[start:end], 'k')
        #plt.plot(depth[start:end], vp_sm[start:end], 'b')
        #plt.plot(depth[start:end], vp[start:end], 'r')
        #plt.title('de-spiked Vp')


    # Compute the time-depth relationship
    # two-way-time to depth relationship
    scaled_dt = 0.1524 * np.nan_to_num(dt) / 1.e6  # scale the sonic log by the sample interval (6 inches or 0.1524 m)
                                                   # and go from usec to sec
    tcum = 2 * np.cumsum(scaled_dt)  # integration
    tdr = tcum + log_start_time
    print(tdr[:10], tdr[-10:])

    # Compute acoustic impedance
    ai = (1e6 / dt) * rho

    # Compute reflection
    rc = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])

    # Compute reflection "my way"
    r0 = rp.intercept(vp, None, rho, None, along_wiggle=True)
    #  The difference between rc and r0 lies basically in the difference in smoothing and clipping of dt vs. vp

    plot = False
    if plot:
        plt.figure(figsize=(18, 4))
        plt.plot(depth[start:end], rc[start:end], 'k')
        plt.plot(depth[start:end], r0[start:end], 'b')
        plt.title('Comparison of reflection coefficients')
        plt.legend(['Notebook way', 'My way'])

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    tops = {}
    for _key in list(wis['WELL_L'].keys()):
        tops[_key] = wis['WELL_L'][_key][0]
    tops_twt = {}
    for _key in list(wis['WELL_L'].keys()):
        tops_twt[_key] = tdr[find_nearest(depth, wis['WELL_L'][_key][0])]

    # RESAMPLING FUNCTION
    t_step = 0.004
    max_t = 3.0
    t = np.arange(0, max_t, t_step)
    ai_t = np.interp(x=t, xp=tdr, fp=ai)
    rc_t = (ai_t[1:] - ai_t[:-1]) / (ai_t[1:] + ai_t[:-1])

    # Compute the depth-time relation
    dtr = np.array([depth[find_nearest(tdr, tt)] for tt in t])

    # Define a Ricker wavelet
    def ricker(_f, _length, _dt):
        _t = np.linspace(-_length / 2, (_length - _dt) / 2, _length / _dt)
        _y = (1. - 2. * (np.pi ** 2) * (_f ** 2) * (_t ** 2)) * np.exp(-(np.pi ** 2) * (_f ** 2) * (_t ** 2))
        return _t, _y

    # Do the convolution
    rc_t = np.nan_to_num(rc_t)
    tw, w = ricker(_f=25, _length=0.512, _dt=0.004)
    synth = np.convolve(w, rc_t, mode='same')

    plot = False
    if plot:
        f2 = plt.figure(figsize=[10, 12])

        ax1 = f2.add_axes([0.05, 0.1, 0.2, 0.9])
        ax1.plot(ai, depth, 'k', alpha=0.75)
        ax1.set_title('impedance')
        ax1.set_ylabel('measured depth ' + '$[m]$', fontsize='12')
        ax1.set_xlabel(r'$kg/m^2s^2$ ', fontsize='16')
        ax1.set_ylim(0, 4500)
        ax1.set_xticks([0.0e7, 0.5e7, 1.0e7, 1.5e7, 2.0e7])
        ax1.invert_yaxis()
        ax1.grid()

        ax2 = f2.add_axes([0.325, 0.1, 0.2, 0.9])
        ppl.wiggle_plot(ax2, dtr[:-1], synth, fill='pos')
        ax2.set_ylim(0, 4500)
        ax2.invert_yaxis()
        ax2.grid()

        ax3 = f2.add_axes([0.675, 0.1, 0.1, 0.9])
        ax3.plot(ai_t, t, 'k', alpha=0.75)
        ax3.set_title('impedance')
        ax3.set_ylabel('two-way time ' + '$[s]$', fontsize='12')
        ax3.set_xlabel(r'$kg/m^2s^2$ ', fontsize='16')
        ax3.set_ylim(0, 3)
        ax3.set_xticks([0.0e7, 0.5e7, 1.0e7, 1.5e7, 2.0e7])
        ax3.invert_yaxis()
        ax3.grid()

        ax4 = f2.add_axes([0.8, 0.1, 0.2, 0.9])
        ppl.wiggle_plot(ax4, t[:-1], synth, scaling=10, fill='pos')
        ax4.set_ylim(0, 3)
        ax4.invert_yaxis()
        ax4.grid()

        for top, depth in tops.items():
            f2.axes[0].axhline(y=float(depth), color='b', lw=2,
                               alpha=0.5, xmin=0.05, xmax=0.95)
            f2.axes[0].text(x=1e7, y=float(depth) - 0.015, s=top,
                            alpha=0.75, color='k',
                            fontsize='12',
                            horizontalalignment='center',
                            verticalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.5, lw=0.5),
                            weight='light')
        for top, depth in tops.items():
            f2.axes[1].axhline(y=float(depth), color='b', lw=2,
                               alpha=0.5, xmin=0.05, xmax=0.95)

        for i in range(2, 4):
            for twt in tops_twt.values():
                f2.axes[i].axhline(y=float(twt), color='b', lw=2,
                                   alpha=0.5, xmin=0.05, xmax=0.95)


def test_synt2():
    """
    This essentially tries to copy test_synt(), which is based on
    https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb
    but using blixt_rp built in functionality instead
    :return:
    """
    import blixt_utils.io.io as uio
    from blixt_rp.core.well import Project

    wp = Project()
    wells = wp.load_all_wells()
    w = wells[list(wells.keys())[0]]  # take first well
    wis = uio.project_working_intervals(wp.project_table)
    log_table = {
        'P velocity': 'vp_dry',
        'S velocity': 'vs_dry',
        'Density': 'rho_dry'
    }

    # when input is in feet and usec
    #depth = ucd.convert(w.block['Logs'].logs['depth'].data, 'ft', 'm')
    #rho_orig = w.block['Logs'].logs['rhob'].data * 1000.  # g/cm3 to kg/m3
    #vp_orig = ucd.convert(w.block['Logs'].logs['dt'].data, 'us/ft', 'm/s')
    #dt_orig = w.block['Logs'].logs['dt'].data * 3.2804  # convert usec/ft to usec/m
    # else
    depth = w.block['Logs'].logs['depth'].data
    rho_orig = w.block['Logs'].logs[log_table['Density']].data * 1000.  # g/cm3 to kg/m3
    vp_orig = w.block['Logs'].logs[log_table['P velocity']].data
    vs_orig = w.block['Logs'].logs[log_table['S velocity']].data

    # when input is in feet and usec
    #rho = w.block['Logs'].logs['rhob'].despike(0.1) * 1000.  # g/cm3 to kg/m3
    #vp = ucd.convert(w.block['Logs'].logs['dt'].despike(5), 'us/ft', 'm/s')
    #dt = w.block['Logs'].logs['dt'].despike(5) * 3.2804  # convert usec/ft to usec/m
    # else
    rho = w.block['Logs'].logs[log_table['Density']].despike(0.1) * 1000.  # g/cm3 to kg/m3
    vp = w.block['Logs'].logs[log_table['P velocity']].despike(200)
    vs = w.block['Logs'].logs[log_table['S velocity']].despike(200)

    start = 13000; end = 14500
    # Plot despiking results
    plot = False
    if plot:
        fig, axes = plt.subplots(3,1, sharex=True, figsize=(18,12))
        for i, y1, y2 in zip([0,1,2], [rho_orig, vp_orig, vs_orig], [rho, vp, vs]):
            axes[i].plot(depth[start:end], y2[start:end], 'y', lw=3)
            axes[i].plot(depth[start:end], y1[start:end], 'k', lw=0.5)
            axes[i].legend(['Smooth & despiked', 'Original'])

    tdr = w.time_to_depth(log_table['P velocity'], debug=False)
    r0 = rp.intercept(vp, None, rho, None, along_wiggle=True)

    plot = True


if __name__ == '__main__':
    vp0 = 2430
    vs0 = 919
    rho0 = 2.11

    vp1 = 3032
    vs1 = 1543
    rho1 = 2.17

    #twolayer(vp0, vs0, rho0, vp1, vs1, rho1)
    test_synt2()

    plt.show()
