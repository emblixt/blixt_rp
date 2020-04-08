"""
https://github.com/seg/tutorials-2017/blob/master/1706_Seismic_rock_physics/seismic_rock_physics.ipynb

ricker from bruges.wavelets
https://github.com/agile-geoscience/bruges/blob/master/bruges/filters/wavelets.py
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import curve_fitting as mycf


def straight_line(x, a, b):
    return a*x + b


def twolayer(vp0, vs0, rho0, vp1, vs1, rho1, angels=None):
    #from bruges.reflection import shuey2
    #from bruges.filters import ricker
    from rp.rp_core import reflectivity, intercept, gradient
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

if __name__ == '__main__':
    vp0 = 2430
    vs0 = 919
    rho0 = 2.11

    vp1 = 3032
    vs1 = 1543
    rho1 = 2.17

    twolayer(vp0, vs0, rho0, vp1, vs1, rho1)

    plt.show()
