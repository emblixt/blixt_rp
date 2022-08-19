import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from itertools import cycle
from scipy.optimize import least_squares

import blixt_utils.misc.curve_fitting as mycf
import blixt_utils.io.io as uio
from blixt_utils.plotting.helpers import wiggle_plot
import blixt_utils.plotting.crossplot as xp

# global variables
logger = logging.getLogger(__name__)
clrs = list(mcolors.BASE_COLORS.keys())
clrs.remove('w')
cclrs = cycle(clrs)  # "infinite" loop of the base colors

def next_color():
    return next(cclrs)

def straight_line(x, a, b):
    return a*x + b

def avo_ig(amp, ang):
    """Calculates the Intercept and Gradient.
    
    Based on avo_IGn in
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/avo_attributes.ipynb
    which in turn is based on
    https://github.com/waynegm/OpendTect-External-Attributes/blob/master/Python_3/Jupyter/AVO_IG.ipynb

    """
    ang_rad = np.sin(np.radians(ang))**2
    m, resid, rank, singval= np.linalg.lstsq(np.c_[ang_rad,np.ones_like(ang_rad)], amp, rcond=None)
    # using only 2 angle stacks residuals  are not computed
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    if amp.shape[0]>2:
        qual = 1 - resid/(amp.shape[0] * np.var(amp,axis=0))
        return m[1],m[0],qual # intercept, gradient, quality factor
    else:
        return m[1],m[0] # intercept, gradient

def pickle_test_data():
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\SEISMIC DATA\KPSDM-NEAR_10deg_cropped.sgy"
    near, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\SEISMIC DATA\KPSDM-MID_18deg_cropped.sgy"
    mid, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\SEISMIC DATA\KPSDM-FAR_26deg_cropped.sgy"
    far, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

    inline = 6550
    xline = 27009
    near_line = near.sel(INLINE=inline)
    mid_line = mid.sel(INLINE=inline)
    far_line = far.sel(INLINE=inline)

    with open('C:\\Users\\marten\\Downloads\\angle_lines.dat', 'wb') as output:
        pickle.dump((twt, near_line, mid_line, far_line), output)

    near_trace = near.sel(INLINE=inline, XLINE=xline)
    mid_trace = mid.sel(INLINE=inline, XLINE=xline)
    far_trace = far.sel(INLINE=inline, XLINE=xline)

    with open('C:\\Users\\marten\\Downloads\\angle_traces.dat', 'wb') as output:
        pickle.dump((twt, near_trace, mid_trace, far_trace), output)


def test_ixg_plot():
    fig, axes = plt.subplots(nrows=1, ncols=2)

    with open('C:\\Users\\marten\\Downloads\\angle_traces.dat', 'rb') as input:
        twt, near_trace, mid_trace, far_trace = pickle.load(input)

    i, g, q = avo_ig(np.array([near_trace.data.flatten(), \
	mid_trace.data.flatten(), \
	far_trace.data.flatten()]), [10., 18., 26.])

    res = least_squares(
        mycf.residuals,
        [1.,1.],
        args=(i, g),
        kwargs={'target_function': straight_line})
    trend_line = 'WS = {:.4f}*I {:.4f} - G'.format(*res.x)
    print(res.status)
    print(res.message)
    print(res.success)
    print(res.fun.shape)

    xp.plot(i, g, cdata=res.fun,
       ctempl = {'full_name': 'Residual', 'colormap': 'seismic', 'min': np.min(res.fun), 'max':np.max(res.fun)}, 
       title='IL: 6550, XL: 27009\n{}'.format(trend_line), 
       fig=fig, ax=axes[0])

    x_new = np.linspace(*axes[0].get_xlim(), 50)
    axes[0].plot(x_new, straight_line(x_new, *res.x), c='r', label='_nolegend_')

    with open('C:\\Users\\marten\\Downloads\\angle_lines.dat', 'rb') as input:
        twt, near_line, mid_line, far_line = pickle.load(input)
    print('Line shape: {}'.format(near_line.data.shape))
    
    i, g, q = avo_ig(np.array([near_line.data.flatten(), \
	mid_line.data.flatten(), \
	far_line.data.flatten()]), [10., 18., 26.])

    res = least_squares(
            mycf.residuals,
            [1.,1.],
            args=(i, g),
            kwargs={'target_function': straight_line}
    )
    trend_line = 'WS = {:.4f}*I {:.4f} - G'.format(*res.x)
    print(res.status)
    print(res.message)
    print(res.success)
    print(res.fun.shape)

    xp.plot(i[::100], g[::100], cdata=res.fun[::100],
       ctempl = {'full_name': 'Residual', 'colormap': 'seismic', 'min': np.min(res.fun), 'max':np.max(res.fun)}, 
       title='IL: 6550\n{}'.format(trend_line), 
       fig=fig, ax=axes[1],
       edge_color=False)

    x_new = np.linspace(*axes[1].get_xlim(), 50)
    axes[1].plot(x_new, straight_line(x_new, *res.x), c='r', label='_nolegend_')

    fig, ax = plt.subplots()
    ax.imshow(np.transpose(res.fun.reshape((401, 501))),
              cmap='seismic',
              vmin=np.min(res.fun),
              vmax=np.max(res.fun),
              interpolation='spline16',
              aspect='auto',
              extent=(0,400,3000,1000))

    #X, Y = np.meshgrid(np.arange(0,401,1), twt)
    #ax.pcolormesh(X, Y, np.transpose(res.fun.reshape((401, 501))),
    #          cmap='seismic',
    #          vmin=np.min(res.fun),
    #          vmax=np.max(res.fun),
    #          )
    #ax.invert_yaxis()
    ax.grid(True)
    ax.set_title('Inline: 6550, Residual')
    ax.set_xlabel('X lines')
    ax.set_ylabel('TWT [ms]')
    plt.show()

def test_plot_amp_vs_offset():
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
    near, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-MID_18deg_cropped.sgy"
    mid, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-FAR_26deg_cropped.sgy"
    far, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

    inline = 6550
    xline = 27009
    t0 = 1245
    near_trace = near.sel(INLINE=inline, XLINE=xline)
    near_max = np.max(np.abs(near_trace.data))
    mid_trace = mid.sel(INLINE=inline, XLINE=xline)
    mid_max = np.max(np.abs(mid_trace.data))
    far_trace = far.sel(INLINE=inline, XLINE=xline)
    far_max = np.max(np.abs(far_trace.data))
    angs = np.array([10., 18., 26.])
    fig, axes = plt.subplots(nrows=1, ncols=2)


    for a, t in zip(angs, [near_trace, mid_trace, far_trace]):
        wiggle_plot(axes[0], twt, t.data, zero_at=a, scaling=10./max([near_max, mid_max, far_max]))

    axes[0].axhline(t0)

    ind = np.argmin(abs(near_trace.TWT.data - t0))

    amps = np.array([near_trace.data[ind], mid_trace.data[ind], far_trace.data[ind]])

    i, g, q = avo_ig(amps, angs)

    xp.plot(angs, amps, cdata='b', fig=fig, ax=axes[1])
    _angs = np.linspace(1, angs[-1]+10)
    axes[1].plot(_angs, i + g*np.sin(np.radians(_angs))**2)

    plt.show()

def test_plot_line():
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
    data, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
    print(header)
    uu={'add_colorbar':False,'robust':True,'interpolation':'spline16'}
    fig, ax = plt.subplots()
    data.plot.imshow(x='XLINE', y='TWT', yincrease=False, ax=ax, **uu)
    plt.show()

def test_amp_spectra():
    from blixt_rp.core.wavelets import plot_cwt
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
    data, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)

    inline = 6550
    xline = 27009
    sr0 = 0.004 # sample rate in seconds
    this_trace = data.sel(INLINE=inline, XLINE=xline)
    this_inline = data.sel(INLINE=inline)

    #fig, ax = plt.subplots()
    #ax.plot(this_trace.data, twt)
    #ax.invert_yaxis()
    #ax.set_title('IL {}, XL {}'.format(inline, xline))
    #ax.set_ylabel('TWT')

    
    #f1, amp1, f_peak1 = fullspec(this_inline.data, sr0)
    #f2, amp2 = ampspec(this_trace.data, sr0, smoothing='median')
    
    #plot_ampspecs([[f1, amp1, f_peak1], [f2, amp2]], ['Whole inline 6550', 'Smooth single trace'])

    #import pandas as pd
    #dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
    #df_nino = pd.read_table(dataset)
    #N = df_nino.shape[0]
    #t0=1871
    #sr0=0.25
    #time = np.arange(0, N) * sr0 + t0
    #signal = df_nino.values.squeeze()

    waveletname = 'cmor'
    desired_freqs = np.linspace(1, 1./(2 * sr0), 10)
    # scales, fs = freq2scale(desired_freqs, waveletname, sr0)
    scales = np.arange(1, 128)
    plot_cwt(twt/1000., this_trace.data, scales, waveletname=waveletname, cmap='jet')
    #plot_cwt(time/1000., signal, desired_freqs)
    # try extracting the scales corresponding to above frequencies
    #scales, fs = freq2scale(desired_freqs, 'morl', sr0)
    ##widths = np.linspace(2, 80, 41)  # pywt.scale2frequency('morl', 2)/0.004 = 101.6 which is less than 125
    #cwtmatr, freqs = pywt.cwt(this_trace.data, scales, 'morl', sr0)
    
    #X, Y = np.meshgrid(freqs, twt)

    #fig2, ax2 = plt.subplots()
    #ax2.contourf(X, Y, abs(cwtmatr.transpose()), 
    #           cmap='jet', 
    #           #vmax=abs(cwtmatr).max(), 
    #           #vmin=-abs(cwtmatr).max()
    #           vmax=np.max(abs(cwtmatr)), 
    #           vmin=np.min(abs(cwtmatr))
    #           )
    #ax2.invert_yaxis()
    #ax2.set_title('IL {}, XL {}'.format(inline, xline))
    #ax2.set_ylabel('TWT')
    #ax2.set_xlabel('Frequency [Hz]')
    #ax2.grid(True)
    plt.show()


if __name__ == '__main__':
    #test_plot_amp_vs_offset()
    #test_amp_spectra()
    test_ixg_plot()
    #pickle_test_data()
