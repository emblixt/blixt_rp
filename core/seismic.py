import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from itertools import cycle
from scipy.optimize import least_squares

import segyio
import xarray as xr
import pywt

import blixt_utils.misc.curve_fitting as mycf
from core.wavelets import freq2scale
from plotting.plot_logs import wiggle_plot
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

def read_segy(f, lag=0, twod=False, byte_il=189, byte_xl=193):
    '''
    read_segy (C) aadm 2018 // using Statoil's segyio
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/playing_with_seismic.ipynb

    Slightly modified and upgraded by Erik Mårten Blixt 2020-08-19
    '''
    if twod:
        with segyio.open(filename, 'r', ignore_geometry=True) as segyfile:
            sr = segyio.tools.dt(segyfile)/1e3
            nsamples = segyfile.samples.size
            twt = segyfile.samples
            ntraces = segyfile.tracecount
            data = segyfile.trace.raw[:]
            header = segyio.tools.wrap(segyfile.text[0])      
    else:
        with segyio.open(f, iline=byte_il, xline=byte_xl) as segyfile:
            sr = segyio.tools.dt(segyfile)/1e3
            nsamples = segyfile.samples.size
            twt = segyfile.samples
            ntraces = segyfile.tracecount
            data = segyio.tools.cube(segyfile)
            header = segyio.tools.wrap(segyfile.text[0])  
            inlines = segyfile.ilines
            crosslines = segyfile.xlines
    size_mb= data.nbytes/1024**2
    info_txt = '[read_segy] reading {}\n'.format(f)
    info_txt += '[read_segy] number of traces: {0}, samples: {1}, sample rate: {2} s\n'.format(ntraces ,nsamples, sr)
    info_txt += '[read_segy] first, last sample twt: {0}, {1} s\n'.format(twt[0],twt[-1])
    info_txt += '[read_segy] size: {:.2f} Mb ({:.2f} Gb)'.format(size_mb, size_mb/1024)
    print(info_txt)
    logger.info(info_txt)
    if not twod:
        info_txt = '[read_segy] inlines: {:.0f}, min={:.0f}, max={:.0f}\n'.format(inlines.size,inlines.min(),inlines.max())
        info_txt += '[read_segy] crosslines: {:.0f}, min={:.0f}, max={:.0f}'.format(crosslines.size,crosslines.min(),crosslines.max())
        print(info_txt)
        logger.info(info_txt)
        return xr.DataArray(data, dims= ['INLINE', 'XLINE', 'TWT'], coords=[inlines, crosslines, twt]), \
            nsamples, sr, twt, ntraces, header, inlines, crosslines
    else:
        return xr.DataArray(data, dims= ['TRACES', 'TWT'], coords=[np.arange(ntraces), twt]), \
            nsamples, sr, twt, ntraces, header, None, None


def plot_ampspec(freq,amp,f_peak,name=None):
    '''
    plot_ampspec (C) aadm 2016-2018
    Plots amplitude spectrum calculated with fullspec (aageofisica.py).

    INPUT
    freq: frequency
    amp: amplitude
    f_peak: average peak frequency
    '''
    db = 20 * np.log10(amp)
    db = db - np.amax(db)
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5),facecolor='w')
    ax[0].plot(freq, amp, '-k', lw=2)
    ax[0].set_ylabel('Power')
    ax[1].plot(freq, db, '-k', lw=2)
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        aa.set_xlim([0,np.amax(freq)/1.5])
        aa.grid()
        aa.axvline(f_peak, color='r', ls='-')
        if name!=None:
            aa.set_title(name, fontsize=16)


def plot_ampspecs(freq_amp_list, names=None):
    '''Plots overlay of multiple amplitude spectras.
 
    A variation of:
    plot_ampspec2 (C) aadm 2016-2018
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/playing_with_seismic.ipynb
    which takes a list of multiple freqency-amplitude pairs (with an optional "average peak frequency") 

    INPUT
        freq_amp_list: list of 
        [frequency (np.ndarray), amplitude spectra (np.ndarray), optional average peak frequency (float)] lists 
        
        names: list of strings, same length as freq_amp_list

    '''
    dbs = []  # list to hold dB values of the amplitude spectra
    for _item in freq_amp_list:
        _db = 20 * np.log10(_item[1])
        dbs.append(_db - np.amax(_db))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5), facecolor='w')
    
    labels = None
    if names is not None:
        if len(freq_amp_list) != len(names):
            raise ValueError('Both input lists must have same length')

        labels = []
        for i, _name in enumerate(names):
            _label = '{}'.format(_name)
            if len(freq_amp_list[i]) > 2:
                if (freq_amp_list[i][2] is not None):
                    _label += ' Fp={:.0f} Hz'.format(freq_amp_list[i][2])
            labels.append(_label)
    if labels is None:
        labels = [''] * len(freq_amp_list)

    saved_colors = []
    for i, _item in enumerate(freq_amp_list):
        tc = next_color()
        saved_colors.append(tc)
        ax[0].plot(_item[0], _item[1], '-{}'.format(tc), lw=2, label=labels[i])
        ax[0].fill_between(_item[0], 0, _item[1], lw=0, facecolor=tc, alpha=0.25)
        ax[1].plot(_item[0], dbs[i], '-{}'.format(tc), lw=2, label=labels[i])

    lower_limit=np.min(ax[1].get_ylim())
    for i, _item in enumerate(freq_amp_list):
        ax[1].fill_between(_item[0], dbs[i], lower_limit, lw=0, facecolor=saved_colors[i], alpha=0.25)

    ax[0].set_ylabel('Power')
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        #aa.set_xlim([0,np.amax(freq)/1.5])
        aa.grid()
        for i, _item in enumerate(freq_amp_list):
            if len(freq_amp_list[i]) > 2:
                if (freq_amp_list[i][2] is not None):
                    aa.axvline(freq_amp_list[i][2], color=saved_colors[i], ls='-')

        aa.legend(fontsize='small')


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
    near, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\SEISMIC DATA\KPSDM-MID_18deg_cropped.sgy"
    mid, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\SEISMIC DATA\KPSDM-FAR_26deg_cropped.sgy"
    far, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)

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
            kwargs={'target_function': straight_line}
    )
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
    near, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-MID_18deg_cropped.sgy"
    mid, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-FAR_26deg_cropped.sgy"
    far, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)

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
    data, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    print(header)
    uu={'add_colorbar':False,'robust':True,'interpolation':'spline16'}
    fig, ax = plt.subplots()
    data.sel(INLINE=6550).plot.imshow(x='XLINE',y='TWT', yincrease=False, ax=ax, **uu)
    plt.show()

def test_amp_spectra():
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
    data, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)

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
