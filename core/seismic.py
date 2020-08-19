import numpy as np
import matplotlib.pyplot as plt
import logging

import segyio
import xarray as xr

from utils.smoothing import smooth as _smooth

# global variables
logger = logging.getLogger(__name__)

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
    info_txt += '[read_segy] number of traces: {0}, samples: {1}, sample rate: {2} s\n'.format(ntraces,nsamples,sr)
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


def ampspec(signal, sr, smoothing=None):
    '''
    ampspec (C) aadm 2016
    Calculates amplitude spectrum of a signal with FFT,  optionally smoothed.
    
    Origin:
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/playing_with_seismic.ipynb

    Slightly modified and upgraded by Erik Mårten Blixt 2020-08-19

    :param signal: 
        1D numpy array

    :param sr: 
        float
        sample rate in s

    :param smoothing: 
        str
        'cubic_spline:  smooths the spectrum using cubic interpolation

    OUTPUT
    freq: frequency
    amp: amplitude
    '''

    SIGNAL = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.size, d=sr)
    keep = freq>=0
    SIGNAL = np.abs(SIGNAL[keep])
    freq = freq[keep]
    if smoothing == 'cubic_spline':
        freq0=np.linspace(freq.min(),freq.max()/2,freq.size*10)
        #f = interp1d(freq, SIGNAL, kind='cubic')
        f = _smooth(SIGNAL, method='cubic_spline', interp_vec=freq)
        return freq0, f(freq0)
    else:
        return freq, SIGNAL

def fullspec(data,sr):
    '''
    fullspec (C) aadm 2016-2018
    Calculates amplitude spectrum of 2D numpy array.

    INPUT
    data: 2D numpy array, shape=(traces, samples)
    sr: sample rate in ms

    OUTPUT
    freq: frequency
    amp: amplitude
    db: amplitude in dB scale
    f_peak: average peak frequency
    '''
    amps, peaks = [], []
    for i in range(data.shape[0]):
        trace = data[i,:]
        freq, amp = ampspec(trace,sr)
        peak = freq[np.argmax(amp)]
        amps.append(amp)
        peaks.append(peak)
    amp0 = np.mean(np.dstack(amps), axis=-1)
    amp0 = np.squeeze(amp0)
    db0 = 20 * np.log10(amp0)
    db0 = db0 - np.amax(db0)
    f_peak = np.mean(peaks)
    print('freq peak: {:.2f} Hz'.format(f_peak))
    return freq,amp0,db0,f_peak

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

def plot_ampspec2(freq1,amp1,f_peak1,freq2,amp2,f_peak2,name1=None,name2=None):
    '''
    plot_ampspec2 (C) aadm 2016-2018
    Plots overlay of 2 amplitude spectra calculated with fullspec.

    INPUT
    freq1, freq2: frequency
    amp1, amp2: amplitude spectra
    f_peak1, f_peak2: average peak frequency
    '''
    db1 = 20 * np.log10(amp1)
    db1 = db1 - np.amax(db1)
    db2 = 20 * np.log10(amp2)
    db2 = db2 - np.amax(db2)
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5),facecolor='w')
    if name1 is not None:
        label1='{:s} Fp={:.0f} Hz'.format(name1,f_peak1)
        label2='{:s} Fp={:.0f} Hz'.format(name2,f_peak2)
    else:
        label1='Fp={:.0f} Hz'.format(f_peak1)
        label2='Fp={:.0f} Hz'.format(f_peak2)
    ax[0].plot(freq1, amp1, '-k', lw=2, label=label1)
    ax[0].plot(freq2, amp2, '-r', lw=2, label=label2)
    ax[0].fill_between(freq1,0,amp1,lw=0, facecolor='k',alpha=0.25)
    ax[0].fill_between(freq2,0,amp2,lw=0, facecolor='r',alpha=0.25)
    ax[0].set_ylabel('Power')
    ax[1].plot(freq1, db1, '-k', lw=2, label=label1)
    ax[1].plot(freq2, db2, '-r', lw=2,label=label2)
    lower_limit=np.min(ax[1].get_ylim())
    ax[1].fill_between(freq1, db1, lower_limit, lw=0, facecolor='k', alpha=0.25)
    ax[1].fill_between(freq2, db2, lower_limit, lw=0, facecolor='r', alpha=0.25)
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        aa.set_xlim([0,np.amax(freq)/1.5])
        aa.grid()
        aa.axvline(f_peak1, color='k', ls='-')
        aa.axvline(f_peak2, color='r', ls='-')
        aa.legend(fontsize='small')

def test():
    filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
    data, nsamples, sr, twt, ntraces, header, ilines, xlines = read_segy(filename, byte_il=189, byte_xl=193)
    print(header)
    uu={'add_colorbar':False,'robust':True,'interpolation':'spline16'}
    fig, ax = plt.subplots()
    data.sel(INLINE=6550).plot.imshow(x='XLINE',y='TWT', yincrease=False, ax=ax, **uu)
    plt.show()

if __name__ == '__main__':
    test()
