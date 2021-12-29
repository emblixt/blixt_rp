import numpy as np
import matplotlib.pyplot as plt
import pywt


def binary_search_scale(f, wavelet_name, dt, start_at_scale=10., freq_tolerance=1., last_scale=None):
    """Returns the nearest scale for a given central frequency using binary search

    INPUT
        f: float 
        the wavelet central frequencies we want to find the nearest scale for
        
        wavelet: str
        name of wavelet
        use
        >>> print(pywt.wavelist(kind='continuous')) 
        to see which are supported

        dt: float
        Sample rate in seconds

    RETURN
        s: float 
        the corresponding scale

        this_freq: float 
        The nearest central frequency(ies) to the input

    """
    if last_scale is None:
        try_this_scale = start_at_scale
    else:
        try_this_scale = (start_at_scale + last_scale)/2.

    this_freq = pywt.scale2frequency(wavelet_name, try_this_scale) / dt 

    if this_freq - f >= freq_tolerance:
        # The center frequency associated with scale is higher than the desired frequency
        # increase the scale 
        this_scale = 2. * try_this_scale
        #print('{:.2f} higher than wanted frequency, trying with scale {:.1e}'.format(this_freq, this_scale))
        return binary_search_scale(f, wavelet_name, dt, start_at_scale=this_scale, last_scale=try_this_scale)

    elif this_freq - f < -1.*freq_tolerance:
        # The center frequency associated with scale is lower than the desired frequency
        # decrease the scale 
        this_scale = 0.5 * try_this_scale
        #print('{:.2f} lower than wanted frequency, trying with scale {:.1e}'.format(this_freq, this_scale))
        return binary_search_scale(f, wavelet_name, dt, start_at_scale=this_scale, last_scale=try_this_scale)
    
    else:
        return try_this_scale, this_freq    


def freq2scale(f, wavelet, dt, max_scale=200, min_scale=1.5):
    """Returns the nearest scale for a given central frequency in a crude way.

    INPUT
        f: float or np.ndarray of floats
        the wavelet central frequencies we want to find the nearest scale for
        
        wavelet: str
        name of wavelet
        use
        >>> print(pywt.wavelist(kind='continuous')) 
        to see which are supported

        dt: float
        Sample rate in seconds

    RETURN
        s: float or np.ndarray of floats
        the corresponding scales

        nearest_f: float or np.ndarray of floats
        The nearest central frequency(ies) to the input

    """
    select_from = np.logspace(np.log10(max_scale), np.log10(min_scale), num=1000)
    freqs = pywt.scale2frequency(wavelet, select_from) / dt
    if isinstance(f, np.ndarray):
        s = np.array([select_from[np.argmin( (freqs - _f)**2 )] for _f in f ])
        nearest_f = np.array([freqs[np.argmin( (freqs - _f)**2 )] for _f in f ])
    elif isinstance(f, float):
        s = select_from[np.argmin( (freqs - f)**2 )]
        nearest_f = freqs[np.argmin( (freqs - f)**2 )]
    else:
        raise ValueError('Input frequencies are neither float or array of floats')

    return s, nearest_f


def plot_cwt(freq_or_scale, twt, cwt_coeffs, 
             ax=None, 
             title='Wavelet transform of signal',
             xlabel='Frequency [Hz]',
             ylabel='TWT [ms]',
             transps=True,
             scale=False,
             **kwargs):
    """Plots the absolute value of the calculated cwt coefficents.
    :param freq_or_scale:
        (M,) sized np.ndarray
        Frequency in Hertz or scale
    :param twt:
        (N,) sized np.ndarray
        Two way traveltime in seconds
    :param cwt_coeffs:
       (M,N) sized real or complex np.ndarray
    """
    if ax is None:
        fig, ax = plt.subplots()

    if transps:
        _amp = np.abs(cwt_coeffs.transpose()) # Transpose it to plot twt on Y axis
        X, Y = np.meshgrid(freq_or_scale, twt * 1000.) # transform twt to ms
    else:
        _amp = np.abs(cwt_coeffs) 
        X, Y = np.meshgrid(twt * 1000., freq_or_scale) # transform twt to ms


    # Trick for plot log2 separated level in contour plot
    # power = _amp**2
    # levels = [2**x for x in np.linspace(np.ceil(np.log2(np.min(power))), np.floor(np.log2(np.max(power))), num=8)] 
    # ax.pcolormesh(X, Y, np.log2(power), levels=np.log2(levels), **kwargs)

    # Trick for plotting log2 scale in frequency
    # ax.pcolormesh(np.log2(X), Y, _amp, **kwargs)
    # xticks = 2**np.arange(np.ceil(np.log2(frequency.min())), np.ceil(np.log2(frequency.max())))
    # ax.set_xticks(np.log2(xticks))
    # ax.set_xticklabels(xticks)

    ax.pcolormesh(X, Y, _amp, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if transps:
        ax.invert_yaxis()
    ax.grid(True)

def test():
    import pickle

    c0 = 0.849 # Center frequency
    #c0 = 4. # Center frequency
    b0 = 1.5 # bandwidth frequency
    #b0 = 6.0 # bandwidth frequency

    # load twt and seismic trace from pickled object
    with open('C:\\Users\\marten\\Downloads\\trace.dat', 'rb') as input:
        twt, trace = pickle.load(input)

    # twt is in ms, so we need to convert it to seconds
    twt = twt/1000.
    dt = twt[1] - twt[0]

    # generate a synthetic signal with same twt
    N = len(twt)
    f_s = 1/dt

    #ta = np.linspace(0, t_n, num=N)
    ta = twt
    #tb = np.linspace(0, t_n/4, num=int(N/4))
    tb = twt[:int(N/4)]

    frequencies = [4, 30, 60, 90]
    y1b = np.sin(2*np.pi*frequencies[0]*tb)
    y2b = np.sin(2*np.pi*frequencies[1]*tb)
    y3b = np.sin(2*np.pi*frequencies[2]*tb)
    y4b = np.sin(2*np.pi*frequencies[3]*tb)
     
    sig1 = np.concatenate([y1b, y2b, y3b, y4b])

    # the synthetic signal become one item shorter than the input signal, so we need to shorten it
    sig2 = trace.data[:-1]
    ta = ta[:-1]


    #fig = plt.figure(constrained_layout=False, figsize=(12,12))
    #gs = fig.add_gridspec(4,2)

    ## plot the signals and their amplitude spectrum
    #for i, sig in enumerate([sig1, sig2]):
    #    ax = fig.add_subplot(gs[i,:])
    #    
    #    color = 'tab:blue'
    #    ax.plot(ta, sig, color=color)
    #    ax.set_xlabel('Time [s]', color=color)
    #    ax.set_ylabel('Signal amp', color=color)
    #    ax.tick_params(axis='x', labelcolor=color)
    #    ax.tick_params(axis='y', labelcolor=color)
    #
    #    color = 'tab:red'
    #    f_values, fft_values = ampspec(sig, dt)   
    #    ax2 = ax.twiny().twinx() # the order of twinx and twiny matters!
    #    ax2.plot(f_values, fft_values, color=color)
    #    ax2.set_xlabel('Frequency [Hz]', color=color)
    #    #ax2.set_xlim(0,150)
    #    ax2.set_ylabel('Spectral amp', color=color)
    #    ax2.tick_params(axis='x', labelcolor=color)
    #    ax2.tick_params(axis='y', labelcolor=color)
    #               
    ## Start CWT calculation
    #wavelet_name = 'cmor{}-{}'.format(str(b0), str(c0))
    lowest_freq = 1. # Hz
    ##highest_freq = 2*f_s # Hz
    highest_freq = 100. # Hz
    #largest_scale, _ = binary_search_scale(lowest_freq, wavelet_name, dt)
    #shortest_scale, _ = binary_search_scale(highest_freq, wavelet_name, dt)
    #scales = 2**np.linspace(np.log2(largest_scale), np.log2(shortest_scale), 20)
    ###scales = np.arange(1, 128)
    ##scales, fs = freq2scale(np.linspace(1,150,10), wavelet_name, T, max_scale=200, min_scale=2)
    #scales = np.linspace(2.5, 250, 20)
    #for i, sig in enumerate([sig1, sig2]):
    #    cwt_attr, f_cwt_val = pywt.cwt(sig, scales, wavelet_name, dt)

    #    ax3 = fig.add_subplot(gs[2+i,0])
    #    ax4 = fig.add_subplot(gs[2+i,1])

    #    plot_cwt(f_cwt_val, ta, cwt_attr, ax=ax3, transps=False, xlabel='Time [ms]', ylabel='Frequency [Hz]', title='Wavelet {}'.format(wavelet_name))
    #    plot_cwt(scales, ta, cwt_attr, ax=ax4, transps=False, scale=True, xlabel='Time [ms]', ylabel='Scale', title='Wavelet {}'.format(wavelet_name))
    #    ax4.invert_yaxis()

    ## plot the CWT for the seismic trace
    #fig, ax = plt.subplots()
    #plot_cwt(f_cwt_val, ta, cwt_attr, ax=ax, title='Wavelet {}'.format(wavelet_name), cmap='jet', vmin=np.min(abs(cwt_attr)),vmax=np.max(abs(cwt_attr)))
    
    
    ##for i in range(3):
    ##    ax[i, 0].set_ylabel('Amplitude')
    ##    ax[i, 1].set_xlim([0, 150])
    ##    ax[i, 2].set_xlim([0, 150])
    ##ax[2, 0].set_xlabel('Time [s]')
    ##ax[2, 1].set_xlabel('Frequency [Hz]')
    #ax5 = fig.add_subplot(gs[2,0])
    #ax6 = fig.add_subplot(gs[2,1])
    #ax5.plot(f_cwt_val2, scales)
    #ax5.set_xlabel('Frequency [Hz]')
    #ax5.set_ylabel('Scale')

    ### get the mexican hat (=ricker) wavelet from pywt
    ###w1, x1  = pywt.ContinuousWavelet('mexh').wavefun() # maximum level of decomposition

    ### get the complex Morlet wavelet
    #w2, x2  = pywt.ContinuousWavelet(wavelet_name).wavefun() # maximum level of decomposition

    #ax6.plot(x2, w2.real)
    #ax6.plot(x2, w2.imag, '--')

    #
    # Try decomposing the synthetic signal for one wavelet
    #
    #scales = 2**np.linspace(np.log2(largest_scale), np.log2(shortest_scale), 10)
    #cwt_attr, f_cwt_val = pywt.cwt(sig1, scales, 'cmor1.5-1.0', dt)
    #fig, ax = plt.subplots()
    #sum = 0.
    #for i, s in enumerate(scales):
    ##    #part = np.real(cwt_attr2[i,:])/(s**0.5)
    #    part = np.real(cwt_attr[i,:])/s**2
    #    wiggle_plot(ax, ta, part, zero_at=i, scaling=1./np.max(part), fill_pos_style=None, fill_neg_style=None)
    #    sum += part
    #wiggle_plot(ax, ta, sum, zero_at=i+2, fill_pos_style=None, fill_neg_style=None)

    #
    # Try decomposing the synthetic signal for multiple wavelets
    #
    wnames = ['cmor1.0-0.5', 'cmor1.5-0.5', 'cmor1.5-0.849' ]
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(ta, sig1)
    ax[0].set_ylabel('Orig signal')

    for i, wname in enumerate(wnames):
        largest_scale, _ = binary_search_scale(lowest_freq, wname, dt)
        shortest_scale, _ = binary_search_scale(highest_freq, wname, dt)
        scales = 2**np.linspace(np.log2(largest_scale), np.log2(shortest_scale), 20)

        cwt_attr, f_cwt_val = pywt.cwt(sig1, scales, wname, dt)
        sum = 0.
        for j, s in enumerate(scales):
            part = np.real(cwt_attr[j,:])/s
            #part = np.real(cwt_attr[j,:])/(s**0.5)
            #part = np.real(cwt_attr[j,:])/s**2
            sum += part
        ax[i+1].plot(ta, sum)
        ax[i+1].set_ylabel('Wavelet {}'.format(wname))


    plt.show()


if __name__ == '__main__':
    test()
