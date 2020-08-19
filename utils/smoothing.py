import numpy as np
from scipy.interpolate import interp1d

def rolling_window(a, window_len):
    """
    After
    https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb
    :param a:
        numpy.ndarray
    :param window_len:
        int
    :return:
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolled


def smooth(x, window_len=11, window='hanning', method='convolution', interp_vec=None):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.


    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

        method:
          'convolution' or 'median' or 'cubic_spline'

        interp_vec:
            np.ndarray 
            when method is set to 'cubic_spline', the input signal is interpolated along this vector

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if method == 'median':
        _tmp = np.median(rolling_window(x, window_len), -1)
        y = np.pad(_tmp, int(window_len/2), mode='edge')
    
    elif method == 'convolution':
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w=numpy.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=numpy.convolve(w/w.sum(), s, mode='valid')

    elif method == 'cubic_spline':
        if interp_vec is None:
            raise ValueError("The interpolate vector 'interp_vec' must be given")
        y = interp1d(interp_vec, x, kind='cubic')

    else:
        raise ValueError("Method is one of 'convolution', 'median', 'cubic_spline'")

        

    return y



