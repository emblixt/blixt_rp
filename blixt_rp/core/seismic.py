import pickle
import os
import sys
import unittest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
from itertools import cycle
from typing import Literal
import bruges
from copy import deepcopy

import pint
from bokeh.models import ColumnDataSource, LinearColorMapper
from scipy.optimize import least_squares

from bokeh.plotting import show, figure
from bokeh.io import output_file

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_utils.misc.curve_fitting as mycf
import blixt_utils.io.io as uio
from blixt_utils.plotting.helpers import wiggle_plot
import blixt_utils.plotting.crossplot as xp
from blixt_rp.core.core import Template, CutoffRule, Cutoffs, LogTable, Header
from blixt_rp.core.log_curve_new import read_las

# global variables
# output_file('C:\\Users\marte\Documents\plot.html')
output_file('C:\\Users\emb\\Documents\\plot.html')
logger = logging.getLogger(__name__)
clrs = list(mcolors.BASE_COLORS.keys())
clrs.remove('w')
cclrs = cycle(clrs)  # "infinite" loop of the base colors
test_data_length = 1000


class SeismicTraces:
    """
    Simple class that contains N number of seismic traces,
    Each trace must have the same depth/time dimension

    """
    def __init__(self,
                 x: np.ndarray | None = None,
                 y: np.ndarray | None = None,
                 traces: np.ndarray | None = None,
                 source: ColumnDataSource | None = None,
                 trace_type: Literal['avo', 'eei', 'index'] | None = None
                 ):
        """

        :param x:
            np.ndarray of length M with values in the x direction
            Contains incidence angles when trace_type is 'avo',
            Chi angles when trace_type is 'eei', and index when trace_type is 'index'
        :param y:
            Numpy array of depth/time values, length N
        :param traces:
            np.ndarray of size MxN, contains M number of seismic traces, each of length N
        :param source:
            ColumnDataSource({'value': [_seismic.traces.T]})
            Must have the key 'value'
        :param trace_type:
            string that describes the type of seismic variation in the x direction
            'avo': Incidence angle (AVO)
            'eei': Chi angle (Extended Elastic Impedance)
            'index': index along an Inline, Xline or arbitrary seismic line
        """
        if x is None:
            x = np.linspace(0,35, 128)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, test_data_length)
        self._y = y
        if trace_type is None:
            trace_type = 'avo'
        self._trace_type = trace_type

        self.source = source

        if traces is None and source is None:
            _synts = SyntheticTraces(trace_length=test_data_length, n_traces=len(x))
            traces = _synts.get_traces(simulate_avo=self._trace_type=='avo')
        self._traces = traces

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def trace_type(self):
        return self._trace_type

    @property
    def traces(self):
        return self._traces


class SyntheticTraces:
    def __init__(self,
                 trace_length:int | None = None,
                 dt: float | None = None,
                 n_traces: int | None = None,
                 n_reflectors: int | None = None,
                 seed: int | None = None):

        if trace_length is None:
            trace_length = 3000
        if dt is None:
            dt = 0.001  # seconds
        if n_traces is None:
            n_traces = 128
        if n_reflectors is None:
            n_reflectors = 3

        x = np.linspace(0, n_traces - 1, n_traces)
        y = np.linspace(0, trace_length - 1, trace_length)
        traces = np.zeros((n_traces, trace_length))

        self.trace_length = trace_length
        self.dt = dt
        self.n_traces = n_traces
        self.seed = seed
        self.n_reflectors = n_reflectors
        self.rng = np.random.default_rng(self.seed)

        self.w_length = 0.082  # Ricker wavelength in seconds
        self.f0 = 25.  # Hz

        self.traces = traces
        self.wavelet = None

    def get_traces(self, simulate_avo=False) -> np.ndarray:
        idx_refl = self.rng.integers(100, self.trace_length, self.n_reflectors)
        refl = np.zeros(self.trace_length)
        refl[idx_refl] = 2 * self.rng.random(self.n_reflectors) - 1

        # Convolve reflectivity model with a Ricker wavelet '''
        this_wavelet = bruges.filters.wavelets.ricker(self.w_length, self.dt, self.f0)
        self.wavelet = this_wavelet.amplitude
        this_trace = np.convolve(refl, self.wavelet, mode='same')
        for i in range(self.traces.shape[0]):
            if simulate_avo:
                _refl = deepcopy(refl)
                _refl[idx_refl[1]] = _refl[idx_refl[1]] * i*2/(self.traces.shape[0] - 1)
                this_trace=  np.convolve(_refl, self.wavelet, mode='same')
            self.traces[i,:] = this_trace

        return self.traces


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


def interpolate_along_offset(offset_traces: np.ndarray, offset_angles: pint.Quantity, inc_angles: pint.Quantity):
    """
    Creates a "Pseudo gather"
    Interpolates the input offset_traces (near, mid, far, ...) in the offset direction so that we get a
    more densely sampled set of traces

    :param offset_traces:
        np.ndarray of shape (M, N)
        M is the number of traces, and N is the number of samples along the trace
    :param offset_angles:
        array like pint Quantity of length M which contains the center incidence angle for each trace
    :param inc_angles:
        array like pint Quantity of length J, with the incidence angles we would like to evaluate the seismic.
        The min & max of inc_angles should not be less / larger than the corresponding min / max of
        offset_angles. Else we would need to extrapolate
    :return:
        np.ndarray of size (J, N)
    """
    from scipy import interpolate

    # calculate sin2(theta)
    s2t_in = np.sin(offset_angles.to('rad').magnitude)**2
    s2t_out = np.sin(inc_angles.to('rad').magnitude)**2

    return interpolate.interp1d(s2t_in, offset_traces, axis=0, fill_value=None)(s2t_out)


def seismic_color_map(min_val=-1, max_val=1, n=256, symmetric=True) -> LinearColorMapper:
    # Construct cmap dictionary
    c_dict = {'red': ((0, 0.6314, 0.6314),
                      (0.33, 0, 0),
                      (0.4, 0.302, 0.302),
                      (0.5, 0.8, 0.8),
                      (0.6, 0.3804, 0.3804),
                      (0.667, 0.749, 0.749),
                      (1, 1, 1)),
              'green': ((0, 1, 1),
                        (0.33, 0, 0),
                        (0.4, 0.302, 0.302),
                        (0.5, 0.8, 0.8),
                        (0.6, 0.2706, 0.2706),
                        (0.667, 0, 0),
                        (1, 1, 1)),
              'blue': ((0, 1, 1),
                       (0.33, 0.749, 0.749),
                       (0.4, 0.302, 0.302),
                       (0.5, 0.8, 0.8),
                       (0.6, 0, 0),
                       (0.667, 0, 0),
                       (1, 0, 0))}

    if symmetric:
        if np.sign(min_val) != np.sign(max_val):
            max_magnitude = np.max([np.abs(min_val), np.abs(max_val)])
            min_val = np.sign(min_val) * max_magnitude
            max_val = np.sign(max_val) * max_magnitude

    cmap = mpl.colors.LinearSegmentedColormap('Seismic', c_dict).reversed()
    _colors = cmap(np.linspace(0, 1, n))
    return LinearColorMapper(palette=[mpl.colors.to_hex(_c) for _c in _colors], low=min_val, high=max_val)

class TestCases(unittest.TestCase):

    def test_interpolate(self):
        from blixt_utils.plotting.log_plotter import LogColumn, LogPlotter
        las_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Tampen\\Wells\\34_3_3S_seismic.las"
        # las_file = "G:\\My Drive\\Work - Current and Recent\\GeoMind\\Clients\\AkerBP\\PL932 Kaldafjell AVO feasibility\\Wells\\34_3_3S_seismic.las"
        t = Template(**{'name': 'Seismic', 'units': 'dimensionless', 'line_color': 'b'})
        log_table = LogTable({'Seismic':
                                  ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2'
                                   ]})
        rename_logs = {
            'near': ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2'],
            'mid': ['CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2'],
            'far': ['CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2'],
            'ufar': ['CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2']
        }
        log_curves, well_info = read_las(las_file, log_table=log_table, template=t, rename_logs=rename_logs)

        offset_angles = pint.Quantity(np.array([10., 18., 26., 34.]), 'deg')
        traces = np.zeros((len(offset_angles), len(log_curves['near'])))
        traces[0,:] = log_curves['near'].values
        traces[1,:] = log_curves['mid'].values
        traces[2,:] = log_curves['far'].values
        traces[3,:] = log_curves['ufar'].values
        # fig1, ax1 = plt.subplots()
        # ax1.imshow(traces.T, aspect=1./1000)
        st_orig = SeismicTraces(x=offset_angles.magnitude,
                                 y=log_curves['near'].depth.values,
                                 # traces=traces,
                                 traces=None,
                                 source=ColumnDataSource({'value': [traces.T]}),
                                 trace_type='avo'
                                 )
        lc_orig = LogColumn('orig',
                            seismic_traces=st_orig)

        new_angles = pint.Quantity(np.linspace(10, 34, 100), 'deg')
        new_traces = interpolate_along_offset(traces, offset_angles, new_angles)
        st_intrp = SeismicTraces(x=new_angles.magnitude,
                                 y=log_curves['near'].depth.values,
                                 # traces=new_traces,
                                 traces=None,
                                 source=ColumnDataSource({'value': [new_traces.T]}),
                                 trace_type='avo')
        lc_intrp = LogColumn('interpolated', seismic_traces=st_intrp)

        plotter = LogPlotter(columns=[lc_orig, lc_intrp])
        grid = plotter.figure()
        show(grid)


        # fig2, ax2 = plt.subplots()
        # ax2.imshow(new_traces.T, aspect=1./100.)
        # plt.show()


    def test_synthetic_traces(self):
        fig, ax = plt.subplots()
        _synts = SyntheticTraces()
        synts = _synts.get_traces(simulate_avo=True)
        print(synts.shape)
        source = ColumnDataSource({'value': [synts.T]})
        print(np.min(synts))
        _seismic_color_map = seismic_color_map(min_val=np.min(synts), max_val=np.max(synts))

        p = figure()
        p.image('value', source=source,
                color_mapper=_seismic_color_map, dh=synts.shape[1], dw=synts.shape[0], x=0, y=0)
        show(p)


    def test_ixg_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2)

        with open('C:\\Users\\marten\\Downloads\\angle_traces.dat', 'rb') as input:
            twt, near_trace, mid_trace, far_trace = pickle.load(input)

        i, g, q = avo_ig(np.array([near_trace.values.flatten(), \
                                   mid_trace.values.flatten(), \
                                   far_trace.values.flatten()]), [10., 18., 26.])

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
        print('Line shape: {}'.format(near_line.values.shape))

        i, g, q = avo_ig(np.array([near_line.values.flatten(), \
                                   mid_line.values.flatten(), \
                                   far_line.values.flatten()]), [10., 18., 26.])

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


    def test_plot_amp_vs_offset(self):
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
        near_max = np.max(np.abs(near_trace.values))
        mid_trace = mid.sel(INLINE=inline, XLINE=xline)
        mid_max = np.max(np.abs(mid_trace.values))
        far_trace = far.sel(INLINE=inline, XLINE=xline)
        far_max = np.max(np.abs(far_trace.values))
        angs = np.array([10., 18., 26.])
        fig, axes = plt.subplots(nrows=1, ncols=2)


        for a, t in zip(angs, [near_trace, mid_trace, far_trace]):
            wiggle_plot(axes[0], twt, t.values, zero_at=a, scaling=10. / max([near_max, mid_max, far_max]))

        axes[0].axhline(t0)

        ind = np.argmin(abs(near_trace.TWT.values - t0))

        amps = np.array([near_trace.values[ind], mid_trace.values[ind], far_trace.values[ind]])

        i, g, q = avo_ig(amps, angs)

        xp.plot(angs, amps, cdata='b', fig=fig, ax=axes[1])
        _angs = np.linspace(1, angs[-1]+10)
        axes[1].plot(_angs, i + g*np.sin(np.radians(_angs))**2)

        plt.show()


    def test_plot_line(self):
        filename = "U:\COMMON\SAAS DEVELOPMENT\TEST_DATA\Test_angle_stacks\KPSDM-NEAR_10deg_cropped.sgy"
        data, nsamples, sr, twt, ntraces, header, ilines, xlines = uio.read_segy(filename, byte_il=189, byte_xl=193)
        print(header)
        uu={'add_colorbar':False,'robust':True,'interpolation':'spline16'}
        fig, ax = plt.subplots()
        data.plot.imshow(x='XLINE', y='TWT', yincrease=False, ax=ax, **uu)
        plt.show()


    def test_amp_spectra(self):
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
        plot_cwt(twt / 1000., this_trace.values, scales, waveletname=waveletname, cmap='jet')
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

