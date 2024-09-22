import unittest
import os
import sys

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp_utils.avo_monte_carlo as havo

sums_and_averages = {
    'shale': {'VpMean': 2200., 'VsMean': 900., 'RhoMean': 2.4,
              'VpStdDev': 70. , 'VsStdDev': 50., 'RhoStdDev': 0.08,
              'VpVsCorrCoef': 0.8, 'VpRhoCorrCoef': -0.03, 'VsRhoCorrCoef': -0.2
              },
    'brine_sst': {'VpMean': 2900., 'VsMean': 1400., 'RhoMean': 2.1,
                  'VpStdDev': 120. , 'VsStdDev': 100., 'RhoStdDev': 0.08,
                  'VpVsCorrCoef': 0.9, 'VpRhoCorrCoef': 0.7, 'VsRhoCorrCoef': 0.6
                  },
    'oil_sst': {'VpMean': 2800., 'VsMean': 1200., 'RhoMean': 2.0,
                  'VpStdDev': 120. , 'VsStdDev': 100., 'RhoStdDev': 0.08,
                  'VpVsCorrCoef': 0.9, 'VpRhoCorrCoef': 0.7, 'VsRhoCorrCoef': 0.6
                  },
}
l1 = {'vp': 3000, 'vs': 1820, 'rho': 2.6}
l2 = {'vp': 2000, 'vs': 1120, 'rho': 2.2}


class TestCase(unittest.TestCase):
    def test_multi_interface(self):
        havo.half_space_mc(
            sums_and_averages,
            [['shale', 'brine_sst', 'b'], ['shale', 'oil_sst', 'g']])
        self.assertTrue(True)

    def test_layered_model(self):
        import blixt_utils.misc.wavelets as bumw
        wavelet = bumw.ricker(0.096, 0.001, 25)
        model = havo.layered_model(0.05, l2, l1, wavelet, verbose=False)

        i, g = havo.evaluate_layered_model(model, wavelet, 2.01, extract_on='nearest min', verbose=True)

        print(i, g)
        self.assertTrue(True)

