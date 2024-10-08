import unittest
import os
import sys
import matplotlib.pyplot as plt

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp_utils.avo_monte_carlo as havo
import blixt_utils.misc.wavelets as bumw

wavelet = bumw.ricker(0.096, 0.001, 25)

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
    def test_half_space(self):
        havo.main(
            sums_and_averages,
            [['shale', 'brine_sst', 'b'], ['shale', 'oil_sst', 'g']],
            n_iter=100
        )
        self.assertTrue(True)

    def test_layered_model(self):
        model = havo.layered_model(0.05, l2, l1, wavelet, verbose=False)

        result = havo.evaluate_layered_model(model, wavelet, 2.01, extract_on='nearest_min', verbose=True)

        print(result['intercept'], result['gradient'])
        print(result['amplitude'])
        self.assertTrue(True)

    def test_execute_layered_mc(self):
        result = havo.execute_monte_carlo(
            sums_and_averages, 'brine_sst', 'shale', 10, (0.05, 0.002), wavelet,
            2.0, 'exact', 'layered_model', verbose=True)

        for _avo in result['amplitude']:
            plt.plot(_avo)
        fig, axs = plt.subplots(ncols=2)
        havo.plot_one_mc_result(result, 'TEST', 'b', axs[0], axs[1])
        plt.show()

    def test_layered_mc(self):
        havo.main(
            sums_and_averages,
            [['shale', 'brine_sst', 'b'], ['shale', 'oil_sst', 'g']],
            n_iter=100,
            thickness=(0.08, 0.001),
            wavelet=wavelet,
            extract_at=2.0,
            extract_on='exact',
            model_type='layered_model',
            verbose=False
        )

