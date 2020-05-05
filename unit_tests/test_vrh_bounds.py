import unittest
import numpy as np

import rp.rp_core as rp


def create_test_data(length, constant_fractions):
    fractions = [c*np.ones(length) for c in constant_fractions]
    constant_params = [1./c for c in constant_fractions]  # This makes the Voigt sum equal to two
    params = [c*np.ones(length) for c in constant_params]
    return fractions, params

class VrhTestCase(unittest.TestCase):
    length = 10
    constant_fractions = [0.2, 0.8]
    constant_params = [1./c for c in constant_fractions]

    wrong_fractions = [0.3, 0.8]  # Sum is not one
    wrong_params = [1./c for c in constant_fractions]

    def test_vrh_bounds(self):
        fractions, params = create_test_data(VrhTestCase.length, VrhTestCase.constant_fractions)
        v, r, vrh = rp.vrh_bounds(fractions, params)
        with self.subTest():
            ans1 = sum([x*y for x, y in zip(VrhTestCase.constant_fractions, VrhTestCase.constant_params)])
            print('Voigt bound should be {}: {}'.format(ans1, v[0]))
            self.assertEqual(ans1, v[0])

        with self.subTest():
            ans2 = 1./sum([x/y for x, y in zip(VrhTestCase.constant_fractions, VrhTestCase.constant_params)])
            print('Reuss bound should be {}: {}'.format(ans2, r[0]))
            self.assertEqual(ans2, r[0])

        with self.subTest():
            print('Voigt-Reuss-Hill bound should be the mean of these: {}'.format(vrh[0]))
            self.assertEqual(0.5*(ans1 + ans2), vrh[0])

    def test_wrong_bounds(self):
        f, p = create_test_data(VrhTestCase.length, VrhTestCase.wrong_fractions)
        io_error = False
        try:
            v, r, vrh = rp.vrh_bounds(f, p)
        except IOError as ioe:
            io_error = True
            print(ioe)
        with self.subTest():
            self.assertTrue(io_error)


