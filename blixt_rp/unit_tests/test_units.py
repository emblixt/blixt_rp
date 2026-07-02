import unittest
import os


import pint
from .. import ureg, Q_


class UnitsTestCase(unittest.TestCase):

    def test_pint(self):
        t1 = Q_(30, 'degC')
        t2 = Q_(30, 'degc')
        print(t1.to('degF'), t2.to('degF'))

        r1 = Q_(300., 'ohm')
        r2 = Q_(300., 'Ohm')
        r3 = Q_(3000., 'Ohmm')
        r5 = Q_(3000., 'ohm m')
        print(type(r3))
        dist = 10 * ureg.meter
        r4 = r3 / dist
        print(f"{r5:~P}", f"{r2:~P}", r3, f"{r4:~#P}")

        p1 = 1000. * ureg('pascal')
        p2 = p1 * dist**2
        print(f"{p2:~P}")

        l1 = Q_(1, 'ft')
        l2 = Q_(1, 'FT')
        print(l1, l2)

        pct1 = Q_(50., 'pct')
        print(pct1.magnitude, pct1.to('dimensionless'))
        frac = Q_(0.5, '')
        print(frac.magnitude, frac.to('dimensionless'), frac.to('dimensionless').magnitude)
        print('Convert 0.1 to percent: ', Q_(0.1, '').to('percent'))

        rho = Q_(10., 'G/cc')
        print(rho)

        a = Q_(10., '')
        b = Q_(5., 'frac')
        print(a.to('dimensionless'))
        print(b.to('dimensionless'))
        print(a/b)

        self.assertIsInstance(r4, pint.Quantity)

    def test_units(self):
        from blixt_rp.core.log_curve_new import fix_units_for_pint, is_equivalent
        print(fix_units_for_pint('m3'))
        nM = ureg.Unit('nM')
        nmol_L = ureg.Unit('nmol/L')
        m = ureg.Unit('m')
        ft = ureg.Unit('ft')
        self.assertTrue(is_equivalent(nM, nmol_L))  # True
        self.assertFalse(is_equivalent(m, ft))  # False

