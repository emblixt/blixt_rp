import unittest
import os


from .. import ureg, Q_


class UnitsTestCase(unittest.TestCase):

    def test_pint(self):
        t1 = Q_(30, 'degC')
        t2 = Q_(30, 'degc')
        print(t1.to('degF'), t2.to('degF'))

        r1 = Q_(300., 'ohm')
        r2 = Q_(300., 'Ohm')
        r3 = Q_(3000., 'ohmm')
        dist = 10 * ureg.meter
        r4 = r3 / dist
        print(f"{r1:~P}", f"{r2:~P}", r3, f"{r4:~#P}")

        p1 = 1000. * ureg('pascal')
        p2 = p1 * dist**2
        print(f"{p2:~P}")

        self.assertTrue(True)

