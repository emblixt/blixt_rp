import unittest
import os


from .. import ureg, Q_


class UnitsTestCase(unittest.TestCase):

    def test_pint(self):
        t1 = Q_(30, 'degC')
        t2 = Q_(30, 'degc')
        print(t1.to('degF'), t2.to('degF'))
        self.assertTrue(True)

