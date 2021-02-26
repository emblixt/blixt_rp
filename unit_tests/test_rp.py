import unittest
import os
import numpy as np
from core.well import Well
from core.well import Project
from blixt_utils.misc.convert_data import convert as cnvrt
import rp.rp_core as rp

class RpTestCase(unittest.TestCase):
    wp = Project()
    well_table = {os.path.join(wp.working_dir, 'test_data/Well D.las'):
                      {'Given well name': 'WELL_D',
                       'logs': {
                           'ac': 'Sonic',
                           'acs': 'Shear sonic',
                           'cali': 'Caliper',
                           'den': 'Density',
                           'gr': 'Gamma ray',
                           'rdep': 'Resistivity',
                           'rmed': 'Resistivity',
                           'rsha': 'Resistivity',
                           'neu': 'Neutron density'},
                       'Note': 'Some notes for well A'}}
    wis = {'WELL_D': {
        'SAND C': [1585.0, 1826.0],
        'SHALE C': [1585.0, 1826.0],
        'SAND D': [1826.0, 1878.0],
        'SAND E': [1878.0, 1984.0],
        'SAND F': [1984.0, 2158.0],
        'SHALE G': [2158.0, 2211.0],
        'SAND H': [2211.0, 2365.0]
    }}

    w = Well()
    w.read_well_table(
        well_table,
        0,
        block_name='Logs')

    def test_step(self):
        i = 5
        theta = 10. # degrees
        x1 = np.linspace(1, 10, 10)
        x2 = np.linspace(2, 11, 10)
        x3 = np.linspace(3, 12, 10)
        d1 = rp.step(x1[i], x1[i+1])
        d2 = rp.step(x1, None, along_wiggle=True)
        incept1 = rp.intercept(x1[i], x1[i+1], x3[i], x3[i+1])
        incept2 = rp.intercept(x1, None, x3, None, along_wiggle=True)
        grad1 = rp.gradient(x1[i], x1[i+1], x2[i], x2[i+1], x3[i], x3[i+1])
        grad2 = rp.gradient(x1, None, x2, None, x3, None, along_wiggle=True)
        func1 = rp.reflectivity(x1[i], x1[i+1], x2[i], x2[i+1], x3[i], x3[i+1])
        func2 = rp.reflectivity(x1, None, x2, None, x3, None, along_wiggle=True)

        with self.subTest():
            print('Layer based step at i {}: {}'.format(i, d1))
            print('Wiggle based step at i {}: {}'.format(i, d2[i]))
            print('Layer based intercept at i {}: {}'.format(i, incept1))
            print('Wiggle based intercept at i {}: {}'.format(i, incept2[i]))
            print('Layer based gradient at i {}: {}'.format(i, grad1))
            print('Wiggle based gradient at i {}: {}'.format(i, grad2[i]))
            print('Layer based refl. coeff. at i {} at {} deg.: {}'.format(i, theta,  func1(theta)))
            print('Wiggle based refl. coeff. at i {} at {} deg.: {}'.format(i, theta, func2(theta)[i]))
            self.assertTrue(True)

    def test_intercept(self):
        """
        Should test if the intercept calculation returns the same result when using 'along_wiggle' as for single layer
        :return:
        """
        rho = RpTestCase.w.block['Logs'].logs['den'].data
        vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].data, 'us/ft', 'm/s')
        incept2 = rp.intercept(vp, None, rho, None, along_wiggle=True)
        i = np.nanargmax(incept2)
        incept1 = rp.intercept(vp[i], vp[i+1], rho[i], rho[i+1])
        incept1_2 = rp.intercept(vp[i+1], vp[i+2], rho[i+1], rho[i+2])

        with self.subTest():
            print('Layer based intercept at i {}: {}'.format(i, incept1))
            print('Layer based intercept at i {}: {}'.format(i+1, incept1_2))
            print('Wiggle based intercept at i {}: {}'.format(i, incept2[i:i+2]))
            self.assertTrue(True)

    def test_gradient(self):
        """
        Should test if the gradient calculation returns the same result when using 'along_wiggle' as for single layer
        :return:
        """
        rho = RpTestCase.w.block['Logs'].logs['den'].data
        vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].data, 'us/ft', 'm/s')
        vs = cnvrt(RpTestCase.w.block['Logs'].logs['acs'].data, 'us/ft', 'm/s')
        grad2 = rp.gradient(vp, None, vs, None, rho, None, along_wiggle=True)
        i = 10637
        grad1 = rp.gradient(vp[i], vp[i+1], vs[i], vs[i+1], rho[i], rho[i+1])
        grad1_2 = rp.gradient(vp[i+1], vp[i+2], vs[i+1], vs[i+2], rho[i+1], rho[i+2])

        with self.subTest():
            print('Layer based gradient at i {}: {}'.format(i, grad1))
            print('Layer based gradient at i {}: {}'.format(i+1, grad1_2))
            print('Wiggle based gradient at i {}: {}'.format(i, grad2[i:i+2]))
            self.assertTrue(True)

    def test_reflectivity(self):
        """
        What happens when the input to the reflectivity is an array?
        :return:
        """
        i = 10637
        theta = 10.  # degrees
        rho = RpTestCase.w.block['Logs'].logs['den'].data
        vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].data, 'us/ft', 'm/s')
        vs = cnvrt(RpTestCase.w.block['Logs'].logs['acs'].data, 'us/ft', 'm/s')

        func1 = rp.reflectivity(vp[i], vp[i+1], vs[i], vs[i+1], rho[i], rho[i+1])
        func1_2 = rp.reflectivity(vp[i+1], vp[i+2], vs[i+1], vs[i+2], rho[i+1], rho[i+2])
        func2 = rp.reflectivity(vp, None, vs, None, rho, None, along_wiggle=True)

        with self.subTest():
            print('Layer based refl. coeff. at i {} at {} deg.: {}'.format(i, theta,  func1(theta)))
            print('Layer based refl. coeff. at i {} at {} deg.: {}'.format(i+1, theta,  func1_2(theta)))
            print('Wiggle based refl. coeff. at i {} at {} deg.: {}'.format(i, theta, func2(theta)[i:i+2]))
            self.assertTrue(True)

    def test_greenberg_castagna(self):
        """
        Example taken from p. 248 in Rock physics handbook, Mavko et al. 1999
        :return:
        """
        vp = 3000.
        f = [0.6, 0.4]
        mono_mins = ['sandstone', 'SHALE']
        gc_coeffs = {
            'sandstone': [0., 0.80416, -0.85588],
            'limestone': [-0.05508, 1.01677, -1.03049],
            'dolomite': [0., 0.58321, -0.07775],
            'shale': [0., 0.76969, -0.86735]
        }
        vs = rp.greenberg_castagna(vp, f, mono_mins, gc_coeffs)

        with self.subTest():
            print('Greenberg Castagna Vs estimate: {}'.format(vs.value))
            self.assertAlmostEqual(vs.value, 1509.58, places=1)
