import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.well import Well
from blixt_rp.core.well import Project
from blixt_utils.misc.convert_data import convert as cnvrt
import blixt_rp.rp.rp_core as rp


class RpTestCase(unittest.TestCase):
    # wp = Project()
    # well_table = {os.path.join(wp.working_dir, 'test_data/Well D.las'):
    #                   {'Given well name': 'WELL_D',
    #                    'logs': {
    #                        'ac': 'Sonic',
    #                        'acs': 'Shear sonic',
    #                        'cali': 'Caliper',
    #                        'den': 'Density',
    #                        'gr': 'Gamma ray',
    #                        'rdep': 'Resistivity',
    #                        'rmed': 'Resistivity',
    #                        'rsha': 'Resistivity',
    #                        'neu': 'Neutron density'},
    #                    'Note': 'Some notes for well A'}}
    # wis = {'WELL_D': {
    #     'SAND C': [1585.0, 1826.0],
    #     'SHALE C': [1585.0, 1826.0],
    #     'SAND D': [1826.0, 1878.0],
    #     'SAND E': [1878.0, 1984.0],
    #     'SAND F': [1984.0, 2158.0],
    #     'SHALE G': [2158.0, 2211.0],
    #     'SAND H': [2211.0, 2365.0]
    # }}

    # w = Well()
    # w.read_well_table(
    #     well_table,
    #     0,
    #     block_name='Logs')

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
        rho = RpTestCase.w.block['Logs'].logs['den'].values
        success, vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].values, 'us/ft', 'm/s')
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
        rho = RpTestCase.w.block['Logs'].logs['den'].values
        success, vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].values, 'us/ft', 'm/s')
        success, vs = cnvrt(RpTestCase.w.block['Logs'].logs['acs'].values, 'us/ft', 'm/s')
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
        rho = RpTestCase.w.block['Logs'].logs['den'].values
        success, vp = cnvrt(RpTestCase.w.block['Logs'].logs['ac'].values, 'us/ft', 'm/s')
        success, vs = cnvrt(RpTestCase.w.block['Logs'].logs['acs'].values, 'us/ft', 'm/s')

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

    def test_hashin_shtrikman(self):
        from bruges.rockphysics.bounds import hashin_shtrikman as bruges_hs
        from blixt_rp.rp.rp_core import hashin_shtrikman as blixt_hs
        f = [0.666, 1.-0.666]
        k = [36.6, 2.56]
        mu = [45., 0.0]
        print(bruges_hs(f, k, mu, 'bulk'))
        print(blixt_hs(f, k, mu, 'bulk'))
        self.assertTrue(True)

    def test_blocky_fluid_sub(self):
        """
        Tries to mimic the Blocky fluid Sub. tool in RokDoc, and compare our results with their

        NOTE:
        Porosity system = Effective porosity
        Single mineral = Quartz

        :return:
        """
        # Input parameters:
        vp_in, vs_in, rho_in = 3.5, 1.6, 2.3  # km/s, km/s, g/cm3
        k_qz, mu_qz, rho_qz = 36.6, 45, 2.65  # GPa, GPa, g/cm3
        k_oil, rho_oil = 1.152, 0.8  # GPa, g/cm3
        k_brine, rho_brine = 2.56, 1.0  # GPa, g/cm3
        sw_1 = 1.  # Initial fluid = 100% brine
        sw_2 = 0.2  # Final fluid = 80% oil, 20% brine

        # RokDoc results:
        k_dry_rd, mu_dry_rd, rho_dry_rd = 17.278, 5.888, 2.088  # GPa, GPa, g/cm3
        phi_rc = 0.212
        vp_rd, vs_rd, rho_rd = 3.4, 1.6, 2.266  # km/s, km/s, g/cm3

        # RokDoc results after porosity perturbation
        phi_pert = 0.2
        vp_rd_pert, vs_rd_pert, rho_rd_pert = 3.5, 1.6, 2.288  # km/s, km/s, g/cm3

        print('Parameter\t RokDoc\t Blixt')

        # Calculate porosity based on mass balance
        phi = rp.por_from_mass_balance(rho_in, rho_qz, rho_brine)
        print('Porosity:\t {:.3}\t {:.3}'.format(phi_rc, phi))

        # Calculate rho dry (mix minerals, assume no weight in pores)
        rho_dry = rp.vrh_bounds([1. - phi, phi], [rho_qz, 0.])[0]  # Voigt mean
        print('Rho_dry:\t {:.3}\t {:.3}'.format(rho_dry_rd, rho_dry))

        # Calculate mu_dry
        mu_dry = rp.mu_from_v(vs_in, rho_in)  # mu_dry = mu_sat according to eq. 1.12 in Avseth
        print('Mu_dry (sat):\t {:.3}\t {:.3}'.format(mu_dry_rd, mu_dry))

        # Calculate k_dry
        k_sat = rp.k_from_v(vp_in, vs_in, rho_in)
        print('K_sat (sat):\t None \t {:.3}'.format(k_sat))
        k_dry = rp.k_dry(k_sat, k_qz, k_brine, phi)  # This is the proper k_dry according to eq. 1.11 in Avseth
        # k_dry = rp.vrh_bounds([1. - phi, phi], [k_qz, 0.])[0]  # Voigt mean. THIS IS TEST AND IS WRONG
        print('K_dry:\t {:.3}\t {:.3}'.format(k_dry_rd, k_dry))

        # Calculate fluid substituted properties
        k_fl_2 = rp.vrh_bounds([sw_2, 1.-sw_2], [k_brine, k_oil])[1]  # Reuss average
        rho_fl_2 = rp.vrh_bounds([sw_2, 1.-sw_2], [rho_brine, rho_oil])[0]  # Voigt average
        print('K_fl_2: {:.3}, Rho_fl_2: {:.3}'.format(k_fl_2, rho_fl_2))

        vp, vs, rho, _k = rp.gassmann_vel(vp_in * 1000., vs_in * 1000., rho_in,
                                      k_brine, rho_brine,
                                      k_fl_2, rho_fl_2,
                                      k_qz, phi)
        print('Vp_2:\t {:.3} \t {:.3}'.format(vp_rd, vp/1000.))
        print('Vs_2:\t {:.3} \t {:.3}'.format(vs_rd, vs/1000.))
        print('Rho_2:\t {:.3} \t {:.3}'.format(rho_rd, rho))

        print('Perturb porosity to {:.3}:'.format(phi_pert))
        vp, vs, rho, _k = rp.vels(k_dry, mu_dry, k_qz, rho_qz, k_fl_2, rho_fl_2, phi_pert)
        print('Vp_3:\t {:.3} \t {:.3}\t Blixt vp decrease with decreasing phi +!'.format(vp_rd_pert, vp/1000.))
        print('Vs_3:\t {:.3} \t {:.3}'.format(vs_rd_pert, vs/1000.))
        print('Rho_3:\t {:.3} \t {:.3}'.format(rho_rd_pert, rho))
        phi_pert = 0.25
        print('\nPerturb porosity to {:.3}:'.format(phi_pert))
        vp_rd_pert, vs_rd_pert, rho_rd_pert = 3.3, 1.6, 2.197  # km/s, km/s, g/cm3
        vp, vs, rho, _k = rp.vels(k_dry, mu_dry, k_qz, rho_qz, k_fl_2, rho_fl_2, phi_pert)
        print('Vp_3:\t {:.3} \t {:.3}\t Blixt vp increase with increasing phi !'.format(vp_rd_pert, vp/1000.))
        print('Vs_3:\t {:.3} \t {:.3}'.format(vs_rd_pert, vs/1000.))
        print('Rho_3:\t {:.3} \t {:.3}'.format(rho_rd_pert, rho))

    def test_k_sat(self):
        k_dry, k_min, rho_min, k_fluid, rho_fluid, phi = [10., 15., 20.], 36.6, 2.65, 1.29, 0.84, np.linspace(0.0, 1.0)

        def t(_phi, _a):
            # second term on right hand side of eq. 1.9 (Avseth) where k_min have been taken out
            # after assuming k_phi = a*k_min
            return 1. + _phi/(_a + k_fluid/(k_min - k_fluid))

        # # Reproduces Figure 1.12 in Avseth (2011)
        # for a in np.linspace(0.05, 0.5, num=5):
        #     plt.plot(phi, 1./t(phi, a))  # k_sat / k_min for different a (pore space stiffness)

        # plot k_sat / k_min
        for _k_dry in k_dry:
            plt.plot(phi, rp.k_sat(_k_dry, k_min, k_fluid, phi)/k_min)
            # plt.axhline(_k_dry / k_min, 0, 1)

        plt.plot(phi, phi*rho_fluid/rho_min + 1 - phi)

        plt.show()

    def test_gassmann(self):
        k_dry, k_min, rho_min, k_fluid, rho_fluid, phi = [10., 15., 20.], 36.6, 2.65, 1.29, 0.84, np.linspace(0.0, 1.0)
        for _k_dry in k_dry:

            # XXX Keeping k_dry and mu_dry constant across this porosity change is non-physical
            vp, vs, rho, k = rp.vels(_k_dry, 5.888, k_min, rho_min, k_fluid, rho_fluid, phi)  # XXX

            plt.plot(phi, vp/1000.)
            # _k, _mu = rp.softsand(k_min, 45, phi)
            # plt.plot(phi, rp.v_p(_k, _mu, rho))

            # _k, _mu = rp.stiffsand(k_min, 45, phi)
            # plt.plot(phi, rp.v_p(_k, _mu, rho))

            # plt.axhline(_k_dry / k_min, 0, 1 )
        plt.show()
        self.assertTrue(True)

    def test_rpt_parameters(self):
        rpt_params = rp.rpt_parameters(verbose=True)
        for key, item in rpt_params.items():
            print(key, item)



