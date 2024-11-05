import unittest
import numpy as np
import os

from blixt_rp.rp import rp_core as rp


def create_test_data(length, constant_fractions):
    fractions = [c*np.ones(length) for c in constant_fractions]
    constant_params = [1./c for c in constant_fractions]  # This makes the Voigt sum equal to two
    params = [c*np.ones(length) for c in constant_params]
    return fractions, params


def create_test_wells():
    import blixt_rp.core.well as brcw
    project_table = str(os.path.dirname(__file__).replace(
        'blixt_rp\\unit_tests', 'excels\\project_table.xlsx'))
    wp = brcw.Project(name='MyProject', project_table=project_table)
    wells = wp.load_all_wells()
    templates = wp.load_all_templates()
    wis = wp.load_all_wis()

    return wp, wells, templates, wis

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

    def test_wells(self):
        import blixt_rp.core.fluids as flds
        import blixt_rp.core.minerals as mnrls

        wp, wells, templates, wis = create_test_wells()
        log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie',
                     'Volume': 'vcl'}

        fm = flds.FluidMix()
        fm.read_excel(wp.project_table)
        fm.calc_elastics(wells, wis, templates)

        mm = mnrls.MineralMix()
        mm.read_excel(wp.project_table)
        mm.calc_elastics(wells, log_table, wis)

        fobj_init = fm.fluids['initial']['WELL_F']
        fobj_fin = fm.fluids['final']['WELL_F']
        for obj in [fobj_init, fobj_fin]:
            for wi, val in obj.items():
                print(wi)
                for this_fm in list(val.keys()):
                    print(' ', this_fm)
                    print('  ', val[this_fm].volume_fraction)
                    print('  ', val[this_fm].k.value)

        well = wells['6306_3_1ST2']

        # rho_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='rho', wis=wis, method='Voigt')
        # k_f1_dict = well.calc_vrh_bounds(fm.fluids['initial'], param='k', wis=wis, method='Reuss')
        rho_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='rho', wis=wis, method='Voigt')
        # k_f2_dict = well.calc_vrh_bounds(fm.fluids['final'], param='k', wis=wis, method='Reuss')
