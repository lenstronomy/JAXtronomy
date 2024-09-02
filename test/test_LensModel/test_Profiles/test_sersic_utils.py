from jaxtronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil as SersicUtil_ref

import numpy as np
import numpy.testing as npt
import pytest


class TestSersicUtils(object):

    def setup_method(self):
        self.sersicutil = SersicUtil(sersic_major_axis=False)
        self.sersicutil_ref = SersicUtil_ref(sersic_major_axis=False)

    def test_k_bn(self):
        n = 3
        Re = 2.1
        k, bn = self.sersicutil.k_bn(n, Re)
        k_ref, bn_ref = self.sersicutil_ref.k_bn(n, Re)
        npt.assert_array_almost_equal(k, k_ref, decimal=6)
        npt.assert_array_almost_equal(bn, bn_ref, decimal=6)

    def test_k_Re(self):
        n = 3
        k = 1.383482
        Re = self.sersicutil.k_Re(n, k)
        Re_ref = self.sersicutil_ref.k_Re(n, k)
        npt.assert_array_almost_equal(Re, Re_ref, decimal=6)

    def test_bn(self):
        n = np.linspace(start=0.2, stop=8, num=30)
        bn = self.sersicutil.b_n(n)
        bn_ref = self.sersicutil_ref.b_n(n)
        npt.assert_array_almost_equal(bn, bn_ref, decimal=6)

    def test_get_distance_from_center(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        e1, e2 = 0.1, -0.2
        center_x, center_y = 1, 0
        r = self.sersicutil.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        r_ref = self.sersicutil_ref.get_distance_from_center(
            x, y, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(r, r_ref, decimal=6)

    def test_x_reduced(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        n_sersic = 1.3
        r_eff = 2.1
        center_x, center_y = 1, 0
        x_reduced = self.sersicutil._x_reduced(
            x, y, n_sersic, r_eff, center_x, center_y
        )
        x_reduced_ref = self.sersicutil_ref._x_reduced(
            x, y, n_sersic, r_eff, center_x, center_y
        )
        npt.assert_array_almost_equal(x_reduced, x_reduced_ref, decimal=6)

    def test_alpha_eff(self):
        r_eff = 1.61298
        n_sersic = 1.4453
        k_eff = 1.288488
        alpha_eff = self.sersicutil._alpha_eff(r_eff, n_sersic, k_eff)
        alpha_eff_ref = self.sersicutil_ref._alpha_eff(r_eff, n_sersic, k_eff)
        npt.assert_array_almost_equal(alpha_eff, alpha_eff_ref, decimal=6)

    def test_alpha_abs(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        r_eff = 1.61298
        n_sersic = 1.4453
        k_eff = 1.288488
        center_x, center_y = 1, 0
        alpha = self.sersicutil.alpha_abs(
            x, y, n_sersic, r_eff, k_eff, center_x, center_y
        )
        alpha_ref = self.sersicutil_ref.alpha_abs(
            x, y, n_sersic, r_eff, k_eff, center_x, center_y
        )
        npt.assert_array_almost_equal(alpha, alpha_ref, decimal=6)

    def test_d_alpha_dr(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        r_eff = 3.61298
        n_sersic = 2.4453
        k_eff = 4.288488
        center_x, center_y = 1, 0
        d_alpha_dr = self.sersicutil.d_alpha_dr(
            x, y, n_sersic, r_eff, k_eff, center_x, center_y
        )
        d_alpha_dr_ref = self.sersicutil_ref.d_alpha_dr(
            x, y, n_sersic, r_eff, k_eff, center_x, center_y
        )
        npt.assert_array_almost_equal(d_alpha_dr, d_alpha_dr_ref, decimal=6)

    def test_density(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        r_eff = 1.81298
        n_sersic = 1.6453
        k_eff = 1.188488
        center_x, center_y = 1, 0
        npt.assert_raises(
            ValueError,
            self.sersicutil.density,
            x,
            y,
            n_sersic,
            r_eff,
            k_eff,
            center_x,
            center_y,
        )

    def test__total_flux(self):
        r_eff = 1.31298
        n_sersic = 1.1453
        I_eff = 1.388488
        total_flux = self.sersicutil._total_flux(r_eff, I_eff, n_sersic)
        total_flux_ref = self.sersicutil_ref._total_flux(r_eff, I_eff, n_sersic)
        npt.assert_array_almost_equal(total_flux, total_flux_ref, decimal=6)

    def test_total_flux(self):
        amp = 4.28319
        R_sersic = 1.388488
        n_sersic = 1.1453
        e1, e2 = 0.12938, -0.32187
        total_flux = self.sersicutil.total_flux(amp, R_sersic, n_sersic, e1, e2)
        total_flux_ref = self.sersicutil_ref.total_flux(amp, R_sersic, n_sersic, e1, e2)
        npt.assert_array_almost_equal(total_flux, total_flux_ref, decimal=6)

    def test_R_stable(self):
        R = np.linspace(start=0.0, stop=0.5, num=50)
        R_stable = self.sersicutil._R_stable(R)
        R_stable_ref = self.sersicutil_ref._R_stable(R)
        npt.assert_array_almost_equal(R_stable, R_stable_ref, decimal=6)

    def test_r_sersic(self):
        R = 0.2178482
        R_sersic = 1.588488
        n_sersic = 2.1453
        r_sersic = self.sersicutil._r_sersic(R, R_sersic, n_sersic)
        r_sersic_ref = self.sersicutil_ref._r_sersic(R, R_sersic, n_sersic)
        npt.assert_array_almost_equal(r_sersic, r_sersic_ref, decimal=6)


class TestSersicUtilsMajorAxis(object):

    def setup_method(self):
        self.sersicutil = SersicUtil(sersic_major_axis=True)
        self.sersicutil_ref = SersicUtil_ref(sersic_major_axis=True)

    def test_get_distance_from_center(self):
        x = np.array([1, 3, 4, 2.3217, -1.3])
        y = np.array([0, 1.123, -2.3, 4, -2])
        e1, e2 = 0.1, -0.2
        center_x, center_y = 1, 0
        r = self.sersicutil.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        r_ref = self.sersicutil_ref.get_distance_from_center(
            x, y, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(r, r_ref, decimal=6)

    def test_total_flux(self):
        amp = 4.28319
        R_sersic = 1.388488
        n_sersic = 1.1453
        e1, e2 = 0.12938, -0.32187
        total_flux = self.sersicutil.total_flux(amp, R_sersic, n_sersic, e1, e2)
        total_flux_ref = self.sersicutil_ref.total_flux(amp, R_sersic, n_sersic, e1, e2)
        npt.assert_array_almost_equal(total_flux, total_flux_ref, decimal=6)


if __name__ == "__main__":
    pytest.main()
