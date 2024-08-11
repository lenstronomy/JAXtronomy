__author__ = "sibirrer"

import jax
import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Hernquist_ref
from jaxtronomy.LensModel.Profiles.hernquist import Hernquist

jax.config.update("jax_enable_x64", True)


class TestHernquist(object):
    def setup_method(self):
        self.profile_ref = Hernquist_ref()
        test_init = Hernquist()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        sigma0 = 0.5
        values_ref = self.profile_ref.function(x, y, sigma0, Rs)
        values = Hernquist.function(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(values_ref, values, decimal=6)
        # NOTE: This test fails with 32 bit floats
        x = np.array([0])
        y = np.array([0])
        Rs = 1.0
        sigma0 = 0.5
        values_ref = self.profile_ref.function(x, y, sigma0, Rs)
        values = Hernquist.function(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(values_ref, values, decimal=6)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values_ref = self.profile_ref.function(x, y, sigma0, Rs)
        values = Hernquist.function(x, y, sigma0, Rs)
        npt.assert_almost_equal(values_ref, values, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        sigma0 = 0.5
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Rs)
        f_x, f_y = Hernquist.derivatives(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)
        # NOTE: This test fails with 32 bit floats
        x = np.array([0])
        y = np.array([0])
        Rs = 1.0
        sigma0 = 0.5
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Rs)
        f_x, f_y = Hernquist.derivatives(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Rs)
        f_x, f_y = Hernquist.derivatives(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        sigma0 = 0.5
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Rs
        )
        f_xx, f_xy, f_yx, f_yy = Hernquist.hessian(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)
        x = np.array([0])
        y = np.array([0])
        Rs = 1.0
        sigma0 = 0.5
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Rs
        )
        f_xx, f_xy, f_yx, f_yy = Hernquist.hessian(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Rs
        )
        f_xx, f_xy, f_yx, f_yy = Hernquist.hessian(x, y, sigma0, Rs)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)

    def test_mass_tot(self):
        rho0 = 1
        Rs = 3
        m_tot_ref = self.profile_ref.mass_tot(rho0, Rs)
        m_tot = Hernquist.mass_tot(rho0, Rs)
        npt.assert_almost_equal(m_tot, m_tot_ref, decimal=6)

    def test_grav_pot(self):
        x, y = 1, 0
        rho0 = 1
        Rs = 3
        grav_pot_ref = self.profile_ref.grav_pot(x, y, rho0, Rs, center_x=0, center_y=0)
        grav_pot = Hernquist.grav_pot(x, y, rho0, Rs, center_x=0, center_y=0)
        npt.assert_almost_equal(grav_pot, grav_pot_ref, decimal=8)

    def test_F(self):
        x = Hernquist._s
        F = Hernquist._F(x)
        F_ref = self.profile_ref._F(x)
        npt.assert_almost_equal(F, F_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
