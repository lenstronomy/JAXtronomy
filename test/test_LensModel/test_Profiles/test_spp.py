__author__ = "sibirrer"

from jaxtronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.spp import SPP as SPP_ref

import numpy as np
import numpy.testing as npt
import pytest


class TestSPP(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.SPP_ref = SPP_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.3
        gamma = 1.9

        values_ref = self.SPP_ref.function(x, y, theta_E, gamma)
        values = SPP.function(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(values, values_ref, decimal=8)

        x = np.array([0])
        y = np.array([0])
        values_ref = self.SPP_ref.function(x, y, theta_E, gamma)
        values = SPP.function(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(values, values_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values_ref = self.SPP_ref.function(x, y, theta_E, gamma)
        values = SPP.function(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(values, values_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.3
        gamma = 1.9

        f_x_ref, f_y_ref = self.SPP_ref.derivatives(x, y, theta_E, gamma)
        f_x, f_y = SPP.derivatives(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([0])
        y = np.array([0])
        f_x_ref, f_y_ref = self.SPP_ref.derivatives(x, y, theta_E, gamma)
        f_x, f_y = SPP.derivatives(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_x_ref, f_y_ref = self.SPP_ref.derivatives(x, y, theta_E, gamma)
        f_x, f_y = SPP.derivatives(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.3
        gamma = 1.9

        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.SPP_ref.hessian(
            x, y, theta_E, gamma
        )
        f_xx, f_xy, f_yx, f_yy = SPP.hessian(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

        x = np.array([0])
        y = np.array([0])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.SPP_ref.hessian(
            x, y, theta_E, gamma
        )
        f_xx, f_xy, f_yx, f_yy = SPP.hessian(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.SPP_ref.hessian(
            x, y, theta_E, gamma
        )
        f_xx, f_xy, f_yx, f_yy = SPP.hessian(x, y, theta_E, gamma)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

    def test_rho2theta(self):
        rho0 = 3.8
        gamma = 1.7
        theta_ref = self.SPP_ref.rho2theta(rho0, gamma)
        theta = SPP.rho2theta(rho0, gamma)
        npt.assert_almost_equal(theta, theta_ref, decimal=8)

    def test_theta2rho(self):
        theta_E = 7.8
        gamma = 2.4
        rho_ref = self.SPP_ref.theta2rho(theta_E, gamma)
        rho = SPP.theta2rho(theta_E, gamma)
        npt.assert_almost_equal(rho, rho_ref, decimal=8)

    def test_mass_3d(self):
        r = 1.1
        rho0 = 9.2347
        gamma = 2.12989
        m_3d_ref = self.SPP_ref.mass_3d(r, rho0, gamma)
        m_3d = SPP.mass_3d(r, rho0, gamma)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_mass_3d_lens(self):
        r = 1.1
        theta_E = 1.00003234
        gamma = 2.12989
        m_3d_ref = self.SPP_ref.mass_3d_lens(r, theta_E, gamma)
        m_3d = SPP.mass_3d_lens(r, theta_E, gamma)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_mass_2d(self):
        r = 1.1
        rho0 = 9.2347
        gamma = 2.12989
        m_2d_ref = self.SPP_ref.mass_2d(r, rho0, gamma)
        m_2d = SPP.mass_2d(r, rho0, gamma)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_mass_2d_lens(self):
        r = 1.1
        theta_E = 1.00003234
        gamma = 2.12989
        m_2d_ref = self.SPP_ref.mass_2d_lens(r, theta_E, gamma)
        m_2d = SPP.mass_2d_lens(r, theta_E, gamma)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_grav_pot(self):
        x, y = 1, 0.27
        rho0 = 1.32
        gamma = 2.49832
        grav_pot_ref = self.SPP_ref.grav_pot(x, y, rho0, gamma, center_x=0, center_y=0)
        grav_pot = SPP.grav_pot(x, y, rho0, gamma, center_x=0, center_y=0)
        npt.assert_almost_equal(grav_pot, grav_pot_ref, decimal=8)

    def test_density(self):
        r = 1.1
        rho0 = 9.2347
        gamma = 2.12989
        density_ref = self.SPP_ref.density(r, rho0, gamma)
        density = SPP.density(r, rho0, gamma)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density_lens(self):
        r = 1.1
        theta_E = 1.00003234
        gamma = 2.12989
        density_ref = self.SPP_ref.density_lens(r, theta_E, gamma)
        density = SPP.density_lens(r, theta_E, gamma)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density_2d(self):
        x = 1.1
        y = 2.38123
        rho0 = 1.00003234
        gamma = 2.12989
        density_ref = self.SPP_ref.density_2d(x, y, rho0, gamma)
        density = SPP.density_2d(x, y, rho0, gamma)
        npt.assert_almost_equal(density, density_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
