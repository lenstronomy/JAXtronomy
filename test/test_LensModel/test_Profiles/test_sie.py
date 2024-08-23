__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
import jaxtronomy.Util.param_util as param_util
from jaxtronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.sie import SIE as SIE_ref


class TestSIE_NIE(object):
    """Tests the SIE methods with NIE = True."""

    def setup_method(self):
        self.sie = SIE(NIE=True)
        self.sie_ref = SIE_ref(NIE=True)

    def test_function(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f = self.sie.function(x, y, theta_E, e1, e2)
        f_ref = self.sie_ref.function(x, y, theta_E, e1, e2)
        npt.assert_almost_equal(f, f_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f_x, f_y = self.sie.derivatives(x, y, theta_E, e1, e2)
        f_x_ref, f_y_ref = self.sie_ref.derivatives(x, y, theta_E, e1, e2)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f_xx, f_xy, f_yx, f_yy = self.sie.hessian(x, y, theta_E, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.sie_ref.hessian(
            x, y, theta_E, e1, e2
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=5)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=5)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=5)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=5)

    def test_theta2rho(self):
        theta_E = np.array([7.8, 1.3, 0.1, 4.5])
        rho_ref = self.sie_ref.theta2rho(theta_E)
        rho = self.sie.theta2rho(theta_E)
        npt.assert_almost_equal(rho, rho_ref, decimal=8)

    def test_mass_3d(self):
        r = 1.1
        rho0 = 9.2347
        m_3d_ref = self.sie_ref.mass_3d(r, rho0)
        m_3d = self.sie.mass_3d(r, rho0)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_mass_3d_lens(self):
        r = 1.1
        theta_E = 1.00003234
        m_3d_ref = self.sie_ref.mass_3d_lens(r, theta_E)
        m_3d = self.sie.mass_3d_lens(r, theta_E)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_mass_2d(self):
        r = 1.1
        rho0 = 9.2347
        m_2d_ref = self.sie_ref.mass_2d(r, rho0)
        m_2d = self.sie.mass_2d(r, rho0)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_mass_2d_lens(self):
        r = 1.1
        theta_E = 1.00003234
        m_2d_ref = self.sie_ref.mass_2d_lens(r, theta_E)
        m_2d = self.sie.mass_2d_lens(r, theta_E)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_grav_pot(self):
        x = np.array([1, -2.4, 2.7])
        y = np.array([2, 0.27, -3.1])
        rho0 = 1.32
        grav_pot_ref = self.sie_ref.grav_pot(x, y, rho0, center_x=3, center_y=1)
        grav_pot = self.sie.grav_pot(x, y, rho0, center_x=3, center_y=1)
        npt.assert_almost_equal(grav_pot, grav_pot_ref, decimal=8)

    def test_density_lens(self):
        r = 1.1
        theta_E = 1.00003234
        density_ref = self.sie_ref.density_lens(r, theta_E)
        density = self.sie.density_lens(r, theta_E)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density(self):
        r = 1.1
        rho0 = 9.2347
        density_ref = self.sie_ref.density(r, rho0)
        density = self.sie.density(r, rho0)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density_2d(self):
        x = np.array([1, -2.4, 2.7])
        y = np.array([2, 0.27, -3.1])
        rho0 = 1.00003234
        density_ref = self.sie_ref.density_2d(x, y, rho0)
        density = self.sie.density_2d(x, y, rho0)
        npt.assert_almost_equal(density, density_ref, decimal=8)


class TestSIE_EPL(object):
    """Tests the SIE methods with NIE = False."""

    def setup_method(self):
        self.sie = SIE(NIE=False)
        self.sie_ref = SIE_ref(NIE=False)

    def test_function(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f = self.sie.function(x, y, theta_E, e1, e2)
        f_ref = self.sie_ref.function(x, y, theta_E, e1, e2)
        npt.assert_almost_equal(f, f_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f_x, f_y = self.sie.derivatives(x, y, theta_E, e1, e2)
        f_x_ref, f_y_ref = self.sie_ref.derivatives(x, y, theta_E, e1, e2)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1, -3.1, 2.7])
        y = np.array([2, 0.7, -2.1])
        theta_E = 1.0
        e1, e2 = 0.1, -0.3
        f_xx, f_xy, f_yx, f_yy = self.sie.hessian(x, y, theta_E, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.sie_ref.hessian(
            x, y, theta_E, e1, e2
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
