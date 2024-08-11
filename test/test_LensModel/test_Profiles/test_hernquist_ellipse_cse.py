__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.hernquist_ellipse_cse import (
    HernquistEllipseCSE as HernquistEllipseCSE_ref,
)
from jaxtronomy.LensModel.Profiles.hernquist_ellipse_cse import HernquistEllipseCSE


import numpy as np
import numpy.testing as npt
import pytest


class TestHernquistEllipseCSE(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.hernquist_cse_ref = HernquistEllipseCSE_ref()
        test_init = HernquistEllipseCSE()

    def test_function(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 2, "Rs": 2, "center_x": 0, "center_y": 0}

        f_ref = self.hernquist_cse_ref.function(x, y, e1=0, e2=0, **kwargs)
        f = HernquistEllipseCSE.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f, f_ref, decimal=8)

    def test_derivatives(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_x_ref, f_y_ref = self.hernquist_cse_ref.derivatives(
            x, y, e1=0, e2=0, **kwargs
        )
        f_x, f_y = HernquistEllipseCSE.derivatives(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.linspace(0.01, 5, 30)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.hernquist_cse_ref.hessian(
            x, y, e1=0, e2=0, **kwargs
        )
        f_xx, f_xy, f_yx, f_yy = HernquistEllipseCSE.hessian(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

    def test_normalization(self):
        sigma0 = 2.1
        Rs = 3
        q = 0.7
        result = HernquistEllipseCSE._normalization(sigma0, Rs, q)
        result_ref = self.hernquist_cse_ref._normalization(sigma0, Rs, q)
        npt.assert_almost_equal(result, result_ref, decimal=8)

    def test_density(self):
        r = 1
        rho0 = 2.1
        Rs = 3
        density = HernquistEllipseCSE.density(r, rho0, Rs)
        density_ref = self.hernquist_cse_ref.density(r, rho0, Rs)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density_lens(self):
        r = 1
        sigma0 = 2.1
        Rs = 3
        density_lens = HernquistEllipseCSE.density_lens(r, sigma0, Rs)
        density_lens_ref = self.hernquist_cse_ref.density_lens(r, sigma0, Rs)
        npt.assert_almost_equal(density_lens, density_lens_ref, decimal=8)

    def test_density_2d(self):
        x = 1.3
        y = 4.2
        rho0 = 2.1
        Rs = 3
        density_2d = HernquistEllipseCSE.density_2d(x, y, rho0, Rs)
        density_2d_ref = self.hernquist_cse_ref.density_2d(x, y, rho0, Rs)
        npt.assert_almost_equal(density_2d, density_2d_ref, decimal=8)

    def test_mass_2d(self):
        R = 1
        rho0 = 2.1
        Rs = 3
        m_2d = HernquistEllipseCSE.mass_2d(R, rho0, Rs)
        m_2d_ref = self.hernquist_cse_ref.mass_2d(R, rho0, Rs)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_mass_2d_lens(self):
        R = 1
        sigma0 = 2.1
        Rs = 3
        m_2d = HernquistEllipseCSE.mass_2d_lens(R, sigma0, Rs)
        m_2d_ref = self.hernquist_cse_ref.mass_2d_lens(R, sigma0, Rs)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

    def test_mass_3d(self):
        R = 1
        rho0 = 2.1
        Rs = 3
        m_3d = HernquistEllipseCSE.mass_3d(R, rho0, Rs)
        m_3d_ref = self.hernquist_cse_ref.mass_3d(R, rho0, Rs)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_mass_3d_lens(self):
        R = 1
        sigma0 = 2.1
        Rs = 3
        m_3d = HernquistEllipseCSE.mass_3d_lens(R, sigma0, Rs)
        m_3d_ref = self.hernquist_cse_ref.mass_3d_lens(R, sigma0, Rs)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
