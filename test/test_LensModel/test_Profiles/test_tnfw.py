__author__ = "sibirrer"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.tnfw import TNFW as TNFW_ref


class TestTNFW(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.tnfw_ref = TNFW_ref()

    def test_function(self):
        x = np.array([0])
        y = np.array([0])
        Rs = 1.4
        alpha_Rs = 3.5
        r_trunc = 3.2
        values_ref = self.tnfw_ref.function(x, y, Rs, alpha_Rs, r_trunc)
        values = TNFW.function(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(values_ref, values, atol=1e-12, rtol=1e-12)

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        Rs = 2.4
        alpha_Rs = 1.5
        r_trunc = 1.2
        values_ref = self.tnfw_ref.function(x, y, Rs, alpha_Rs, r_trunc)
        values = TNFW.function(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(values_ref, values, atol=1e-12, rtol=1e-12)

        Rs = 4
        alpha_Rs = 0.5
        r_trunc = 8
        values_ref = self.tnfw_ref.function(x, y, Rs, alpha_Rs, r_trunc)
        values = TNFW.function(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(values_ref, values, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        x = np.array([0])
        y = np.array([0])
        Rs = 1.4
        alpha_Rs = 3.5
        r_trunc = 3.2
        f_x_ref, f_y_ref = self.tnfw_ref.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        f_x, f_y = TNFW.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_x_ref, f_x, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y_ref, f_y, atol=1e-12, rtol=1e-12)

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        Rs = 2.4
        alpha_Rs = 1.5
        r_trunc = 1.2
        f_x_ref, f_y_ref = self.tnfw_ref.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        f_x, f_y = TNFW.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_x_ref, f_x, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y_ref, f_y, atol=1e-12, rtol=1e-12)

        Rs = 4
        alpha_Rs = 0.5
        r_trunc = 8
        f_x_ref, f_y_ref = self.tnfw_ref.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        f_x, f_y = TNFW.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_x_ref, f_x, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y_ref, f_y, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        x = np.array([0])
        y = np.array([0])
        Rs = 1.4
        alpha_Rs = 3.5
        r_trunc = 3.2
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.tnfw_ref.hessian(
            x, y, Rs, alpha_Rs, r_trunc
        )
        f_xx, f_xy, f_yx, f_yy = TNFW.hessian(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_xx_ref, f_xx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy_ref, f_xy, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yx_ref, f_yx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy_ref, f_yy, atol=1e-12, rtol=1e-12)

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        Rs = 2.4
        alpha_Rs = 1.5
        r_trunc = 1.2
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.tnfw_ref.hessian(
            x, y, Rs, alpha_Rs, r_trunc
        )
        f_xx, f_xy, f_yx, f_yy = TNFW.hessian(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_xx_ref, f_xx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy_ref, f_xy, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yx_ref, f_yx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy_ref, f_yy, atol=1e-12, rtol=1e-12)

        Rs = 4
        alpha_Rs = 0.5
        r_trunc = 8
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.tnfw_ref.hessian(
            x, y, Rs, alpha_Rs, r_trunc
        )
        f_xx, f_xy, f_yx, f_yy = TNFW.hessian(x, y, Rs, alpha_Rs, r_trunc)
        npt.assert_allclose(f_xx_ref, f_xx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy_ref, f_xy, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yx_ref, f_yx, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy_ref, f_yy, atol=1e-12, rtol=1e-12)

    def test_density(self):
        R = np.linspace(0, 10, 100)
        Rs = 3.2
        rho0 = 1.1
        r_trunc = 1.4
        density_ref = self.tnfw_ref.density(R, Rs, rho0, r_trunc)
        density = TNFW.density(R, Rs, rho0, r_trunc)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.2
        rho0 = 1.3
        r_trunc = 0.6
        density_ref = self.tnfw_ref.density(R, Rs, rho0, r_trunc)
        density = TNFW.density(R, Rs, rho0, r_trunc)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        density_ref = self.tnfw_ref.density(R, Rs, rho0, r_trunc)
        density = TNFW.density(R, Rs, rho0, r_trunc)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)

    def test_density_2d(self):
        x = np.array([0])
        y = np.array([0])
        Rs = 3.2
        rho0 = 1.1
        r_trunc = 1.54
        density_ref = self.tnfw_ref.density_2d(x, y, Rs, rho0, r_trunc)
        density = TNFW.density_2d(x, y, Rs, rho0, r_trunc)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        density_ref = self.tnfw_ref.density_2d(x, y, Rs, rho0, r_trunc)
        density = TNFW.density_2d(x, y, Rs, rho0, r_trunc)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)

    def test_mass_2d(self):
        R = np.linspace(0, 10, 100)
        Rs = 1.2
        rho0 = 1.3
        r_trunc = 1.5
        m_2d_ref = self.tnfw_ref.mass_2d(R, Rs, rho0, r_trunc)
        m_2d = TNFW.mass_2d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_2d, m_2d_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.3
        rho0 = 1.5
        r_trunc = 1.3
        m_2d_ref = self.tnfw_ref.mass_2d(R, Rs, rho0, r_trunc)
        m_2d = TNFW.mass_2d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_2d, m_2d_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        m_2d_ref = self.tnfw_ref.mass_2d(R, Rs, rho0, r_trunc)
        m_2d = TNFW.mass_2d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_2d, m_2d_ref, atol=1e-12, rtol=1e-12)

    def test_mass_3d(self):
        R = np.linspace(0, 10, 100)
        Rs = 1.2
        rho0 = 1.3
        r_trunc = 1.5
        m_3d_ref = self.tnfw_ref.mass_3d(R, Rs, rho0, r_trunc)
        m_3d = TNFW.mass_3d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_3d, m_3d_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.3
        rho0 = 1.5
        r_trunc = 1.3
        m_3d_ref = self.tnfw_ref.mass_3d(R, Rs, rho0, r_trunc)
        m_3d = TNFW.mass_3d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_3d, m_3d_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        m_3d_ref = self.tnfw_ref.mass_3d(R, Rs, rho0, r_trunc)
        m_3d = TNFW.mass_3d(R, Rs, rho0, r_trunc)
        npt.assert_allclose(m_3d, m_3d_ref, atol=1e-12, rtol=1e-12)

    def test_tnfw_potential(self):
        R = np.linspace(0, 10, 100)
        Rs = 1.2
        rho0 = 1.3
        r_trunc = 1.5
        potential_ref = self.tnfw_ref.tnfw_potential(R, Rs, rho0, r_trunc)
        potential = TNFW.tnfw_potential(R, Rs, rho0, r_trunc)
        npt.assert_allclose(potential, potential_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.3
        rho0 = 1.5
        r_trunc = 1.3
        potential_ref = self.tnfw_ref.tnfw_potential(R, Rs, rho0, r_trunc)
        potential = TNFW.tnfw_potential(R, Rs, rho0, r_trunc)
        npt.assert_allclose(potential, potential_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        potential_ref = self.tnfw_ref.tnfw_potential(R, Rs, rho0, r_trunc)
        potential = TNFW.tnfw_potential(R, Rs, rho0, r_trunc)
        npt.assert_allclose(potential, potential_ref, atol=1e-12, rtol=1e-12)

    def test_tnfw_alpha(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        R = np.sqrt(x**2 + y**2)
        Rs = 1.2
        rho0 = 1.3
        r_trunc = 1.5
        alpha_ref = self.tnfw_ref.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        alpha = TNFW.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.3
        rho0 = 1.5
        r_trunc = 1.3
        alpha_ref = self.tnfw_ref.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        alpha = TNFW.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        alpha_ref = self.tnfw_ref.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        alpha = TNFW.tnfw_alpha(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)

    def test_tnfw_gamma(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)[::-1]
        R = np.sqrt(x**2 + y**2)
        Rs = 1.2
        rho0 = 1.3
        r_trunc = 1.5
        gamma_ref = self.tnfw_ref.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        gamma = TNFW.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(gamma, gamma_ref, atol=1e-12, rtol=1e-12)

        Rs = 2.3
        rho0 = 1.5
        r_trunc = 1.3
        gamma_ref = self.tnfw_ref.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        gamma = TNFW.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(gamma, gamma_ref, atol=1e-12, rtol=1e-12)

        Rs = 4
        rho0 = 1.1
        r_trunc = 8
        gamma_ref = self.tnfw_ref.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        gamma = TNFW.tnfw_gamma(R, Rs, rho0, r_trunc, x, y)
        npt.assert_allclose(gamma, gamma_ref, atol=1e-12, rtol=1e-12)

    def test__L(self):
        x = np.linspace(0, 5, 100)
        tau = 0.3
        _L_ref = self.tnfw_ref._L(x, tau)
        _L = TNFW._L(x, tau)
        npt.assert_allclose(_L, _L_ref, atol=1e-12, rtol=1e-12)

        tau = 1.3
        _L_ref = self.tnfw_ref._L(x, tau)
        _L = TNFW._L(x, tau)
        npt.assert_allclose(_L, _L_ref, atol=1e-12, rtol=1e-12)

        tau = 3.3
        _L_ref = self.tnfw_ref._L(x, tau)
        _L = TNFW._L(x, tau)
        npt.assert_allclose(_L, _L_ref, atol=1e-12, rtol=1e-12)

    def test_F(self):
        x = np.linspace(0, 4, 100)
        F_ref = self.tnfw_ref.F(x)
        F = TNFW.F(x)
        npt.assert_allclose(F, F_ref, atol=1e-12, rtol=1e-12)

    def test__F(self):
        x = np.linspace(0, 5, 100)
        tau = 0.3
        _F_ref = self.tnfw_ref._F(x, tau)
        _F = TNFW._F(x, tau)
        npt.assert_allclose(_F, _F_ref, atol=1e-12, rtol=1e-12)

        tau = 1.3
        _F_ref = self.tnfw_ref._F(x, tau)
        _F = TNFW._F(x, tau)
        npt.assert_allclose(_F, _F_ref, atol=1e-12, rtol=1e-12)

        tau = 3.3
        _F_ref = self.tnfw_ref._F(x, tau)
        _F = TNFW._F(x, tau)
        npt.assert_allclose(_F, _F_ref, atol=1e-12, rtol=1e-12)

    def test__g(self):
        x = np.linspace(0, 5, 100)
        tau = 0.3
        _g_ref = self.tnfw_ref._g(x, tau)
        _g = TNFW._g(x, tau)
        npt.assert_allclose(_g, _g_ref, atol=1e-12, rtol=1e-12)

        tau = 1.3
        _g_ref = self.tnfw_ref._g(x, tau)
        _g = TNFW._g(x, tau)
        npt.assert_allclose(_g, _g_ref, atol=1e-12, rtol=1e-12)

        tau = 3.3
        _g_ref = self.tnfw_ref._g(x, tau)
        _g = TNFW._g(x, tau)
        npt.assert_allclose(_g, _g_ref, atol=1e-12, rtol=1e-12)

    def test_cos_function(self):
        x = np.linspace(0, 4, 100)
        cos_ref = self.tnfw_ref._cos_function(x)
        cos = TNFW._cos_function(x)
        npt.assert_allclose(cos, cos_ref, atol=1e-12, rtol=1e-12)

    def test__h(self):
        x = np.linspace(0, 5, 100)
        tau = 0.3
        _h_ref = self.tnfw_ref._h(x, tau)
        _h = TNFW._h(x, tau)
        npt.assert_allclose(_h, _h_ref, atol=1e-12, rtol=1e-12)

        tau = 1.3
        _h_ref = self.tnfw_ref._h(x, tau)
        _h = TNFW._h(x, tau)
        npt.assert_allclose(_h, _h_ref, atol=1e-12, rtol=1e-12)

        tau = 3.3
        _h_ref = self.tnfw_ref._h(x, tau)
        _h = TNFW._h(x, tau)
        npt.assert_allclose(_h, _h_ref, atol=1e-12, rtol=1e-12)

    def test_alpha2rho(self):
        alpha_Rs = 1.5
        Rs = 1.9
        rho_ref = self.tnfw_ref.alpha2rho0(alpha_Rs, Rs)
        rho = TNFW.alpha2rho0(alpha_Rs, Rs)
        npt.assert_allclose(rho, rho_ref, atol=1e-12, rtol=1e-12)

        alpha_Rs = 0.3
        Rs = 0.1
        rho_ref = self.tnfw_ref.alpha2rho0(alpha_Rs, Rs)
        rho = TNFW.alpha2rho0(alpha_Rs, Rs)
        npt.assert_allclose(rho, rho_ref, atol=1e-12, rtol=1e-12)

    def test_rho2alpha(self):
        rho0 = 1.5
        Rs = 1.9
        alpha_ref = self.tnfw_ref.alpha2rho0(rho0, Rs)
        alpha = TNFW.alpha2rho0(rho0, Rs)
        npt.assert_allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)

        rho0 = 0.3
        Rs = 0.1
        alpha_ref = self.tnfw_ref.rho02alpha(rho0, Rs)
        alpha = TNFW.rho02alpha(rho0, Rs)
        npt.assert_allclose(alpha, alpha_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
