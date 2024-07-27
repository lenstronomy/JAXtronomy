__author__ = "sibirrer"

from jaxtronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe
from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe as Pjaffe_ref

import numpy as np
import numpy.testing as npt
import pytest
import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp


class TestP_JAFFW(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.profile = PseudoJaffe()
        self.profile_ref = Pjaffe_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        values = self.profile.function(x, y, sigma0, Ra, Rs)
        values_ref = self.profile_ref.function(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values, values_ref, decimal=6)

        x = np.array([0])
        y = np.array([0])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        values = self.profile.function(x, y, sigma0, Ra, Rs)
        values_ref = self.profile_ref.function(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values, values_ref, decimal=7)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, sigma0, Ra, Rs)
        values_ref = self.profile_ref.function(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values, values_ref, decimal=6)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        f_x, f_y = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=7)

        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, sigma0, Ra, Rs)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=6)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, Ra, Rs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Ra, Rs
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_yx, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, Ra, Rs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Ra, Rs
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_yx, decimal=8)

    def test_mass_tot(self):
        rho0 = 1.0
        Ra, Rs = 0.5, 0.8
        values = self.profile.mass_tot(rho0, Ra, Rs)
        values_ref = self.profile_ref.mass_tot(rho0, Ra, Rs)
        npt.assert_almost_equal(values, values_ref, decimal=8)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, sigma0=1, Ra=0.5, Rs=0.8)
        mass_ref = self.profile_ref.mass_3d_lens(r=1, sigma0=1, Ra=0.5, Rs=0.8)
        npt.assert_almost_equal(mass, mass_ref, decimal=8)

    def test_grav_pot(self):
        x = 1
        y = 2
        rho0 = 1.0
        r = jnp.sqrt(x**2 + y**2)
        Ra, Rs = 0.5, 0.8
        grav_pot = self.profile.grav_pot(r, rho0, Ra, Rs)
        grav_pot_ref = self.profile_ref.grav_pot(r, rho0, Ra, Rs)
        npt.assert_almost_equal(grav_pot, grav_pot_ref, decimal=8)

    def test_density(self):
        x = 1
        y = 2
        rho0 = 1.0
        r = jnp.sqrt(x**2 + y**2)
        Ra, Rs = 0.5, 0.8
        density = self.profile.density(r, rho0, Ra, Rs)
        density_ref = self.profile_ref.density(r, rho0, Ra, Rs)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_density_2d(self):
        x = 1
        y = 2
        rho0 = 1.0
        Ra, Rs = 0.5, 0.8
        density2d = self.profile.density_2d(x, y, rho0, Ra, Rs)
        density2d_ref = self.profile_ref.density_2d(x, y, rho0, Ra, Rs)
        npt.assert_almost_equal(density2d, density2d_ref, decimal=8)

    def test_mass_2d(self):
        x = 1
        y = 2
        rho0 = 1.0
        r = jnp.sqrt(x**2 + y**2)
        Ra, Rs = 0.5, 0.8
        mass2d = self.profile.mass_2d(r, rho0, Ra, Rs)
        mass2d_ref = self.profile_ref.mass_2d(r, rho0, Ra, Rs)
        npt.assert_almost_equal(mass2d, mass2d_ref, decimal=8)

    def test_jax_jit(self):
        x = jnp.array([1])
        y = jnp.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        jitted = jax.jit(self.profile.function)
        npt.assert_almost_equal(
            self.profile.function(x, y, sigma0, Ra, Rs),
            jitted(x, y, sigma0, Ra, Rs),
            decimal=8,
        )

        jitted = jax.jit(self.profile.derivatives)
        npt.assert_array_almost_equal(
            self.profile.derivatives(x, y, sigma0, Ra, Rs),
            jitted(x, y, sigma0, Ra, Rs),
            decimal=8,
        )

        jitted = jax.jit(self.profile.hessian)
        npt.assert_array_almost_equal(
            self.profile.hessian(x, y, sigma0, Ra, Rs),
            jitted(x, y, sigma0, Ra, Rs),
            decimal=8,
        )


if __name__ == "__main__":
    pytest.main()
