__author__ = "sibirrer"

import jax
import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensModel.Profiles.gaussian import Gaussian as Gaussian_ref
from jaxtronomy.LensModel.Profiles.gaussian import Gaussian

from lenstronomy.LensModel.Profiles.gaussian_potential import (
    GaussianPotential as GaussianPotential_ref,
)
from jaxtronomy.LensModel.Profiles.gaussian_potential import GaussianPotential

jax.config.update("jax_enable_x64", True)


class TestGaussian(object):
    def setup_method(self):
        self.profile_ref = Gaussian_ref()
        self.profile = Gaussian()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma = 0.5
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=4)

        x = np.array([0])
        y = np.array([0])
        amp = 1.3
        sigma = 0.5
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=4)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_almost_equal(values_ref, values, decimal=4)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma = 0.5
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=6)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=6)
        # NOTE: This test fails with 32 bit floats
        x = np.array([0])
        y = np.array([0])
        amp = 1.3
        sigma = 0.5
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=6)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=6)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=6)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=6)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma = 0.5
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=6)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=6)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=6)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=6)

        x = np.array([0])
        y = np.array([0])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=6)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=6)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=6)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=6)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=6)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=6)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=6)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=6)

    def test_density(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        density = self.profile.density(r, amp, sigma)
        density_ref = self.profile_ref.density(r, amp, sigma)
        npt.assert_array_almost_equal(density_ref, density, decimal=6)

    def test_density2d(self):
        x = np.array([0, 1.0, 2.7, 3.4, 5.9])
        y = np.array([0, 1.3, 2.1, 3.2, 5.2])
        amp = 1.7
        sigma = 0.4
        density = self.profile.density_2d(x, y, amp, sigma)
        density_ref = self.profile_ref.density_2d(x, y, amp, sigma)
        npt.assert_array_almost_equal(density_ref, density, decimal=6)

    def test_mass2d(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        mass2d = self.profile.mass_2d(r, amp, sigma)
        mass2d_ref = self.profile_ref.mass_2d(r, amp, sigma)
        npt.assert_array_almost_equal(mass2d_ref, mass2d, decimal=6)

    def test_mass2d_lens(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        mass2d = self.profile.mass_2d_lens(r, amp, sigma)
        mass2d_ref = self.profile_ref.mass_2d_lens(r, amp, sigma)
        npt.assert_array_almost_equal(mass2d_ref, mass2d, decimal=6)

    def test_alpha_abs(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        alpha_abs = self.profile.alpha_abs(r, amp, sigma)
        alpha_abs_ref = self.profile_ref.alpha_abs(r, amp, sigma)
        npt.assert_array_almost_equal(alpha_abs_ref, alpha_abs, decimal=6)

    def test_d_alpha_dr(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma_x = 0.4
        sigma_y = 0.1
        d_alpha_dr = self.profile.d_alpha_dr(r, amp, sigma_x, sigma_y)
        d_alpha_dr_ref = self.profile_ref.d_alpha_dr(r, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(d_alpha_dr_ref, d_alpha_dr, decimal=6)

    def test_mass3d(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        mass3d = self.profile.mass_3d(r, amp, sigma)
        mass3d_ref = self.profile_ref.mass_3d(r, amp, sigma)
        npt.assert_array_almost_equal(mass3d_ref, mass3d, decimal=6)

    def test_mass3d_lens(self):
        r = np.array([0, 1.0, 2.7, 3.4, 5.9])
        amp = 1.7
        sigma = 0.4
        mass3d = self.profile.mass_3d_lens(r, amp, sigma)
        mass3d_ref = self.profile_ref.mass_3d_lens(r, amp, sigma)
        npt.assert_array_almost_equal(mass3d_ref, mass3d, decimal=6)

    def test_amp3d_to_2d(self):
        amp = 1.7
        sigma_x = 0.4
        sigma_y = 0.8
        amp2d = self.profile._amp3d_to_2d(amp, sigma_x, sigma_y)
        amp2d_ref = self.profile_ref._amp3d_to_2d(amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(amp2d_ref, amp2d, decimal=6)

    def test_amp2d_to_3d(self):
        amp = 1.7
        sigma_x = 0.4
        sigma_y = 0.8
        amp3d = self.profile._amp2d_to_3d(amp, sigma_x, sigma_y)
        amp3d_ref = self.profile_ref._amp2d_to_3d(amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(amp3d_ref, amp3d, decimal=6)

    def test_num_integral(self):
        r = np.array([0.1, 1.0, 2.7, 3.4, 5.9, 85])
        c = 0.4
        result_ref = []
        for i in range(len(r)):
            result_ref.append(self.profile_ref._num_integral(r[i], c))
        result_ref = np.array(result_ref)

        result = self.profile._num_integral(r, c)
        npt.assert_array_almost_equal(result_ref, result, decimal=8)


class TestGaussianPotential(object):
    def setup_method(self):
        self.profile_ref = GaussianPotential_ref()
        self.profile = GaussianPotential()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        values_ref = self.profile_ref.function(x, y, amp, sigma_x, sigma_y)
        values = self.profile.function(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        # NOTE: This test fails with 32 bit floats
        x = np.array([0])
        y = np.array([0])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        values_ref = self.profile_ref.function(x, y, amp, sigma_x, sigma_y)
        values = self.profile.function(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma_x, sigma_y)
        values = self.profile.function(x, y, amp, sigma_x, sigma_y)
        npt.assert_almost_equal(values_ref, values, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma_x, sigma_y)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)
        # NOTE: This test fails with 32 bit floats
        x = np.array([0])
        y = np.array([0])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma_x, sigma_y)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, amp, sigma_x, sigma_y)
        f_x, f_y = self.profile.derivatives(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma_x, sigma_y
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)
        x = np.array([0])
        y = np.array([0])
        amp = 1.3
        sigma_x, sigma_y = 0.5, 0.7
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma_x, sigma_y
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, amp, sigma_x, sigma_y
        )
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, amp, sigma_x, sigma_y)
        npt.assert_array_almost_equal(f_xx_ref, f_xx, decimal=8)
        npt.assert_array_almost_equal(f_xy_ref, f_xy, decimal=8)
        npt.assert_array_almost_equal(f_yx_ref, f_yx, decimal=8)
        npt.assert_array_almost_equal(f_yy_ref, f_yy, decimal=8)


if __name__ == "__main__":
    pytest.main()
