__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.epl import EPL as EPL_ref
from lenstronomy.LensModel.Profiles.epl import EPLMajorAxis as EPLMajorAxis_ref
from lenstronomy.LensModel.Profiles.epl import EPLQPhi as EPLQPhi_ref
from jaxtronomy.LensModel.Profiles.epl import EPL, EPLMajorAxis, EPLQPhi

import numpy as np
import numpy.testing as npt
import pytest
import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp


class TestEPL(object):

    def setup_method(self):
        self.profile = EPL()
        self.profile_ref = EPL_ref()

    def test_param_conv(self):
        theta_E = 12.3
        gamma, e1, e2 = 1.7, 0.3, -0.4
        self.profile.set_static(theta_E, gamma, e1, e2)
        self.profile_ref.set_static(theta_E, gamma, e1, e2)

        b_static, t_static, q_static, phi_static = self.profile.param_conv(
            theta_E, gamma, e1, e2
        )

        (
            b_static_ref,
            t_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.profile_ref.param_conv(theta_E, gamma, e1, e2)
        npt.assert_allclose(b_static, b_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(t_static, t_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(q_static, q_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(phi_static, phi_static_ref, atol=1e-12, rtol=1e-12)

        theta_E = 11.3
        gamma, e1, e2 = 1.6, 0.2, -0.3
        self.profile.set_dynamic()
        self.profile_ref.set_dynamic()
        b, t, q, phi_G = self.profile.param_conv(theta_E, gamma, e1, e2)
        b_ref, t_ref, q_ref, phi_G_ref = self.profile_ref.param_conv(
            theta_E, gamma, e1, e2
        )
        npt.assert_allclose(b, b_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(t, t_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(q, q_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(phi_G, phi_G_ref, atol=1e-12, rtol=1e-12)

        # Do the same test as before to make sure that the
        # static variables are correctly updated with new values
        theta_E = 4.3
        gamma, e1, e2 = 2.3, 0.4, -0.2
        self.profile.set_static(theta_E, gamma, e1, e2)
        self.profile_ref.set_static(theta_E, gamma, e1, e2)

        b_static, t_static, q_static, phi_static = self.profile.param_conv(
            theta_E, gamma, e1, e2
        )

        (
            b_static_ref,
            t_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.profile_ref.param_conv(theta_E, gamma, e1, e2)
        npt.assert_allclose(b_static, b_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(t_static, t_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(q_static, q_static_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(phi_static, phi_static_ref, atol=1e-12, rtol=1e-12)

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 12.3
        gamma, e1, e2 = 1.7, 0.3, -0.2
        values = self.profile.function(x, y, theta_E, gamma, e1, e2)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, e1, e2)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, theta_E, gamma, e1, e2)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, e1, e2)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, e1, e2 = 1.7, -0.1, 0.2
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, e1, e2)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, e1, e2)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, e1, e2 = 2.2, 0.1, -0.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, e1, e2
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, e1, e2
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        mass_ref = self.profile_ref.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        npt.assert_allclose(mass, mass_ref, atol=1e-12, rtol=1e-12)

    def test_density_lens(self):
        x = 1
        y = 2
        theta_E = 1.0
        r = jnp.sqrt(x**2 + y**2)
        gamma = 2.1
        density = self.profile.density_lens(r, theta_E, gamma)
        density_ref = self.profile_ref.density_lens(r, theta_E, gamma)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)


class TestEPLMajorAxis(object):

    def setup_method(self):
        self.profile = EPLMajorAxis()
        self.profile_ref = EPLMajorAxis_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        b, t, q = 1.7, 0.7, 0.7
        values = self.profile.function(x, y, b, t, q)
        values_ref = self.profile_ref.function(x, y, b, t, q)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, b, t, q)
        values_ref = self.profile_ref.function(x, y, b, t, q)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        b, t, q = 1.7, 0.9, 0.8
        f_x, f_y = self.profile.derivatives(x, y, b, t, q)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, b, t, q)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        y = np.linspace(-10, 10, 101)
        x = np.ones_like(y)
        list_of_q = np.linspace(0.03, 1, 200)[::-1]
        for q in list_of_q:
            f_x, f_y = self.profile.derivatives(x, y, b, t, q)
            f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, b, t, q)
            npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
            npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        b, t, q = 2.2, 0.6, 0.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, b, t, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(x, y, b, t, q)
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, b, t, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(x, y, b, t, q)
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)


class TestEPLQPhi(object):

    def setup_method(self):
        self.profile = EPLQPhi()
        self.profile_ref = EPLQPhi_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 12.3
        gamma, q, phi = 1.7, 0.3, -2.2
        values = self.profile.function(x, y, theta_E, gamma, q, phi)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, q, phi)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, theta_E, gamma, q, phi)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, q, phi)
        npt.assert_allclose(values, values_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, q, phi = 1.7, 0.7, 2.2
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, q, phi)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, q, phi)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, q, phi)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, q, phi)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, q, phi = 2.2, 0.4, 1.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, q, phi)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, q, phi
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, q, phi)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, q, phi
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_xy, f_yx, atol=1e-12, rtol=1e-12)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        mass_ref = self.profile_ref.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        npt.assert_allclose(mass, mass_ref, atol=1e-12, rtol=1e-12)

    def test_density_lens(self):
        x = 3
        y = 2
        theta_E = 1.0
        r = jnp.sqrt(x**2 + y**2)
        gamma = 2.1
        density = self.profile.density_lens(r, theta_E, gamma)
        density_ref = self.profile_ref.density_lens(r, theta_E, gamma)
        npt.assert_allclose(density, density_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
