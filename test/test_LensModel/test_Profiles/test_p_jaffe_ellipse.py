__author__ = "sibirrer"


from jaxtronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential import (
    PseudoJaffeEllipsePotential,
)
from lenstronomy.LensModel.Profiles.pseudo_jaffe_ellipse_potential import (
    PseudoJaffeEllipsePotential as PJaffe_Ellipse_ref,
)
import jaxtronomy.Util.param_util as param_util
import lenstronomy.Util.param_util as param_util_ref

import numpy as np
import numpy.testing as npt
import pytest
import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp


class TestP_JAFFW(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.profile = PseudoJaffeEllipsePotential()
        self.profile_ref = PJaffe_Ellipse_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.profile.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        e1, e2 = param_util_ref.phi_q2_ellipticity(phi_G, q)
        values_ref = self.profile_ref.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values, values_ref, decimal=8)

        x = np.array([0])
        y = np.array([0])
        values = self.profile.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        values_ref = self.profile_ref.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(values, values_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        values_ref = self.profile_ref.function(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        e1, e2 = param_util_ref.phi_q2_ellipticity(phi_G, q)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        f_x_ref, f_y_ref = self.profile_ref.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        f_x_ref, f_y_ref = self.profile_ref.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        e1, e2 = param_util_ref.phi_q2_ellipticity(phi_G, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=6)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_yx, decimal=6)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, sigma0=1, Ra=0.5, Rs=0.8, e1=0, e2=0)
        mass_ref = self.profile_ref.mass_3d_lens(
            r=1, sigma0=1, Ra=0.5, Rs=0.8, e1=0, e2=0
        )
        npt.assert_almost_equal(mass, mass_ref, decimal=8)

    def test_jax_jit(self):
        x = jnp.array([1])
        y = jnp.array([2])
        sigma0 = 1.0
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0.1
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        jitted = jax.jit(self.profile.function)
        npt.assert_almost_equal(
            self.profile.function(x, y, sigma0, Ra, Rs, e1, e2),
            jitted(x, y, sigma0, Ra, Rs, e1, e2),
            decimal=8,
        )

        jitted = jax.jit(self.profile.derivatives)
        npt.assert_array_almost_equal(
            self.profile.derivatives(x, y, sigma0, Ra, Rs, e1, e2),
            jitted(x, y, sigma0, Ra, Rs, e1, e2),
            decimal=8,
        )

        jitted = jax.jit(self.profile.hessian)
        npt.assert_array_almost_equal(
            self.profile.hessian(x, y, sigma0, Ra, Rs, e1, e2),
            jitted(x, y, sigma0, Ra, Rs, e1, e2),
            decimal=8,
        )


if __name__ == "__main__":
    pytest.main()
