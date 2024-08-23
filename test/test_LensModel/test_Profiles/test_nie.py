__author__ = "sibirrer"


import numpy as np
import numpy.testing as npt
import pytest
import jax
from jaxtronomy.LensModel.Profiles.nie import NIE, NIEMajorAxis
from lenstronomy.LensModel.Profiles.nie import NIE as NIE_ref
from lenstronomy.LensModel.Profiles.nie import NIEMajorAxis as NIEMajorAxis_ref

jax.config.update("jax_enable_x64", True)


class TestNIE(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.nie = NIE()
        self.nie_ref = NIE_ref()

    def test_function(self):
        y = np.array([1.0, 2])
        x = np.array([0.0, 0.0])
        theta_E = 1.0
        e1 = 0.1
        e2 = 0.3
        s = 0.00001

        f_ = self.nie.function(x, y, theta_E, e1, e2, s)
        f_ref = self.nie_ref.function(x, y, theta_E, e1, e2, s)
        npt.assert_almost_equal(f_, f_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        theta_E = 1.0
        e1 = 0.1
        e2 = 0.3
        s = 0.00001

        f_x, f_y = self.nie.derivatives(x, y, theta_E, e1, e2, s)
        f_x_ref, f_y_ref = self.nie_ref.derivatives(x, y, theta_E, e1, e2, s)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        theta_E = 1.0
        e1 = 0.1
        e2 = 0.3
        s = 0.00001

        f_xx, f_xy, f_yx, f_yy = self.nie.hessian(x, y, theta_E, e1, e2, s)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.nie_ref.hessian(
            x, y, theta_E, e1, e2, s
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=5)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=5)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=5)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=5)

    def test_density_lens(self):
        r = 3
        theta_E = 12.3
        s_scale, e1, e2 = 0.1, 0.3, -0.4

        density = self.nie.density_lens(r, theta_E, e1, e2, s_scale)
        density_ref = self.nie_ref.density_lens(r, theta_E, e1, e2, s_scale)
        npt.assert_almost_equal(density, density_ref, decimal=7)

    def test_mass_3d(self):
        r = 3
        theta_E = 12.3
        s_scale, e1, e2 = 0.1, 0.3, -0.4

        mass_3d = self.nie.mass_3d_lens(r, theta_E, e1, e2, s_scale)
        mass_3d_ref = self.nie_ref.mass_3d_lens(r, theta_E, e1, e2, s_scale)
        npt.assert_almost_equal(mass_3d, mass_3d_ref, decimal=7)

    def test_param_conv(self):
        theta_E = 12.3
        s_scale, e1, e2 = 0.1, 0.3, -0.4
        self.nie.set_static(theta_E, e1, e2, s_scale)
        self.nie_ref.set_static(theta_E, e1, e2, s_scale)

        b_static, s_static, q_static, phi_static = self.nie.param_conv(
            theta_E, e1, e2, s_scale
        )

        (
            b_static_ref,
            s_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.nie_ref.param_conv(theta_E, e1, e2, s_scale)
        npt.assert_almost_equal(b_static, b_static_ref, decimal=7)
        npt.assert_almost_equal(s_static, s_static_ref, decimal=7)
        npt.assert_almost_equal(q_static, q_static_ref, decimal=7)
        npt.assert_almost_equal(phi_static, phi_static_ref, decimal=7)

        theta_E = 11.3
        s_scale, e1, e2 = 0.2, 0.2, -0.3
        self.nie.set_dynamic()
        self.nie_ref.set_dynamic()
        b, s, q, phi_G = self.nie.param_conv(theta_E, e1, e2, s_scale)
        b_ref, s_ref, q_ref, phi_G_ref = self.nie_ref.param_conv(
            theta_E, e1, e2, s_scale
        )
        npt.assert_almost_equal(b, b_ref, decimal=7)
        npt.assert_almost_equal(s, s_ref, decimal=7)
        npt.assert_almost_equal(q, q_ref, decimal=7)
        npt.assert_almost_equal(phi_G, phi_G_ref, decimal=7)

        # Do the same test as before to make sure that the
        # static variables are correctly updated with new values
        theta_E = 4.3
        s_scale, e1, e2 = 2.3, 0.4, -0.2
        self.nie.set_static(theta_E, e1, e2, s_scale)
        self.nie_ref.set_static(theta_E, e1, e2, s_scale)

        b_static, s_static, q_static, phi_static = self.nie.param_conv(
            theta_E, e1, e2, s_scale
        )

        (
            b_static_ref,
            s_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.nie_ref.param_conv(theta_E, e1, e2, s_scale)
        npt.assert_almost_equal(b_static, b_static_ref, decimal=7)
        npt.assert_almost_equal(s_static, s_static_ref, decimal=7)
        npt.assert_almost_equal(q_static, q_static_ref, decimal=7)
        npt.assert_almost_equal(phi_static, phi_static_ref, decimal=7)

    def test_theta_E_prod_average2major_axis(self):
        theta_E = np.array([1.1, 2.1, 0.3])
        q = np.array([0.999999999, 0.000000001, 0.3])

        result = self.nie._theta_E_prod_average2major_axis(theta_E, q)
        result_ref = self.nie_ref._theta_E_prod_average2major_axis(theta_E, q)
        npt.assert_almost_equal(result, result_ref, decimal=6)


class TestNIEMajorAxis(object):

    def setup_method(self):
        self.nie = NIEMajorAxis  # Class methods have all been made static in jaxtronomy
        self.nie_ref = NIEMajorAxis_ref()
        test_init = NIEMajorAxis()

    def test_function(self):
        y = np.array([1.0, 2])
        x = np.array([0.0, 0.0])
        b, s, q = 0.1, 0.2, 0.3

        f_ = self.nie.function(x, y, b, s, q)
        f_ref = self.nie_ref.function(x, y, b, s, q)
        npt.assert_almost_equal(f_, f_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        b, s, q = 0.1, 0.2, 0.3

        f_x, f_y = self.nie.derivatives(x, y, b, s, q)
        f_x_ref, f_y_ref = self.nie_ref.derivatives(x, y, b, s, q)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        b, s, q = 0.1, 0.2, 0.3

        f_xx, f_xy, f_yx, f_yy = self.nie.hessian(x, y, b, s, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.nie_ref.hessian(x, y, b, s, q)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=6)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=6)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=6)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=6)

    def test_kappa(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        b, s, q = 0.1, 0.2, 0.3

        kappa = self.nie.kappa(x, y, b, s, q)
        kappa_ref = self.nie_ref.kappa(x, y, b, s, q)
        npt.assert_almost_equal(kappa, kappa_ref, decimal=6)

    def test_psi(self):
        x = np.array([1, 3, 4])
        y = np.array([2, -1, 3])
        s, q = 0.2, 0.3

        psi = self.nie._psi(x, y, s, q)
        psi_ref = self.nie_ref._psi(x, y, s, q)
        npt.assert_almost_equal(psi, psi_ref, decimal=6)


if __name__ == "__main__":
    pytest.main()
