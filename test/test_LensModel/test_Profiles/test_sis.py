__author__ = "sibirrer"

from jaxtronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.sis import SIS as SIS_ref

import numpy as np
import numpy.testing as npt
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class TestSIS(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.SIS = SIS()
        self.SIS_ref = SIS_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        values = self.SIS.function(x, y, phi_E)
        values_ref = self.SIS_ref.function(x, y, phi_E)
        npt.assert_almost_equal(values[0], 2.2360679774997898, decimal=9)
        npt.assert_almost_equal(values, values_ref, decimal=9)
        x = np.array([0])
        y = np.array([0])
        phi_E = 1.0
        values = self.SIS.function(x, y, phi_E)
        assert values[0] == 0

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.SIS.function(x, y, phi_E)
        values_ref = self.SIS_ref.function(x, y, phi_E)
        npt.assert_almost_equal(values[0], 2.2360679774997898, decimal=9)
        npt.assert_almost_equal(values[1], 3.1622776601683795, decimal=9)
        npt.assert_almost_equal(values[2], 4.1231056256176606, decimal=9)
        npt.assert_almost_equal(values, values_ref, decimal=9)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        f_x, f_y = self.SIS.derivatives(x, y, phi_E)
        f_x_ref, f_y_ref = self.SIS_ref.derivatives(x, y, phi_E)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=9)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=9)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.SIS.derivatives(x, y, phi_E)
        assert f_x[0] == 0
        assert f_y[0] == 0

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        f_xx, f_xy, f_yx, f_yy = self.SIS.hessian(x, y, phi_E)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.SIS_ref.hessian(x, y, phi_E)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=9)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=9)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=9)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=9)
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.SIS.hessian(x, y, phi_E)
        values_ref = self.SIS_ref.hessian(x, y, phi_E)
        npt.assert_almost_equal(values, values_ref)

    def test_theta2rho(self):
        theta_E = 2.0
        rho0 = self.SIS.theta2rho(theta_E)
        theta_E_new = self.SIS.rho2theta(rho0)
        rho0_ref = self.SIS_ref.theta2rho(theta_E)
        theta_E_new_ref = self.SIS_ref.rho2theta(rho0)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=7)
        npt.assert_almost_equal(theta_E_new, theta_E_new_ref, decimal=7)


if __name__ == "__main__":
    pytest.main()