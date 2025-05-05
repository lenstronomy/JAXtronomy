from jax import config

config.update("jax_enable_x64", True)

from jaxtronomy.LensModel.Profiles.multipole import (
    Multipole,
    EllipticalMultipole,
    _phi_ell,
    _G_m_1,
    _F_m1_1_hat,
    _F_m1_1_hat_derivative,
    _potential_m1_1,
    _alpha_m1_1,
    _hessian_m1_1,
    _A_3_1,
    _F_m3_1_hat,
    _F_m3_1_hat_derivative,
    _potential_m3_1,
    _alpha_m3_1,
    _hessian_m3_1,
    _F_m4_1,
    _F_m4_1_derivative,
    _F_m4_2,
    _F_m4_2_derivative,
)
from lenstronomy.LensModel.Profiles.multipole import (
    Multipole as Multipole_ref,
    EllipticalMultipole as EllipticalMultipole_ref,
    _phi_ell as _phi_ell_ref,
    _G_m_1 as _G_m_1_ref,
    _F_m1_1_hat as _F_m1_1_hat_ref,
    _F_m1_1_hat_derivative as _F_m1_1_hat_derivative_ref,
    _potential_m1_1 as _potential_m1_1_ref,
    _alpha_m1_1 as _alpha_m1_1_ref,
    _hessian_m1_1 as _hessian_m1_1_ref,
    _A_3_1 as _A_3_1_ref,
    _F_m3_1_hat as _F_m3_1_hat_ref,
    _F_m3_1_hat_derivative as _F_m3_1_hat_derivative_ref,
    _potential_m3_1 as _potential_m3_1_ref,
    _alpha_m3_1 as _alpha_m3_1_ref,
    _hessian_m3_1 as _hessian_m3_1_ref,
    _F_m4_1 as _F_m4_1_ref,
    _F_m4_1_derivative as _F_m4_1_derivative_ref,
    _F_m4_2 as _F_m4_2_ref,
    _F_m4_2_derivative as _F_m4_2_derivative_ref,
)

import numpy as np
import pytest
import numpy.testing as npt


class TestMultipole(object):
    """Tests the Multipole methods."""

    def setup_method(self):
        self.multipole_ref = Multipole_ref()
        self.multipole = Multipole()

    def test_function(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 2,
            "a_m": 1.1,
            "phi_m": 1.5,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.1,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 2,
            "a_m": 1.1,
            "phi_m": 1.5,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.1,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 2,
            "a_m": 1.1,
            "phi_m": 1.5,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.1,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


class TestEllipticalMultipole(object):

    def setup_method(self):
        self.multipole_ref = EllipticalMultipole_ref()
        self.multipole = EllipticalMultipole()

    def test_function(self):

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 3,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 4,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 6,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.9999999999999999999999,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.function(x, y, **kwargs)
        result_ref = self.multipole_ref.function(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 7,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }
        result = self.multipole.function(x, y, **kwargs)
        npt.assert_array_equal(result, np.ones_like(x) * 1e18)

    def test_derivatives(self):

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 3,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 4,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 6,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.9999999999999999999999,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.derivatives(x, y, **kwargs)
        result_ref = self.multipole_ref.derivatives(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 7,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }
        result = self.multipole.derivatives(x, y, **kwargs)
        npt.assert_array_equal(result, (np.ones_like(x) * 1e18,) * 2)

    def test_hessian(self):

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-6, 6, 100)

        kwargs = {
            "m": 1,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 3,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 4,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.4,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 6,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.9999999999999999999999,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }

        result = self.multipole.hessian(x, y, **kwargs)
        result_ref = self.multipole_ref.hessian(x, y, **kwargs)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        kwargs = {
            "m": 7,
            "a_m": 1.4,
            "phi_m": 1.3,
            "q": 0.3,
            "center_x": 0.1,
            "center_y": -0.3,
            "r_E": 1.3,
        }
        result = self.multipole.hessian(x, y, **kwargs)
        npt.assert_array_equal(result, (np.ones_like(x) * 1e18,) * 4)


class TestMiscFunctions(object):

    def setup_method(self):
        self.q = 0.7
        self.phi = np.linspace(-2.8, 2.3, 10)

    def test_phi_ell(self):
        result = _phi_ell(self.phi, self.q)
        result_ref = _phi_ell_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_G_m_1(self):
        result = _G_m_1(4, self.phi, self.q)
        result_ref = _G_m_1_ref(4, self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m1_hat(self):
        result = _F_m1_1_hat(self.phi, self.q)
        result_ref = _F_m1_1_hat_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m1_hat_derivative(self):
        result = _F_m1_1_hat_derivative(self.phi, self.q)
        result_ref = _F_m1_1_hat_derivative_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_potential_m1_1(self):
        r = np.linspace(0.00001, 5, 10)
        r_E = 0.3
        result = _potential_m1_1(r, self.phi, self.q, r_E)
        result_ref = _potential_m1_1_ref(r, self.phi, self.q, r_E)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_alpha_m1_1(self):
        r = np.linspace(0.00001, 5, 10)
        r_E = 0.3
        result = _alpha_m1_1(r, self.phi, self.q, r_E)
        result_ref = _alpha_m1_1_ref(r, self.phi, self.q, r_E)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_hessian_m1_1(self):
        r = np.linspace(0.00001, 5, 10)
        result = _hessian_m1_1(r, self.phi, self.q)
        result_ref = _hessian_m1_1_ref(r, self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_A_3_1(self):
        result = _A_3_1(self.q)
        result_ref = _A_3_1_ref(self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m3_1_hat(self):
        result = _F_m3_1_hat(self.phi, self.q)
        result_ref = _F_m3_1_hat_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m3_1_hat(self):
        result = _F_m3_1_hat_derivative(self.phi, self.q)
        result_ref = _F_m3_1_hat_derivative_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_potential_m3_1(self):
        r = np.linspace(0.00001, 5, 10)
        r_E = 0.3
        result = _potential_m3_1(r, self.phi, self.q, r_E)
        result_ref = _potential_m3_1_ref(r, self.phi, self.q, r_E)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_alpha_m3_1(self):
        r = np.linspace(0.00001, 5, 10)
        r_E = 0.3
        result = _alpha_m3_1(r, self.phi, self.q, r_E)
        result_ref = _alpha_m3_1_ref(r, self.phi, self.q, r_E)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_hessian_m3_1(self):
        r = np.linspace(0.00001, 5, 10)
        result = _hessian_m3_1(r, self.phi, self.q)
        result_ref = _hessian_m3_1_ref(r, self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m4_1(self):
        result = _F_m4_1(self.phi, self.q)
        result_ref = _F_m4_1_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m4_1_derivative(self):
        result = _F_m4_1_derivative(self.phi, self.q)
        result_ref = _F_m4_1_derivative_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m4_2(self):
        result = _F_m4_2(self.phi, self.q)
        result_ref = _F_m4_2_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_F_m4_2_derivative(self):
        result = _F_m4_2_derivative(self.phi, self.q)
        result_ref = _F_m4_2_derivative_ref(self.phi, self.q)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
