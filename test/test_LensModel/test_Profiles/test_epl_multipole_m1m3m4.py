from jax import config

config.update("jax_enable_x64", True)

import numpy.testing as npt
import numpy as np
import pytest

from jaxtronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
    EPL_MULTIPOLE_M1M3M4,
    EPL_MULTIPOLE_M1M3M4_ELL,
)
from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
    EPL_MULTIPOLE_M1M3M4 as EPL_MULTIPOLE_M1M3M4_ref,
    EPL_MULTIPOLE_M1M3M4_ELL as EPL_MULTIPOLE_M1M3M4_ELL_ref,
)


class TestEPL_MULTIPOLE_M1M3M4(object):
    """Tests EPL_MULTIPOLE_M1M3M4."""

    def setup_method(self):

        self.epl_m1m3m4 = EPL_MULTIPOLE_M1M3M4()
        self.epl_m1m3m4_ref = EPL_MULTIPOLE_M1M3M4_ref()
        self.kwargs1_m1m3m4 = {
            "x": np.linspace(-5, 5, 100),
            "y": np.linspace(-5, 5, 100),
            "theta_E": 1.2,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a1_a": 0.1,
            "delta_phi_m1": 0.2,
            "a3_a": 0.05,
            "delta_phi_m3": 0.2,
            "a4_a": -0.05,
            "delta_phi_m4": 0.3,
        }
        self.kwargs2_m1m3m4 = {
            "x": np.linspace(-5, 5, 100),
            "y": np.linspace(-5, 5, 100),
            "theta_E": 1.5,
            "center_x": -0.4,
            "center_y": 0.1,
            "e1": 0.2,
            "e2": -0.3,
            "gamma": 2.1,
            "a1_a": 0.2,
            "delta_phi_m1": 0.1,
            "a3_a": 0.02,
            "delta_phi_m3": 0.1,
            "a4_a": -0.03,
            "delta_phi_m4": 0.1,
        }

    def test_function(self):
        result = self.epl_m1m3m4.function(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.function(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.function(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.function(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        result = self.epl_m1m3m4.derivatives(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.derivatives(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.derivatives(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.derivatives(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        result = self.epl_m1m3m4.hessian(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.hessian(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.hessian(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.hessian(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


class TestEPL_MULTIPOLE_M1M3M4_ELL(object):
    """Tests EPL_MULTIPOLE_M1M3M4_ELL."""

    def setup_method(self):

        self.epl_m1m3m4 = EPL_MULTIPOLE_M1M3M4_ELL()
        self.epl_m1m3m4_ref = EPL_MULTIPOLE_M1M3M4_ELL_ref()
        self.kwargs1_m1m3m4 = {
            "x": np.linspace(-5, 5, 100),
            "y": np.linspace(-5, 5, 100),
            "theta_E": 1.2,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a1_a": 0.1,
            "delta_phi_m1": 0.2,
            "a3_a": 0.05,
            "delta_phi_m3": 0.2,
            "a4_a": -0.05,
            "delta_phi_m4": 0.3,
        }
        self.kwargs2_m1m3m4 = {
            "x": np.linspace(-5, 5, 100),
            "y": np.linspace(-5, 5, 100),
            "theta_E": 1.5,
            "center_x": -0.4,
            "center_y": 0.1,
            "e1": 0.2,
            "e2": -0.3,
            "gamma": 2.1,
            "a1_a": 0.2,
            "delta_phi_m1": 0.1,
            "a3_a": 0.02,
            "delta_phi_m3": 0.1,
            "a4_a": -0.03,
            "delta_phi_m4": 0.1,
        }

    def test_function(self):
        result = self.epl_m1m3m4.function(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.function(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.function(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.function(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_derivatives(self):
        result = self.epl_m1m3m4.derivatives(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.derivatives(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.derivatives(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.derivatives(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    def test_hessian(self):
        result = self.epl_m1m3m4.hessian(**self.kwargs1_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.hessian(**self.kwargs1_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        result = self.epl_m1m3m4.hessian(**self.kwargs2_m1m3m4)
        result_ref = self.epl_m1m3m4_ref.hessian(**self.kwargs2_m1m3m4)
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
