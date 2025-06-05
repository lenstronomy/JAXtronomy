__author__ = "sibirrer"

from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim import de_lens as de_lens_ref
from jaxtronomy.ImSim import de_lens

import pytest


class TestDeLens(object):
    def setup_method(self):
        pass

    def test_get_param_WLS(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([1, 1, 1])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=True)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=True
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(cov_error, cov_error_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)

        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=False
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)
        assert cov_error is None
        assert cov_error_ref is None

    def test_wls_stability(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([0, 0, 0])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=True)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=True
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(cov_error, cov_error_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)

        A = np.array([[1, 2, 1], [1, 2, 1]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=False
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)
        assert cov_error is None
        assert cov_error_ref is None

        C_D_inv = np.array([1, 1, 1])
        A = np.array([[1.0, 2.0, 1.0 + 10 ** (-8.9)], [1.0, 2.0, 1.0]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=False
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)

        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=True)
        result_ref, cov_error_ref, image_ref = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=True
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(cov_error, cov_error_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(image, image_ref, atol=1e-12, rtol=1e-12)

    def test_marginalisation_const(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([1, 1, 1])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens_ref.get_param_WLS(
            A, C_D_inv, d, inv_bool=True
        )
        logL_marg = de_lens.marginalisation_const(cov_error)
        logL_marg_ref = de_lens_ref.marginalisation_const(cov_error)
        npt.assert_allclose(logL_marg, logL_marg_ref, atol=1e-12, rtol=1e-12)

        M_inv = np.array([[1, 0], [0, 1]])
        marg_const = de_lens.marginalisation_const(M_inv)
        marg_const_ref = de_lens_ref.marginalisation_const(M_inv)
        npt.assert_allclose(marg_const, marg_const_ref, atol=1e-12, rtol=1e-12)

    def test_margnialization_new(self):
        M_inv = np.array([[1, -0.5, 1], [-0.5, 3, 0], [1, 0, 2]])
        log_det = de_lens.marginalization_new(M_inv, d_prior=1000)
        log_det_ref = de_lens_ref.marginalization_new(M_inv, d_prior=1000)
        npt.assert_allclose(log_det, log_det_ref, atol=1e-12, rtol=1e-12)

        M_inv = np.array([[1, 1, 1], [0.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
        log_det = de_lens.marginalization_new(M_inv, d_prior=10)
        log_det_ref = de_lens_ref.marginalization_new(M_inv, d_prior=10)
        npt.assert_allclose(log_det, log_det_ref, atol=1e-12, rtol=1e-12)

        log_det = de_lens.marginalization_new(M_inv, d_prior=None)
        log_det_ref = de_lens_ref.marginalization_new(M_inv, d_prior=None)
        npt.assert_allclose(log_det, log_det_ref, atol=1e-12, rtol=1e-12)

    def test_stable_inv(self):
        m = np.diag(np.ones(10) * 2)
        m_inv = de_lens._stable_inv(m)
        m_inv_ref = de_lens_ref._stable_inv(m)
        npt.assert_allclose(m_inv, m_inv_ref, atol=1e-12, rtol=1e-12)

        m = np.ones((10, 10))
        m_inv = de_lens._stable_inv(m)
        m_inv_ref = de_lens_ref._stable_inv(m)
        npt.assert_allclose(m_inv, m_inv_ref, atol=1e-12, rtol=1e-12)

    def test_solve_stable(self):
        m = np.array([[2, 1], [1, 2]])
        r = np.array([2, 1])
        b = de_lens._solve_stable(m, r)
        b_ref = de_lens_ref._solve_stable(m, r)
        npt.assert_allclose(b, b_ref, atol=1e-12, rtol=1e-12)

        m = np.array([[0, 0], [0, 0]])
        r = np.array([1, 1])
        b = de_lens._solve_stable(m, r)
        b_ref = de_lens_ref._solve_stable(m, r)
        npt.assert_allclose(b, b_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
