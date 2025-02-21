import pytest

import jax
import numpy as np, numpy.testing as npt
from numpy.polynomial.hermite import hermval as hermval_ref
from scipy.special import eval_hermite as eval_hermite_ref

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent witsh numpy
from jaxtronomy.Util.herm_util import eval_hermite, hermval

A_TOL = 1e-10
R_TOL = 1e-12


def test_eval_hermite():
    n_test = 3
    x_array = np.array([[1, 4, 6], [2, 3, 5]])
    npt.assert_allclose(
        eval_hermite(n_test, x_array), eval_hermite_ref(n_test, x_array), R_TOL, A_TOL
    )

    n_test = 7
    x_array = np.array([[1, 4, 6], [2, 3, 5]])
    npt.assert_allclose(
        eval_hermite(n_test, x_array), eval_hermite_ref(n_test, x_array), R_TOL, A_TOL
    )

    n_test = 38
    x_array = np.array([[1.3, 4.49, -6], [2.344, -3.194, 5.392]])
    npt.assert_allclose(
        eval_hermite(n_test, x_array), eval_hermite_ref(n_test, x_array), R_TOL, A_TOL
    )


def test_hermval():
    n_array = np.linspace(0.5, 1.5, 4)
    x_array = np.array([[1.35, 2.234], [-3.45654, 4.465], [5.236, -6.26]], dtype=float)
    npt.assert_allclose(
        hermval(x_array, n_array), hermval_ref(x_array, n_array), R_TOL, A_TOL
    )

    n_array = np.linspace(0.5, 32, 50)
    x_array = np.array([[1.35, 2.234], [-3.45654, 4.465], [5.236, -6.26]], dtype=float)
    npt.assert_allclose(
        hermval(x_array, n_array), hermval_ref(x_array, n_array), R_TOL, A_TOL
    )


if __name__ == "__main__":
    pytest.main()
