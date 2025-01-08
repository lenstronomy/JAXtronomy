import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Util import util
import jaxtronomy.Util.param_util as param_util
import lenstronomy.Util.param_util as param_util_ref
import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp


def test_cart2polar():
    # singel 2d coordinate transformation
    center_x, center_y = 0, 0
    x = 1
    y = 1
    r, phi = param_util.cart2polar(x, y, center_x, center_y)
    r_ref, phi_ref = param_util_ref.cart2polar(x, y, center_x, center_y)
    npt.assert_almost_equal(r, r_ref, decimal=8)
    npt.assert_almost_equal(phi, phi_ref, decimal=8)

    # array of 2d coordinates
    x = np.array([1, 2])
    y = np.array([1, 1])
    r, phi = param_util.cart2polar(x, y, center_x, center_y)
    r_ref, phi_ref = param_util_ref.cart2polar(x, y, center_x, center_y)
    npt.assert_array_almost_equal(r, r_ref, decimal=7)
    npt.assert_array_almost_equal(phi, phi_ref, decimal=7)


def test_polar2cart():
    # singel 2d coordinate transformation
    center = np.array([0, 0])
    r = 1
    phi = np.pi
    x, y = param_util.polar2cart(r, phi, center)
    x_ref, y_ref = param_util_ref.polar2cart(r, phi, center)
    npt.assert_almost_equal(x, x_ref, decimal=8)
    npt.assert_almost_equal(y, y_ref, decimal=8)


def test_phi_q2_ellipticity():
    phi, q = 0, 1
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    e1_ref, e2_ref = param_util_ref.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_ref, decimal=8)
    npt.assert_almost_equal(e2, e2_ref, decimal=8)

    phi, q = 1, 1
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    e1_ref, e2_ref = param_util_ref.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_ref, decimal=8)
    npt.assert_almost_equal(e2, e2_ref, decimal=8)

    phi, q = 2.0, 0.95
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    e1_ref, e2_ref = param_util_ref.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_ref, decimal=8)
    npt.assert_almost_equal(e2, e2_ref, decimal=8)

    phi, q = 0, 0.9
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    e1_ref, e2_ref = param_util_ref.phi_q2_ellipticity(phi, q)
    npt.assert_almost_equal(e1, e1_ref, decimal=8)
    npt.assert_almost_equal(e2, e2_ref, decimal=8)


def test_ellipticity2phi_q():
    e1, e2 = 0.3, 0
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    phi_ref, q_ref = param_util_ref.ellipticity2phi_q(e1, e2)
    npt.assert_almost_equal(phi, phi_ref, decimal=8)
    npt.assert_almost_equal(q, q_ref, decimal=8)

    # Works on np arrays as well
    e1 = np.array([0.3, 0.9])
    e2 = np.array([0.0, 0.9])
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    phi_ref, q_ref = param_util_ref.ellipticity2phi_q(e1, e2)
    npt.assert_array_almost_equal(phi, phi_ref, decimal=8)
    npt.assert_array_almost_equal(q, q_ref, decimal=8)


def test_transform_e1e2_product_average():
    e1 = 0.01
    e2 = -0.03
    x = 0.0
    y = 1.0
    x_, y_ = param_util.transform_e1e2_product_average(
        x, y, e1, e2, center_x=-0.3, center_y=0.4
    )
    x_ref, y_ref = param_util_ref.transform_e1e2_product_average(
        x, y, e1, e2, center_x=-0.3, center_y=0.4
    )
    npt.assert_almost_equal(x_, x_ref, decimal=8)
    npt.assert_almost_equal(y_, y_ref, decimal=8)

    e1 = 0.0
    e2 = 0.0
    x_, y_ = param_util.transform_e1e2_product_average(
        x, y, e1, e2, center_x=-0.3, center_y=0.4
    )
    x_ref, y_ref = param_util_ref.transform_e1e2_product_average(
        x, y, e1, e2, center_x=-0.3, center_y=0.4
    )
    npt.assert_almost_equal(x_, x_ref, decimal=8)
    npt.assert_almost_equal(y_, y_ref, decimal=8)


def test_shear_polar2cartesian():
    phi = -1.0
    gamma = 0.1
    e1, e2 = param_util.shear_polar2cartesian(phi, gamma)
    e1_ref, e2_ref = param_util_ref.shear_polar2cartesian(phi, gamma)
    npt.assert_almost_equal(e1, e1_ref, decimal=8)
    npt.assert_almost_equal(e2, e2_ref, decimal=8)


def test_shear_cartesian2polar():
    e1, e2 = 0.1, 0.06
    phi, gamma = param_util.shear_cartesian2polar(e1, e2)
    phi_ref, gamma_ref = param_util_ref.shear_cartesian2polar(e1, e2)
    npt.assert_almost_equal(phi, phi_ref, decimal=8)
    npt.assert_almost_equal(gamma, gamma_ref, decimal=8)


def test_transform_e1e2_square_average():
    x, y = np.array([1, 0]), np.array([0, 1])
    e1 = 0.1
    e2 = 0
    center_x, center_y = 0, 0
    x_, y_ = param_util.transform_e1e2_square_average(
        x, y, e1, e2, center_x=center_x, center_y=center_y
    )
    x_ref, y_ref = param_util_ref.transform_e1e2_square_average(
        x, y, e1, e2, center_x=center_x, center_y=center_y
    )
    npt.assert_array_almost_equal(x_, x_ref, decimal=8)
    npt.assert_array_almost_equal(y_, y_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
