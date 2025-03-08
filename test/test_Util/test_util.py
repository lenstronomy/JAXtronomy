__author__ = "sibirrer"

import lenstronomy.Util.util as util_ref
import jaxtronomy.Util.util as util

import jax
import numpy as np
import pytest
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)


def test_array2image():
    x = np.array([0, 1, 2, 5, 7, 3], dtype=float)
    npt.assert_raises(ValueError, util.array2image, x)

    x_image = util.array2image(x, 2, 3)
    x_image_ref = util_ref.array2image(x, 2, 3)
    npt.assert_array_almost_equal(x_image, x_image_ref, decimal=8)

    x = np.array([0, 1, 2, 5, 7, 3, 1, 8, 9], dtype=float)

    x_image = util.array2image(x)
    x_image_ref = util_ref.array2image(x)
    npt.assert_array_almost_equal(x_image, x_image_ref, decimal=8)


def test_displaceAbs():
    x = np.array([0, 1, 2])
    y = np.array([3, 2, 1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    result_ref = util_ref.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    npt.assert_array_almost_equal(result, result_ref, decimal=7)


def test_fwhm2sigma():
    fwhm = np.array([0.32345, 1.0, 2.3, 3.5, 4.7])
    sigma = util.fwhm2sigma(fwhm)
    sigma_ref = util_ref.fwhm2sigma(fwhm)
    npt.assert_array_almost_equal(sigma, sigma_ref, decimal=8)


def test_image2array():
    x = np.array([[0, 1, 2], [5, 7, 3], [1, 5, 8]], dtype=float)
    x_array = util.image2array(x)
    x_array_ref = util_ref.image2array(x)
    npt.assert_array_almost_equal(x_array, x_array_ref, decimal=8)


def test_map_coord2pix():
    M = np.array([[1, 3], [-1, 2]])
    x_0, y_0 = 3, 2
    ra = np.array([0.3435, 0.29384, 1.32989])
    dec = np.array([1.3482, 2.4823, 23.8345])

    result = util.map_coord2pix(ra, dec, x_0, y_0, M)
    result_ref = util_ref.map_coord2pix(ra, dec, x_0, y_0, M)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)

    M = np.array([[1.453, 2.432], [-0.2354, 1.2342]])
    x_0, y_0 = 2, -2
    ra = np.array([0.43435, 0.29384, 12.32989])
    dec = np.array([1.482, 23.823, 2.8345])

    result = util.map_coord2pix(ra, dec, x_0, y_0, M)
    result_ref = util_ref.map_coord2pix(ra, dec, x_0, y_0, M)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)


def test_rotate():
    x = np.array([0, 1, 2, 10])
    y = np.array([3, 2, 1, 8])
    angle = 0.012788
    result = util.rotate(x, y, angle)
    result_ref = util_ref.rotate(x, y, angle)
    npt.assert_array_almost_equal(result, result_ref, decimal=8)


def test_sigma2fwhm():
    sigma = np.array([0.32345, 1.0, 2.3, 3.5, 4.7])
    fwhm = util.sigma2fwhm(sigma)
    fwhm_ref = util_ref.sigma2fwhm(sigma)
    npt.assert_array_almost_equal(fwhm, fwhm_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
