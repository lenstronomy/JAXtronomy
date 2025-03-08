__author__ = "sibirrer"

import pytest
from jax import config
import numpy as np
import numpy.testing as npt
from scipy.ndimage import shift as shift_ref

import lenstronomy.Util.image_util as image_util_ref
import jaxtronomy.Util.image_util as image_util

config.update("jax_enable_x64", True)


def test_add_layer2image():
    kernel = np.ones((45, 45))
    kernel[44] = 2
    test = np.ones((256, 256))
    x_pos = 27.3287
    y_pos = -13.3248

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    kernel = np.ones((31, 31))
    kernel[13] = 2
    kernel[27] = 5
    kernel[:, 12] = 7
    test = np.linspace(-50, 50, 50 * 50).reshape((50, 50))
    x_pos = -12.89543
    y_pos = 37.2783

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    npt.assert_raises(
        ValueError, image_util.add_layer2image, test, x_pos, y_pos, kernel, order=2
    )


def test_add_layer2image_int():

    kernel = np.ones((45, 45))
    kernel[44] = 2
    test = np.ones((256, 256))
    x_pos = 270
    y_pos = -10

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    kernel = np.ones((31, 31))
    kernel[13] = 2
    kernel[27] = 5
    kernel[:, 12] = 7
    test = np.linspace(-50, 50, 50 * 50).reshape((50, 50))
    x_pos = -12
    y_pos = 37

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)


def test_re_size():
    grid = np.ones((200, 100))
    grid[101, 57] = 4
    grid[12, 37] = 17

    grid_same = image_util.re_size(grid, factor=1)
    npt.assert_equal(grid, grid_same)

    grid_small = image_util.re_size(grid, factor=2)
    grid_small_ref = image_util_ref.re_size(grid, factor=2)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)

    grid_small = image_util.re_size(grid, factor=4)
    grid_small_ref = image_util_ref.re_size(grid, factor=4)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)

    grid_small = image_util.re_size(grid, factor=5)
    grid_small_ref = image_util_ref.re_size(grid, factor=5)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)

    npt.assert_raises(ValueError, image_util.re_size, grid, factor=0.5)
    npt.assert_raises(ValueError, image_util.re_size, grid, factor=3)


def test_shift():
    input_array = np.linspace(-5, 5, 280 * 280, dtype=float).reshape((280, 280))
    shift = [-0.5, -0.3]
    result = image_util.shift(input=input_array, shift=shift)
    result_ref = shift_ref(input=input_array, shift=shift, order=1)
    npt.assert_allclose(result, result_ref, atol=1e-15, rtol=1e-15)

    input_array = np.linspace(-15, 15, 150 * 150, dtype=float).reshape((150, 150))
    shift = [0.7, -0.1]
    result = image_util.shift(input=input_array, shift=shift)
    result_ref = shift_ref(input=input_array, shift=shift, order=1)
    npt.assert_allclose(result, result_ref, atol=1e-15, rtol=1e-15)

    input_array = np.linspace(-15, 15, 150 * 150, dtype=float).reshape((150, 150))
    shift = 0.423
    result = image_util.shift(input=input_array, shift=shift)
    result_ref = shift_ref(input=input_array, shift=shift, order=1)
    npt.assert_allclose(result, result_ref, atol=1e-15, rtol=1e-15)

    _3d_array = np.ones((3, 3, 3))
    npt.assert_raises(ValueError, image_util.shift, _3d_array, 0.23)

    npt.assert_raises(ValueError, image_util.shift, input_array, shift=[0.1, 0.1, 0.1])


if __name__ == "__main__":
    pytest.main()
