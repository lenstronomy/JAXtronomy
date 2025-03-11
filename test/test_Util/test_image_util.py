__author__ = "sibirrer"

import pytest
from jax import config
import numpy as np
import numpy.testing as npt

import lenstronomy.Util.image_util as image_util_ref
import jaxtronomy.Util.image_util as image_util

config.update("jax_enable_x64", True)


def test_add_layer2image():
    kernel = np.ones((45, 47))
    kernel[44] = 2
    test = np.ones((256, 256))
    x_pos = 27.3287
    y_pos = -13.3248

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    kernel = np.ones((31, 33))
    kernel[13] = 2
    kernel[27] = 5
    kernel[:, 12] = 7
    test = np.linspace(-50, 50, 50 * 50).reshape((50, 50))
    x_pos = -12.89543
    y_pos = 37.2783

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    x_pos = 25.89543
    y_pos = 25.2783

    result = image_util.add_layer2image_int(test, x_pos, y_pos, kernel)
    result_ref = image_util_ref.add_layer2image_int(test, x_pos, y_pos, kernel)
    np.testing.assert_array_equal(result, result_ref)

    npt.assert_raises(
        ValueError, image_util.add_layer2image, test, x_pos, y_pos, kernel, order=2
    )
    kernel = np.ones((10, 11))
    npt.assert_raises(
        ValueError, image_util.add_layer2image, test, x_pos, y_pos, kernel, order=1
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

    kernel = np.ones((10, 11))
    npt.assert_raises(
        ValueError, image_util.add_layer2image_int, test, x_pos, y_pos, kernel
    )


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


if __name__ == "__main__":
    pytest.main()
