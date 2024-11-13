__author__ = "sibirrer"

import pytest
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.image_util as image_util_ref
import jaxtronomy.Util.image_util as image_util

def test_re_size():
    grid = np.ones((200, 100))
    grid[101, 57] = 4
    grid[12, 37] = 17
    grid_small = image_util.re_size(grid, factor=2)
    grid_small_ref = image_util_ref.re_size(grid, factor=2)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)

    grid_small = image_util.re_size(grid, factor=4)
    grid_small_ref = image_util_ref.re_size(grid, factor=4)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)

    grid_small = image_util.re_size(grid, factor=5)
    grid_small_ref = image_util_ref.re_size(grid, factor=5)
    npt.assert_array_almost_equal(grid_small, grid_small_ref, decimal=6)


if __name__ == "__main__":
    pytest.main()
