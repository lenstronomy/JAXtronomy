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

def test_image2array():
    x = np.array([[0, 1, 2], [5, 7, 3], [1, 5, 8]], dtype=float)
    x_array = util.image2array(x)
    x_array_ref = util_ref.image2array(x)
    npt.assert_array_almost_equal(x_array, x_array_ref, decimal=8)

def test_makesubgrid():
    ra_coord = np.array([1, 3, 5, 7, 1, 9, 4, 2, 6], dtype=float)
    dec_coord = np.array([3, 8, 4, 7, 5, 2, 9, 7, 6], dtype=float)
    ra_coord_sub, dec_coord_sub = util.make_subgrid(ra_coord, dec_coord)
    ra_coord_sub_ref, dec_coord_sub_ref = util_ref.make_subgrid(ra_coord, dec_coord)
    npt.assert_array_almost_equal(ra_coord_sub, ra_coord_sub_ref, decimal=8)
    npt.assert_array_almost_equal(dec_coord_sub, dec_coord_sub_ref, decimal=8)

    subgrid_res = 4
    ra_coord_sub, dec_coord_sub = util.make_subgrid(ra_coord, dec_coord, subgrid_res)
    ra_coord_sub_ref, dec_coord_sub_ref = util_ref.make_subgrid(ra_coord, dec_coord, subgrid_res)
    npt.assert_array_almost_equal(ra_coord_sub, ra_coord_sub_ref, decimal=8)
    npt.assert_array_almost_equal(dec_coord_sub, dec_coord_sub_ref, decimal=8)

def test_rotate():
    x = np.array([0, 1, 2, 10])
    y = np.array([3, 2, 1, 8])
    angle = 0.012788
    result = util.rotate(x, y, angle)
    result_ref = util_ref.rotate(x, y, angle)
    npt.assert_array_almost_equal(result, result_ref, decimal=8)



if __name__ == "__main__":
    pytest.main()
