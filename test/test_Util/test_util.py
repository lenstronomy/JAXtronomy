__author__ = "sibirrer"

import lenstronomy.Util.util as util_ref
import jaxtronomy.Util.util as util

import jax
import numpy as np
import pytest
import numpy.testing as npt
import unittest

jax.config.update("jax_enable_x64", True)


def test_rotate():
    x = np.array([0, 1, 2, 10])
    y = np.array([3, 2, 1, 8])
    angle = 0.012788
    result = util.rotate(x, y, angle)
    result_ref = util_ref.rotate(x, y, angle)
    npt.assert_array_almost_equal(result, result_ref, decimal=8)


def test_displaceAbs():
    x = np.array([0, 1, 2])
    y = np.array([3, 2, 1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    result_ref = util_ref.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    npt.assert_array_almost_equal(result, result_ref, decimal=7)


if __name__ == "__main__":
    pytest.main()
