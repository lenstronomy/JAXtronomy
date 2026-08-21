__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.hessian import Hessian as Hessian_ref
from jaxtronomy.LensModel.Profiles.hessian import Hessian

import numpy as np
import numpy.testing as npt
import pytest

class TestHessian(object):
    """Tests the Hessian methods."""

    def setup_method(self):
        self.Hessian_ref = Hessian_ref()
        self.kwargs_lens = {
            "f_xx": 0.1,
            "f_xy": -0.1,
            "f_yx": -0.1,
            "f_yy": 0.3,
            "ra_0": 0.5,
            "dec_0": -0.1,
        }

    def test_function(self):
        x = np.array([0.5])
        y = np.array([3])
        values = Hessian.function(x, y, **self.kwargs_lens)
        values_ref = self.Hessian_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)
        x = np.array([0])
        y = np.array([0])
        values = Hessian.function(x, y, **self.kwargs_lens)
        values_ref = self.Hessian_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = Hessian.function(x, y, **self.kwargs_lens)
        values_ref = self.Hessian_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

    def test_derivatives(self):
        x = np.array([0.5])
        y = np.array([3])
        f_x, f_y = Hessian.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.Hessian_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = Hessian.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.Hessian_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

    def test_hessian(self):
        x = np.array([0.5])
        y = np.array([3])
        f_xx, f_xy, f_yx, f_yy = Hessian.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.Hessian_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = Hessian.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.Hessian_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)


if __name__ == "__main__":
    pytest.main()
