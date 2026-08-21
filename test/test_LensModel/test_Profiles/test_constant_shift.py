__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.constant_shift import Shift as Shift_ref
from jaxtronomy.LensModel.Profiles.constant_shift import Shift

import numpy as np
import numpy.testing as npt
import pytest


class TestShift(object):
    """Tests the Shift methods."""

    def setup_method(self):
        self.Shift_ref = Shift_ref()
        self.kwargs_lens = {"alpha_x": 0.3, "alpha_y": 0.5}

    def test_function(self):
        x = np.array([0.5])
        y = np.array([3])
        values = Shift.function(x, y, **self.kwargs_lens)
        values_ref = self.Shift_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)
        x = np.array([0])
        y = np.array([0])
        values = Shift.function(x, y, **self.kwargs_lens)
        values_ref = self.Shift_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = Shift.function(x, y, **self.kwargs_lens)
        values_ref = self.Shift_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

    def test_derivatives(self):
        x = np.array([0.5])
        y = np.array([3])
        f_x, f_y = Shift.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.Shift_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = Shift.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.Shift_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

    def test_hessian(self):
        x = np.array([0.5])
        y = np.array([3])
        f_xx, f_xy, f_yx, f_yy = Shift.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.Shift_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = Shift.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.Shift_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)


if __name__ == "__main__":
    pytest.main()
