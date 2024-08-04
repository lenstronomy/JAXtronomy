__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.convergence import Convergence as Convergence_ref
from jaxtronomy.LensModel.Profiles.convergence import Convergence

import numpy as np
import numpy.testing as npt
import pytest


class TestConvergence(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.convergence_ref = Convergence_ref()
        self.kwargs_lens = {"kappa": 0.1}

    def test_function(self):
        x = np.array([1])
        y = np.array([0])
        values = Convergence.function(x, y, **self.kwargs_lens)
        values_ref = self.convergence_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)
        x = np.array([0])
        y = np.array([0])
        values = Convergence.function(x, y, **self.kwargs_lens)
        values_ref = self.convergence_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = Convergence.function(x, y, **self.kwargs_lens)
        values_ref = self.convergence_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = Convergence.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.convergence_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = Convergence.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.convergence_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        f_xx, f_xy, f_yx, f_yy = Convergence.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.convergence_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = Convergence.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.convergence_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=7)


if __name__ == "__main__":
    pytest.main()
