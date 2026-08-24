__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.perturber_model import (
    PerturberModel as PerturberModel_ref,
)
from jaxtronomy.LensModel.Profiles.perturber_model import PerturberModel

import numpy as np
import numpy.testing as npt
import pytest


class TestPerturberModel(object):
    """Tests the PerturberModel methods."""

    def setup_method(self):
        self.PerturberModel = PerturberModel("SIS", ra_0=0.1, dec_0=-0.2)
        self.PerturberModel_ref = PerturberModel_ref("SIS", ra_0=0.1, dec_0=-0.2)
        self.kwargs_lens = {"theta_E": 1.3123}

    def test_function(self):
        x = np.array([0.5])
        y = np.array([3])
        values = self.PerturberModel.function(x, y, **self.kwargs_lens)
        values_ref = self.PerturberModel_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)
        x = np.array([0])
        y = np.array([0])
        values = self.PerturberModel.function(x, y, **self.kwargs_lens)
        values_ref = self.PerturberModel_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.PerturberModel.function(x, y, **self.kwargs_lens)
        values_ref = self.PerturberModel_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=12)

    def test_derivatives(self):
        x = np.array([0.5])
        y = np.array([3])
        f_x, f_y = self.PerturberModel.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.PerturberModel_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.PerturberModel.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.PerturberModel_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=12)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=12)

    def test_hessian(self):
        x = np.array([0.5])
        y = np.array([3])
        f_xx, f_xy, f_yx, f_yy = self.PerturberModel.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.PerturberModel_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.PerturberModel.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.PerturberModel_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=12)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=12)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=12)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=12)


if __name__ == "__main__":
    pytest.main()
