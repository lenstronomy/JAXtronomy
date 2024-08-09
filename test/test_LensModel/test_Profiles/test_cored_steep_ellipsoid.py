__author__ = "sibirrer"


import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.cored_steep_ellipsoid import (
    CSE,
    CSEMajorAxis,
    CSEMajorAxisSet,
    CSEProductAvg,
    CSEProductAvgSet,
)
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSE as CSE_ref
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import (
    CSEMajorAxisSet as CSEMajorAxisSet_ref,
)
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import (
    CSEProductAvgSet as CSEProductAvgSet_ref,
)


class TestCSE_product_avg(object):
    """Tests the cored steep ellipsoid (CSE)"""

    def setup_method(self):
        self.CSE = CSE(axis="product_avg")
        self.CSE_ref = CSE_ref(axis="product_avg")
        test_init = CSEProductAvg()
        npt.assert_raises(ValueError, CSE, axis="not_available")

    def test_function(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.1, "e2": -0.3, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_ = self.CSE.function(x, y, **kwargs)
        f_ref = self.CSE_ref.function(x, y, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

    def test_derivatives(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.0, "e2": 0.0, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_x, f_y = self.CSE.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.CSE_ref.derivatives(x, y, **kwargs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.0, "e2": 0.0, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.3, 0.5])
        f_xx, f_xy, f_yx, f_yy = self.CSE.hessian(x, y, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.CSE_ref.hessian(x, y, **kwargs)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)


class TestCSE_product_avg_set(object):
    """Tests the cored steep ellipsoid (CSE)"""

    def setup_method(self):
        self.CSEProductAvgSet_ref = CSEProductAvgSet_ref()
        test_init = CSEProductAvgSet()

    def test_function(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_ = CSEProductAvgSet.function(x, y, **kwargs)
        f_ref = self.CSEProductAvgSet_ref.function(x, y, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

    def test_derivatives(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_x, f_y = CSEProductAvgSet.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.CSEProductAvgSet_ref.derivatives(x, y, **kwargs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        x = np.array([1.0, 2.0])
        y = np.array([2.3, 0.5])
        f_xx, f_xy, f_yx, f_yy = CSEProductAvgSet.hessian(x, y, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.CSEProductAvgSet_ref.hessian(
            x, y, **kwargs
        )
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)


class TestCSE_major(object):
    """Tests the cored steep ellipsoid (CSE)"""

    def setup_method(self):
        self.CSE = CSE(axis="major")
        self.CSE_ref = CSE_ref(axis="major")
        test_init = CSEMajorAxis()

    def test_function(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.1, "e2": -0.3, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_ = self.CSE.function(x, y, **kwargs)
        f_ref = self.CSE_ref.function(x, y, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

    def test_derivatives(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.0, "e2": 0.0, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_x, f_y = self.CSE.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.CSE_ref.derivatives(x, y, **kwargs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        kwargs = {"a": 2, "s": 1, "e1": 0.0, "e2": 0.0, "center_x": 0, "center_y": 0}

        x = np.array([1.0, 2.0])
        y = np.array([2.3, 0.5])
        f_xx, f_xy, f_yx, f_yy = self.CSE.hessian(x, y, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.CSE_ref.hessian(x, y, **kwargs)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)


class TestCSE_major_set(object):
    """Tests the cored steep ellipsoid (CSE)"""

    def setup_method(self):
        self.CSEMajorAxisSet_ref = CSEMajorAxisSet_ref()
        test_init = CSEMajorAxisSet()

    def test_function(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_ = CSEMajorAxisSet.function(x, y, **kwargs)
        f_ref = self.CSEMajorAxisSet_ref.function(x, y, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

    def test_derivatives(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_x, f_y = CSEMajorAxisSet.derivatives(x, y, **kwargs)
        f_x_ref, f_y_ref = self.CSEMajorAxisSet_ref.derivatives(x, y, **kwargs)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        kwargs = {
            "a_list": np.asarray([1.0, 4.0, 3.0]),
            "s_list": np.asarray([1.0, 0.0, 0.0]),
            "q": 0.3,
        }

        # NOTE: If x and y are made to be integers, the tests fails due to a bug with lenstronomy.
        #       Change the test when the bug in lenstronomy is fixed
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 0])
        f_xx, f_xy, f_yx, f_yy = CSEMajorAxisSet.hessian(x, y, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.CSEMajorAxisSet_ref.hessian(
            x, y, **kwargs
        )
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
