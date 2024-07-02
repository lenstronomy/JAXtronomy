__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from jaxtronomy.LensModel.LineOfSight.LOSModels.los import LOS
from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS as LOS_ref


class TestLOS(object):
    """Tests the LOS profile."""

    def setup_method(self):
        self.LOS = LOS()
        self.LOS_ref = LOS_ref()

    def test_distort_vector(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        x = 1
        y = 1

        x_distorted, y_distorted = self.LOS.distort_vector(
            x, y, kappa, gamma1, gamma2, omega
        )
        x_distorted_ref, y_distorted_ref = self.LOS_ref.distort_vector(
            x, y, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(x_distorted, 0.8, decimal=9)
        npt.assert_almost_equal(y_distorted, 0.8, decimal=9)
        npt.assert_almost_equal(x_distorted, x_distorted_ref, decimal=9)
        npt.assert_almost_equal(y_distorted, y_distorted_ref, decimal=9)

    def test_left_multiply(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        fxx = 1
        fxy = 1
        fyx = 1
        fyy = 1

        f_xx, f_xy, f_yx, f_yy = self.LOS.left_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.LOS_ref.left_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(f_xx, 0.8)
        npt.assert_almost_equal(f_xy, 0.8)
        npt.assert_almost_equal(f_yx, 0.8)
        npt.assert_almost_equal(f_yy, 0.8)

        npt.assert_almost_equal(f_xx, f_xx_ref)
        npt.assert_almost_equal(f_xy, f_xy_ref)
        npt.assert_almost_equal(f_yx, f_yx_ref)
        npt.assert_almost_equal(f_yy, f_yy_ref)

    def test_right_multiply(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        fxx = 1
        fxy = 1
        fyx = 1
        fyy = 1

        f_xx, f_xy, f_yx, f_yy = self.LOS.right_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.LOS_ref.right_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(f_xx, 0.4)
        npt.assert_almost_equal(f_xy, 1.2)
        npt.assert_almost_equal(f_yx, 0.4)
        npt.assert_almost_equal(f_yy, 1.2)

        npt.assert_almost_equal(f_xx, f_xx_ref)
        npt.assert_almost_equal(f_xy, f_xy_ref)
        npt.assert_almost_equal(f_yx, f_yx_ref)
        npt.assert_almost_equal(f_yy, f_yy_ref)

    def test_set_static(self):

        p = self.LOS.set_static()

    def test_set_dynamic(self):

        d = self.LOS.set_dynamic()


if __name__ == "__main__":
    pytest.main("-k TestLOS")
