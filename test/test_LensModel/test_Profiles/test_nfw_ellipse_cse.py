__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE
from lenstronomy.LensModel.Profiles.nfw_ellipse_cse import (
    NFW_ELLIPSE_CSE as NFW_ELLIPSE_CSE_ref,
)


class TestNFWELLIPSE(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.high_accuracy = NFW_ELLIPSE_CSE(high_accuracy=True)
        self.high_accuracy_ref = NFW_ELLIPSE_CSE_ref(high_accuracy=True)
        self.low_accuracy = NFW_ELLIPSE_CSE(high_accuracy=False)
        self.low_accuracy_ref = NFW_ELLIPSE_CSE_ref(high_accuracy=False)

    def test_function(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"alpha_Rs": 2, "Rs": 2, "center_x": 0, "center_y": 0}

        f_ = self.high_accuracy.function(x, y, e1=0, e2=0, **kwargs)
        f_ref = self.high_accuracy_ref.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

        f_ = self.low_accuracy.function(x, y, e1=0, e2=0, **kwargs)
        f_ref = self.low_accuracy_ref.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f_, f_ref, decimal=8)

    def test_derivatives(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"alpha_Rs": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_x, f_y = self.high_accuracy.derivatives(x, y, e1=0, e2=0, **kwargs)
        f_x_ref, f_y_ref = self.high_accuracy_ref.derivatives(
            x, y, e1=0, e2=0, **kwargs
        )
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

        f_x, f_y = self.low_accuracy.derivatives(x, y, e1=0, e2=0, **kwargs)
        f_x_ref, f_y_ref = self.low_accuracy_ref.derivatives(x, y, e1=0, e2=0, **kwargs)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"alpha_Rs": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_xx, f_xy, f_yx, f_yy = self.high_accuracy.hessian(x, y, e1=0, e2=0, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.high_accuracy_ref.hessian(
            x, y, e1=0, e2=0, **kwargs
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

        f_xx, f_xy, f_yx, f_yy = self.low_accuracy.hessian(x, y, e1=0, e2=0, **kwargs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.low_accuracy_ref.hessian(
            x, y, e1=0, e2=0, **kwargs
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        m_3d_ref = self.high_accuracy_ref.mass_3d_lens(R, Rs, alpha_Rs)
        m_3d = self.high_accuracy.mass_3d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_3d_ref, m_3d, decimal=8)

    def test_normalization(self):
        Rs = 3.1
        alpha_Rs = 1.3
        q = 0.1
        const = self.high_accuracy._normalization(alpha_Rs, Rs, q)
        const_ref = self.high_accuracy_ref._normalization(alpha_Rs, Rs, q)
        npt.assert_almost_equal(const, const_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
