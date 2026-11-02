import numpy as np
import pytest
import numpy.testing as npt

from jaxtronomy.LightModel.Profiles.shapelets_ellipse import (
    ShapeletSetEllipseStatic as ShapeletSetStatic,
)
from lenstronomy.LightModel.Profiles.shapelets_ellipse import (
    ShapeletSetEllipse as ShapeletSet_ref,
)

class TestShapeletSet(object):

    def setup_method(self):
        self.shapeletset_ref = ShapeletSet_ref()
        self.shapeletset_off = ShapeletSetStatic(n_max=-1)
        self.shapeletset4 = ShapeletSetStatic(n_max=4)
        self.shapeletset12 = ShapeletSetStatic(n_max=12)

    def test_init(self):
        npt.assert_raises(ValueError, ShapeletSetStatic, 5.5)

    def test_function(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        beta = 3.287
        e1 = -0.33215
        e2 = 0.2123
        center_x = -0.7
        center_y = 1.7

        n_max = -1
        amp = []
        result = self.shapeletset_off.function(
            x, y, amp, beta, e1, e2, center_x, center_y
        )
        result_ref = self.shapeletset_ref.function(
            x, y, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        n_max = 4
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset4.function(x, y, amp, beta, e1, e2, center_x, center_y)
        result_ref = self.shapeletset_ref.function(
            x, y, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        n_max = 12
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset12.function(y, x, amp, beta, e1, e2, center_x, center_y)
        result_ref = self.shapeletset_ref.function(
            y, x, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


    def test_function_split(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        beta = 3.287
        e1 = 0.2178783
        e2 = -0.12983289
        center_x = -0.7
        center_y = 1.7

        n_max = -1
        amp = []
        result = self.shapeletset_off.function_split(
            x, y, amp, beta, e1, e2, center_x, center_y
        )
        result_ref = self.shapeletset_ref.function_split(
            x, y, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        n_max = 4
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset4.function_split(
            x, y, amp, beta, e1, e2, center_x, center_y
        )
        result_ref = self.shapeletset_ref.function_split(
            x, y, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

        n_max = 12
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset12.function_split(
            y, x, amp, beta, e1, e2, center_x, center_y
        )
        result_ref = self.shapeletset_ref.function_split(
            y, x, amp, n_max, beta, e1, e2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    pytest.main()
