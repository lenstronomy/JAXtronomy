import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensModel.Profiles.flexion import Flexion as Flexion_ref
from jaxtronomy.LensModel.Profiles.flexion import Flexion


class TestFlexion(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.flexion_ref = Flexion_ref()
        self.kwargs_lens = {
            "g1": -0.03,
            "g2": -0.04,
            "g3": 0.07,
            "g4": -0.01,
        }

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        values = Flexion.function(x, y, **self.kwargs_lens)
        values_ref = self.flexion_ref.function(x, y, **self.kwargs_lens)
        npt.assert_allclose(values, values_ref, atol=1e-15, rtol=1e-15)
        x = np.array([0])
        y = np.array([0])
        values = Flexion.function(x, y, **self.kwargs_lens)
        values_ref = self.flexion_ref.function(x, y, **self.kwargs_lens)
        npt.assert_allclose(values, values_ref, atol=1e-15, rtol=1e-15)

        x = np.tile(np.linspace(-10, 10, 20), 20)
        y = np.repeat(np.linspace(-10, 10, 20), 20)
        values = Flexion.function(x, y, **self.kwargs_lens)
        values_ref = self.flexion_ref.function(x, y, **self.kwargs_lens)
        npt.assert_allclose(values, values_ref, atol=1e-15, rtol=1e-15)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = Flexion.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.flexion_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-15, rtol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = Flexion.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.flexion_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-15, rtol=1e-15)

        x = np.tile(np.linspace(-10, 10, 20), 20)
        y = np.repeat(np.linspace(-10, 10, 20), 20)
        f_x, f_y = Flexion.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.flexion_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-15, rtol=1e-15)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])

        f_xx, f_xy, f_yx, f_yy = Flexion.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.flexion_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yx, f_yx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-15, rtol=1e-15)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = Flexion.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.flexion_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yx, f_yx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-15, rtol=1e-15)

        x = np.tile(np.linspace(-10, 10, 20), 20)
        y = np.repeat(np.linspace(-10, 10, 20), 20)
        f_xx, f_xy, f_yx, f_yy = Flexion.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.flexion_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yx, f_yx_ref, atol=1e-15, rtol=1e-15)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    pytest.main()
