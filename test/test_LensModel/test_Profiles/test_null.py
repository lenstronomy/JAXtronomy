import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.null import Null

class TestGaussian(object):
    def setup_method(self):
        self.profile = Null()

    def test_function(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)
        values = self.profile.function(x, y)
        npt.assert_allclose(values, np.zeros_like(x), atol=1e-16)

    def test_derivatives(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)
        f_x, f_y = self.profile.derivatives(x, y)
        npt.assert_allclose(f_x, np.zeros_like(x), atol=1e-16)
        npt.assert_allclose(f_y, np.zeros_like(x), atol=1e-16)

    def test_hessian(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)
        f_xx, f_xy, f_yx, f_yy = self.profile.derivatives(x, y)
        npt.assert_allclose(f_xx, np.zeros_like(x), atol=1e-16)
        npt.assert_allclose(f_xy, np.zeros_like(x), atol=1e-16)
        npt.assert_allclose(f_yx, np.zeros_like(x), atol=1e-16)
        npt.assert_allclose(f_yy, np.zeros_like(x), atol=1e-16)


if __name__ == "__main__":
    pytest.main()
