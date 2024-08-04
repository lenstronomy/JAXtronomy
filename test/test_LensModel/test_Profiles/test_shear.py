__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.shear import Shear as Shear_ref
from lenstronomy.LensModel.Profiles.shear import ShearGammaPsi as ShearGammaPsi_ref
from lenstronomy.LensModel.Profiles.shear import ShearReduced as ShearReduced_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref

from jaxtronomy.LensModel.Profiles.shear import Shear, ShearGammaPsi, ShearReduced
from jaxtronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest


class TestShear(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.shear_ref = Shear_ref()
        gamma1, gamma2 = 0.1, 0.1
        self.kwargs_lens = {"gamma1": gamma1, "gamma2": gamma2}

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        values = Shear.function(x, y, **self.kwargs_lens)
        values_ref = self.shear_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)
        x = np.array([0])
        y = np.array([0])
        values = Shear.function(x, y, **self.kwargs_lens)
        values_ref = self.shear_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = Shear.function(x, y, **self.kwargs_lens)
        values_ref = self.shear_ref.function(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = Shear.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.shear_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = Shear.derivatives(x, y, **self.kwargs_lens)
        f_x_ref, f_y_ref = self.shear_ref.derivatives(x, y, **self.kwargs_lens)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=7)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])

        f_xx, f_xy, f_yx, f_yy = Shear.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.shear_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=7)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = Shear.hessian(x, y, **self.kwargs_lens)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.shear_ref.hessian(
            x, y, **self.kwargs_lens
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=7)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)

        gamma1, gamma2 = 0.1, -0.1
        kwargs = {"gamma1": gamma1, "gamma2": gamma2}
        lensModel = LensModel(["SHEAR"])
        lensModel_ref = LensModel_ref(["SHEAR"])
        gamma1, gamma2 = lensModel.gamma(x, y, [kwargs])
        gamma1_ref, gamma2_ref = lensModel_ref.gamma(x, y, [kwargs])
        npt.assert_array_almost_equal(gamma1, gamma1_ref, decimal=9)
        npt.assert_array_almost_equal(gamma2, gamma2_ref, decimal=9)


class TestShearGammaPsi(object):

    def setup_method(self):
        self.sheargammapsi_ref = ShearGammaPsi_ref()
        test_init = ShearGammaPsi()

    def test_function(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma, psi = 0.1, 0.5
        values = ShearGammaPsi.function(x, y, gamma, psi)
        values_ref = self.sheargammapsi_ref.function(x, y, gamma, psi)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma, psi = 0.1, 0.5
        values = ShearGammaPsi.derivatives(x, y, gamma, psi)
        values_ref = self.sheargammapsi_ref.derivatives(x, y, gamma, psi)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma, psi = 0.1, 0.5
        values = ShearGammaPsi.hessian(x, y, gamma, psi)
        values_ref = self.sheargammapsi_ref.hessian(x, y, gamma, psi)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)


class TestShearReduced(object):

    def setup_method(self):
        self.shearreduced_ref = ShearReduced_ref()
        test_init = ShearReduced()

    def test_kappa_reduced(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma1, gamma2 = -0.4, 0.4
        kappa, gamma1_, gamma2_ = ShearReduced.kappa_reduced(gamma1, gamma2)
        kappa_ref, gamma1_ref_, gamma2_ref_ = self.shearreduced_ref._kappa_reduced(
            gamma1, gamma2
        )
        npt.assert_array_almost_equal(kappa, kappa_ref, decimal=7)
        npt.assert_array_almost_equal(gamma1_, gamma1_ref_, decimal=7)
        npt.assert_array_almost_equal(gamma2_, gamma2_ref_, decimal=7)

    def test_function(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma1, gamma2 = 0.1, 0.5
        values = ShearReduced.function(x, y, gamma1, gamma2)
        values_ref = self.shearreduced_ref.function(x, y, gamma1, gamma2)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_derivatives(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma1, gamma2 = 0.2, 0.4
        values = ShearReduced.derivatives(x, y, gamma1, gamma2)
        values_ref = self.shearreduced_ref.derivatives(x, y, gamma1, gamma2)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_hessian(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        gamma1, gamma2 = 0.1, 0.3
        values = ShearReduced.hessian(x, y, gamma1, gamma2)
        values_ref = self.shearreduced_ref.hessian(x, y, gamma1, gamma2)
        npt.assert_array_almost_equal(values, values_ref, decimal=7)


if __name__ == "__main__":
    pytest.main()
