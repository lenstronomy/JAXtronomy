import numpy as np
import pytest
import numpy.testing as npt

import jax

jax.config.update("jax_enable_x64", True)

from jaxtronomy.LightModel.Profiles.shapelets import Shapelets, ShapeletSet
from lenstronomy.LightModel.Profiles.shapelets import (
    Shapelets as Shapelets_ref,
    ShapeletSet as ShapeletSet_ref,
)


class TestShapelets_StableCut(object):

    def setup_method(self):
        self.shapelet = Shapelets()
        self.shapelet_ref = Shapelets_ref()

    def test_init(self):
        npt.assert_raises(ValueError, Shapelets, interpolation=True)

    def test_hermval(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        n_array = [2, 3.3, 4.6, 6.7, 1.4]

        result = self.shapelet.hermval(x, n_array)
        result_ref = self.shapelet_ref.hermval(x, n_array)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.hermval(y, n_array)
        result_ref = self.shapelet_ref.hermval(y, n_array)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_function(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        amp = 31.3584
        beta = 3.287
        n1 = 6
        n2 = 8
        center_x = np.linspace(-0.3, 0.4, 6)
        center_y = np.linspace(-0.5, 0.1, 6)

        result = self.shapelet.function(x, y, amp, beta, n1, n2, center_x, center_y)
        result_ref = self.shapelet_ref.function(
            x, y, amp, beta, n1, n2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        n1, n2 = 2, 3
        precalc_shapelet = Shapelets(precalc=True)
        precalc_shapelet_ref = Shapelets_ref(precalc=True)
        result = precalc_shapelet.function(x, y, amp, beta, n1, n2, center_x, center_y)
        result_ref = precalc_shapelet_ref.function(
            x, y, amp, beta, n1, n2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_Hn(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        n1 = 6
        n2 = 8

        result = self.shapelet.H_n(n1, x)
        result_ref = self.shapelet_ref.H_n(n1, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.H_n(n2, x)
        result_ref = self.shapelet_ref.H_n(n2, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_phin(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        n1 = 6
        n2 = 8

        result = self.shapelet.phi_n(n1, x)
        result_ref = self.shapelet_ref.phi_n(n1, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.phi_n(n2, x)
        result_ref = self.shapelet_ref.phi_n(n2, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_precalc(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        beta = 3.287
        n1 = 6
        center_x = np.linspace(-0.3, 0.4, 6)
        center_y = np.linspace(-0.5, 0.1, 6)
        phi_x, phi_y = self.shapelet.pre_calc(x, y, beta, n1, center_x, center_y)
        phi_x_ref, phi_y_ref = self.shapelet_ref.pre_calc(
            x, y, beta, n1, center_x, center_y
        )
        npt.assert_allclose(phi_x, phi_x_ref, atol=1e-10, rtol=1e-15)
        npt.assert_allclose(phi_y, phi_y_ref, atol=1e-10, rtol=1e-15)
        npt.assert_raises(
            ValueError, self.shapelet.pre_calc, x, y, beta, 171, center_x, center_y
        )


class TestShapelets_NoStableCut(object):

    def setup_method(self):
        self.shapelet = Shapelets(stable_cut=False)
        self.shapelet_ref = Shapelets_ref(stable_cut=False)

    def test_hermval(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        n_array = [2, 3.3, 4.6, 6.7, 1.4]

        result = self.shapelet.hermval(x, n_array)
        result_ref = self.shapelet_ref.hermval(x, n_array)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.hermval(y, n_array)
        result_ref = self.shapelet_ref.hermval(y, n_array)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_function(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        amp = 31.3584
        beta = 3.287
        n1 = 6
        n2 = 8
        center_x = np.linspace(-0.3, 0.4, 6)
        center_y = np.linspace(-0.5, 0.1, 6)

        result = self.shapelet.function(x, y, amp, beta, n1, n2, center_x, center_y)
        result_ref = self.shapelet_ref.function(
            x, y, amp, beta, n1, n2, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_Hn(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        n1 = 6
        n2 = 8

        result = self.shapelet.H_n(n1, x)
        result_ref = self.shapelet_ref.H_n(n1, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.H_n(n2, x)
        result_ref = self.shapelet_ref.H_n(n2, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_phin(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        n1 = 6
        n2 = 8

        result = self.shapelet.phi_n(n1, x)
        result_ref = self.shapelet_ref.phi_n(n1, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        result = self.shapelet.phi_n(n2, x)
        result_ref = self.shapelet_ref.phi_n(n2, x)
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

    def test_precalc(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        beta = 3.287
        n1 = 6
        n2 = 8
        center_x = np.linspace(-0.3, 0.4, 6)
        center_y = np.linspace(-0.5, 0.1, 6)
        phi_x, phi_y = self.shapelet.pre_calc(x, y, beta, n1, center_x, center_y)
        phi_x_ref, phi_y_ref = self.shapelet_ref.pre_calc(
            x, y, beta, n1, center_x, center_y
        )
        npt.assert_allclose(phi_x, phi_x_ref, atol=1e-10, rtol=1e-15)
        npt.assert_allclose(phi_y, phi_y_ref, atol=1e-10, rtol=1e-15)


class TestShapeletSet(object):

    def setup_method(self):
        self.shapeletset = ShapeletSet()
        self.shapeletset_ref = ShapeletSet_ref()

    def test_function(self):
        x = np.array([1.3, 3.5, 6.7, 2.5, 13.54, 99])
        y = np.array([-0.6, 2.5, 6.1, -2.6, 13.54, 99])
        beta = 3.287
        center_x = np.linspace(-0.3, 0.4, 6)
        center_y = np.linspace(-0.5, 0.1, 6)

        n_max = 4
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset.function(x, y, amp, n_max, beta, center_x, center_y)
        result_ref = self.shapeletset_ref.function(
            x, y, amp, n_max, beta, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)

        n_max = 12
        num_param = int((n_max + 1) * (n_max + 2) / 2)
        amp = np.linspace(1.0, 100.0, num_param)
        result = self.shapeletset.function(y, x, amp, n_max, beta, center_x, center_y)
        result_ref = self.shapeletset_ref.function(
            y, x, amp, n_max, beta, center_x, center_y
        )
        npt.assert_allclose(result, result_ref, atol=1e-10, rtol=1e-15)


if __name__ == "__main__":
    pytest.main()
