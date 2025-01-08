__author__ = "sibirrer"

import jax
import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LightModel.Profiles.gaussian import (
    Gaussian as Gaussian_ref,
    GaussianEllipse as GaussianEllipse_ref,
    MultiGaussian as MultiGaussian_ref,
    MultiGaussianEllipse as MultiGaussianEllipse_ref,
)
from jaxtronomy.LightModel.Profiles.gaussian import (
    Gaussian,
    GaussianEllipse,
    MultiGaussian,
    MultiGaussianEllipse,
)


class TestGaussian(object):
    def setup_method(self):
        self.profile_ref = Gaussian_ref()
        self.profile = Gaussian()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma = 0.5
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0])
        y = np.array([0])
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_total_flux(self):
        amp = 1.3
        sigma = 0.5
        values_ref = self.profile_ref.total_flux(amp, sigma)
        values = self.profile.total_flux(amp, sigma)
        npt.assert_equal(values_ref, values)
        npt.assert_equal(amp, values)

    def test_light_3d(self):
        r = np.array([2])
        amp = 1.387344
        sigma = 0.74385
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([3.534])
        amp = 1.387344
        sigma = 0.74385
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([2, 9.2398, 2.183])
        amp = 2.387344
        sigma = 1.74385
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)


class TestGaussianEllipse(object):
    def setup_method(self):
        self.profile_ref = GaussianEllipse_ref()
        self.profile = GaussianEllipse()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        amp = 1.3
        sigma = 0.5
        e1 = 0.1
        e2 = -0.3
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0])
        y = np.array([0])
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_total_flux(self):
        amp = 1.3
        sigma = 0.5
        values_ref = self.profile_ref.total_flux(amp, sigma)
        values = self.profile.total_flux(amp, sigma)
        npt.assert_equal(values_ref, values)
        npt.assert_equal(amp, values)

    def test_light_3d(self):
        r = np.array([2])
        amp = 1.387344
        sigma = 0.74385
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([3.534])
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([2, 9.2398, 2.183])
        amp = 2.387344
        sigma = 1.74385
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)


class TestMultiGaussian(object):
    def setup_method(self):
        self.profile_ref = MultiGaussian_ref()
        self.profile = MultiGaussian()

    def test_function(self):
        x = np.array([1.0])
        y = np.array([2.0])
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma)
        values = self.profile.function(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_function_split(self):
        x = np.array([1.2345])
        y = np.array([2.2345123])
        amp = [1.3656, 1.51432, 2.34123]
        sigma = [0.55678, 1.19876, 3.745567]
        values_ref = self.profile_ref.function_split(x, y, amp, sigma)
        values = self.profile.function_split(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function_split(x, y, amp, sigma)
        values = self.profile.function_split(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.0354, 3.0567, 4.0345])
        y = np.array([1.0456, 1.0876, 1.0876])
        values_ref = self.profile_ref.function_split(x, y, amp, sigma)
        values = self.profile.function_split(x, y, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_total_flux(self):
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        values_ref = self.profile_ref.total_flux(amp, sigma)
        values = self.profile.total_flux(amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)
        npt.assert_array_almost_equal(np.sum(amp), values, decimal=7)

    def test_light_3d(self):
        r = np.array([2.0])
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([3.534])
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([2, 9.2398, 2.183])
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)


class TestMultiGaussianEllipse(object):
    def setup_method(self):
        self.profile_ref = MultiGaussianEllipse_ref()
        self.profile = MultiGaussianEllipse()

    def test_function(self):
        x = np.array([1.0])
        y = np.array([2.0])
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        e1 = 0.1345236
        e2 = -0.33452
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])
        values_ref = self.profile_ref.function(x, y, amp, sigma, e1, e2)
        values = self.profile.function(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_function_split(self):
        x = np.array([1.4568])
        y = np.array([2.8567])
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        e1 = 0.1345236
        e2 = -0.33452
        values_ref = self.profile_ref.function_split(x, y, amp, sigma, e1, e2)
        values = self.profile.function_split(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function_split(x, y, amp, sigma, e1, e2)
        values = self.profile.function_split(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)

        x = np.array([2.02345, 3.01234, 4.02345])
        y = np.array([1.01234, 1.05234, 1.02345])
        values_ref = self.profile_ref.function_split(x, y, amp, sigma, e1, e2)
        values = self.profile.function_split(x, y, amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

    def test_total_flux(self):
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        e1 = 0.1345236
        e2 = -0.33452
        values_ref = self.profile_ref.total_flux(amp, sigma, e1, e2)
        values = self.profile.total_flux(amp, sigma, e1, e2)
        npt.assert_array_almost_equal(values_ref, values, decimal=7)
        npt.assert_array_almost_equal(np.sum(amp), values, decimal=7)

    def test_light_3d(self):
        r = np.array([2.0])
        amp = [1.3, 1.5, 2.3]
        sigma = [0.5, 1.1, 3.7]
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([3.534])
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)

        r = np.array([2, 9.2398, 2.183])
        values_ref = self.profile_ref.light_3d(r, amp, sigma)
        values = self.profile.light_3d(r, amp, sigma)
        npt.assert_array_almost_equal(values_ref, values, decimal=8)


if __name__ == "__main__":
    pytest.main()
