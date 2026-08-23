__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LightModel.Profiles.mge_set import MGESet as MGESet_ref
from lenstronomy.LightModel.Profiles.mge_ellipse import MGEEllipse as MGEEllipse_ref

from jaxtronomy.LightModel.Profiles.mge_set import MGESet
from jaxtronomy.LightModel.Profiles.mge_ellipse import MGEEllipse


class TestMGESet(object):
    def setup_method(self):
        self.profile_ref = MGESet_ref(15)
        self.profile = MGESet(15)
        self.amp = np.linspace(0.1, 300, 15)
        self.sigma_min = 0.1
        self.sigma_width = 121

    def test_function(self):
        x = np.array([1.0])
        y = np.array([2.0])
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function(x, y, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function(x, y, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function(x, y, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

    def test_function_split(self):
        x = np.array([1.2345])
        y = np.array([2.2345123])
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

    def test_total_flux(self):
        values_ref = self.profile_ref.total_flux(
            self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.total_flux(self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)
        npt.assert_array_almost_equal(np.sum(self.amp), values, decimal=12)

    def test_light_3d(self):
        r = np.array([2.0])
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.light_3d(r, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        r = np.array([3.534])
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.light_3d(r, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        r = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width
        )
        values = self.profile.light_3d(r, self.amp, self.sigma_min, self.sigma_width)
        npt.assert_array_almost_equal(values_ref, values, decimal=12)


class TestMGEEllipse(object):
    def setup_method(self):
        self.profile_ref = MGEEllipse_ref(4)
        self.profile = MGEEllipse(4)
        self.amp = np.linspace(0.1, 3, 4)
        self.sigma_min = 43.1
        self.sigma_width = 0.03
        self.e1 = 0.3
        self.e2 = -0.17

    def test_function(self):
        x = np.array([1.0])
        y = np.array([2.0])
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

    def test_function_split(self):
        x = np.array([1.2345])
        y = np.array([2.2345123])
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.array([0.0])
        y = np.array([0.0])
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        x = np.linspace(-50, 50, 100)
        y = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.function_split(
            x, y, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

    def test_total_flux(self):
        values_ref = self.profile_ref.total_flux(
            self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.total_flux(
            self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)
        npt.assert_array_almost_equal(np.sum(self.amp), values, decimal=12)

    def test_light_3d(self):
        r = np.array([2.0])
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        r = np.array([3.534])
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)

        r = np.linspace(-50, 50, 100)
        values_ref = self.profile_ref.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        values = self.profile.light_3d(
            r, self.amp, self.sigma_min, self.sigma_width, self.e1, self.e2
        )
        npt.assert_array_almost_equal(values_ref, values, decimal=12)


if __name__ == "__main__":
    pytest.main()
