__author__ = "sibirrer"


from jaxtronomy.LightModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.sersic import Sersic as Sersic_ref

import numpy as np
import pytest
import numpy.testing as npt


class TestSersicMajorAxis(object):
    """Tests the Sersic methods."""

    def setup_method(self):
        self.sersic = Sersic(smoothing=0.02, sersic_major_axis=True)
        self.sersic_ref = Sersic_ref(smoothing=0.02, sersic_major_axis=True)

    def test_sersic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        center_x = 0.1
        center_y = -0.3
        values = self.sersic.function(x, y, I0, R_sersic, n_sersic, center_x, center_y)
        values_ref = self.sersic_ref.function(
            x, y, I0, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)


class TestSersic(object):
    """Tests the Sersic methods."""

    def setup_method(self):
        self.sersic = Sersic(smoothing=0.02, sersic_major_axis=False)
        self.sersic_ref = Sersic_ref(smoothing=0.02, sersic_major_axis=False)

    def test_sersic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        center_x = 0.1
        center_y = -0.3
        values = self.sersic.function(x, y, I0, R_sersic, n_sersic, center_x, center_y)
        values_ref = self.sersic_ref.function(
            x, y, I0, R_sersic, n_sersic, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
