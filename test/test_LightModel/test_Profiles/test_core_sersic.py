__author__ = "sibirrer"


from jaxtronomy.LightModel.Profiles.core_sersic import CoreSersic
from lenstronomy.LightModel.Profiles.sersic import CoreSersic as CoreSersic_ref

import numpy as np
import pytest
import numpy.testing as npt


class TestCoreSersicMajorAxis(object):
    """Tests the CoreSersic methods."""

    def setup_method(self):
        self.core_sersic = CoreSersic(smoothing=0.02, sersic_major_axis=True)
        self.core_sersic_ref = CoreSersic_ref(smoothing=0.02, sersic_major_axis=True)

    def test_core_sersic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1.2
        R_sersic = 1
        Rb = 0.7
        n_sersic = 3
        gamma = 4
        e1 = 0.1
        e2 = -0.3
        center_x = 0.1
        center_y = -0.3
        values = self.core_sersic.function(
            x, y, I0, R_sersic, Rb, n_sersic, gamma, e1, e2, center_x, center_y
        )
        values_ref = self.core_sersic_ref.function(
            x, y, I0, R_sersic, Rb, n_sersic, gamma, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)


class TestCoreSersic(object):
    """Tests the CoreSersic methods."""

    def setup_method(self):
        self.core_sersic = CoreSersic(smoothing=0.02, sersic_major_axis=False)
        self.core_sersic_ref = CoreSersic_ref(smoothing=0.02, sersic_major_axis=False)

    def test_core_sersic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1.2
        R_sersic = 1
        Rb = 0.7
        n_sersic = 3
        gamma = 4
        e1 = 0.1
        e2 = -0.3
        center_x = 0.1
        center_y = -0.3
        values = self.core_sersic.function(
            x, y, I0, R_sersic, Rb, n_sersic, gamma, e1, e2, center_x, center_y
        )
        values_ref = self.core_sersic_ref.function(
            x, y, I0, R_sersic, Rb, n_sersic, gamma, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
