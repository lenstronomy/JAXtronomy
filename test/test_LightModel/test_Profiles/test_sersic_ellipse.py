__author__ = "sibirrer"


from jaxtronomy.LightModel.Profiles.sersic_ellipse import (
    SersicElliptic,
    SersicElliptic_qPhi,
)
from lenstronomy.LightModel.Profiles.sersic import (
    SersicElliptic as SersicElliptic_ref,
    SersicElliptic_qPhi as SersicElliptic_qPhi_ref,
)

import numpy as np
import pytest
import numpy.testing as npt


class TestSersicEllipseMajorAxis(object):
    """Tests the SersicElliptic and SersicElliptic_qPhi methods."""

    def setup_method(self):
        self.sersic_elliptic = SersicElliptic(smoothing=0.02, sersic_major_axis=True)
        self.sersic_elliptic_qphi = SersicElliptic_qPhi(
            smoothing=0.02, sersic_major_axis=True
        )
        self.sersic_elliptic_ref = SersicElliptic_ref(
            smoothing=0.02, sersic_major_axis=True
        )
        self.sersic_elliptic_qphi_ref = SersicElliptic_qPhi_ref(
            smoothing=0.02, sersic_major_axis=True
        )

    def test_sersic_elliptic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        e1 = 0.1
        e2 = -0.3
        center_x = 0.1
        center_y = -0.3
        values = self.sersic_elliptic.function(
            x, y, I0, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        values_ref = self.sersic_elliptic_ref.function(
            x, y, I0, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=8)

    def test_sersic_elliptic_qphi(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        q = 0.7
        phi = np.pi / 8
        center_x = 0.1
        center_y = -0.3
        values = self.sersic_elliptic_qphi.function(
            x, y, I0, R_sersic, n_sersic, q, phi, center_x, center_y
        )
        values_ref = self.sersic_elliptic_qphi_ref.function(
            x, y, I0, R_sersic, n_sersic, q, phi, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=7)


class TestSersicEllipse(object):
    """Tests the SersicElliptic and SersicElliptic_qPhi methods."""

    def setup_method(self):
        self.sersic_elliptic = SersicElliptic(smoothing=0.02, sersic_major_axis=False)
        self.sersic_elliptic_qphi = SersicElliptic_qPhi(
            smoothing=0.02, sersic_major_axis=False
        )
        self.sersic_elliptic_ref = SersicElliptic_ref(
            smoothing=0.02, sersic_major_axis=False
        )
        self.sersic_elliptic_qphi_ref = SersicElliptic_qPhi_ref(
            smoothing=0.02, sersic_major_axis=False
        )

    def test_sersic_elliptic(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        e1 = 0.1
        e2 = -0.3
        center_x = 0.1
        center_y = -0.3
        values = self.sersic_elliptic.function(
            x, y, I0, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        values_ref = self.sersic_elliptic_ref.function(
            x, y, I0, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=7)

    def test_sersic_elliptic_qphi(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        I0 = 1
        R_sersic = 1
        n_sersic = 3
        q = 0.7
        phi = np.pi / 8
        center_x = 0.1
        center_y = -0.3
        values = self.sersic_elliptic_qphi.function(
            x, y, I0, R_sersic, n_sersic, q, phi, center_x, center_y
        )
        values_ref = self.sersic_elliptic_qphi_ref.function(
            x, y, I0, R_sersic, n_sersic, q, phi, center_x, center_y
        )
        npt.assert_array_almost_equal(values, values_ref, decimal=7)


if __name__ == "__main__":
    pytest.main()
