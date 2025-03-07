from jaxtronomy.PointSource.Types.unlensed import Unlensed
from lenstronomy.PointSource.Types.unlensed import Unlensed as Unlensed_ref
import pytest
import numpy.testing as npt


# NOTE: Even though this is a simple class, we still test against lenstronomy to ensure that the API remains the same
class TestUnlensed(object):
    def setup_method(self):
        self.ps = Unlensed()
        self.ps_ref = Unlensed_ref()
        self.kwargs = {"point_amp": [2, 1], "ra_image": [0, 1], "dec_image": [1, 0]}

    def test_image_position(self):
        x_img, y_img = self.ps.image_position(self.kwargs)
        x_img_ref, y_img_ref = self.ps_ref.image_position(self.kwargs)
        npt.assert_array_equal(x_img, x_img_ref)
        npt.assert_array_equal(y_img, y_img_ref)

    def test_source_position(self):
        x_src, y_src = self.ps.source_position(self.kwargs)
        x_src_ref, y_src_ref = self.ps_ref.source_position(self.kwargs)
        npt.assert_array_equal(x_src, x_src_ref)
        npt.assert_array_equal(y_src, y_src_ref)

    def test_image_amplitude(self):
        amp = self.ps.image_amplitude(
            self.kwargs,
            kwargs_lens=None,
            x_pos=None,
            y_pos=None,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        amp_ref = self.ps_ref.image_amplitude(
            self.kwargs,
            kwargs_lens=None,
            x_pos=None,
            y_pos=None,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        npt.assert_array_equal(amp, amp_ref)

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(self.kwargs, kwargs_lens=None)
        amp_ref = self.ps_ref.source_amplitude(self.kwargs, kwargs_lens=None)
        npt.assert_almost_equal(amp, amp_ref)


if __name__ == "__main__":
    pytest.main()
