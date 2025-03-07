from jaxtronomy.PointSource.Types.lensed_position import LensedPositions
from lenstronomy.PointSource.Types.lensed_position import (
    LensedPositions as LensedPositions_ref,
)

from jaxtronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref

import pytest
import numpy.testing as npt


class TestLensedPosition(object):
    def setup_method(self):
        lens_model = LensModel(lens_model_list=["SIS", "SIS"])
        lens_model_ref = LensModel_ref(lens_model_list=["SIS", "SIS"])
        self.kwargs_lens = [
            {"theta_E": 2.3874, "center_x": 0.1345, "center_y": -0.23847},
            {"theta_E": 1.3874, "center_x": 0.2345, "center_y": 0.123847},
        ]
        self.ps_fixed_mag = LensedPositions(
            lens_model=lens_model,
            fixed_magnification=True,
            index_lens_model_list=[[0], [0, 1]],
            point_source_frame_list=[0, 1],
        )
        self.ps_fixed_mag_ref = LensedPositions_ref(
            lens_model=lens_model_ref,
            fixed_magnification=True,
            index_lens_model_list=[[0], [0, 1]],
            point_source_frame_list=[0, 1],
        )
        self.kwargs_fixed_mag = {
            "source_amp": 2,
            "ra_image": [0, 1.2],
            "dec_image": [0, 0],
        }

        self.ps = LensedPositions(
            lens_model=lens_model,
            fixed_magnification=False,
            index_lens_model_list=[[0], [0, 1]],
            point_source_frame_list=[0, 1],
        )
        self.ps_ref = LensedPositions_ref(
            lens_model=lens_model_ref,
            fixed_magnification=False,
            index_lens_model_list=[[0], [0, 1]],
            point_source_frame_list=[0, 1],
        )
        self.kwargs = {
            "point_amp": [2, 1],
            "ra_image": [0.32, 1.3],
            "dec_image": [0.345, -0.923],
        }

    def test_image_position(self):
        x_img, y_img = self.ps.image_position(self.kwargs)
        x_img_ref, y_img_ref = self.ps_ref.image_position(self.kwargs)
        npt.assert_allclose(x_img, x_img_ref, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(y_img, y_img_ref, rtol=1e-10, atol=1e-10)

        x_img, y_img = self.ps_fixed_mag.image_position(self.kwargs_fixed_mag)
        x_img_ref, y_img_ref = self.ps_fixed_mag_ref.image_position(
            self.kwargs_fixed_mag
        )
        npt.assert_allclose(x_img, x_img_ref, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(y_img, y_img_ref, rtol=1e-10, atol=1e-10)

        npt.assert_raises(
            ValueError, self.ps.image_position, self.kwargs, additional_images=True
        )

    def test_source_position(self):
        x_src, y_src = self.ps.source_position(
            self.kwargs, kwargs_lens=self.kwargs_lens
        )
        x_src_ref, y_src_ref = self.ps_ref.source_position(
            self.kwargs, kwargs_lens=self.kwargs_lens
        )
        npt.assert_allclose(x_src, x_src_ref, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(y_src, y_src_ref, rtol=1e-10, atol=1e-10)

        x_src, y_src = self.ps_fixed_mag.source_position(
            self.kwargs_fixed_mag, kwargs_lens=self.kwargs_lens
        )
        x_src_ref, y_src_ref = self.ps_fixed_mag_ref.source_position(
            self.kwargs_fixed_mag, kwargs_lens=self.kwargs_lens
        )
        npt.assert_allclose(x_src, x_src_ref, rtol=1e-10, atol=1e-10)
        npt.assert_allclose(y_src, y_src_ref, rtol=1e-10, atol=1e-10)

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
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 1

        x_pos = [0.1238, -0.23478, -1.478734, 1.382]
        y_pos = [0.98324, 0.123, 0.352489, -1.38743]

        amp = self.ps.image_amplitude(
            self.kwargs,
            kwargs_lens=None,
            x_pos=x_pos,
            y_pos=y_pos,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        amp_ref = self.ps_ref.image_amplitude(
            self.kwargs,
            kwargs_lens=None,
            x_pos=x_pos,
            y_pos=y_pos,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 1

        amp = self.ps_fixed_mag.image_amplitude(
            self.kwargs_fixed_mag,
            kwargs_lens=self.kwargs_lens,
            x_pos=None,
            y_pos=None,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        amp_ref = self.ps_fixed_mag_ref.image_amplitude(
            self.kwargs_fixed_mag,
            kwargs_lens=self.kwargs_lens,
            x_pos=None,
            y_pos=None,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 1

        x_pos = [0.1238, -0.23478]
        y_pos = [0.98324, 0.123]

        amp = self.ps_fixed_mag.image_amplitude(
            self.kwargs_fixed_mag,
            kwargs_lens=self.kwargs_lens,
            x_pos=x_pos,
            y_pos=y_pos,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        amp_ref = self.ps_fixed_mag_ref.image_amplitude(
            self.kwargs_fixed_mag,
            kwargs_lens=self.kwargs_lens,
            x_pos=x_pos,
            y_pos=y_pos,
            magnification_limit=None,
            kwargs_lens_eqn_solver=None,
        )
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 1

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens)
        amp_ref = self.ps_ref.source_amplitude(
            self.kwargs, kwargs_lens=self.kwargs_lens
        )
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 0

        amp = self.ps_fixed_mag.source_amplitude(
            self.kwargs_fixed_mag, kwargs_lens=None
        )
        amp_ref = self.ps_fixed_mag_ref.source_amplitude(
            self.kwargs_fixed_mag, kwargs_lens=None
        )
        npt.assert_allclose(amp, amp_ref, rtol=1e-10, atol=1e-10)
        assert amp.ndim == 0


if __name__ == "__main__":
    pytest.main()
