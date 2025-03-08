from lenstronomy.ImSim.Numerics.point_source_rendering import (
    PointSourceRendering as PointSourceRendering_ref,
)
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF

from jaxtronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering

from jax import config
import numpy as np
import numpy.testing as npt
import pytest
import unittest

config.update("jax_enable_x64", True)


class TestPointSourceRendering(object):
    def setup_method(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": 0,
            "dec_at_xy_0": 0,
            "transform_pix2angle": Mpix2coord,
            "nx": 10,
            "ny": 10,
        }
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((7, 7))
        kernel[1:6, 1:6] = 2
        kernel[4:4] = 7
        kwargs_psf = {
            "kernel_point_source": kernel,
            "psf_type": "PIXEL",
            "psf_variance_map": np.ones_like(kernel) * kernel**2,
        }
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(
            pixel_grid, supersampling_factor=1, psf=psf_class
        )
        self._ps_rendering_ref = PointSourceRendering_ref(
            pixel_grid, supersampling_factor=1, psf=psf_class
        )

    def test_point_source_rendering(self):
        amp = [16.435, 10.563]
        ra_pos, dec_pos = [0.12, 1.345], [1.563, 0.4567]
        model = self._ps_rendering.point_source_rendering(ra_pos, dec_pos, amp)
        model_ref = self._ps_rendering_ref.point_source_rendering(ra_pos, dec_pos, amp)
        npt.assert_allclose(model, model_ref, atol=1e-8, rtol=1e-8)

        amp = [92.435, 37.563]
        ra_pos, dec_pos = [-0.12, 1.55], [2.563, 1.4567]
        model = self._ps_rendering.point_source_rendering(ra_pos, dec_pos, amp)
        model_ref = self._ps_rendering_ref.point_source_rendering(ra_pos, dec_pos, amp)
        npt.assert_allclose(model, model_ref, atol=1e-8, rtol=1e-8)

        amp = [73.435, 35.563]
        ra_pos, dec_pos = [6.12, 5.55], [7.563, 4.4567]
        model = self._ps_rendering.point_source_rendering(ra_pos, dec_pos, amp)
        model_ref = self._ps_rendering_ref.point_source_rendering(ra_pos, dec_pos, amp)
        npt.assert_allclose(model, model_ref, atol=1e-8, rtol=1e-8)

    def test_psf_variance_map(self):
        ra_pos, dec_pos = [5], [5]
        data = np.ones((10, 10)) - 0.12389
        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        image_ref = self._ps_rendering_ref.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        npt.assert_allclose(image, image_ref, atol=1e-8, rtol=1e-8)

        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=5, data=data, fix_psf_variance_map=True
        )
        image_ref = self._ps_rendering_ref.psf_variance_map(
            ra_pos, dec_pos, amp=5, data=data, fix_psf_variance_map=True
        )
        npt.assert_allclose(image, image_ref, atol=1e-8, rtol=1e-8)

        ra_pos, dec_pos = [7], [7]
        data = np.ones((10, 10)) * 1.2892
        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        image_ref = self._ps_rendering_ref.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        npt.assert_allclose(image, image_ref, atol=1e-8, rtol=1e-8)


# Same tests as above but with supersampling = 3
class TestPointSourceRenderingSuperSampling(TestPointSourceRendering):
    def setup_method(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": 0,
            "dec_at_xy_0": 0,
            "transform_pix2angle": Mpix2coord,
            "nx": 10,
            "ny": 10,
        }
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((7, 7))
        kernel[1:6, 1:6] = 2
        kernel[4:4] = 7
        kwargs_psf = {
            "kernel_point_source": kernel,
            "psf_type": "PIXEL",
            "psf_variance_map": np.ones_like(kernel) * kernel**2,
            "point_source_supersampling_factor": 3,
        }
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(
            pixel_grid, supersampling_factor=None, psf=psf_class
        )
        self._ps_rendering_ref = PointSourceRendering_ref(
            pixel_grid, supersampling_factor=None, psf=psf_class
        )


class TestRaise(unittest.TestCase):
    def test_raise(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": 0,
            "dec_at_xy_0": 0,
            "transform_pix2angle": Mpix2coord,
            "nx": 10,
            "ny": 10,
        }
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {
            "kernel_point_source": kernel,
            "psf_type": "PIXEL",
            "psf_variance_map": np.ones_like(kernel),
        }
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(
            pixel_grid, supersampling_factor=1, psf=psf_class
        )
        with self.assertRaises(ValueError):
            self._ps_rendering.point_source_rendering(
                ra_pos=[1, 1], dec_pos=[0, 1], amp=[1]
            )


if __name__ == "__main__":
    pytest.main()
