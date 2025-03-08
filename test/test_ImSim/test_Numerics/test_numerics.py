import pytest
import jax
import numpy.testing as npt

import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util

from jaxtronomy.ImSim.Numerics.convolution import (
    PixelKernelConvolution,
    SubgridKernelConvolution,
    GaussianConvolution,
)
from jaxtronomy.ImSim.Numerics.grid import RegularGrid

from jaxtronomy.ImSim.Numerics.numerics import Numerics
from lenstronomy.ImSim.Numerics.numerics import Numerics as Numerics_ref

from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel

jax.config.update("jax_enable_x64", True)


class TestNumerics(object):
    def setup_method(self):
        # we define a model consisting of a single Sersic profile

        light_model_list = ["SERSIC_ELLIPSE"]
        self.lightModel = LightModel(light_model_list=light_model_list)
        self.kwargs_light = [
            {
                "amp": 100,
                "R_sersic": 0.5,
                "n_sersic": 3,
                "e1": -0.3123,
                "e2": 0.1234,
                "center_x": 0.02,
                "center_y": 0,
            }
        ]

        # we define a pixel grid and a higher resolution super sampling factor
        self._supersampling_factor = 5
        numPix = 60  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        (
            x,
            y,
            ra_at_xy_0,
            dec_at_xy_0,
            x_at_radec_0,
            y_at_radec_0,
            Mpix2coord,
            Mcoord2pix,
        ) = util.make_grid_with_coordtransform(
            numPix=numPix,
            deltapix=deltaPix,
            subgrid_res=1,
            left_lower=False,
            inverse=False,
        )
        self.flux = self.lightModel.surface_brightness(
            x, y, kwargs_list=self.kwargs_light
        )

        (
            x,
            y,
            ra_at_xy_0,
            dec_at_xy_0,
            x_at_radec_0,
            y_at_radec_0,
            Mpix2coord,
            Mcoord2pix,
        ) = util.make_grid_with_coordtransform(
            numPix=numPix * self._supersampling_factor,
            deltapix=deltaPix / self._supersampling_factor,
            subgrid_res=1,
            left_lower=False,
            inverse=False,
        )
        self.flux_supersampled = self.lightModel.surface_brightness(
            x, y, kwargs_list=self.kwargs_light
        )

        self.kernel_super = kernel_util.kernel_gaussian(
            num_pix=11 * self._supersampling_factor,
            delta_pix=deltaPix / self._supersampling_factor,
            fwhm=0.1,
        )

        kwargs_grid = {
            "nx": numPix,
            "ny": numPix,
            "transform_pix2angle": Mpix2coord,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
        }
        self.pixel_grid = PixelGrid(**kwargs_grid)

        kwargs_psf_pixel = {
            "psf_type": "PIXEL",
            "kernel_point_source": self.kernel_super,
            "point_source_supersampling_factor": self._supersampling_factor,
            "kernel_point_source_normalisation": True,
        }
        self.psf_class_pixel = PSF(**kwargs_psf_pixel)

        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
            "kernel_point_source_normalisation": True,
            "pixel_size": 0.11,
        }
        self.psf_class_gaussian = PSF(**kwargs_psf_gaussian)
        kwargs_psf_none = {
            "psf_type": "NONE",
        }
        self.psf_class_none = PSF(**kwargs_psf_none)

    def test_supersampling_cut_kernel(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
        )
        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
        )
        cut_kernel = numerics._supersampling_cut_kernel(
            self.kernel_super, 5, self._supersampling_factor
        )
        cut_kernel_ref = numerics_ref._supersampling_cut_kernel(
            self.kernel_super, 5, self._supersampling_factor
        )
        npt.assert_array_almost_equal(cut_kernel, cut_kernel_ref, decimal=8)

        cut_kernel = numerics._supersampling_cut_kernel(
            self.kernel_super, None, self._supersampling_factor
        )
        cut_kernel_ref = numerics_ref._supersampling_cut_kernel(
            self.kernel_super, None, self._supersampling_factor
        )
        npt.assert_array_equal(cut_kernel, cut_kernel_ref)

    def test_no_supersampling_pixel_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=1,
            compute_mode="regular",
            supersampling_convolution=False,
            convolution_kernel_size=7,
            convolution_type="fft",
        )
        assert numerics.grid_supersampling_factor == 1
        assert isinstance(numerics.convolution_class, PixelKernelConvolution)
        assert isinstance(numerics.grid_class, RegularGrid)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=1,
            compute_mode="regular",
            supersampling_convolution=False,
            convolution_kernel_size=7,
            convolution_type="fft",
        )

        re_size_convolve = numerics.re_size_convolve(self.flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux)
        npt.assert_array_almost_equal(re_size_convolve, re_size_convolve_ref, decimal=8)

        re_size_convolve = numerics.re_size_convolve(self.flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(
            self.flux, unconvolved=True
        )
        npt.assert_array_equal(re_size_convolve, re_size_convolve_ref)

    def test_supersampling_pixel_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode="regular",
            supersampling_convolution=True,
            supersampling_kernel_size=5,
            convolution_kernel_size=8,
            convolution_type="fft",
        )
        assert numerics.grid_supersampling_factor == 5
        assert isinstance(numerics.convolution_class, SubgridKernelConvolution)
        assert isinstance(numerics.grid_class, RegularGrid)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode="regular",
            supersampling_convolution=True,
            supersampling_kernel_size=5,
            convolution_kernel_size=8,
            convolution_type="fft",
        )
        npt.assert_array_equal(
            numerics.coordinates_evaluate, numerics_ref.coordinates_evaluate
        )

        re_size_convolve = numerics.re_size_convolve(self.flux_supersampled)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux_supersampled)
        npt.assert_array_almost_equal(re_size_convolve, re_size_convolve_ref, decimal=8)

        re_size_convolve = numerics.re_size_convolve(
            self.flux_supersampled, unconvolved=True
        )
        re_size_convolve_ref = numerics_ref.re_size_convolve(
            self.flux_supersampled, unconvolved=True
        )
        # These should actually be equal but there's some floating point precision nonsense happening
        npt.assert_array_almost_equal(
            re_size_convolve, re_size_convolve_ref, decimal=16
        )

    def test_no_supersampling_gaussian_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            compute_mode="regular",
            supersampling_factor=1,
            supersampling_convolution=False,
            convolution_kernel_size=None,
        )
        assert numerics.grid_supersampling_factor == 1
        assert isinstance(numerics.convolution_class, GaussianConvolution)
        assert isinstance(numerics.grid_class, RegularGrid)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            compute_mode="regular",
            supersampling_factor=1,
            supersampling_convolution=False,
            convolution_kernel_size=None,
        )

        re_size_convolve = numerics.re_size_convolve(self.flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux)
        npt.assert_array_almost_equal(re_size_convolve, re_size_convolve_ref, decimal=5)

        re_size_convolve = numerics.re_size_convolve(self.flux + 5.12838)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux + 5.12838)
        npt.assert_array_almost_equal(re_size_convolve, re_size_convolve_ref, decimal=5)

        re_size_convolve = numerics.re_size_convolve(self.flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(
            self.flux, unconvolved=True
        )
        npt.assert_array_equal(re_size_convolve, re_size_convolve_ref)

    def test_supersampling_gaussian_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            supersampling_factor=self._supersampling_factor,
            compute_mode="regular",
            supersampling_convolution=True,
        )
        assert numerics.grid_supersampling_factor == 5
        assert isinstance(numerics.convolution_class, GaussianConvolution)
        assert isinstance(numerics.grid_class, RegularGrid)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            supersampling_factor=self._supersampling_factor,
            compute_mode="regular",
            supersampling_convolution=True,
        )

        re_size_convolve = numerics.re_size_convolve(self.flux_supersampled)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux_supersampled)
        npt.assert_array_almost_equal(re_size_convolve, re_size_convolve_ref, decimal=5)

        re_size_convolve = numerics.re_size_convolve(
            self.flux_supersampled, unconvolved=True
        )
        re_size_convolve_ref = numerics_ref.re_size_convolve(
            self.flux_supersampled, unconvolved=True
        )
        # These should actually be equal but there's some floating point precision nonsense happening
        npt.assert_array_almost_equal(
            re_size_convolve, re_size_convolve_ref, decimal=16
        )

    def test_psf_none(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
        )
        assert numerics.convolution_class is None
        assert isinstance(numerics.grid_class, RegularGrid)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
        )

        re_size_convolve = numerics.re_size_convolve(self.flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(self.flux)
        npt.assert_array_equal(re_size_convolve, re_size_convolve_ref)

    def test_init_raise_errors(self):
        npt.assert_raises(
            TypeError,
            Numerics,
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            supersampling_factor=1.0,
        )
        npt.assert_raises(
            ValueError,
            Numerics,
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            compute_mode="incorrect",
        )
        npt.assert_raises(
            ValueError,
            Numerics,
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            compute_mode="adaptive",
        )
        self.psf_class_none.psf_type = "incorrect"
        npt.assert_raises(
            ValueError, Numerics, pixel_grid=self.pixel_grid, psf=self.psf_class_none
        )


if __name__ == "__main__":
    pytest.main()
