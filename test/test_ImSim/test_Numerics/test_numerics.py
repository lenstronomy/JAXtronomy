import pytest
import numpy.testing as npt

import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util

from jaxtronomy.ImSim.Numerics.convolution import (
    PixelKernelConvolution,
    SubgridKernelConvolution,
    PartialSubgridKernelConvolution,
    GaussianConvolution,
)
from jaxtronomy.ImSim.Numerics.grid import RegularGrid, AdaptiveGrid

from jaxtronomy.ImSim.Numerics.numerics import Numerics
from lenstronomy.ImSim.Numerics.numerics import Numerics as Numerics_ref

from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel


# Test compute_mode = "regular"
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
        num_pix = 60  # cutout pixel size
        delta_pix = 0.05  # pixel size in arcsec (area per pixel = delta_pix**2)
        (
            x,
            y,
            ra_at_xy_0,
            dec_at_xy_0,
            x_at_radec_0,
            y_at_radec_0,
            transform_pix2coord,
            transform_coord2pix,
        ) = util.make_grid_with_coordtransform(
            num_pix=num_pix,
            delta_pix=delta_pix,
            subgrid_res=1,
            left_lower=False,
            inverse=False,
        )

        (
            x,
            y,
            ra_at_xy_0,
            dec_at_xy_0,
            x_at_radec_0,
            y_at_radec_0,
            transform_pix2coord,
            transform_coord2pix,
        ) = util.make_grid_with_coordtransform(
            num_pix=num_pix * self._supersampling_factor,
            delta_pix=delta_pix / self._supersampling_factor,
            subgrid_res=1,
            left_lower=False,
            inverse=False,
        )

        self.kernel_super = kernel_util.kernel_gaussian(
            num_pix=11 * self._supersampling_factor,
            delta_pix=delta_pix / self._supersampling_factor,
            fwhm=0.1,
        )

        kwargs_grid = {
            "nx": num_pix,
            "ny": num_pix,
            "transform_pix2angle": transform_pix2coord,
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
        self._compute_mode = "regular"
        self._backend = "cpu"

    def test_supersampling_cut_kernel(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            compute_mode=self._compute_mode,
        )
        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            compute_mode=self._compute_mode,
        )
        cut_kernel = numerics._supersampling_cut_kernel(
            self.kernel_super, 5, self._supersampling_factor
        )
        cut_kernel_ref = numerics_ref._supersampling_cut_kernel(
            self.kernel_super, 5, self._supersampling_factor
        )
        npt.assert_allclose(cut_kernel, cut_kernel_ref, atol=1e-16, rtol=1e-16)

        cut_kernel = numerics._supersampling_cut_kernel(
            self.kernel_super, None, self._supersampling_factor
        )
        cut_kernel_ref = numerics_ref._supersampling_cut_kernel(
            self.kernel_super, None, self._supersampling_factor
        )
        npt.assert_array_equal(cut_kernel, cut_kernel_ref)

    # supersampled ray shooting by not supersampled convolution
    def test_no_supersampling_pixel_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=False,
            convolution_kernel_size=7,
            convolution_type="fft",
        )
        assert numerics.grid_supersampling_factor == 5
        if self._compute_mode == "regular":
            grid_class = RegularGrid
        elif self._compute_mode == "adaptive":
            grid_class = AdaptiveGrid

        assert isinstance(numerics.convolution_class, PixelKernelConvolution)
        assert isinstance(numerics.grid_class, grid_class)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=False,
            convolution_kernel_size=7,
            convolution_type="fft",
        )

        x, y = numerics_ref.coordinates_evaluate
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)

        re_size_convolve = numerics.re_size_convolve(flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

        re_size_convolve = numerics.re_size_convolve(flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux, unconvolved=True)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

    # supersample both ray shooting and convolution
    def test_supersampling_pixel_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=True,
            supersampling_kernel_size=5,
            convolution_kernel_size=8,
            convolution_type="fft",
            backend=self._backend,
        )
        assert numerics.grid_supersampling_factor == 5

        if self._compute_mode == "regular":
            grid_class = RegularGrid
            atol = 1e-16
            rtol = 1e-16
        elif self._compute_mode == "adaptive":
            grid_class = AdaptiveGrid
            # adaptive implementation differs from lenstronomy
            atol = 5e-5
            rtol = 1e-5

        assert isinstance(
            numerics.convolution_class, PartialSubgridKernelConvolution
        ) or isinstance(numerics.convolution_class, SubgridKernelConvolution)
        assert isinstance(numerics.grid_class, grid_class)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_pixel,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=True,
            supersampling_kernel_size=5,
            convolution_kernel_size=8,
            convolution_type="fft",
        )
        npt.assert_array_equal(
            numerics.coordinates_evaluate, numerics_ref.coordinates_evaluate
        )

        x, y = numerics_ref.coordinates_evaluate
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)

        re_size_convolve = numerics.re_size_convolve(flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=atol, rtol=rtol
        )

        re_size_convolve = numerics.re_size_convolve(flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux, unconvolved=True)
        # These should actually be equal but there's some floating point precision nonsense happening
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

    # supersampled ray shooting by not supersampled convolution
    def test_no_supersampling_gaussian_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            compute_mode=self._compute_mode,
            supersampling_factor=self._supersampling_factor,
            supersampling_convolution=False,
            convolution_kernel_size=None,
        )
        assert numerics.grid_supersampling_factor == 5
        assert isinstance(numerics.convolution_class, GaussianConvolution)
        if self._compute_mode == "regular":
            grid_class = RegularGrid
        elif self._compute_mode == "adaptive":
            grid_class = AdaptiveGrid
        assert isinstance(numerics.grid_class, grid_class)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            compute_mode=self._compute_mode,
            supersampling_factor=self._supersampling_factor,
            supersampling_convolution=False,
            convolution_kernel_size=None,
        )

        x, y = numerics_ref.coordinates_evaluate
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)

        re_size_convolve = numerics.re_size_convolve(flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

        re_size_convolve = numerics.re_size_convolve(flux + 5.12838)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux + 5.12838)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

        re_size_convolve = numerics.re_size_convolve(flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux, unconvolved=True)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

    # supersampled ray shooting and supersampled convolution
    def test_supersampling_gaussian_psf(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=True,
        )
        assert numerics.grid_supersampling_factor == 5
        assert isinstance(numerics.convolution_class, GaussianConvolution)
        if self._compute_mode == "regular":
            grid_class = RegularGrid
        elif self._compute_mode == "adaptive":
            grid_class = AdaptiveGrid
        assert isinstance(numerics.grid_class, grid_class)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_gaussian,
            supersampling_factor=self._supersampling_factor,
            compute_mode=self._compute_mode,
            supersampling_convolution=True,
        )

        x, y = numerics_ref.coordinates_evaluate
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)

        re_size_convolve = numerics.re_size_convolve(flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux)
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

        re_size_convolve = numerics.re_size_convolve(flux, unconvolved=True)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux, unconvolved=True)
        # These should actually be equal but there's some floating point precision nonsense happening
        npt.assert_allclose(
            re_size_convolve, re_size_convolve_ref, atol=1e-16, rtol=1e-16
        )

    def test_psf_none(self):
        numerics = Numerics(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            compute_mode=self._compute_mode,
        )
        assert numerics.convolution_class is None
        if self._compute_mode == "regular":
            grid_class = RegularGrid
        elif self._compute_mode == "adaptive":
            grid_class = AdaptiveGrid
        assert isinstance(numerics.grid_class, grid_class)

        numerics_ref = Numerics_ref(
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            compute_mode=self._compute_mode,
        )

        x, y = numerics_ref.coordinates_evaluate
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)

        re_size_convolve = numerics.re_size_convolve(flux)
        re_size_convolve_ref = numerics_ref.re_size_convolve(flux)
        npt.assert_array_equal(re_size_convolve, re_size_convolve_ref)

    def test_init_raise_errors(self):

        # supersampling factor needs to be an int
        npt.assert_raises(
            TypeError,
            Numerics,
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            supersampling_factor=1.0,
        )

        # incorrect compute mode
        npt.assert_raises(
            ValueError,
            Numerics,
            pixel_grid=self.pixel_grid,
            psf=self.psf_class_none,
            compute_mode="incorrect",
        )

        # incorrect psf type
        self.psf_class_none.psf_type = "incorrect"
        npt.assert_raises(
            ValueError, Numerics, pixel_grid=self.pixel_grid, psf=self.psf_class_none
        )


# Same as above but testing Adaptive compute mode
# Skip Gaussian convolution tests due to lenstronomy incompatibility with adaptive compute mode
class TestNumerics2(object):
    def setup_method(self):
        TestNumerics.setup_method(self)
        self._compute_mode = "adaptive"

    def test_supersampling_pixel_psf(self):
        TestNumerics.test_supersampling_pixel_psf(self)

    def test_no_supersampling_pixel_psf(self):
        TestNumerics.test_no_supersampling_pixel_psf(self)

    def test_psf_none(self):
        TestNumerics.test_psf_none(self)


# Same as above but testing the GPU version of SubgridKernelConvolution
class TestNumerics3(object):
    def setup_method(self):
        TestNumerics.setup_method(self)
        self._backend = "gpu"

    def test_supersampling_pixel_psf(self):
        TestNumerics.test_supersampling_pixel_psf(self)


if __name__ == "__main__":
    pytest.main()
