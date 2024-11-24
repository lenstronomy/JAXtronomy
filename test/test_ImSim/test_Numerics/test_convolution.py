__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
from jaxtronomy.ImSim.Numerics.convolution import (
    PixelKernelConvolution,
    SubgridKernelConvolution,
)
from lenstronomy.ImSim.Numerics.convolution import (
    PixelKernelConvolution as PixelKernelConvolution_ref,
    SubgridKernelConvolution as SubgridKernelConvolution_ref,
)
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as util
import pytest


class TestPixelKernelConvolution(object):
    def setup_method(self):
        lightModel = LightModel(light_model_list=["GAUSSIAN"])
        self.delta_pix = 1
        x, y = util.make_grid(10, deltapix=self.delta_pix)
        kwargs = [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)

    def test_init(self):
        kernel = np.ones((3, 3)) * 2
        kernel[1, 1] = 1
        kernel = kernel / np.sum(kernel)

        npt.assert_raises(
            ValueError,
            PixelKernelConvolution,
            kernel=kernel,
            convolution_type="fft_static",
        )
        npt.assert_raises(
            ValueError,
            PixelKernelConvolution,
            kernel=kernel,
            convolution_type="incorrect",
        )

    def test_convolve2d_fft(self):
        kernel = np.ones((3, 3)) * 2
        kernel[1, 1] = 1
        kernel = kernel / np.sum(kernel)

        pixel_conv = PixelKernelConvolution(kernel=kernel)
        pixel_conv_ref = PixelKernelConvolution_ref(kernel=kernel)
        image_convolved = pixel_conv.convolution2d(self.model)
        image_convolved_ref = pixel_conv_ref.convolution2d(self.model)
        npt.assert_almost_equal(image_convolved, image_convolved_ref, decimal=5)

    def test_convolve2d_grid(self):
        kernel = np.ones((3, 3)) * 2
        kernel[1, 1] = 1
        kernel = kernel / np.sum(kernel)

        pixel_conv = PixelKernelConvolution(kernel=kernel, convolution_type="grid")
        pixel_conv_ref = PixelKernelConvolution_ref(
            kernel=kernel, convolution_type="grid"
        )
        image_convolved = pixel_conv.convolution2d(self.model)
        image_convolved_ref = pixel_conv_ref.convolution2d(self.model)
        npt.assert_almost_equal(image_convolved, image_convolved_ref, decimal=8)

    def test_convolve2d_incorrect(self):
        kernel = np.ones((3, 3)) * 2
        kernel[1, 1] = 1
        kernel = kernel / np.sum(kernel)

        pixel_conv = PixelKernelConvolution(kernel=kernel, convolution_type="grid")
        pixel_conv.convolution_type = "incorrect"
        npt.assert_raises(ValueError, pixel_conv.convolution2d, self.model)

    def test_copy_transpose(self):
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1.0 / 7.0
        kernel[2, 0] = 3.0 / 7.0
        kernel[0, 2] = 3.0 / 7.0
        pixel_conv = PixelKernelConvolution(kernel=kernel)
        pixel_conv_t = pixel_conv.copy_transpose()
        image_convolved = pixel_conv.convolution2d(self.model)
        image_convolved_t = pixel_conv_t.convolution2d(self.model)
        npt.assert_array_almost_equal(image_convolved, image_convolved_t, decimal=8)

    def test_pixel_kernel(self):
        kernel = np.zeros((5, 5))
        kernel[1, 1] = 1.0 / 3.0
        kernel[2, 0] = 2.0 / 3.0
        pixel_conv = PixelKernelConvolution(kernel=kernel)
        npt.assert_equal(pixel_conv.pixel_kernel(), kernel)
        npt.assert_equal(
            pixel_conv.pixel_kernel(num_pix=3),
            kernel[1:-1, 1:-1] / np.sum(kernel[1:-1, 1:-1]),
        )

    def test_re_size_convolve(self):
        kernel = np.ones((3, 3)) * 2
        kernel[1, 1] = 1
        kernel = kernel / np.sum(kernel)

        pixel_conv = PixelKernelConvolution(kernel=kernel)
        pixel_conv_ref = PixelKernelConvolution_ref(kernel=kernel)
        image_convolved = pixel_conv.re_size_convolve(self.model)
        image_convolved_ref = pixel_conv_ref.re_size_convolve(self.model)
        npt.assert_almost_equal(image_convolved, image_convolved_ref, decimal=5)


class TestSubgridKernelConvolution(object):
    def setup_method(self):
        self.supersampling_factor = 3
        lightModel = LightModel(light_model_list=["GAUSSIAN"])
        self.delta_pix = 1.0
        x, y = util.make_grid(20, deltapix=self.delta_pix)
        x_sub, y_sub = util.make_grid(
            20 * self.supersampling_factor,
            deltapix=self.delta_pix / self.supersampling_factor,
        )
        kwargs = [{"amp": 1, "sigma": 2, "center_x": 0, "center_y": 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)
        flux_sub = lightModel.surface_brightness(x_sub, y_sub, kwargs)
        self.model_sub = util.array2image(flux_sub)

        x, y = util.make_grid(5, deltapix=self.delta_pix)
        kwargs_kernel = [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}]
        kernel = lightModel.surface_brightness(x, y, kwargs_kernel)
        self.kernel = util.array2image(kernel) / np.sum(kernel)

        x_sub, y_sub = util.make_grid(
            5 * self.supersampling_factor,
            deltapix=self.delta_pix / self.supersampling_factor,
        )
        kernel_sub = lightModel.surface_brightness(x_sub, y_sub, kwargs_kernel)
        self.kernel_sub = util.array2image(kernel_sub) / np.sum(kernel_sub)

    def test_convolve2d(self):
        subgrid_conv = SubgridKernelConvolution(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=None,
            convolution_type="fft",
        )
        model_subgrid_conv = subgrid_conv.convolution2d(self.model_sub)

        subgrid_conv_ref = SubgridKernelConvolution_ref(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=None,
            convolution_type="fft",
        )
        model_subgrid_conv_ref = subgrid_conv_ref.convolution2d(self.model_sub)

        npt.assert_array_almost_equal(
            model_subgrid_conv, model_subgrid_conv_ref, decimal=6
        )

    def test_convolve2d_low_res(self):
        subgrid_conv_split = SubgridKernelConvolution(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=5,
            convolution_type="fft",
        )
        model_subgrid_conv = subgrid_conv_split.convolution2d(self.model_sub)

        subgrid_conv_split_ref = SubgridKernelConvolution_ref(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=5,
            convolution_type="fft",
        )
        model_subgrid_conv_ref = subgrid_conv_split_ref.convolution2d(self.model_sub)

        npt.assert_array_almost_equal(
            model_subgrid_conv, model_subgrid_conv_ref, decimal=6
        )

    def test_re_size_convolve(self):
        subgrid_conv = SubgridKernelConvolution(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=None,
            convolution_type="fft",
        )
        re_size_conv = subgrid_conv.re_size_convolve(self.model, self.model_sub)

        subgrid_conv_ref = SubgridKernelConvolution_ref(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=None,
            convolution_type="fft",
        )
        re_size_conv_ref = subgrid_conv_ref.re_size_convolve(self.model, self.model_sub)

        npt.assert_array_almost_equal(re_size_conv, re_size_conv_ref, decimal=6)

    def test_re_size_convolve_low_res(self):
        subgrid_conv = SubgridKernelConvolution(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=5,
            convolution_type="fft",
        )
        re_size_conv = subgrid_conv.re_size_convolve(self.model, self.model_sub)

        subgrid_conv_ref = SubgridKernelConvolution_ref(
            self.kernel_sub,
            self.supersampling_factor,
            supersampling_kernel_size=5,
            convolution_type="fft",
        )
        re_size_conv_ref = subgrid_conv_ref.re_size_convolve(self.model, self.model_sub)

        npt.assert_array_almost_equal(re_size_conv, re_size_conv_ref, decimal=6)


if __name__ == "__main__":
    pytest.main()
