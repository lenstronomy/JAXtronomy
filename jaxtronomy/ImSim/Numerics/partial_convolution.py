from functools import partial
from jax import jit, numpy as jnp
import numpy as np

from jaxtronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import image_util

__all__ = ["PartialConvolution", "SubgridPartialConvolution"]


class PartialConvolution(object):
    """Class to convolve explicit pixels only. This class is the JAXtronomy version of
    lenstronomy's NumbaConvolution class. Although the implementation differs significantly,
    the end result is the same.
    """

    def __init__(
        self,
        kernel,
        conv_pixels,
        compute_pixels=None,
    ):
        """

        :param kernel: convolution kernel in units of the image pixels provided, odd length per axis
        :param conv_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other
            pixels
        """
        self._kernel = kernel
        self._conv_pixels = conv_pixels
        self._nx, self._ny = np.shape(conv_pixels)
        if compute_pixels is None:
            compute_pixels = np.ones_like(conv_pixels)
            compute_pixels = np.array(compute_pixels, dtype=bool)
        assert np.shape(conv_pixels) == np.shape(compute_pixels)
        self._mask = compute_pixels
        self._pixel_conv = PixelKernelConvolution(kernel, convolution_type="fft")

    @partial(jit, static_argnums=0)
    def convolve2d(self, image):
        """2d convolution.

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """
        conv_image = self._pixel_conv.convolution2d(image * self._conv_pixels)
        return conv_image * self._mask


class SubgridPartialConvolution(object):
    """Class that inputs a supersampled grid and convolution kernel and computes the
    response on the regular grid. This makes use of the regular PixelKernelConvolution class as
    a loop through the different sub-pixel positions, mirroring the functionality of lenstronomy's
    SubgridNumbaConvolution class."""

    def __init__(
        self,
        kernel_super,
        supersampling_factor,
        conv_pixels,
        compute_pixels=None,
        kernel_size=None,
    ):
        """

        :param kernel_super: convolution kernel in units of super sampled pixels provided, odd length per axis
        :param supersampling_factor: factor of supersampling relative to pixel grid
        :param conv_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        """
        self._nx, self._ny = conv_pixels.shape
        self._supersampling_factor = supersampling_factor
        # loop through the different supersampling sectors
        self._numba_conv_list = []
        if compute_pixels is None:
            compute_pixels = np.ones_like(conv_pixels)
            compute_pixels = np.array(compute_pixels, dtype=bool)
        self._compute_pixels = compute_pixels

        self._kernel_ij = []
        for i in range(supersampling_factor):
            for j in range(supersampling_factor):
                # compute shifted psf kernel
                kernel = self._partial_kernel(kernel_super, i, j)
                if kernel_size is not None:
                    kernel = image_util.cut_edges(kernel, kernel_size)
                self._kernel_ij.append(kernel)

        self._conv = PixelKernelConvolution(
            kernel=None,
            convolution_type="fft"
        )

    @partial(jit, static_argnums=0)
    def convolve2d(self, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved and re-bined to regular resolution
        :return: convolved and re-bind image
        """
        conv_image = jnp.zeros((self._nx, self._ny))
        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_select = self._partial_image(image_high_res, i, j) * self._compute_pixels
                conv_image += self._conv.convolution2d(image_select, self._kernel_ij[count])
                count += 1
        return conv_image * self._compute_pixels

    def _partial_image(self, image_high_res, i, j):
        """

        :param image_high_res: 2d array supersampled
        :param i: index of super-sampled position in first axis
        :param j: index of super-sampled position in second axis
        :return: 2d array only selected the specific supersampled position within a regular pixel
        """
        return image_high_res[
            i :: self._supersampling_factor, j :: self._supersampling_factor
        ]

    def _partial_kernel(self, kernel_super, i, j):
        """

        :param kernel_super: supersampled kernel
        :param i: index of super-sampled position in first axis
        :param j: index of super-sampled position in second axis
        :return: effective kernel rebinned to regular grid resulting from the subpersampled position (i,j)
        """
        n = len(kernel_super)
        kernel_size = int(round(n / float(self._supersampling_factor) + 1.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        n_match = kernel_size * self._supersampling_factor
        kernel_super_match = np.zeros((n_match, n_match))
        delta = int((n_match - n - self._supersampling_factor) / 2) + 1
        i0 = delta  # index where to start kernel for i=0
        j0 = delta  # index where to start kernel for j=0  (should be symmetric)
        kernel_super_match[i0 + i : i0 + i + n, j0 + j : j0 + j + n] = kernel_super
        # kernel_super_match = image_util.cut_edges(kernel_super_match, numPix=n)
        kernel = image_util.re_size(
            kernel_super_match, factor=self._supersampling_factor
        )
        return kernel
