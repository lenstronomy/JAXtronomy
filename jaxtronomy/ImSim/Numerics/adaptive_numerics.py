from jaxtronomy.ImSim.Numerics.partial_convolution import (
    SubgridPartialConvolution,
    PartialConvolution,
)
from jaxtronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from jaxtronomy.Util import image_util
from lenstronomy.Util import kernel_util, image_util as image_util_lenstronomy

from jax import jit
from functools import partial

__all__ = ["AdaptiveConvolution"]


class AdaptiveConvolution(object):
    """
    This class performs convolutions of a subset of pixels at higher supersampled resolution
    Goal: Allows ray tracing to be done supersampled only on a subset of pixels.

    strategy:
    1. lower resolution FFT convolution over full image
    2. higher resolution FFT convolution over selected subset of pixels (with smaller kernel)
    3. the same subset of pixels with low resolution FFT convolution (with same kernel as step 2)
    adaptive solution is 1 + 2 - 3

    """

    def __init__(
        self,
        kernel_super,
        supersampling_factor,
        conv_supersample_pixels,
        supersampling_kernel_size=None,
        compute_pixels=None,
    ):
        """

        :param kernel_super: convolution kernel in units of super sampled pixels provided, odd length per axis
        :param supersampling_factor: factor of supersampling relative to pixel grid
        :param conv_supersample_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
         supersampled kernel
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        """
        kernel = kernel_util.degrade_kernel(
            kernel_super, degrading_factor=supersampling_factor
        )
        self._low_res_conv = PixelKernelConvolution(kernel, convolution_type="fft")
        if supersampling_kernel_size is None:
            supersampling_kernel_size = len(kernel)

        n_cut_super = supersampling_kernel_size * supersampling_factor
        if n_cut_super % 2 == 0:
            n_cut_super += 1
        kernel_super_cut = image_util_lenstronomy.cut_edges(kernel_super, n_cut_super)
        kernel_cut = kernel_util.degrade_kernel(
            kernel_super_cut, degrading_factor=supersampling_factor
        )

        self._low_res_partial = PartialConvolution(
            kernel_cut,
            conv_supersample_pixels,
            compute_pixels=compute_pixels,
        )
        self._hig_res_partial = SubgridPartialConvolution(
            kernel_super_cut,
            supersampling_factor,
            conv_supersample_pixels,
            compute_pixels=compute_pixels,
        )
        self._supersampling_factor = supersampling_factor

    @partial(jit, static_argnums=0)
    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_low_res: regular sampled image/model
        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_low_res_conv = self._low_res_conv.convolution2d(image_low_res)
        image_low_res_partial_conv = self._low_res_partial.convolve2d(image_low_res)
        image_high_res_partial_conv = self._hig_res_partial.convolve2d(image_high_res)
        return (
            image_low_res_conv
            + image_high_res_partial_conv
            - image_low_res_partial_conv
        )

    @partial(jit, static_argnums=0)
    def convolve2d(self, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_low_res = image_util.re_size(
            image_high_res, factor=self._supersampling_factor
        )
        return self.re_size_convolve(image_low_res, image_high_res)
