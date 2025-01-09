from jax import jit, numpy as jnp, tree_util
from jax.scipy import signal
import numpy as np

from jaxtronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.Util import kernel_util
from jaxtronomy.Util import image_util, util
from functools import partial

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class PixelKernelConvolution(object):
    """Class to compute convolutions for a given pixelized kernel (fft, grid)"""

    def __init__(self, kernel, convolution_type="fft_static"):
        """

        :param kernel: 2d array, convolution kernel
        :param convolution_type: string, 'fft', 'grid', mode of 2d convolution
        """
        self._kernel = kernel
        if convolution_type not in ["fft", "grid"]:
            if convolution_type == "fft_static":
                self.fftconvolve_static = jit(
                    partial(signal.fftconvolve, in2=kernel, mode="same")
                )
            else:
                raise ValueError(
                    "convolution_type %s not supported!" % convolution_type
                )
        self.convolution_type = convolution_type

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (but there's no need to recompile if it changes to a previous value)
    def _tree_flatten(self):
        children = (self._kernel,)
        aux_data = {"convolution_type": self.convolution_type}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # ---------------------------------------------------------------------------------

    def pixel_kernel(self, num_pix=None):
        """Access pixelated kernel.

        :param num_pix: size of returned kernel (odd number per axis). If None, return
            the original kernel.
        :return: pixel kernel centered
        """
        if num_pix is not None:
            return kernel_util.cut_psf(self._kernel, num_pix)
        return self._kernel

    def copy_transpose(self):
        """

        :return: copy of the class with kernel set to the transpose of original one
        """
        return PixelKernelConvolution(
            self._kernel.T, convolution_type=self.convolution_type
        )

    @jit
    def convolution2d(self, image):
        """

        :param image: 2d array (image) to be convolved
        :return: fft convolution
        """
        if self.convolution_type == "fft":
            image_conv = signal.fftconvolve(image, self._kernel, mode="same")
        elif self.convolution_type == "fft_static":
            image_conv = self.fftconvolve_static(image)
        else:
            image_conv = signal.convolve2d(image, self._kernel, mode="same")
        return image_conv

    @jit
    def re_size_convolve(self, image_low_res, image_high_res=None):
        """

        :param image_low_res: regular sampled image/model
        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res)


@export
class SubgridKernelConvolution(object):
    """Class to compute the convolution on a supersampled grid with partial convolution
    computed on the regular grid."""

    def __init__(
        self,
        kernel_supersampled,
        supersampling_factor,
        supersampling_kernel_size=None,
        convolution_type="fft_static",
    ):
        """

        :param kernel_supersampled: kernel in supersampled pixels
        :param supersampling_factor: supersampling factor relative to the image pixel grid
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
         supersampled kernel
        """
        self._supersampling_factor = supersampling_factor
        if supersampling_kernel_size is None:
            kernel_low_res, kernel_high_res = np.zeros((3, 3)), kernel_supersampled
            self._low_res_convolution = False
        else:
            kernel_low_res, kernel_high_res = kernel_util.split_kernel(
                kernel_supersampled,
                supersampling_kernel_size,
                self._supersampling_factor,
            )
            self._low_res_convolution = True
        self._low_res_conv = PixelKernelConvolution(
            kernel_low_res, convolution_type=convolution_type
        )
        self._high_res_conv = PixelKernelConvolution(
            kernel_high_res, convolution_type=convolution_type
        )

    @partial(jit, static_argnums=0)
    def convolution2d(self, image):
        """

        :param image: 2d array (high resoluton image) to be convolved and re-sized
        :return: convolved image
        """

        image_high_res_conv = self._high_res_conv.convolution2d(image)
        image_resized_conv = image_util.re_size(
            image_high_res_conv, self._supersampling_factor
        )
        if self._low_res_convolution is True:
            image_resized = image_util.re_size(image, self._supersampling_factor)
            image_resized_conv += self._low_res_conv.convolution2d(image_resized)
        return image_resized_conv

    @partial(jit, static_argnums=0)
    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_high_res_conv = self._high_res_conv.convolution2d(image_high_res)
        image_resized_conv = image_util.re_size(
            image_high_res_conv, self._supersampling_factor
        )
        if self._low_res_convolution is True:
            image_resized_conv += self._low_res_conv.convolution2d(image_low_res)
        return image_resized_conv


class GaussianConvolution(object):
    """Class to perform a convolution consisting of multiple 2d Gaussians.

    Since JAX does not have an ndimage.gaussian_filter function, to perform Gaussian
    convolutions, we first create Gaussian psf kernels and convolve them using
    fftconvolve.
    """

    def __init__(
        self,
        sigma,
        pixel_scale,
        supersampling_factor=1,
        supersampling_convolution=False,
        truncation=2,
    ):
        """
        :param sigma: std value of Gaussian kernel
        :param pixel_scale: scale of pixel width (to convert sigmas into units of pixels)
        :param supersampling_factor: int, ratio of the number of pixels of the high resolution grid
            to the number of pixels of the original image
        :param supersampling convolution: bool, determines whether convolution uses supersampled grid or not
        :param truncation: float. Truncate the filter at this many standard deviations.
         Default is 4.0.
        """
        self._sigma_scaled = sigma / pixel_scale
        self._supersampling_factor = supersampling_factor
        self._supersampling_convolution = supersampling_convolution
        self._truncation = truncation

        if supersampling_convolution is True:
            self._sigma_scaled *= supersampling_factor

        # This num_pix definition is equivalent to that of the scipy ndimage.gaussian_filter
        # num_pix = 2r + 1 where r = round(truncation * sigma) is the radius of the gaussian kernel
        kernel_radius = max(round(self._sigma_scaled * self._truncation), 1)
        num_pix = 2 * kernel_radius + 1
        kernel = self.pixel_kernel(num_pix)

        # Before convolution, images will be padded
        # Even though kernel_radius is already an int, we need to apply int because of some JAX nonsense
        self._pad_width = int(kernel_radius)

        self.PixelKernelConv = PixelKernelConvolution(
            kernel, convolution_type="fft_static"
        )

    @partial(jit, static_argnums=0)
    def convolution2d(self, image):
        """Convolve the image via FFT convolution.

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """

        # Pads the image before convolution. This is equivalent to performing scipy.ndimage.gaussian_filter
        # with mode "nearest"
        image = jnp.pad(image, pad_width=self._pad_width, mode="edge")
        image_conv = jnp.zeros_like(image)

        image_conv = self.PixelKernelConv.convolution2d(image)

        # Removes the padding from the final image
        image_conv = image_conv[
            self._pad_width : -self._pad_width, self._pad_width : -self._pad_width
        ]

        return image_conv

    @partial(jit, static_argnums=0)
    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        if self._supersampling_convolution is True:
            image_high_res_conv = self.convolution2d(image_high_res)
            image_resized_conv = image_util.re_size(
                image_high_res_conv, self._supersampling_factor
            )
        else:
            image_resized_conv = self.convolution2d(image_low_res)
        return image_resized_conv

    def pixel_kernel(self, num_pix):
        """Computes a pixelized kernel from the Gaussian parameters.

        :param num_pix: int, size of kernel (odd number per axis) should be equal to 2 *
            sigma_scaled * truncation + 1 to be consistent with
            scipy.ndimage.gaussian_filter
        :return: pixel kernel centered
        """

        if num_pix % 2 == 0:
            raise ValueError("num_pix must be an odd integer")
        if num_pix < 3:
            raise ValueError("psf kernel size must be 3 or greater")

        gaussian = Gaussian()
        # Since sigma is in units of pixels, deltapix is trivially 1 in units of pixels
        x, y = util.make_grid(numPix=num_pix, deltapix=1)
        kernel = gaussian.function(x, y, amp=1, sigma=self._sigma_scaled)
        kernel = util.array2image(kernel)
        return kernel / jnp.sum(kernel)


tree_util.register_pytree_node(
    PixelKernelConvolution,
    PixelKernelConvolution._tree_flatten,
    PixelKernelConvolution._tree_unflatten,
)
