from jax import jit, numpy as jnp, tree_util
from jax.scipy import signal
import lenstronomy.Util.kernel_util as kernel_util
import jaxtronomy.Util.image_util as image_util
from functools import partial

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class PixelKernelConvolution(object):
    """Class to compute convolutions for a given pixelized kernel (fft, grid)"""

    def __init__(self, kernel, convolution_type="fft"):
        """

        :param kernel: 2d array, convolution kernel
        :param convolution_type: string, 'fft', 'grid', mode of 2d convolution
        """
        self._kernel = kernel
        if convolution_type not in ["fft", "grid"]:
            if convolution_type == "fft_static":
                raise ValueError("fft_static convolution not implemented in JAXtronomy")
            raise ValueError("convolution_type %s not supported!" % convolution_type)
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
        convolution_type="fft",
    ):
        """

        :param kernel_supersampled: kernel in supersampled pixels
        :param supersampling_factor: supersampling factor relative to the image pixel grid
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
         supersampled kernel
        """
        self._supersampling_factor = supersampling_factor
        if supersampling_kernel_size is None:
            kernel_low_res, kernel_high_res = jnp.zeros((3, 3)), kernel_supersampled
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


tree_util.register_pytree_node(
    PixelKernelConvolution,
    PixelKernelConvolution._tree_flatten,
    PixelKernelConvolution._tree_unflatten,
)
