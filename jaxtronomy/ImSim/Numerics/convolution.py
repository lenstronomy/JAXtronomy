from jax import jit, numpy as jnp
from jax.scipy import signal
import numpy as np

from jaxtronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.Util import kernel_util
from jaxtronomy.Util import image_util, util
from functools import partial


class PixelKernelConvolution(object):
    """Class to compute convolutions for a given pixelized kernel (fft, grid).

    convolution_type="fft_static" does not result in a speedup in JAXtronomy, thus it is
    not implemented.
    """

    def __init__(self, kernel=None, convolution_type="fft"):
        """

        :param kernel: 2d array, convolution kernel, can also be supplied at runtime
        :param convolution_type: string, 'fft', 'grid', mode of 2d convolution
        """
        if kernel is not None:
            kernel = jnp.asarray(kernel, dtype=float)
        self._kernel = kernel
        if convolution_type not in ["fft", "grid"]:
            raise ValueError("convolution_type %s not supported!" % convolution_type)
        self.convolution_type = convolution_type

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

    @partial(jit, static_argnums=0)
    def convolution2d(self, image, kernel=None):
        """

        :param image: 2d array (image) to be convolved
        :return: fft convolution
        """
        if kernel is None:
            kernel = self._kernel
        if self.convolution_type == "fft":
            image_conv = signal.fftconvolve(image, kernel, mode="same")
        elif self.convolution_type == "grid":
            image_conv = signal.convolve2d(image, kernel, mode="same")
        else:
            raise ValueError(
                "convolution_type %s not supported!" % self.convolution_type
            )
        return image_conv

    @partial(jit, static_argnums=0)
    def re_size_convolve(self, image_low_res, image_high_res=None):
        """

        :param image_low_res: regular sampled image/model
        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res)


class SubgridKernelConvolution(object):
    """Class to compute the convolution on a supersampled grid with partial convolution
    computed on the regular grid.

    For high resolution convolution, one single convolution is performed using the high
    resolution kernel and high resolution image.
    """

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


class PartialSubgridKernelConvolution(object):
    """Class to compute the convolution on a supersampled grid with partial convolution
    computed on the regular grid.

    For high resolution convolution, rather than performing one single convolution over
    the entire image, N^2 low-resolution convolutions are performed and summed, where N
    is the supersampling factor. On CPU, this is a performance boost over performing one
    high resolution convolution. This class is effectively the same as lenstronomy's
    SubgridNumbaConvolution, but with FFT instead of real space convolution.
    """

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
            kernel_low_res, kernel_high_res = np.zeros((3, 3)), kernel_supersampled
            self._low_res_convolution = False
        else:
            kernel_low_res, kernel_high_res = kernel_util.split_kernel(
                kernel_supersampled,
                supersampling_kernel_size,
                self._supersampling_factor,
            )
            self._low_res_convolution = True
        self._kernel_low_res = kernel_low_res

        self._kernel_high_res_ij = []
        for i in range(supersampling_factor):
            for j in range(supersampling_factor):
                # compute shifted psf kernel
                kernel = self._partial_kernel(kernel_high_res, i, j)
                self._kernel_high_res_ij.append(jnp.asarray(kernel, dtype=float))

        self._conv = PixelKernelConvolution(kernel=None, convolution_type="fft")

    @partial(jit, static_argnums=0)
    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_low_res: 2d array, image at native resolution
        :param image_high_res: 2d array, supersampled image
        :return: convolved and re-sized image
        """
        image_conv = jnp.zeros(image_low_res.shape)

        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_select = self._partial_image(image_high_res, i, j)
                image_conv += self._conv.convolution2d(
                    image_select, self._kernel_high_res_ij[count]
                )
                count += 1

        if self._low_res_convolution is True:
            image_conv += self._conv.convolution2d(image_low_res, self._kernel_low_res)
        return image_conv

    @partial(jit, static_argnums=0)
    def convolution2d(self, image):
        """

        :param image: 2d array (high resoluton image) to be convolved and re-sized
        :return: convolved image
        """
        n_rows, n_cols = image.shape
        n_rows = int(n_rows / self._supersampling_factor)
        n_cols = int(n_cols / self._supersampling_factor)
        image_conv = jnp.zeros((n_rows, n_cols))

        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_ij = self._partial_image(image, i, j)
                image_conv += self._conv.convolution2d(
                    image_ij, self._kernel_high_res_ij[count]
                )
                count += 1

        if self._low_res_convolution is True:
            image_resized = image_util.re_size(image, self._supersampling_factor)
            image_conv += self._conv.convolution2d(
                image_resized, kernel=self._kernel_low_res
            )
        return image_conv

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
        :return: effective kernel rebinned to regular grid resulting from the supersampled position (i,j)
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

        self.PixelKernelConv = PixelKernelConvolution(kernel, convolution_type="fft")

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
        # Since sigma is in units of pixels, delta_pix is trivially 1 in units of pixels
        x, y = util.make_grid(num_pix=num_pix, delta_pix=1)
        kernel = gaussian.function(x, y, amp=1, sigma=self._sigma_scaled)
        kernel = util.array2image(kernel)
        return kernel / jnp.sum(kernel)
