import numpy as np
from jax import jit, numpy as jnp

from lenstronomy.Util.package_util import exporter
from functools import partial

export, __all__ = exporter()


@export
class ImageNoise(object):
    """Class that deals with noise properties of imaging data."""

    # NOTE: JIT-compiled functions need to be recompiled each time a new instance of the class is created.

    def __init__(
        self,
        image_data,
        exposure_time=None,
        background_rms=None,
        noise_map=None,
        flux_scaling=1,
        gradient_boost_factor=None,
        verbose=True,
    ):
        """

        :param image_data: numpy array, pixel data values
        :param exposure_time: int or array of size the data; exposure time
         (common for all pixels or individually for each individual pixel)
         Units of data and exposure map should result in:
         number of flux counts = data * exposure_map
        :param background_rms: root-mean-square value of Gaussian background noise
        :param noise_map: int or array of size the data; joint noise sqrt(variance) of each individual pixel.
         Overwrites meaning of background_rms and exposure_time.
        :param flux_scaling: scales the model amplitudes to match the imaging data units. This can be used, for example,
         when modeling multiple exposures that have different magnitude zero points (or flux normalizations) but demand
         the same model normalization
        :type flux_scaling: float or int (default=1)
        :param gradient_boost_factor: None or float, variance terms added in quadrature scaling with
         gradient^2 * gradient_boost_factor. NOTE: NOT supported in Jaxtronomy
        """

        # Set exposure time
        if exposure_time is None:
            if noise_map is None:
                raise ValueError(
                    "Exposure map has not been specified in Noise() class!"
                )
        else:
            # make sure no negative exposure values are present no dividing by zero
            self.exposure_map = jnp.where(
                exposure_time <= 10 ** (-10), 10 ** (-10), exposure_time
            )

        # Set background rms
        if background_rms is None:
            if noise_map is None:
                raise ValueError(
                    "rms background value as 'background_rms' not specified!"
                )
            self.background_rms = np.median(noise_map)
        else:
            self.background_rms = background_rms

        self.data = jnp.array(image_data)
        self.flux_scaling = flux_scaling

        if noise_map is not None:
            assert np.shape(noise_map) == np.shape(image_data)
            self._noise_map = jnp.array(noise_map)
        else:
            self._noise_map = noise_map
            if background_rms is not None and exposure_time is not None:
                if np.any(background_rms * exposure_time < 1) and verbose is True:
                    print(
                        "WARNING! sigma_b*f %s < 1 count may introduce unstable error estimates with a Gaussian"
                        " error function for a Poisson distribution with mean < 1."
                        % (background_rms * np.max(exposure_time))
                    )

        # Covariance matrix of all pixel values in 2d numpy array (only diagonal component)
        # The covariance matrix is estimated from the data.
        # WARNING: For low count statistics, the noise in the data may lead to biased estimates of the covariance matrix.
        if self._noise_map is not None:
            self.C_D = self._noise_map**2
        else:
            self.C_D = covariance_matrix(
                self.data,
                self.background_rms,
                self.exposure_map,
            )

        if gradient_boost_factor is not None:
            raise ValueError(
                "gradient_boost_factor not supported in JAXtronomy. Please use lenstronomy instead"
            )

    @partial(jit, static_argnums=0)
    def C_D_model(self, model):
        """

        :param model: model (same as data but without noise)
        :return: estimate of the noise per pixel based on the model flux
        """

        if self._noise_map is not None:
            return self._noise_map**2
        else:
            return covariance_matrix(model, self.background_rms, self.exposure_map)


@export
@jit
def covariance_matrix(data, background_rms, exposure_map):
    """Returns a diagonal matrix for the covariance estimation which describes the
    error.

    Notes:

    - the exposure map must be positive definite. Values that deviate too much from the mean exposure time will be
        given a lower limit to not under-predict the Poisson component of the noise.

    - the data must be positive semi-definite for the Poisson noise estimate.
        Values < 0 (Possible after mean subtraction) will not have a Poisson component in their noise estimate.


    :param data: data array, eg in units of photons/second
    :param background_rms: background noise rms, eg. in units (photons/second)^2
    :param exposure_map: exposure time per pixel, e.g. in units of seconds
    :param gradient_boost_factor: None or float, variance terms added in quadrature scaling with
         gradient^2 * gradient_boost_factor
    :return: len(d) x len(d) matrix that give the error of background and Poisson components; (photons/second)^2
    """
    d_pos = jnp.where(data >= 0, data, 0)
    sigma = d_pos / exposure_map + background_rms**2
    return sigma
