from jaxtronomy.PointSource.Types.base_ps import PSBase
from jax import jit, numpy as jnp

__all__ = ["Unlensed"]


class Unlensed(PSBase):
    """
    class of a single point source in the image plane, aka star
    Name within the PointSource module: 'UNLENSED'
    This model can deal with arrays of point sources.
    parameters: ra_image, dec_image, point_amp

    """

    @staticmethod
    @jit
    def image_position(kwargs_ps, *args, **kwargs):
        """On-sky position.

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y image positions
        """
        ra_image = kwargs_ps["ra_image"]
        dec_image = kwargs_ps["dec_image"]
        return jnp.array(ra_image), jnp.array(dec_image)

    @staticmethod
    @jit
    def source_position(kwargs_ps, *args, **kwargs):
        """Original physical position (identical for this object)

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y source positions
        """
        ra_image = kwargs_ps["ra_image"]
        dec_image = kwargs_ps["dec_image"]
        return jnp.array(ra_image), jnp.array(dec_image)

    @staticmethod
    @jit
    def image_amplitude(kwargs_ps, *args, **kwargs):
        """Amplitudes as observed on the sky.

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this
            object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps["point_amp"]
        return jnp.array(point_amp)

    @staticmethod
    @jit
    def source_amplitude(kwargs_ps, *args, **kwargs):
        """Intrinsic source amplitudes.

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this
            object
        :return: numpy array of amplitudes
        """
        point_amp = kwargs_ps["point_amp"]
        return jnp.array(point_amp)
