from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import numpy as np
from jax import jit, numpy as jnp
from functools import partial

__all__ = ["PSBase", "_expand_to_array", "_shrink_array"]


class PSBase(object):
    """Base point source type class."""

    def __init__(
        self,
        lens_model=None,
        fixed_magnification=False,
        additional_images=False,
        index_lens_model_list=None,
        point_source_frame_list=None,
        redshift=None,
    ):
        """

        :param lens_model: instance of the LensModel() class
        :param fixed_magnification: bool. If True, magnification
            ratio of point sources is fixed to the one given by the lens model
        :param additional_images: bool. If True, search for additional images of the same source is conducted.
        :param index_lens_model_list: list (length of different patches/bands) of integer lists, evaluating a subset of
            the lens models per individual bands. e.g., [[0], [2, 3], [1]] assigns the 0th lens model to the 0th band,
            the 2nd and 3rd lens models to the 1st band, and the 1st lens model to the 2nd band.
            If this keyword is set, the image positions need to have a specified band/frame assigned to it
        :param point_source_frame_list: list of ints assigning each image to a specific band/frame. Only relevant in
            LENSED_POSITION. e.g. if LENSED_POSITION contains 4 images, we can assign them each to one of the bands with
            point_source_frame_list = [1, 2, 0, 1], where point_source_frame_list[i] = n means that the i-th image belongs
            to band n.
        :param redshift: redshift of the source, only required for multiple source redshifts
        :type redshift: None or float
        """
        if redshift is not None:
            raise ValueError("multiple source redshifts not supported in jaxtronomy")
        self._redshift = redshift
        self._lens_model = lens_model

        # Combine point_source_frame_list and index_lens_model_list to obtain k_list,
        # which assigns each image the corresponding lens models from its band
        # e.g. in the example used above, k_list = [[2, 3], [1], [0], [2, 3]]
        if index_lens_model_list is not None:
            k_list = []
            for point_source_frame in point_source_frame_list:
                # Needs to be a tuple in jaxtronomy
                k_list.append(tuple(index_lens_model_list[point_source_frame]))
            self.k_list = k_list
        else:
            # if none, then all images are assigned all lens models
            self.k_list = None
        if self._lens_model is None:
            self._solver = None
        else:
            self._solver = LensEquationSolver(lens_model)
        self._fixed_magnification = fixed_magnification
        if additional_images:
            raise ValueError("additional images not supported in jaxtronomy")
        self.additional_images = additional_images
        # if fixed_magnification is True and additional_images is True:
        #    Warning(
        #        "The combination of fixed_magnification=True and additional_image=True is not optimal for the "
        #        "current computation. If you see this warning, please approach the developers."
        #    )

    def image_position(self, kwargs_ps, **kwargs):
        """On-sky position.

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y image positions
        """
        raise ValueError(
            "image_position definition is not defined in the profile you want to execute."
        )

    def source_position(self, kwargs_ps, **kwargs):
        """Original unlensed position.

        :param kwargs_ps: keyword argument of point source model
        :return: numpy array of x, y source positions
        """
        raise ValueError(
            "source_position definition is not defined in the profile you want to execute."
        )

    def image_amplitude(self, kwargs_ps, *args, **kwargs):
        """Amplitudes as observed on the sky.

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call
        :return: numpy array of amplitudes
        """
        raise ValueError(
            "source_position definition is not defined in the profile you want to execute."
        )

    def source_amplitude(self, kwargs_ps, **kwargs):
        """Intrinsic source amplitudes (without lensing magnification, but still
        apparent)

        :param kwargs_ps: keyword argument of point source model
        :param kwargs: keyword arguments of function call (which are not used for this
            object
        :return: numpy array of amplitudes
        """
        raise ValueError(
            "source_position definition is not defined in the profile you want to execute."
        )

    # NOTE: Updating class instance attributes not supported in jaxtronomy
    def update_lens_model(self, lens_model_class):
        """Update LensModel() and LensEquationSolver() instance.

        :param lens_model_class: LensModel() class instance
        :return: internal `LensModel` class updated
        """
        raise ValueError(
            "Updating class instance attributes not supported in jaxtronomy"
        )

    #    self._lens_model = lens_model_class
    #    if lens_model_class is None:
    #        self._solver = None
    #    else:
    #        self._solver = LensEquationSolver(lens_model_class)


@partial(jit, static_argnums=1)
def _expand_to_array(array, num):
    """
    Case 1: If array is a scalar, returns jnp.ones(num) * array.
    Case 2: If len(array) < num, appends zeros to array until it is of size num.
    Case 3: If len(array) >= num, returns array with no modifications.

    :param array: float/int or numpy array
    :param num: number of array entries expected in array
    :return: array of size num
    """
    array = jnp.array(array, dtype=float)
    if jnp.isscalar(array):
        return jnp.ones(num) * array
    elif array.size < num:
        out = jnp.zeros(num)
        out = out.at[0 : array.size].set(array)
        return out
    else:
        return array


@partial(jit, static_argnums=1)
def _shrink_array(array, num):
    """
    Case 1: If array is a scalar, returns it with no modifications, regardless of num
    Case 2: If len(array) < num, raises error
    Case 3: If len(array) >= num, truncates the end of array until it is of size num

    :param array: float/int or numpy array
    :param num: number of array entries expected in array
    :return: array of size num, or scalar if array is a scalar
    """
    array = jnp.array(array, dtype=float)
    if array.shape == ():
        return array
    elif array.size < num:
        raise ValueError(
            "the length of the array (%s) needs to be larger or equal than the designated length %s "
            % (array.size, num)
        )
    else:
        return array.at[0:num].get()
