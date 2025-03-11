import copy
from functools import partial
from jax import jit, numpy as jnp
import numpy as np


__all__ = ["PointSource"]

# NOTE: In jaxtronomy, only UNLENSED and LENSED_POSITION with additional_images=False models are currently supported
#       The lens equation solver is not used for these classes at the moment.
_SUPPORTED_MODELS = ["UNLENSED", "LENSED_POSITION", "SOURCE_POSITION"]


class PointSource(object):
    def __init__(
        self,
        point_source_type_list,
        lens_model=None,
        fixed_magnification_list=None,
        additional_images_list=None,
        flux_from_point_source_list=None,
        magnification_limit=None,
        save_cache=False,
        kwargs_lens_eqn_solver=None,
        index_lens_model_list=None,
        point_source_frame_list=None,
        redshift_list=None,
    ):
        """

        :param point_source_type_list: list of point source types
        :param lens_model: instance of the LensModel() class
        :param fixed_magnification_list: list of booleans (same length as point_source_type_list).
            If True, magnification ratio of point sources is fixed to the one given by the lens model.
            This option then requires to provide a 'source_amp' amplitude of the source brightness instead of
            'point_amp' the list of image brightnesses.
        :param additional_images_list: list of booleans (same length as point_source_type_list). If True, search for
            additional images of the same source is conducted.
        :param flux_from_point_source_list: list of booleans (optional), if set, will only return image positions
            (for imaging modeling) for the subset of the point source lists that =True. This option enables to model
            imaging data with transient point sources, when the point source positions are measured and present at a
            different time than the imaging data, or when the image position is not known (such as for lensed GW)
        :param magnification_limit: float >0 or None, if float is set and additional images are computed, only those
            images will be computed that exceed the lensing magnification (absolute value) limit
        :param save_cache: bool, saves image positions and only if delete_cache is executed, a new solution of the lens
            equation is conducted with the lens model parameters provided. This can increase the speed as multiple times
            the image positions are requested for the same lens model. Attention in usage!
        :param kwargs_lens_eqn_solver: keyword arguments specifying the numerical settings for the lens equation solver
            see LensEquationSolver() class for details, such as:
            min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100
        :param index_lens_model_list: list (length of different patches/bands) of integer lists, evaluating a subset of
            the lens models per individual bands. e.g., [[0], [2, 3], [1]] assigns the 0th lens model to the 0th band,
            the 2nd and 3rd lens models to the 1st band, and the 1st lens model to the 2nd band.
            If this keyword is set, the image positions need to have a specified band/frame assigned to it
        :param point_source_frame_list: list of list of ints, assigns each model in point_source_type_list a frame list.
            Only relevent for LENSED_POSITION. e.g. if point_source_type_list = ["UNLENSED", "LENSED_POSITION", "LENSED_POSITION"]
            with point_source_frame_list = [None, [0, 1, 2], [1, 2, 0, 1]], then the first LENSED_POSITION will have a frame list of
            [0, 1, 2] and the second LENSED_POSITION will have a frame list of [1, 2, 0, 1]. See docstring for point_source_frame_list
            in PSBase for further details.
        :param redshift_list: None or list of redshifts (only required for multiple source redshifts)
        """
        if "LENSED_POSITION" in point_source_type_list:
            if index_lens_model_list is not None and point_source_frame_list is None:
                raise ValueError(
                    "with specified index_lens_model_list, a specified point_source_frame_list argument is "
                    "required for LENSED_POSITION"
                )
            if index_lens_model_list is None:
                point_source_frame_list = [None] * len(point_source_type_list)
        if save_cache:
            raise ValueError("saving cache is not supported in jaxtronomy")
        self._index_lens_model_list = index_lens_model_list
        self._point_source_frame_list = point_source_frame_list
        self._lens_model = lens_model
        self.point_source_type_list = point_source_type_list
        self._point_source_list = []
        if fixed_magnification_list is None:
            fixed_magnification_list = [False] * len(point_source_type_list)
        self._fixed_magnification_list = fixed_magnification_list
        if additional_images_list is None:
            additional_images_list = [False] * len(point_source_type_list)
        if flux_from_point_source_list is None:
            flux_from_point_source_list = [True] * len(point_source_type_list)
        self._flux_from_point_source_list = flux_from_point_source_list
        if redshift_list is None:
            redshift_list = [None] * len(point_source_type_list)
        self._redshift_list = redshift_list
        for i, model in enumerate(point_source_type_list):
            if model == "UNLENSED":
                from jaxtronomy.PointSource.Types.unlensed import Unlensed

                self._point_source_list.append(Unlensed())
            elif model == "LENSED_POSITION":
                from jaxtronomy.PointSource.Types.lensed_position import LensedPositions

                self._point_source_list.append(
                    LensedPositions(
                        lens_model,
                        fixed_magnification=fixed_magnification_list[i],
                        additional_images=additional_images_list[i],
                        index_lens_model_list=index_lens_model_list,
                        point_source_frame_list=point_source_frame_list[i],
                        redshift=redshift_list[i],
                    )
                )
            elif model == "SOURCE_POSITION":
                raise ValueError("source position not supported in jaxtronomy")
            else:
                raise ValueError(
                    "Point-source model %s not available. Supported models are %s ."
                    % (model, _SUPPORTED_MODELS)
                )
        if kwargs_lens_eqn_solver is None:
            kwargs_lens_eqn_solver = {}
        self._kwargs_lens_eqn_solver = kwargs_lens_eqn_solver
        self._magnification_limit = magnification_limit
        self._save_cache = save_cache

    def update_search_window(
        self,
        search_window,
        x_center,
        y_center,
        min_distance=None,
        only_from_unspecified=False,
    ):
        """Update the search area for the lens equation solver.

        :param search_window: search_window: window size of the image position search
            with the lens equation solver.
        :param x_center: center of search window
        :param y_center: center of search window
        :param min_distance: minimum search distance
        :param only_from_unspecified: bool, if True, only sets keywords that previously
            have not been set
        :return: updated self instances
        """
        raise ValueError(
            "Updating class instance attributes not supported in jaxtronomy"
        )
        # if (
        #     min_distance is not None
        #     and "min_distance" not in self._kwargs_lens_eqn_solver
        #     and only_from_unspecified
        # ):
        #     self._kwargs_lens_eqn_solver["min_distance"] = min_distance
        # if only_from_unspecified:
        #     self._kwargs_lens_eqn_solver["search_window"] = (
        #         self._kwargs_lens_eqn_solver.get("search_window", search_window)
        #     )
        #     self._kwargs_lens_eqn_solver["x_center"] = self._kwargs_lens_eqn_solver.get(
        #         "x_center", x_center
        #     )
        #     self._kwargs_lens_eqn_solver["y_center"] = self._kwargs_lens_eqn_solver.get(
        #         "y_center", y_center
        #     )
        # else:
        #     self._kwargs_lens_eqn_solver["search_window"] = search_window
        #     self._kwargs_lens_eqn_solver["x_center"] = x_center
        #     self._kwargs_lens_eqn_solver["y_center"] = y_center

    def update_lens_model(self, lens_model_class):
        """

        :param lens_model_class: instance of LensModel class
        :return: update instance of lens model class
        """
        raise ValueError(
            "Updating class instance attributes not supported in jaxtronomy"
        )
        # self._lens_model = lens_model_class
        # for model in self._point_source_list:
        #    model.update_lens_model(lens_model_class=lens_model_class)

    def k_list(self, k):
        """

        :param k: index of point source model
        :return: list of lengths of images with corresponding lens models in the frame (or None if not multi-frame)
        """
        if self._index_lens_model_list is not None:
            k_list = []
            for point_source_frame in self._point_source_frame_list[k]:
                k_list.append(self._index_lens_model_list[point_source_frame])
        else:
            k_list = None
        return k_list

    @partial(jit, static_argnums=0)
    def source_position(self, kwargs_ps, kwargs_lens):
        """Intrinsic source positions of the point sources.

        :param kwargs_ps: keyword argument list of point source models
        :param kwargs_lens: keyword argument list of lens models
        :return: array of source positions for each point source model
        """
        x_source_list = []
        y_source_list = []
        for i, model in enumerate(self._point_source_list):
            kwargs = kwargs_ps[i]
            x_source, y_source = model.source_position(kwargs, kwargs_lens)
            x_source_list.append(x_source)
            y_source_list.append(y_source)
        return x_source_list, y_source_list

    @partial(jit, static_argnums=(0, 3, 5))
    def image_position(
        self,
        kwargs_ps,
        kwargs_lens,
        k=None,
        original_position=False,
        additional_images=False,
    ):
        """Image positions as observed on the sky of the point sources.

        :param kwargs_ps: point source parameter keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None or int; only returns a subset of the model predictions
        :param original_position: boolean (only applies to 'LENSED_POSITION' models),
            returns the image positions in the model parameters and does not re-compute
            images (which might be differently ordered) in case of the lens equation
            solver
        :param additional_images: if True, solves the lens equation for additional
            images
        :type additional_images: bool
        :return: list of: list of image positions per point source model component
        """
        x_image_list = []
        y_image_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                kwargs = kwargs_ps[i]
                x_image, y_image = model.image_position(
                    kwargs,
                    kwargs_lens,
                    magnification_limit=self._magnification_limit,
                    kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver,
                    additional_images=additional_images,
                )
                ## this takes action when new images are computed not necessarily in order
                # if (
                #    original_position is True
                #    and additional_images is True
                #    and self.point_source_type_list[i] == "LENSED_POSITION"
                # ):
                #    x_o, y_o = kwargs["ra_image"], kwargs["dec_image"]
                #    x_image, y_image = _sort_position_by_original(
                #        x_o, y_o, x_image, y_image
                #    )

                x_image_list.append(x_image)
                y_image_list.append(y_image)
        return x_image_list, y_image_list

    @partial(jit, static_argnums=(0, 3, 4))
    def point_source_list(self, kwargs_ps, kwargs_lens, k=None, with_amp=True):
        """Returns the image coordinates and image amplitudes of all point sources in a
        single array.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None or int; selects a subset of the point source models in the return
        :param with_amp: bool, if False, ignores the amplitude parameters in the return
            and instead provides ones for each point source image
        :return: ra_array, dec_array, amp_array
        """
        # we make sure we do not re-compute the image positions twice when evaluating position and their amplitudes
        ra_list, dec_list = self.image_position(kwargs_ps, kwargs_lens, k=k)
        if with_amp is True:
            amp_list = self.image_amplitude(kwargs_ps, kwargs_lens, k=k)

        # In lenstronomy, we get rid of images with 0 amplitude so that they are not rendered
        # However we cannot do that here since the amplitudes are not known at compile time,
        # and therefore the final array size would not be known at compile time.
        ra_array, dec_array, amp_array = [], [], []
        for i in range(len(ra_list)):
            for j in range(ra_list[i].size):
                ra_array.append(ra_list[i][j])
                dec_array.append(dec_list[i][j])
                if with_amp:
                    amp_array.append(amp_list[i][j])
                else:
                    amp_array.append(1.0)
        return jnp.array(ra_array), jnp.array(dec_array), jnp.array(amp_array)

    @partial(jit, static_argnums=(0, 3))
    def image_amplitude(self, kwargs_ps, kwargs_lens, k=None):
        """Returns the image amplitudes.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param k: None or int, selects a subset of the point source models in the return
        :return: list of image amplitudes per model component
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            if k is None or k == i:
                image_amp = model.image_amplitude(
                    kwargs_ps=kwargs_ps[i],
                    kwargs_lens=kwargs_lens,
                    kwargs_lens_eqn_solver=self._kwargs_lens_eqn_solver,
                )
                if self._flux_from_point_source_list[i]:
                    amp_list.append(image_amp)
                else:
                    amp_list.append(jnp.zeros_like(image_amp))

        return amp_list

    @partial(jit, static_argnums=(0))
    def source_amplitude(self, kwargs_ps, kwargs_lens):
        """Intrinsic (unlensed) point source amplitudes.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: list of intrinsic (unlensed) point source amplitudes
        """
        amp_list = []
        for i, model in enumerate(self._point_source_list):
            source_amp = model.source_amplitude(
                kwargs_ps=kwargs_ps[i], kwargs_lens=kwargs_lens
            )
            if self._flux_from_point_source_list[i]:
                amp_list.append(source_amp)
            else:
                amp_list.append(jnp.zeros_like(source_amp))
        return amp_list

    @partial(jit, static_argnums=(0,))
    def check_image_positions(self, kwargs_ps, kwargs_lens, tolerance=0.001):
        """Checks whether the point sources in kwargs_ps satisfy the lens equation with
        a tolerance (computed by ray-tracing to the source plane)

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param tolerance: Euclidian distance between the source positions ray-traced
            backwards to be tolerated
        :return: bool: True, if requirement on tolerance is fulfilled, False if not.
        """
        x_image_list, y_image_list = self.image_position(kwargs_ps, kwargs_lens)
        within_tolerance = True
        for i, model in enumerate(self.point_source_type_list):
            if model in ["LENSED_POSITION", "SOURCE_POSITION"]:
                x_pos = x_image_list[i]
                y_pos = y_image_list[i]
                # TODO: ray-trace to specific source redshift
                x_source, y_source = self._lens_model.ray_shooting(
                    x_pos, y_pos, kwargs_lens
                )
                dist = jnp.sqrt(
                    (x_source - x_source[0]) ** 2 + (y_source - y_source[0]) ** 2
                )
                within_tolerance = jnp.where(
                    jnp.max(dist) > tolerance, False, within_tolerance
                )
        return within_tolerance

    # This function should be called outside of a JIT'd environment
    def set_amplitudes(self, amp_list, kwargs_ps):
        """Translates the amplitude parameters into the convention of the keyword
        argument list currently only used in SimAPI to transform magnitudes to
        amplitudes in the lenstronomy conventions.

        :param amp_list: list of model amplitudes for each point source model. This list
            should include all of the point source models even if flux_from_point_source
            is False for any of them. In that case, the amplitudes will not be changed
            for those models.
        :param kwargs_ps: list of point source keywords
        :return: overwrites kwargs_ps with new amplitudes
        """
        kwargs_list = copy.deepcopy(kwargs_ps)
        for i, model in enumerate(self.point_source_type_list):
            if self._flux_from_point_source_list[i]:
                amp = amp_list[i]
                if model == "UNLENSED":
                    kwargs_list[i]["point_amp"] = amp
                elif model in ["LENSED_POSITION", "SOURCE_POSITION"]:
                    if self._fixed_magnification_list[i] is True:
                        kwargs_list[i]["source_amp"] = amp
                    else:
                        kwargs_list[i]["point_amp"] = amp
        return kwargs_list


# def _sort_position_by_original(x_o, y_o, x_solved, y_solved):
#    """Sorting new image positions such that the old order is best preserved.
#
#    :param x_o: numpy array; original image positions
#    :param y_o: numpy array; original image positions
#    :param x_solved: numpy array; solved image positions with potentially more or fewer
#        images
#    :param y_solved: numpy array; solved image positions with potentially more or fewer
#        images
#    :return: sorted new image positions with the order best matching the original
#        positions first, and then all other images in the same order as solved for
#    """
#    if len(x_o) > len(x_solved):
#        # if new images are less , then return the original images (no sorting required)
#        x_solved_new, y_solved_new = x_o, y_o
#    else:
#        x_solved_new, y_solved_new = [], []
#        for i in range(len(x_o)):
#            x, y = x_o[i], y_o[i]
#            r2_i = (x - x_solved) ** 2 + (y - y_solved) ** 2
#            # index of minimum radios
#            index = np.argmin(r2_i)
#            x_solved_new.append(x_solved[index])
#            y_solved_new.append(y_solved[index])
#            # delete this index
#            x_solved = np.delete(x_solved, index)
#            y_solved = np.delete(y_solved, index)
#        # now we append the remaining additional images in the same order behind the original ones
#        x_solved_new = np.append(np.array(x_solved_new), x_solved)
#        y_solved_new = np.append(np.array(y_solved_new), y_solved)
#    return x_solved_new, y_solved_new
