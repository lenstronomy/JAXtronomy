__author__ = "sibirrer"

import jax
from jax import jit, lax, numpy as jnp
from jaxtronomy.LensModel.profile_list_base import ProfileListBase, _select_kwargs
from functools import partial
import numpy as np

jax.config.update("jax_enable_x64", True)

__all__ = ["SinglePlaneGPU"]


class SinglePlaneGPU(ProfileListBase):
    """This class should be used whenever either of the following conditions are met:

    1) ray tracing is performed on GPU
    2) ray tracing is performed on CPU and the number of profile components exceeds 500,
        making the usual LensModel class unusable due to exploding compile times.
    """

    def __init__(
        self,
        unique_lens_model_list,
        profile_kwargs_list=None,
    ):
        """
        :param unique_lens_model_list: a list of the unique lens models that will be used to perform ray shooting.
        :param profile_kwargs_list: list of dictionaries, keyword arguments used to intiialize different lens model profiles
        """

        self.unique_lens_model_list = unique_lens_model_list
        if isinstance(profile_kwargs_list, list):
            profile_kwargs_list += [{}]

        ProfileListBase.__init__(
            self,
            unique_lens_model_list + ["NULL"],
            profile_kwargs_list,
            lens_redshift_list=None,
            z_source_convention=None,
        )

        self._derivatives_list = [
            _select_kwargs(self.func_list[i], self.param_name_list[i])
            for i in range(len(self.func_list))
        ]

    # This function is called outside of JIT
    def prepare_ray_shooting_kwargs(
        self, lens_model_list, kwargs_lens, num_deflectors=None
    ):
        """This is a helper functon which should be used to convert a lens_model_list
        and kwargs_lens from the typical lenstronomy convention to a format that is
        compatible with JAX scan.

        :param lens_model_list: list of lens models in the usual lenstronomy convention
        :param kwargs_lens: list of dictionaries for all keyword arguments for each lens
            model in the same order of the lens_model_list (same as in lenstronomy)
        :param num_deflectors: optional int, fills the lens_model_list with NULL lens
            models until the length of lens_model_list is equal to num_deflectors + 1
            (the source will be treated as a deflector). Keeping num_deflectors fixed
            will avoid the recompilation of JAX functions, even if lens_model_list
            changes (since JAX needs to recompile functions whenever input arrays have a
            new shape)
        :return: ray_shooting_kwargs, a dictionary of kwargs for the ray_shooting()
            function. See docstring for ray_shooting().
        """
        if len(lens_model_list) != len(kwargs_lens):
            raise ValueError(
                f"length of lens model list {len(lens_model_list)} and length of kwargs_lens {len(kwargs_lens)} do not match"
            )

        if num_deflectors is None:
            num_deflectors = len(lens_model_list)
        elif num_deflectors < len(lens_model_list):
            raise ValueError(
                f"The provided num_deflectors {num_deflectors} is smaller than the length of lens_model_list {len(lens_model_list)}"
            )

        num_filler = num_deflectors - len(kwargs_lens)

        # Converts lens_model_list to a list of indices
        index_list = []
        for lens_model in lens_model_list:
            index_list.append(self.unique_lens_model_list.index(lens_model))

        # fills the rest of the list with empty lens models
        index_list += [len(self.unique_lens_model_list)] * num_filler
        index_list = np.array(index_list)

        # Converts kwargs_lens from typical lenstronomy convention to a format required for JAX scan
        all_kwargs = {}
        unique_kwargs = set(
            param for sublist in self.param_name_list for param in sublist
        )

        for kwarg in unique_kwargs:
            all_kwargs[kwarg] = []
            for kwargs_profile in kwargs_lens:
                value = kwargs_profile.get(kwarg, 0)
                all_kwargs[kwarg].append(value)

            # Fill arrays with zeros to keep them all the same size, equal to num_deflectors
            all_kwargs[kwarg] += [0] * num_filler
            all_kwargs[kwarg] = np.array(all_kwargs[kwarg])

        ray_shooting_kwargs = {"all_kwargs": all_kwargs, "index_list": index_list}
        return ray_shooting_kwargs

    @partial(jit, static_argnums=0)
    def ray_shooting(self, x, y, all_kwargs, index_list):
        """This function computes the ray tracing through all of the lens models given
        by index_list. The arguments all_kwargs and index_list should be obtained by
        calling SinglePlaneGPU.prepare_ray_shooting_kwargs().

        :param x: x-position (preferentially arcsec)
        :param y: y-position (preferentially arcsec)
        :param all_kwargs: dictionary of JAX or numpy arrays, containing all parameters
            for all lens models
        :param index_list: list of ints, maps each element in lens_model_list to the
            unique_lens_model_list that was provided at class initialization
        """

        def body_fun(xs):
            all_kwargs, index = xs[0], xs[1]
            return lax.switch(index, self._derivatives_list, x, y, all_kwargs)

        f_x, f_y = lax.map(body_fun, (all_kwargs, index_list))

        return x - jnp.sum(f_x, axis=0), y - jnp.sum(f_y, axis=0)