from jaxtronomy.LensModel.profile_list_base import ProfileListBase, _select_kwargs
from lenstronomy.Cosmo.background import Background
from lenstronomy.Util.cosmo_util import get_astropy_cosmology

from astropy.cosmology import *
from functools import partial
from jax import jit, lax, numpy as jnp
import numpy as np


class MultiPlaneGPU(ProfileListBase):
    """This class should be used whenever either of the following conditions are met:

    1) ray tracing is performed on GPU
    2) ray tracing is performed on CPU and the number of profile components exceeds 500,
        making the usual LensModel class unusable due to exploding compile times.
    """

    def __init__(
        self,
        unique_lens_model_list,
        profile_kwargs_list=None,
        cosmo=None,
        cosmology_model="FlatLambdaCDM",
    ):
        """
        :param unique_lens_model_list: a list of the unique lens models that will be used to perform ray shooting.
        :param profile_kwargs_list: list of dictionaries, keyword arguments used to intiialize different lens model profiles
        :param cosmo: instance of astropy cosmology. If None, an instance will be created based off of cosmology_model
        :param cosmology_model: string, used to initialize an instance of astropy cosmology if one was not already provided
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

        if cosmo is None and cosmology_model == "FlatLambdaCDM":
            cosmo = default_cosmology.get()
        elif cosmo is None and cosmology_model != "FlatLambdaCDM":
            cosmo = get_astropy_cosmology(cosmology_model=cosmology_model)
        self._cosmo_bkg = Background(cosmo)

    # This function is called outside of JIT
    def prepare_ray_shooting_kwargs(
        self,
        lens_model_list,
        kwargs_lens,
        z_source,
        lens_redshift_list,
        num_deflectors=None,
    ):
        """This is a helper functon which should be used to convert a lens_model_list
        and kwargs_lens from the typical lenstronomy convention to a format that is
        compatible with JAX scan.

        :param lens_model_list: list of lens models in the usual lenstronomy convention
        :param kwargs_lens: list of dictionaries for all keyword arguments for each lens
            model in the same order of the lens_model_list (same as in lenstronomy)
        :param z_source: float, redshift of source
        :param lens_redshift_list: list of redshifts for each deflector
        :param num_deflectors: optional int, fills the lens_model_list with NULL lens
            models until the length of lens_model_list is equal to num_deflectors + 1
            (the source will be treated as a deflector). Keeping num_deflectors fixed
            will avoid the recompilation of JAX functions, even if lens_model_list
            changes (since JAX needs to recompile functions whenever input arrays have a
            new shape)
        :return: ray_shooting_kwargs, a dictionary of kwargs for the ray_shooting()
            function. See docstring for ray_shooting().
        """
        if (
            len(lens_model_list) != len(kwargs_lens)
            or len(kwargs_lens) != len(lens_redshift_list)
            or len(lens_model_list) != len(lens_redshift_list)
        ):
            raise ValueError(
                f"length of lens model list {len(lens_model_list)}, length of kwargs_lens {len(kwargs_lens)}, and length of redshift list {len(lens_redshift_list)} should match"
            )

        if num_deflectors is None:
            num_deflectors = len(lens_model_list)
        elif num_deflectors < len(lens_model_list):
            raise ValueError(
                f"The provided num_deflectors {num_deflectors} is smaller than the length of lens_model_list {len(lens_model_list)}"
            )

        lens_model_list, lens_redshift_list, kwargs_lens = self._sort_lists_by_redshift(
            lens_model_list, lens_redshift_list, kwargs_lens
        )

        # Converts lens_model_list from typical lenstronomy convention to a list of indices, required for jax.lax.scan
        index_list = []
        for lens_model in lens_model_list:
            index_list.append(self.unique_lens_model_list.index(lens_model))

        # fills the rest of the list with empty lens models so that the array has size num_deflectors + 1
        num_filler = num_deflectors - len(lens_model_list)
        index_list += [len(self.unique_lens_model_list)] * (num_filler + 1)
        index_list = np.array(index_list)

        # Converts kwargs_lens from typical lenstronomy convention to a format required for JAX
        all_kwargs = {}
        unique_kwargs = set(
            param for sublist in self.param_name_list for param in sublist
        )

        for kwarg in unique_kwargs:
            all_kwargs[kwarg] = []
            for kwargs_profile in kwargs_lens:
                value = kwargs_profile.get(kwarg, 0)
                all_kwargs[kwarg].append(value)

            # Fill arrays with zeros so that all arrays have size num_deflectors + 1
            all_kwargs[kwarg] += [0] * (num_filler + 1)
            all_kwargs[kwarg] = np.array(all_kwargs[kwarg])

        # Compute transverse comoving distances with the source included
        T_ij_list, T_z_list, reduced2physical_factor = self._set_T_zs_and_T_ijs(
            lens_redshift_list, z_source, self._cosmo_bkg
        )

        # Fills distance lists so that they are of size num_deflectors + 1
        T_ij_list += [0] * num_filler
        T_ij_list = np.array(T_ij_list)
        T_z_list += [T_z_list[-1]] * num_filler
        T_z_list = np.array(T_z_list)
        reduced2physical_factor += [0] * (num_filler + 1)
        reduced2physical_factor = np.array(reduced2physical_factor)

        ray_shooting_kwargs = {
            "all_kwargs": all_kwargs,
            "index_list": index_list,
            "T_ij_list": T_ij_list,
            "T_z_list": T_z_list,
            "reduced2physical_factor": reduced2physical_factor,
        }
        return ray_shooting_kwargs

    @partial(jit, static_argnums=(0,))
    def ray_shooting(
        self,
        alpha_x,
        alpha_y,
        all_kwargs,
        index_list,
        T_ij_list,
        T_z_list,
        reduced2physical_factor,
    ):
        """Ray-tracing (backwards light cone) from redshift z=0 to z=z_source.

        :param alpha_x: ray angle at z_start=0 [arcsec]
        :param alpha_y: ray angle at z_start=0 [arcsec]
        :param all_kwargs: dictionary of JAX or numpy arrays, containing all parameters
            for all lens models
        :param index_list: list of ints, maps each element in lens_model_list to the
            unique_lens_model_list that was provided at class initialization
        :param T_ij_list: array of transverse angular distances between z=0 to the first
            deflector, then between the first deflector to the second deflector, etc
        :param T_z_list: array of transverse angular distances between z=0 to each
            deflector
        :param reduced2physical_factor: array of conversion factors from reduced
            deflection angles to physical deflection angles
        :return: angles at the source plane
        """
        alpha_x = jnp.array(alpha_x, dtype=float)
        alpha_y = jnp.array(alpha_y, dtype=float)
        x = jnp.zeros_like(alpha_x)
        y = jnp.zeros_like(alpha_y)

        # This function is called iteratively by jax.lax.scan to ray trace through all deflectors
        def body_fun(carry, xs):
            alpha_x, alpha_y, x, y = carry[0], carry[1], carry[2], carry[3]
            all_kwargs, index, delta_T, T_z, reduced2physical_factor = (
                xs[0],
                xs[1],
                xs[2],
                xs[3],
                xs[4],
            )
            x += alpha_x * delta_T
            y += alpha_y * delta_T
            alpha_x_red, alpha_y_red = lax.switch(
                index, self._derivatives_list, x / T_z, y / T_z, all_kwargs
            )
            alpha_x -= alpha_x_red * reduced2physical_factor
            alpha_y -= alpha_y_red * reduced2physical_factor
            return (alpha_x, alpha_y, x, y), 0

        (alpha_x, alpha_y, x, y), _ = lax.scan(
            body_fun,
            init=(alpha_x, alpha_y, x, y),
            xs=(all_kwargs, index_list, T_ij_list, T_z_list, reduced2physical_factor),
        )

        beta_x = x / T_z_list[-1]
        beta_y = y / T_z_list[-1]
        return beta_x, beta_y

    # This function is called outside of jit
    @staticmethod
    def _sort_lists_by_redshift(lens_model_list, lens_redshift_list, kwargs_lens):
        """Sorts the lens model list, lens redshift list, and kwargs_lens by increasing
        redshift.

        :param lens_model_list: list of lens models in the usual lenstronomy convention
        :param kwargs_lens: list of dictionaries for all keyword arguments for each lens
            model in the same order of the lens_model_list (same as in lenstronomy)
        :param lens_redshift_list: list of redshifts for each deflector
        :returns: same as the inputs but sorted by increasing redshift
        """

        lens_redshift_list = np.array(lens_redshift_list)
        sorted_indices = np.argsort(lens_redshift_list)

        lens_model_list = [lens_model_list[index] for index in sorted_indices]
        lens_redshift_list = lens_redshift_list[sorted_indices]
        kwargs_lens = [kwargs_lens[index] for index in sorted_indices]

        return lens_model_list, lens_redshift_list, kwargs_lens

    # This function is called outside of jit
    @staticmethod
    def _set_T_zs_and_T_ijs(lens_redshift_list, z_source, cosmo_bkg):
        """Set the transverse comoving distances between the observer and the lens
        planes and between the lens planes.

        :param lens_redshift_list: a SORTED np.array of redshifts
        :param z_source: float, redshift of source
        :cosmo_bkg: instance of lenstronomy.Cosmo.background.Background class

        Returns:
            T_ij_list: list of transverse comoving distances going from deflector i to deflector i+1,
                also includes the distance between the last deflector and z_source.
            T_z_list: list of transverse comoving distances between z=0 to each deflector and z_source.
            reduced2physical_factor: list of conversion factors from reduced deflection angles to physical deflection angles.
        """

        # reduced2physical_factor calculation
        z_source_array = np.ones(lens_redshift_list.shape) * z_source
        reduced2physical_factor = cosmo_bkg.d_xy(0, z_source) / cosmo_bkg.d_xy(
            lens_redshift_list, z_source_array
        )

        # Transverse comoving distance calculations
        T_ij_list = []
        T_z_list = []
        z_before = 0
        T_z = 0
        for z in np.append(lens_redshift_list, z_source):
            if z_before == z:
                delta_T = 0
            else:
                T_z = cosmo_bkg.T_xy(0, z)
                delta_T = cosmo_bkg.T_xy(z_before, z)
            T_ij_list.append(delta_T)
            T_z_list.append(T_z)
            z_before = z

        return T_ij_list, T_z_list, reduced2physical_factor.tolist()
