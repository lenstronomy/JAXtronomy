from jaxtronomy.LensModel.single_plane_gpu import SinglePlaneGPU
from jaxtronomy.LensModel.MultiPlane.multi_plane_gpu import MultiPlaneGPU

from functools import partial
from jax import jit

__all__ = ["LensModelGPU"]


class LensModelGPU(object):
    """This class should be used whenever either of the following conditions are met:

    1) ray tracing is performed on GPU
    2) ray tracing is performed on CPU and the number of profile components exceeds 500,
        making the usual LensModel class unusable due to exploding compile times.
    """

    def __init__(
        self,
        unique_lens_model_list,
        multi_plane,
        profile_kwargs_list=None,
        cosmo=None,
        cosmology_model="FlatLambdaCDM",
    ):
        """
        :param unique_lens_model_list: list of strings with unique lens model names
            not required for the core functionalities in the single plane mode.
        :param multi_plane: bool, if True, uses multi-plane mode. Default is False.
        :param profile_kwargs_list: list of dicts, keyword arguments used to initialize profile classes
            in the same order of the lens_model_list. If any of the profile_kwargs are None, then that
            profile will be initialized using default settings.
        :param cosmo: instance of the astropy cosmology class. If not specified, uses the default cosmology.
        :param cosmology_model: string, used to initialize an instance of astropy cosmology if one was not already provided
        """
        if profile_kwargs_list is None:
            profile_kwargs_list = [{} for _ in range(len(unique_lens_model_list))]
        self.profile_kwargs_list = profile_kwargs_list

        # Multi-plane or single-plane lensing?
        if multi_plane:
            self.lens_model = MultiPlaneGPU(
                unique_lens_model_list=unique_lens_model_list,
                cosmo=cosmo,
                profile_kwargs_list=profile_kwargs_list,
                cosmology_model=cosmology_model,
            )
            self.type = "MultiPlaneGPU"
        else:
            self.lens_model = SinglePlaneGPU(
                unique_lens_model_list=unique_lens_model_list,
                profile_kwargs_list=profile_kwargs_list,
            )
            self.type = "SinglePlaneGPU"

        # Save these for convenience if class reinitialization is required
        self.init_kwargs = {
            "unique_lens_model_list": unique_lens_model_list,
            "cosmo": cosmo,
            "multi_plane": multi_plane,
            "profile_kwargs_list": profile_kwargs_list,
            "cosmology_model": cosmology_model,
        }

    def prepare_ray_shooting_kwargs(
        self,
        lens_model_list,
        kwargs_lens,
        z_source=None,
        lens_redshift_list=None,
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
        multi_plane_kwargs = {}
        if self.type == "MultiPlaneGPU":
            if lens_redshift_list is None:
                raise ValueError(
                    "In multi-plane lensing, you need to specify the redshifts of the lensing planes."
                )
            if z_source is None:
                raise ValueError(
                    "z_source needs to be set for multi-plane lens modelling."
                )
            multi_plane_kwargs = {
                "z_source": z_source,
                "lens_redshift_list": lens_redshift_list,
            }

        return self.lens_model.prepare_ray_shooting_kwargs(
            lens_model_list=lens_model_list,
            kwargs_lens=kwargs_lens,
            num_deflectors=num_deflectors,
            **multi_plane_kwargs
        )

    @partial(jit, static_argnums=(0,))
    def ray_shooting(self, x, y, ray_shooting_kwargs):
        """Maps image to source position (inverse deflection)

        :param x: x-position (preferentially arcsec)
        :param y: y-position (preferentially arcsec)
        :param ray_shooting_kwargs: dict of ray shooting kwargs, should be obtained by
            first calling prepare_ray_shooting_kwargs(). For more details about these
            kwargs, see docstring for SinglePlaneGPU.ray_shooting() or
            MultiPlaneGPU.ray_shooting()
        :return: source plane positions corresponding to (x, y) in the image plane
        """
        return self.lens_model.ray_shooting(x, y, **ray_shooting_kwargs)
