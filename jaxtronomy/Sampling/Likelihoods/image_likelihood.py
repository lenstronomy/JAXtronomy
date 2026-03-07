from functools import partial
from jax import jit, numpy as jnp
from jaxtronomy.Util import class_creator

__all__ = ["ImageLikelihood"]


class ImageLikelihood(object):
    """Manages imaging data likelihoods."""

    def __init__(
        self,
        multi_band_list,
        multi_band_type,
        kwargs_model,
        bands_compute=None,
        image_likelihood_mask_list=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        kwargs_pixelbased=None,
        linear_solver=True,
    ):
        """
        :param multi_band_list: list of imaging band configurations [[kwargs_data, kwargs_psf, kwargs_numerics],[...], ...]
        :param multi_band_type: string, can be "single-band" only in jaxtronomy. "multi-linear" and "joint-liner" not supported yet
        :param kwargs_model: dict containing model option keyword arguments. See arguments to class_creator.create_class_instances() for options.
        :param bands_compute: bool list to indicate which band to be included in the modeling, only relevant for joint-linear and multi-linear.
            For single-band, the band index is zero by default.
        :param image_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be
            evaluated in the likelihood
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :param check_positive_flux: checks whether the results of the linear solver amplitudes are positive, and
            applies a logL penalty if amplitudes are negative (for profiles whose amplitudes have a physical interpretation)
        :param kwargs_pixelbased: should be None; not supported in jaxtronomy
        :param linear_solver: bool, determines whether to solve for linear parameters (amplitudes of sources)
        """
        if kwargs_pixelbased is not None:
            raise ValueError("pixelbased solver not supported in jaxtronomy")

        self.imSim = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=bands_compute,
            image_likelihood_mask_list=image_likelihood_mask_list,
            kwargs_pixelbased=kwargs_pixelbased,
            linear_solver=linear_solver,
        )
        self._model_type = self.imSim.type
        self._source_marg = source_marg
        self._linear_prior = linear_prior
        self._check_positive_flux = check_positive_flux

    @partial(jit, static_argnums=0)
    def logL(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_special=None,
        kwargs_extinction=None,
        **kwargs,
    ):
        """Computes the log likelihood of the data given a model. The model kwargs are
        used to simulate an image which is compared to the data image.

        :param kwargs_lens: lens model keyword argument list according to LensModel
            module
        :param kwargs_source: source light keyword argument list according to LightModel
            module
        :param kwargs_lens_light: deflector light (not lensed) keyword argument list
            according to LightModel module
        :param kwargs_ps: point source keyword argument list according to PointSource
            module
        :param kwargs_special: special keyword argument list as part of the Param module
        :param kwargs_extinction: extinction parameter keyword argument list according
            to LightModel module
        :return: log likelihood of the data given the model, linear parameter inversion
            list
        """
        logL, param = self.imSim.likelihood_data_given_model(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction=kwargs_extinction,
            kwargs_special=kwargs_special,
            source_marg=self._source_marg,
            linear_prior=self._linear_prior,
            check_positive_flux=self._check_positive_flux,
        )
        logL = jnp.nan_to_num(logL, nan=-(10**15))
        return logL, param

    @property
    def num_data(self):
        """

        :return: number of image data points
        """
        return self.imSim.num_data_evaluate

    @partial(jit, static_argnums=0)
    def num_param_linear(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_special=None,
        kwargs_extinction=None,
        kwargs_tracer_source=None,
    ):
        """

        :return:  number of linear parameters solved for during the image reconstruction process
        """
        return self.imSim.num_param_linear(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )

    # NOTE: PointSource cache not supported in jaxtronomy
    # def reset_point_source_cache(self, cache=True):
    #    """

    #    :param cache: boolean
    #    :return: None
    #    """
    #    self.imSim.reset_point_source_cache(cache=cache)
