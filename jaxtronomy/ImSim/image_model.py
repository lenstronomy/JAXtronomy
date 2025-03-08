__author__ = "sibirrer"

from jaxtronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.PointSource.point_source import PointSource
from jaxtronomy.Util import util

from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
from lenstronomy.Util import util as util_lenstronomy

from functools import partial
from jax import jit, numpy as jnp
import numpy as np

__all__ = ["ImageModel"]

# TODO: Implement PointSource and extinction in JAXtronomy
# Probably will not implement pixelbased solver at all


class ImageModel(object):
    """This class uses functions of lens_model and source_model to make a lensed
    image."""

    def __init__(
        self,
        data_class,
        psf_class=None,
        lens_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_class=None,
        extinction_class=None,
        kwargs_numerics=None,
        likelihood_mask=None,
        psf_error_map_bool_list=None,
        kwargs_pixelbased=None,
    ):
        """
        :param data_class: instance of ImageData() or PixelGrid() class
        :param psf_class: instance of PSF() class
        :param lens_model_class: instance of LensModel() class
        :param source_model_class: instance of LightModel() class describing the source parameters
        :param lens_light_model_class: instance of LightModel() class describing the lens light parameters
        :param point_source_class: instance of PointSource() class describing the point sources
        :param kwargs_numerics: keyword arguments with various numeric description (see ImageNumerics class for options)
        :param likelihood_mask: 2d boolean array of pixels to be counted in the likelihood calculation.
            Must be a np array; cannot be a jnp array.
        :param psf_error_map_bool_list: list of boolean of length of point source models.
            Indicates whether PSF error map is used for the point source model stated as the index.
        :param kwargs_pixelbased: kwargs for pixelbased solver; not supported in jaxtronomy. Must be None
        """

        self.type = "single-band"
        self.num_bands = 1
        self.PSF = psf_class
        self.Data = data_class
        if hasattr(self.Data, "flux_scaling"):
            self._flux_scaling = self.Data.flux_scaling
        else:
            self._flux_scaling = 1
        self.PSF.set_pixel_size(self.Data.pixel_width)
        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.ImageNumerics = NumericsSubFrame(
            pixel_grid=self.Data, psf=self.PSF, **kwargs_numerics
        )
        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class
        if point_source_class is None:
            point_source_class = PointSource(
                point_source_type_list=[], lens_model=lens_model_class
            )

        # If user initiated point source class without the lens model class,
        # re-initiate the class with the lens models included
        if point_source_class._lens_model is None:
            point_source_class = PointSource(
                point_source_type_list=point_source_class.point_source_type_list,
                lens_model=lens_model_class,
                fixed_magnification_list=point_source_class._fixed_magnification_list,
                flux_from_point_source_list=point_source_class._flux_from_point_source_list,
                magnification_limit=point_source_class._magnification_limit,
                save_cache=point_source_class._save_cache,
                kwargs_lens_eqn_solver=point_source_class._kwargs_lens_eqn_solver,
                index_lens_model_list=point_source_class._index_lens_model_list,
                point_source_frame_list=point_source_class._point_source_frame_list,
                redshift_list=point_source_class._redshift_list,
            )
        self.PointSource = point_source_class

        x_center, y_center = self.Data.center
        search_window = np.max(self.Data.width)
        # either have the pixel resolution of the window resolved in 200x200 grid
        min_distance = min(self.Data.pixel_width, search_window / 200)
        # TODO: Lens equation solver is not used in jaxtronomy PointSource yet.
        #       Re-initialize the class with updated kwargs_lens_eqn_solver
        #       instead of calling update_search_window
        # self.PointSource.update_search_window(
        #     search_window=search_window,
        #     x_center=x_center,
        #     y_center=y_center,
        #     min_distance=min_distance,
        #     only_from_unspecified=True,
        # )
        if source_model_class is None:
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        if lens_light_model_class is None:
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class
        self._kwargs_numerics = kwargs_numerics
        if extinction_class is None:
            extinction_class = DifferentialExtinction(optical_depth_model=[])
        self._extinction = extinction_class
        if kwargs_pixelbased is None:
            kwargs_pixelbased = {}
        else:
            raise ValueError("pixelbased solver not supported in JAXtronomy")
        self.source_mapping = Image2SourceMapping(
            lens_model=lens_model_class, source_model=source_model_class
        )
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(
                self.PointSource.point_source_type_list
            )
        self._psf_error_map_bool_list = psf_error_map_bool_list
        self._psf_error_map = self.PSF.psf_variance_map_bool

        # NOTE: likelihood mask cannot be a traced jnp array; must be concrete np array
        if likelihood_mask is None:
            likelihood_mask = np.ones(data_class.num_pixel_axes)
        self.likelihood_mask = np.array(likelihood_mask, dtype=bool)

        # number of pixels used in likelihood calculation
        self.num_data_evaluate = np.sum(self.likelihood_mask)
        # conversion of likelihood mask into 1d array
        self._mask1d = util_lenstronomy.image2array(self.likelihood_mask)

        # primary beam is not supported yet
        self._pb = data_class.primary_beam
        if self._pb is not None:
            raise ValueError("primary beam not supported in jaxtronomy")
            # self._pb_1d = util.image2array(self._pb)
        else:
            self._pb_1d = None

    # def reset_point_source_cache(self, cache=True):
    #    """Deletes all the cache in the point source class and saves it from then on.

    #    :param cache: boolean, if True, saves the next occuring point source positions
    #        in the cache
    #    :return: None
    #    """
    #    self.PointSource.delete_lens_model_cache()
    #    self.PointSource.set_save_cache(cache)

    @partial(jit, static_argnums=(0, 7, 8, 9))
    def likelihood_data_given_model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
    ):
        """Computes the likelihood of the data given a model This is specified with the
        non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_source: list of dicts, keyword arguments corresponding to the
            superposition of different source light profiles in the same order of
            light_model_list
        :param kwargs_lens_light: list of dicts, keyword arguments corresponding to
            different lens light surface brightness profiles in the same order of
            lens_light_model_list
        :param kwargs_ps: list of dicts, keyword arguments for the points source models
            in the same order of point_source_type_list
        :param kwargs_extinction: list of dicts, keyword arguments corresponding to
            different light profiles in the optical_depth_model
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :param check_positive_flux: bool, if True, checks whether the linear inversion
            resulted in non-negative flux components and applies a punishment in the
            likelihood if so.
        :param linear_solver: bool, if True (default) fixes the linear amplitude
            parameters 'amp' (avoid sampling) such that they get overwritten by the
            linear solver solution. Should always be false in jaxtronomy
        :return: log likelihood (natural logarithm), linear parameter list
        """
        if check_positive_flux:
            raise ValueError("check positive flux is not supported in jaxtronomy")
        # generate image
        im_sim = ImageModel.image(
            self,
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
        )
        model_error = self._error_map_model(
            kwargs_lens, kwargs_ps=kwargs_ps, kwargs_special=kwargs_special
        )
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)
        return logL

    @partial(jit, static_argnums=(0, 5, 6, 7, 8))
    def source_surface_brightness(
        self,
        kwargs_source,
        kwargs_lens=None,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
        de_lensed=False,
        k=None,
        update_pixelbased_mapping=True,
    ):
        """Computes the source surface brightness distribution.

        :param kwargs_source: list of dicts, keyword arguments corresponding to the
            superposition of different source light profiles in the same order of
            light_model_list
        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_extinction: list of dicts, keyword arguments corresponding to
            different light profiles in the optical_depth_model
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param unconvolved: if True: returns the unconvolved light distribution (prefect
            seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness
            profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :return: 2d array of surface brightness pixels
        """
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros(self.Data.num_pixel_axes)

        return self._source_surface_brightness_analytical(
            kwargs_source,
            kwargs_lens=kwargs_lens,
            kwargs_extinction=kwargs_extinction,
            kwargs_special=kwargs_special,
            unconvolved=unconvolved,
            de_lensed=de_lensed,
            k=k,
        )

    @partial(jit, static_argnums=(0, 5, 6, 7))
    def _source_surface_brightness_analytical(
        self,
        kwargs_source,
        kwargs_lens=None,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
        de_lensed=False,
        k=None,
    ):
        """Computes the source surface brightness distribution.

        :param kwargs_source: list of dicts, keyword arguments corresponding to the
            superposition of different source light profiles in the same order of
            light_model_list
        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_extinction: list of dicts, keyword arguments corresponding to
            different light profiles in the optical_depth_model
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param unconvolved: if True: returns the unconvolved light distribution (prefect
            seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness
            profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :return: 2d array of surface brightness pixels
        """
        source_light = self._source_surface_brightness_analytical_numerics(
            kwargs_source,
            kwargs_lens,
            kwargs_extinction,
            kwargs_special=kwargs_special,
            de_lensed=de_lensed,
            k=k,
        )

        source_light_final = self.ImageNumerics.re_size_convolve(
            source_light, unconvolved=unconvolved
        )
        return source_light_final

    @partial(jit, static_argnums=(0, 5, 6))
    def _source_surface_brightness_analytical_numerics(
        self,
        kwargs_source,
        kwargs_lens=None,
        kwargs_extinction=None,
        kwargs_special=None,
        de_lensed=False,
        k=None,
    ):
        """Computes the source surface brightness distribution.

        :param kwargs_source: list of dicts, keyword arguments corresponding to the
            superposition of different source light profiles in the same order of
            light_model_list
        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_extinction: list of dicts, keyword arguments corresponding to
            different light profiles in the optical_depth_model
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param de_lensed: if True: returns the un-lensed source surface brightness
            profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :return: 2d array of surface brightness pixels
        """
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(
                ra_grid, dec_grid, kwargs_source, k=k
            )
        else:
            source_light = self.source_mapping.image_flux_joint(
                ra_grid,
                dec_grid,
                kwargs_lens,
                kwargs_source,
                kwargs_special=kwargs_special,
                k=k,
            )
            # dicts and lists evaluate to true if not empty
            if kwargs_extinction is not None and kwargs_extinction:
                raise ValueError("Extinction is not implemented in JAXtronomy yet")
            # source_light *= self._extinction.extinction(
            #    ra_grid,
            #    dec_grid,
            #    kwargs_extinction=kwargs_extinction,
            #    kwargs_special=kwargs_special,
            # )

        # multiply with primary beam before convolution (not supported yet in jaxtronomy)
        # if self._pb is not None:
        #    source_light *= self._pb_1d
        return source_light * self._flux_scaling

    @partial(jit, static_argnums=(0, 2, 3))
    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """Computes the lens surface brightness distribution.

        :param kwargs_lens_light: list of keyword arguments corresponding to different
            lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect
            seeing), otherwise convolved with PSF kernel
        :return: 2d array of surface brightness pixels
        """

        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(
            ra_grid, dec_grid, kwargs_lens_light, k=k
        )

        # multiply with primary beam before convolution (not supported yet in jaxtronomy)
        # if self._pb is not None:
        #    lens_light *= self._pb_1d

        lens_light_final = self.ImageNumerics.re_size_convolve(
            lens_light, unconvolved=unconvolved
        )
        return lens_light_final * self._flux_scaling

    @partial(jit, static_argnums=(0, 4, 5))
    def point_source(
        self,
        kwargs_ps,
        kwargs_lens=None,
        kwargs_special=None,
        unconvolved=False,
        k=None,
    ):
        """Computes the point source positions and paints PSF convolutions on them.

        :param kwargs_ps: list of dicts, keyword arguments for each point source model
            in the same order of the point_source_type_list
        :param kwargs_lens: list of dicts, keyword arguments for the full set of lens
            models in the same order of the lens_model_list
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param unconvolved: bool, includes point source images if False, excludes ps if
            True
        :param k: optional int, include only the k-th point source model. If None,
            includes all
        :return: rendered point source images
        """
        point_source_image = jnp.zeros((self.Data.num_pixel_axes))
        if unconvolved or self.PointSource is None:
            return point_source_image
        ra_pos, dec_pos, amp = self.PointSource.point_source_list(
            kwargs_ps, kwargs_lens=kwargs_lens, k=k
        )
        ra_pos, dec_pos = self._displace_astrometry(
            ra_pos, dec_pos, kwargs_special=kwargs_special
        )
        point_source_image += self.ImageNumerics.point_source_rendering(
            ra_pos, dec_pos, amp
        )
        return point_source_image * self._flux_scaling

    @partial(jit, static_argnums=(0, 7, 8, 9, 10))
    def image(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=True,
    ):
        """Make an image with a realisation of linear parameter values "param".

        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_source: list of dicts, keyword arguments corresponding to the
            superposition of different source light profiles in the same order of
            light_model_list
        :param kwargs_lens_light: list of dicts, keyword arguments corresponding to
            different lens light surface brightness profiles in the same order of
            lens_light_model_list
        :param kwargs_ps: list of dicts, keyword arguments for the points source models
            in the same order of point_source_type_list
        :param kwargs_extinction: list of dicts, keyword arguments corresponding to
            different light profiles in the optical_depth_model
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :param unconvolved: if True: returns the unconvolved light distribution (prefect
            seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 2d array of surface brightness pixels of the simulation
        """
        model = jnp.zeros(self.Data.num_pixel_axes)
        if source_add is True:
            model += ImageModel.source_surface_brightness(
                self,
                kwargs_source,
                kwargs_lens,
                kwargs_extinction=kwargs_extinction,
                kwargs_special=kwargs_special,
                unconvolved=unconvolved,
            )
        if lens_light_add is True:
            model += ImageModel.lens_surface_brightness(
                self, kwargs_lens_light, unconvolved=unconvolved
            )
        if point_source_add is True:
            model += ImageModel.point_source(
                self,
                kwargs_ps,
                kwargs_lens,
                kwargs_special=kwargs_special,
                unconvolved=unconvolved,
            )
        return model

    # def extinction_map(self, kwargs_extinction=None, kwargs_special=None):
    #     """Differential extinction per pixel.

    #     :param kwargs_extinction: list of keyword arguments corresponding to the optical
    #         depth models tau, such that extinction is exp(-tau)
    #     :param kwargs_special: keyword arguments, additional parameter to the extinction
    #     :return: 2d array of size of the image
    #     """
    #     ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
    #     extinction = self._extinction.extinction(
    #         ra_grid,
    #         dec_grid,
    #         kwargs_extinction=kwargs_extinction,
    #         kwargs_special=kwargs_special,
    #     )
    #     print(extinction, "test extinction")
    #     extinction_array = np.ones_like(ra_grid) * extinction
    #     extinction = (
    #         self.ImageNumerics.re_size_convolve(extinction_array, unconvolved=True)
    #         / self.ImageNumerics.grid_class.pixel_width**2
    #     )
    #     return extinction

    @partial(jit, static_argnums=0)
    def reduced_residuals(self, model, error_map=0):
        """

        :param model: 2d numpy array of the modeled image
        :param error_map: 2d numpy array of additional noise/error terms from model components (such as PSF model uncertainties)
        :return: 2d numpy array of reduced residuals per pixel
        """
        mask = self.likelihood_mask
        C_D = self.Data.C_D_model(model)
        residual = (model - self.Data.data) / jnp.sqrt(C_D + jnp.abs(error_map)) * mask
        return residual

    @partial(jit, static_argnums=0)
    def reduced_chi2(self, model, error_map=0):
        """Returns reduced chi2.

        :param model: 2d numpy array of a model predicted image
        :param error_map: same format as model, additional error component (such as PSF
            errors)
        :return: reduced chi2.
        """
        norm_res = self.reduced_residuals(model, error_map)
        return jnp.sum(norm_res**2) / self.num_data_evaluate

    @partial(jit, static_argnums=0)
    def image2array_masked(self, image):
        """Returns 1d array of values in image that are not masked out for the
        likelihood computation/linear minimization.

        :param image: 2d numpy array of full image
        :return: 1d array.
        """
        array = util.image2array(image)
        return array[self._mask1d]

    @partial(jit, static_argnums=0)
    def array_masked2image(self, array):
        """Converts the 1d masked array into a 2d image.

        :param array: 1d array of values not masked out
        :return: 2d array of full image
        """
        nx, ny = self.Data.num_pixel_axes
        grid1d = jnp.zeros(nx * ny)
        grid1d = grid1d.at[self._mask1d].set(array)
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    @property
    def data_response(self):
        """Returns the 1d array of the data element that is fitted for (including
        masking)

        :return: 1d numpy array.
        """
        d = self.image2array_masked(self.Data.data)
        return d

    @partial(jit, static_argnums=0)
    def error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """Returns the 1d array of the error estimate corresponding to the data
        response.

        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_ps: list of dicts, keyword arguments for the points source models
            in the same order of point_source_type_list
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :return: 1d numpy array of response, 2d array of additional errors (e.g. point
            source uncertainties)
        """
        model_error = self._error_map_model(kwargs_lens, kwargs_ps, kwargs_special)
        # adding the uncertainties estimated from the data with the ones from the model
        C_D_response = self.image2array_masked(self.Data.C_D + model_error)
        return C_D_response, model_error

    @partial(jit, static_argnums=0)
    def _error_map_model(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Noise estimate (variances as diagonal of the pixel covariance matrix)
        resulted from inherent model uncertainties. This term is currently the psf error
        map.

        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_ps: list of dicts, keyword arguments for the points source models
            in the same order of point_source_type_list
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :return: 2d array corresponding to the pixels in terms of variance in noise
        """
        return self._error_map_psf(kwargs_lens, kwargs_ps, kwargs_special)

    @partial(jit, static_argnums=0)
    def _error_map_psf(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Map of image with error terms (sigma**2) expected from inaccuracies in the
        PSF modeling.

        :param kwargs_lens: list of dicts, keyword arguments corresponding to the
            superposition of different lens profiles in the same order of the
            lens_model_list
        :param kwargs_ps: list of dicts, keyword arguments for the points source models
            in the same order of point_source_type_list
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :return: 2d array of size of the image
        """
        error_map = jnp.zeros(self.Data.num_pixel_axes)
        if self._psf_error_map is True:
            for k, bool_ in enumerate(self._psf_error_map_bool_list):
                if bool_ is True:
                    ra_pos, dec_pos, _ = self.PointSource.point_source_list(
                        kwargs_ps, kwargs_lens=kwargs_lens, k=k, with_amp=False
                    )
                    if len(ra_pos) > 0:
                        ra_pos, dec_pos = self._displace_astrometry(
                            ra_pos, dec_pos, kwargs_special=kwargs_special
                        )
                        error_map += self.ImageNumerics.psf_variance_map(
                            ra_pos,
                            dec_pos,
                            None,
                            self.Data.data,
                            fix_psf_variance_map=False,
                        )
        return error_map

    @staticmethod
    @jit
    def _displace_astrometry(x_pos, y_pos, kwargs_special=None):
        """Displaces point sources by shifts specified in kwargs_special.

        :param x_pos: list of point source positions according to point source model
            list
        :param y_pos: list of point source positions according to point source model
            list
        :param kwargs_special: optional dict including keys "delta_x_image" and
            "delta_y_image" and array/list values indicating how much to shift each
            point source image in units of arcseconds
        :return: shifted image positions in same format as input
        """
        if kwargs_special is not None:
            if "delta_x_image" in kwargs_special:
                delta_x, delta_y = (
                    kwargs_special["delta_x_image"],
                    kwargs_special["delta_y_image"],
                )
                delta_x_new = jnp.zeros(len(x_pos))
                delta_x_new = delta_x_new.at[0 : len(delta_x)].set(delta_x)
                delta_y_new = jnp.zeros(len(y_pos))
                delta_y_new = delta_y_new.at[0 : len(delta_y)].set(delta_y)
                x_pos = x_pos + delta_x_new
                y_pos = y_pos + delta_y_new
        return x_pos, y_pos

    def update_psf(self, psf_class):
        """Update the psf class.

        Not supported in jaxtronomy.
        """
        raise ValueError(
            "Updating psf class not supported in jaxtronomy. Create a new instance of ImageModel instead."
        )

    def update_data(self, data_class):
        """Update the data class.

        Not supported in jaxtronomy.
        """
        raise ValueError(
            "Updating data class not supported in jaxtronomy. Create a new instance of ImageModel instead."
        )
