from functools import partial
from jax import config, debug, jit, lax, numpy as jnp
import warnings

# from lenstronomy.Util.cosmo_util import get_astropy_cosmology

config.update("jax_enable_x64", True)

__all__ = ["PositionLikelihood"]


class PositionLikelihood(object):
    """Likelihood of positions of multiply imaged point sources."""

    def __init__(
        self,
        point_source_class,
        image_position_uncertainty=0.005,
        astrometric_likelihood=False,
        image_position_likelihood=False,
        ra_image_list=None,
        dec_image_list=None,
        source_position_likelihood=False,
        source_position_tolerance=None,
        source_position_sigma=0.001,
        force_no_add_image=False,
        restrict_image_number=False,
        max_num_images=None,
    ):
        """

        :param point_source_class: Instance of PointSource() class
        :param image_position_uncertainty: uncertainty in image position uncertainty (1-sigma Gaussian radially),
            this is applicable for astrometric uncertainties as well as if image positions are provided as data
        :param astrometric_likelihood: bool, if True, evaluates the astrometric uncertainty of the predicted and modeled
            image positions with an offset 'delta_x_image' and 'delta_y_image'
        :param image_position_likelihood: bool, if True, evaluates the likelihood of the model predicted image position
            given the data/measured image positions
        :param ra_image_list: list of lists; RA image positions per model component
        :param dec_image_list: list of lists; DEC image positions per model component
        :param source_position_likelihood: bool, if True, ray-traces image positions back to source plane and evaluates
            relative errors in respect ot the position_uncertainties in the image plane (image_position_uncertainty)
        :param source_position_tolerance: tolerance level (in arc seconds in the source plane) of the different images.
            If set =! None, then the backwards ray tracing is performed on the images and demand on the same position of
            the source is meant to match the requirements, otherwise a punishing likelihood term is introduced
        :type source_position_tolerance: None or float
        :param source_position_sigma: r.m.s. value corresponding to a 1-sigma Gaussian likelihood accepted by the model
            precision in matching the source position transformed from the image plane
        :param force_no_add_image: bool, if True, will punish additional images appearing in the frame of the modelled
            image(first calculate them)
        :param restrict_image_number: bool, if True, searches for all appearing images in the frame of the data and
            compares with max_num_images
        :param max_num_images: integer, maximum number of appearing images. Default is the number of  images given in
            the Param() class
        """
        self._pointSource = point_source_class
        # TODO replace with public function of ray_shooting
        self._lensModel = point_source_class._lens_model

        # TODO: Implement restrict image number and force no add image
        if force_no_add_image:
            raise ValueError("force_no_add_image is not supported in jaxtronomy yet")
        self._force_no_add_image = force_no_add_image

        if restrict_image_number:
            raise ValueError("restrict_image_number is not supported in jaxtronomy yet")
        self._restrict_number_images = restrict_image_number
        self._max_num_images = max_num_images
        # if max_num_images is None and restrict_image_number is True:
        #     raise ValueError(
        #         "max_num_images needs to be provided when restrict_number_images is True!"
        #     )

        self._astrometric_likelihood = astrometric_likelihood

        self._source_position_likelihood = source_position_likelihood
        self._source_position_sigma = source_position_sigma
        self._bound_source_position_tolerance = source_position_tolerance
        if (
            source_position_tolerance is not None
            and source_position_likelihood is False
        ):
            warnings.warn(
                "source_position_tolerance has been set but source_position_likelihood is False. \n"
                "In order to use the source_position_tolerance, set source_position_likelihood to True"
            )

        self._image_position_likelihood = image_position_likelihood
        self._image_position_sigma = image_position_uncertainty

        self._ra_image_list, self._dec_image_list = [], []
        if ra_image_list is not None:
            for ra_image in ra_image_list:
                self._ra_image_list.append(jnp.array(ra_image, dtype=float))
        if dec_image_list is not None:
            for dec_image in dec_image_list:
                self._dec_image_list.append(jnp.array(dec_image, dtype=float))

    @partial(jit, static_argnums=(0, 4))
    def logL(self, kwargs_lens, kwargs_ps, kwargs_special, verbose=False):
        """

        :param kwargs_lens: lens model parameter keyword argument list
        :param kwargs_ps: point source model parameter keyword argument list
        :param kwargs_special: special keyword arguments
        :param verbose: bool
        :return: log likelihood of the optional likelihoods being computed
        """

        logL = 0

        # TODO: Cosmology sampling not in jaxtronomy yet
        # if self._lensModel.cosmology_sampling:
        #     cosmo = get_astropy_cosmology(
        #         cosmology_model=self._lensModel.cosmology_model,
        #         param_kwargs=kwargs_special,
        #     )
        #     self._lensModel.update_cosmology(cosmo)

        if self._astrometric_likelihood is True:
            logL_astrometry = self.astrometric_likelihood(
                kwargs_ps, kwargs_special, self._image_position_sigma
            )
            logL += logL_astrometry
            if verbose is True:
                debug.print("Astrometric likelihood = {}", logL_astrometry)

        # TODO: Implement force_no_add_image and restrict_number_images
        #       Though these won't actually be useful for gradient based sampling since
        #       the gradient of a step function is 0
        # if self._force_no_add_image:
        #     additional_image_bool = self.check_additional_images(kwargs_ps, kwargs_lens)
        #     if additional_image_bool is True:
        #         logL -= 10.0**5
        #         if verbose is True:
        #             print(
        #                 "force no additional image penalty as additional images are found!"
        #             )
        # if self._restrict_number_images is True:
        #     ra_image_list, dec_image_list = self._pointSource.image_position(
        #         kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens
        #     )
        #     if len(ra_image_list[0]) > self._max_num_images:
        #         logL -= 10.0**5
        #         if verbose is True:
        #             print(
        #                 "Number of images found %s exceeded the limited number allowed %s"
        #                 % (len(ra_image_list[0]), self._max_num_images)
        #             )
        if self._source_position_likelihood:
            logL_source_pos = self.source_position_likelihood(
                kwargs_lens,
                kwargs_ps,
                self._source_position_sigma,
                hard_bound_rms=self._bound_source_position_tolerance,
                verbose=verbose,
            )
            logL += logL_source_pos
            if verbose is True:
                debug.print("source position likelihood {}", logL_source_pos)
        if self._image_position_likelihood is True:
            logL_image_pos = self.image_position_likelihood(
                kwargs_ps=kwargs_ps,
                kwargs_lens=kwargs_lens,
                sigma=self._image_position_sigma,
            )
            logL += logL_image_pos
            if verbose is True:
                debug.print("image position likelihood {}", logL_image_pos)
        return logL

    # def check_additional_images(self, kwargs_ps, kwargs_lens):
    #     """Checks whether additional images have been found and placed in kwargs_ps.

    #     :param kwargs_ps: point source kwargs
    #     :param kwargs_lens: lens model keyword arguments
    #     :return: bool, True if more image positions are found than originally been
    #         assigned
    #     """
    #     ra_image_list, dec_image_list = self._pointSource.image_position(
    #         kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, additional_images=True
    #     )
    #     for i in range(len(ra_image_list)):
    #         if "ra_image" in kwargs_ps[i]:
    #             if len(ra_image_list[i]) > len(kwargs_ps[i]["ra_image"]):
    #                 return True
    #     return False

    @staticmethod
    @jit
    def astrometric_likelihood(kwargs_ps, kwargs_special, sigma):
        """Evaluates the astrometric uncertainty of the model plotted point sources
        (only available for 'LENSED_POSITION' point source model) and predicted image
        position by the lens model including an astrometric correction term.

        :param kwargs_ps: point source model kwargs list
        :param kwargs_special: kwargs list, should include the astrometric corrections
            'delta_x', 'delta_y'
        :param sigma: 1-sigma Gaussian uncertainty in the astrometry
        :return: log likelihood of the astrometirc correction between predicted image
            positions and model placement of the point sources
        """
        # TODO: make it compatible with multiple source instances
        if len(kwargs_ps) == 0:
            return 0
        if "ra_image" not in kwargs_ps[0]:
            return 0
        if "delta_x_image" in kwargs_special:
            delta_x, delta_y = jnp.array(
                kwargs_special["delta_x_image"], dtype=float
            ), jnp.array(kwargs_special["delta_y_image"], dtype=float)
            dist = (delta_x**2 + delta_y**2) / sigma**2 / 2
            logL = -jnp.sum(dist)
            logL = jnp.nan_to_num(logL, nan=-(10**15))
            return logL
        else:
            return 0

    @partial(jit, static_argnums=0)
    def image_position_likelihood(
        self,
        kwargs_ps,
        kwargs_lens,
        sigma,
    ):
        """Computes the likelihood of the model predicted image position relative to
        measured image positions with an astrometric error. This routine requires the
        'ra_image_list' and 'dec_image_list' being declared in the initiation of the
        class.

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param sigma: 1-sigma uncertainty in the measured position of the images
        :return: log likelihood of the model predicted image positions given the
            data/measured image positions.
        """

        ra_image_list, dec_image_list = self._pointSource.image_position(
            kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, original_position=True
        )
        logL = 0
        for i in range(
            len(ra_image_list)
        ):  # sum over the images of the different model components
            len_i = min(len(self._ra_image_list[i]), len(ra_image_list[i]))
            logL += -jnp.sum(
                (
                    (ra_image_list[i][:len_i] - self._ra_image_list[i][:len_i]) ** 2
                    + (dec_image_list[i][:len_i] - self._dec_image_list[i][:len_i]) ** 2
                )
                / sigma**2
                / 2
            )
        return logL

    @partial(jit, static_argnums=(0, 5))
    def source_position_likelihood(
        self,
        kwargs_lens,
        kwargs_ps,
        sigma,
        hard_bound_rms=None,
        verbose=False,
    ):
        """Computes a likelihood/punishing factor of how well the source positions of
        multiple images match given the image position and a lens model. The likelihood
        level is computed in respect of a displacement in the image plane and transposed
        through the Hessian into the source plane.

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source keyword argument list
        :param sigma: 1-sigma Gaussian uncertainty in the image plane
        :param hard_bound_rms: hard bound deviation between the mapping of the images
            back to the source plane (in source frame)
        :param verbose: bool, if True provides print statements with useful information.
        :return: log likelihood of the model reproducing the correct image positions
            given an image position uncertainty
        """
        if len(kwargs_ps) < 1:
            return 0
        if verbose:

            def true_fun(i, k, delta, hard_bound_rms):
                debug.print(
                    "Source positions of image {i} of model {k} do not match to the same source position to the required "
                    "precision. Achieved: {delta}, Required: {hard_bound_rms}.",
                    i=i,
                    k=k,
                    delta=delta,
                    hard_bound_rms=hard_bound_rms,
                )

            def false_fun(i, k, delta, hard_bound_rms):
                pass

        logL = 0
        source_x, source_y = self._pointSource.source_position(kwargs_ps, kwargs_lens)
        # redshift_list = self._pointSource._redshift_list

        for k in range(len(kwargs_ps)):
            if (
                "ra_image" in kwargs_ps[k]
                and self._pointSource.point_source_type_list[k] == "LENSED_POSITION"
            ):
                x_image = kwargs_ps[k]["ra_image"]
                y_image = kwargs_ps[k]["dec_image"]
                # self._lensModel.change_source_redshift(redshift_list[k])
                # calculating the individual source positions from the image positions
                k_list = self._pointSource.k_list(k)
                for i in range(len(x_image)):
                    if k_list is not None:
                        k_lens = tuple(k_list[i])
                    else:
                        k_lens = None
                    x_source_i, y_source_i = self._lensModel.ray_shooting(
                        x_image[i], y_image[i], kwargs_lens, k=k_lens
                    )
                    f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(
                        x_image[i], y_image[i], kwargs_lens, k=k_lens
                    )
                    A = jnp.array([[1 - f_xx, -f_xy], [-f_yx, 1 - f_yy]], dtype=float)
                    Sigma_theta = jnp.array([[1, 0], [0, 1]], dtype=float) * sigma**2
                    Sigma_beta = image2source_covariance(A, Sigma_theta)
                    delta = jnp.array(
                        [source_x[k] - x_source_i, source_y[k] - y_source_i],
                        dtype=float,
                    )
                    if hard_bound_rms is not None:
                        bound_hit = jnp.where(
                            delta[0] ** 2 + delta[1] ** 2 > hard_bound_rms**2,
                            True,
                            False,
                        )
                        logL = jnp.where(bound_hit, logL - 10**3, logL)
                        if verbose is True:
                            lax.cond(
                                bound_hit,
                                true_fun,
                                false_fun,
                                i,
                                k,
                                delta,
                                hard_bound_rms,
                            )
                    a, b, c, d = (
                        Sigma_beta[0][0],
                        Sigma_beta[0][1],
                        Sigma_beta[1][0],
                        Sigma_beta[1][1],
                    )
                    det = a * d - b * c
                    Sigma_inv = jnp.array([[d, -b], [-c, a]])
                    logL = jnp.where(
                        det == 0,
                        -(10**15),
                        logL - delta.T.dot(Sigma_inv.dot(delta)) / (2 * det),
                    )
        return logL

    @property
    def num_data(self):
        """

        :return: integer, number of data points associated with the class instance
        """
        num = 0
        if self._image_position_likelihood is True:
            for i in range(
                len(self._ra_image_list)
            ):  # sum over the images of the different model components
                num += len(self._ra_image_list[i]) * 2
        return num


# Equation (13) in Birrer & Treu 2019
@jit
def image2source_covariance(A, Sigma_theta):
    """
    computes error covariance in the source plane
    A: Hessian lensing matrix
    Sigma_theta: image plane covariance matrix of uncertainties
    """
    ATSigma = jnp.matmul(A.T, Sigma_theta)
    return jnp.matmul(ATSigma, A)
