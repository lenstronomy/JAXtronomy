__author__ = "sibirrer"

from jax import numpy as jnp, grad
import pytest
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from jaxtronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.likelihood import LikelihoodModule as LikelihoodModule_ref


class TestLikelihoodModule(object):
    """Test the fitting sequences."""

    def setup_method(self):
        np.random.seed(42)

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 50  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_model = {
            "lens_model_list": ["EPL"],
            "lens_light_model_list": ["SERSIC"],
            "source_light_model_list": ["SERSIC"],
        }

        # PSF specification
        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": deltaPix}
        psf_class = PSF(**kwargs_psf)
        print(np.shape(psf_class.kernel_point_source), "test kernel shape -")
        kwargs_spep = {
            "theta_E": 1.0,
            "gamma": 1.95,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        self.kwargs_lens = [kwargs_spep]
        kwargs_sersic = {
            "amp": 21.3,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {
            "amp": 11.7,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0,
            "center_y": 0,
        }

        self.kwargs_lens_light = [kwargs_sersic]
        self.kwargs_source = [kwargs_sersic_ellipse]

        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
        }
        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**kwargs_model)
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
        )

        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim

        self.data_class = data_class
        self.psf_class = psf_class
        self.kwargs_model = kwargs_model
        self.kwargs_numerics = kwargs_numerics

        def condition_definition(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps=None,
            kwargs_special=None,
            kwargs_extinction=None,
            **kwargs,
        ):
            logL = jnp.where(
                kwargs_lens_light[0]["R_sersic"] > kwargs_source[0]["R_sersic"],
                -(10**15),
                0,
            )
            return logL

        kwargs_likelihood = {
            "prior_lens": [[0, "theta_E", 1, 0.1]],
            "custom_logL_addition": condition_definition,
        }
        self.kwargs_data_joint = {
            "multi_band_list": [[kwargs_data, kwargs_psf, kwargs_numerics]],
            "multi_band_type": "single-band",
        }

        self.param_class = Param(self.kwargs_model, linear_solver=False)
        self.imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.Likelihood = LikelihoodModule(
            kwargs_data_joint=self.kwargs_data_joint,
            kwargs_model=kwargs_model,
            param_class=self.param_class,
            check_bounds=True,
            **kwargs_likelihood,
        )
        self.Likelihood_ref = LikelihoodModule_ref(
            kwargs_data_joint=self.kwargs_data_joint,
            kwargs_model=kwargs_model,
            param_class=self.param_class,
            **kwargs_likelihood,
        )
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.numPix = numPix

    def test_raises(self):
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            time_delay_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            tracer_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            image_position_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            source_position_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            flux_ratio_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            kinematic_2d_likelihood=True,
        )
        npt.assert_raises(
            ValueError,
            LikelihoodModule,
            self.kwargs_data_joint,
            self.kwargs_model,
            self.param_class,
            astrometric_likelihood=True,
        )

    def test_logL(self):
        args = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )

        logL = self.Likelihood(args)
        logL_ref = self.Likelihood_ref(args)
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

        logL = self.Likelihood.logL(args, verbose=True)
        logL_ref = self.Likelihood_ref.logL(args, verbose=True)
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

        args[3] += 0.1

        logL = self.Likelihood.likelihood(args)
        npt.assert_raises(AssertionError, npt.assert_almost_equal, logL, logL_ref)
        logL_ref = self.Likelihood_ref.likelihood(args)
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

        logL = self.Likelihood.negativelogL(args)
        logL_ref = self.Likelihood_ref.negativelogL(args)
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

        num_data = self.Likelihood.num_data
        num_data_ref = self.Likelihood_ref.num_data
        assert num_data == num_data_ref

        num_data_effective = self.Likelihood.effective_num_data_points()
        num_data_effective_ref = self.Likelihood_ref.effective_num_data_points()
        assert num_data_effective == num_data_effective_ref

    def test_grad_logL(self):
        args = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        grad_func = grad(self.Likelihood.logL)
        assert len(args) == len(grad_func(args))


    def test_check_bounds(self):
        lower_limit, upper_limit = self.Likelihood.param_limits
        lower_limit_ref, upper_limit_ref = self.Likelihood_ref.param_limits
        npt.assert_array_equal(lower_limit, lower_limit_ref)
        npt.assert_array_equal(upper_limit, upper_limit_ref)

        penalty, bound_hit = self.Likelihood.check_bounds(
            args=[0, 1], lowerLimit=[1, 0], upperLimit=[2, 2], verbose=True
        )
        assert bound_hit

        penalty, bound_hit = self.Likelihood.check_bounds(
            args=[1, 3], lowerLimit=[1, 0], upperLimit=[2, 2], verbose=True
        )
        assert bound_hit

        penalty, bound_hit = self.Likelihood.check_bounds(
            args=[1, 2], lowerLimit=[1, 0], upperLimit=[2, 2], verbose=True
        )
        assert not bound_hit

    def test_kwargs_imaging(self):
        kwargs_imaging = self.Likelihood.kwargs_imaging
        kwargs_imaging_ref = self.Likelihood_ref.kwargs_imaging
        assert kwargs_imaging == kwargs_imaging_ref

    def test_no_multiband(self):
        args = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )

        self.kwargs_data_joint["multi_band_list"] = None
        Likelihood = LikelihoodModule(
            kwargs_data_joint=self.kwargs_data_joint,
            kwargs_model=self.kwargs_model,
            param_class=self.param_class,
            check_bounds=True,
        )
        Likelihood_ref = LikelihoodModule_ref(
            kwargs_data_joint=self.kwargs_data_joint,
            kwargs_model=self.kwargs_model,
            param_class=self.param_class,
            check_bounds=True,
        )

        assert Likelihood.logL(args) == 0

        args[0] = 1000000
        assert Likelihood.logL(args) == -1e18
        assert Likelihood.logL(args) == Likelihood_ref.logL(args)


if __name__ == "__main__":
    pytest.main()
