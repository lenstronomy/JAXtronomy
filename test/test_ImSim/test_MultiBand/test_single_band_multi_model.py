__author__ = "sibirrer"

import copy
import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.Util import simulation_util as sim_util, param_util
from lenstronomy.ImSim.MultiBand.single_band_multi_model import (
    SingleBandMultiModel as SingleBandMultiModel_ref,
)

from jaxtronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel


class TestSingleBandMultiModel(object):
    """Tests the SingleBandMultiModel routines."""

    def setup_method(self):
        self._setup_method(linear_solver=False)

    def _setup_method(self, linear_solver):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        self.numPix = numPix = 100  # cutout pixel size
        self.numPix2 = numPix2 = 120
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )
        kwargs_data["image_data"] = np.ones((numPix, numPix)) * 30.5

        kwargs_data2 = sim_util.data_configure_simple(
            numPix2, deltaPix, exp_time + 30, sigma_bkg + 0.01, inverse=True
        )
        kwargs_data2["image_data"] = np.ones((numPix2, numPix2)) * 50.1

        # Create likelihood masks
        likelihood_mask = np.ones((numPix, numPix))
        likelihood_mask[50][::2] -= likelihood_mask[50][::2]
        likelihood_mask[25][::3] -= likelihood_mask[25][::3]

        likelihood_mask2 = np.ones((numPix2, numPix2))
        likelihood_mask2[60][::2] -= likelihood_mask2[60][::2]
        likelihood_mask2[30][::3] -= likelihood_mask2[30][::3]
        likelihood_mask_list = [likelihood_mask, likelihood_mask2]

        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        kernel = np.zeros((17, 17))
        kernel[5:-5, 5:-5] = 1
        kernel[7:-7, 7:-7] = 3
        kernel[9, 9] = 7
        kwargs_psf2 = {
            "psf_type": "PIXEL",
            "kernel_point_source": kernel,
            "psf_variance_map": np.ones_like(kernel) * kernel**2,
        }

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.0,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_epl = {
            "theta_E": 3.0,
            "gamma": 1.7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_epl2 = {
            "theta_E": 2.3,
            "gamma": 1.9,
            "center_x": 0.1,
            "center_y": -0.3,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SIE", "EPL", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_epl, kwargs_shear]
        self.kwargs_lens2 = [kwargs_spemd, kwargs_epl2, kwargs_shear]

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 31.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 73.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]

        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]

        point_source_type_list = ["UNLENSED", "LENSED_POSITION", "LENSED_POSITION"]
        fixed_magnification_list = [False, True, False]
        kwargs_unlensed = {
            "ra_image": [5.342],
            "dec_image": [2.8743],
            "point_amp": [18.23],
        }
        kwargs_lensed_fixed_mag = {
            "ra_image": [2.342],
            "dec_image": [-3.8743],
            "source_amp": 13.23,
        }
        kwargs_lensed = {
            "ra_image": [1.342, -3.23498],
            "dec_image": [-5.8743, 4.2384],
            "point_amp": [19.23, 28.543],
        }
        self.kwargs_ps = [kwargs_unlensed, kwargs_lensed_fixed_mag, kwargs_lensed]
        self.kwargs_special = {
            "delta_x_image": [-0.334],
            "delta_y_image": [3.3287],
        }

        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
        }
        kwargs_numerics2 = {
            "supersampling_factor": 3,
            "supersampling_convolution": False,
        }

        multi_band_list = [
            [kwargs_data, kwargs_psf, kwargs_numerics],
            [kwargs_data2, kwargs_psf2, kwargs_numerics2],
        ]
        self.multi_band_list = multi_band_list

        # Band 0: SIE + SHEAR, SERSIC, SERSIC_ELLIPSE, UNLENSED + LENSED_POSITION 1
        # Band 1: EPL + SHEAR, SERSIC, SERSIC_ELLIPSE, UNLENSED + LENSED_POSITION 2
        kwargs_model = {
            "lens_model_list": lens_model_list,  # ["SIE", "EPL", "SHEAR"]
            "source_light_model_list": source_model_list,  # ["SERSIC_ELLIPSE"]
            "lens_light_model_list": lens_light_model_list,  # ["SERSIC"]
            "point_source_model_list": point_source_type_list,  # ["UNLENSED", "LENSED_POSITION", "LENSED_POSITION"]
            "fixed_magnification_list": fixed_magnification_list,
            "index_lens_model_list": [[0, 2], [1, 2]],
            "index_source_light_model_list": [[0], [0]],
            "index_lens_light_model_list": [[0], [0]],
            "index_point_source_model_list": [[0, 1], [0, 2]],
        }
        self.kwargs_model = kwargs_model
        self.singleband0 = SingleBandMultiModel(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=0,
            kwargs_pixelbased=None,
            linear_solver=linear_solver,
        )
        self.singleband1 = SingleBandMultiModel(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=1,
            kwargs_pixelbased=None,
            linear_solver=linear_solver,
        )
        self.singleband0_ref = SingleBandMultiModel_ref(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=0,
            kwargs_pixelbased=None,
            linear_solver=linear_solver,
        )
        self.singleband1_ref = SingleBandMultiModel_ref(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=1,
            kwargs_pixelbased=None,
            linear_solver=linear_solver,
        )

    def test_image(self):
        image0 = self.singleband0.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        image0_ref = self.singleband0_ref.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_allclose(image0, image0_ref, atol=1e-10, rtol=1e-10)
        assert image0.shape == (self.numPix, self.numPix)

        image1 = self.singleband1.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        image1_ref = self.singleband1_ref.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_allclose(image1, image1_ref, atol=1e-10, rtol=1e-10)
        assert image1.shape == (self.numPix2, self.numPix2)

        # Use kwargs_lens2 and make sure we get a different result
        image1 = self.singleband1.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_raises(
            AssertionError,
            npt.assert_allclose,
            image1,
            image1_ref,
            atol=1e-10,
            rtol=1e-10,
        )
        image1_ref = self.singleband1_ref.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_allclose(image1, image1_ref, atol=1e-10, rtol=1e-10)

    def test_source_surface_brightness(self):
        flux0 = self.singleband0.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        flux0_ref = self.singleband0_ref.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        npt.assert_allclose(flux0, flux0_ref, atol=1e-10, rtol=1e-10)

        flux1 = self.singleband1.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        flux1_ref = self.singleband1_ref.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        npt.assert_allclose(flux1, flux1_ref, atol=1e-10, rtol=1e-10)

    def test_lens_surface_brightness(self):
        flux0 = self.singleband0.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        flux0_ref = self.singleband0_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_allclose(flux0, flux0_ref, atol=1e-10, rtol=1e-10)

        flux1 = self.singleband1.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        flux1_ref = self.singleband1_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_allclose(flux1, flux1_ref, atol=1e-10, rtol=1e-10)

    def test_point_source(self):
        flux = self.singleband0.point_source(
            self.kwargs_ps,
            self.kwargs_lens,
            self.kwargs_special,
        )
        flux_ref = self.singleband0_ref.point_source(
            self.kwargs_ps,
            self.kwargs_lens,
            self.kwargs_special,
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

        flux = self.singleband0.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, k=1
        )
        flux_ref = self.singleband0_ref.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, k=1
        )
        npt.assert_allclose(flux, flux_ref, atol=5e-7, rtol=5e-7)

        flux = self.singleband0.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, unconvolved=True
        )
        flux_ref = self.singleband0_ref.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, unconvolved=True
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

        flux = self.singleband1.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special
        )
        flux_ref = self.singleband1_ref.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

    def test_likelihood_data_given_model(self):
        likelihood0, _ = self.singleband0.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        likelihood0_ref, _ = self.singleband0_ref.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_allclose(likelihood0, likelihood0_ref, atol=1e-10, rtol=1e-10)

        likelihood1, _ = self.singleband1.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        likelihood1_ref, _ = self.singleband1_ref.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_allclose(likelihood1, likelihood1_ref, atol=1e-10, rtol=1e-10)

    def test_error_response(self):
        c_d, error0 = self.singleband0.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        c_d_ref, error0_ref = self.singleband0_ref.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_allclose(c_d, c_d_ref, atol=1e-10, rtol=1e-10)
        npt.assert_allclose(error0, error0_ref, atol=1e-10, rtol=1e-10)

        c_d, error1 = self.singleband1.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        c_d_ref, error1_ref = self.singleband1_ref.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_allclose(c_d, c_d_ref, atol=1e-10, rtol=1e-10)
        npt.assert_allclose(error1, error1_ref, atol=1e-10, rtol=1e-10)

    def test_num_param_linear(self):
        num_param_linear = self.singleband0.num_param_linear(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        num_param_linear_ref = self.singleband0_ref.num_param_linear(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert num_param_linear == num_param_linear_ref
        assert num_param_linear == 0

    def test_error_map_source(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)
        error_map = self.singleband0.error_map_source(self.kwargs_source, x, y, cov_param=None)
        error_map_ref = self.singleband0_ref.error_map_source(self.kwargs_source, x, y, cov_param=None)
        npt.assert_array_equal(error_map, error_map_ref)
        npt.assert_array_equal(error_map, np.zeros_like(x))

    def test_linear_param_from_kwargs(self):
        param = self.singleband0.linear_param_from_kwargs(
            self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        param_ref = self.singleband0_ref.linear_param_from_kwargs(
            self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        npt.assert_array_equal(param, param_ref)
        npt.assert_array_equal(param, [])

    def test_select_kwargs(self):
        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.singleband0.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        (
            kwargs_lens_i_ref,
            kwargs_source_i_ref,
            kwargs_lens_light_i_ref,
            kwargs_ps_i_ref,
            kwargs_extinction_i_ref,
        ) = self.singleband0_ref.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        assert kwargs_lens_i == kwargs_lens_i_ref
        assert kwargs_source_i == kwargs_source_i_ref
        assert kwargs_lens_light_i == kwargs_lens_light_i_ref
        assert kwargs_ps_i == kwargs_ps_i_ref
        assert kwargs_extinction_i == kwargs_extinction_i_ref

        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.singleband1.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        (
            kwargs_lens_i_ref,
            kwargs_source_i_ref,
            kwargs_lens_light_i_ref,
            kwargs_ps_i_ref,
            kwargs_extinction_i_ref,
        ) = self.singleband1_ref.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        assert kwargs_lens_i == kwargs_lens_i_ref
        assert kwargs_source_i == kwargs_source_i_ref
        assert kwargs_lens_light_i == kwargs_lens_light_i_ref
        assert kwargs_ps_i == kwargs_ps_i_ref
        assert kwargs_extinction_i == kwargs_extinction_i_ref

        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.singleband0.select_kwargs()
        (
            kwargs_lens_i_ref,
            kwargs_source_i_ref,
            kwargs_lens_light_i_ref,
            kwargs_ps_i_ref,
            kwargs_extinction_i_ref,
        ) = self.singleband0_ref.select_kwargs()
        assert kwargs_lens_i == kwargs_lens_i_ref
        assert kwargs_source_i == kwargs_source_i_ref
        assert kwargs_lens_light_i == kwargs_lens_light_i_ref
        assert kwargs_ps_i == kwargs_ps_i_ref
        assert kwargs_extinction_i == kwargs_extinction_i_ref

        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.singleband0.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        (
            kwargs_lens_i_ref,
            kwargs_source_i_ref,
            kwargs_lens_light_i_ref,
            kwargs_ps_i_ref,
            kwargs_extinction_i_ref,
        ) = self.singleband1_ref.select_kwargs(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        assert kwargs_lens_i != kwargs_lens_i_ref


class TestSingleBandMultiModel_LinearSolver(object):

    # Same setup as before but with linear_solver=True
    def setup_method(self):
        self.test = TestSingleBandMultiModel()
        self.test._setup_method(linear_solver=True)

    def test_image_linear_solve(self):

        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        model, model_error, cov_param, param = self.test.singleband0.image_linear_solve(
            kwargs_lens=self.test.kwargs_lens,
            kwargs_source=self.test.kwargs_source,
            kwargs_lens_light=self.test.kwargs_lens_light,
            kwargs_ps=self.test.kwargs_ps,
            kwargs_special=self.test.kwargs_special,
            inv_bool=False,
        )
        model_ref, model_error_ref, cov_param_ref, param_ref = (
            self.test.singleband0_ref.image_linear_solve(
                kwargs_lens=self.test.kwargs_lens,
                kwargs_source=self.test.kwargs_source,
                kwargs_lens_light=self.test.kwargs_lens_light,
                kwargs_ps=self.test.kwargs_ps,
                kwargs_special=self.test.kwargs_special,
                inv_bool=False,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(model_error, model_error_ref, atol=1e-12, rtol=1e-12)
        assert cov_param is None and cov_param_ref is None
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        model, model_error, cov_param, param = self.test.singleband0.image_linear_solve(
            kwargs_lens=self.test.kwargs_lens,
            kwargs_source=self.test.kwargs_source,
            kwargs_lens_light=self.test.kwargs_lens_light,
            kwargs_ps=self.test.kwargs_ps,
            kwargs_special=self.test.kwargs_special,
            inv_bool=True,
        )
        model_ref, model_error_ref, cov_param_ref, param_ref = (
            self.test.singleband0_ref.image_linear_solve(
                kwargs_lens=self.test.kwargs_lens,
                kwargs_source=self.test.kwargs_source,
                kwargs_lens_light=self.test.kwargs_lens_light,
                kwargs_ps=self.test.kwargs_ps,
                kwargs_special=self.test.kwargs_special,
                inv_bool=True,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(model_error, model_error_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(cov_param, cov_param_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        error_map = self.test.singleband0.error_map_source(
            self.test.kwargs_source, x, y, cov_param
        )
        error_map_ref = self.test.singleband0_ref.error_map_source(
            self.test.kwargs_source, x, y, cov_param_ref
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-12, rtol=1e-12)

        error_map = self.test.singleband0.error_map_source(
            self.test.kwargs_source, x, y, cov_param, model_index_select=False
        )
        error_map_ref = self.test.singleband0_ref.error_map_source(
            self.test.kwargs_source, x, y, cov_param_ref, model_index_select=False
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-12, rtol=1e-12)

        model, model_error, cov_param, param = self.test.singleband1.image_linear_solve(
            kwargs_lens=self.test.kwargs_lens,
            kwargs_source=self.test.kwargs_source,
            kwargs_lens_light=self.test.kwargs_lens_light,
            kwargs_ps=self.test.kwargs_ps,
            kwargs_special=self.test.kwargs_special,
            inv_bool=False,
        )
        model_ref, model_error_ref, cov_param_ref, param_ref = (
            self.test.singleband1_ref.image_linear_solve(
                kwargs_lens=self.test.kwargs_lens,
                kwargs_source=self.test.kwargs_source,
                kwargs_lens_light=self.test.kwargs_lens_light,
                kwargs_ps=self.test.kwargs_ps,
                kwargs_special=self.test.kwargs_special,
                inv_bool=False,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(model_error, model_error_ref, atol=1e-12, rtol=1e-12)
        assert cov_param is None and cov_param_ref is None
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        model, model_error, cov_param, param = self.test.singleband1.image_linear_solve(
            kwargs_lens=self.test.kwargs_lens,
            kwargs_source=self.test.kwargs_source,
            kwargs_lens_light=self.test.kwargs_lens_light,
            kwargs_ps=self.test.kwargs_ps,
            kwargs_special=self.test.kwargs_special,
            inv_bool=True,
        )
        model_ref, model_error_ref, cov_param_ref, param_ref = (
            self.test.singleband1_ref.image_linear_solve(
                kwargs_lens=self.test.kwargs_lens,
                kwargs_source=self.test.kwargs_source,
                kwargs_lens_light=self.test.kwargs_lens_light,
                kwargs_ps=self.test.kwargs_ps,
                kwargs_special=self.test.kwargs_special,
                inv_bool=True,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(model_error, model_error_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(cov_param, cov_param_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        error_map = self.test.singleband1.error_map_source(
            self.test.kwargs_source, x, y, cov_param
        )
        error_map_ref = self.test.singleband1_ref.error_map_source(
            self.test.kwargs_source, x, y, cov_param_ref
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-12, rtol=1e-12)

        error_map = self.test.singleband1.error_map_source(
            self.test.kwargs_source, x, y, cov_param, model_index_select=False
        )
        error_map_ref = self.test.singleband1_ref.error_map_source(
            self.test.kwargs_source, x, y, cov_param_ref, model_index_select=False
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-12, rtol=1e-12)

    def test_likelihood_data_given_model(self):
        self.test.test_likelihood_data_given_model()

    def test_update_linear_kwargs(self):
        param = [1, 2, 3, 4]
        kwargs_source = copy.deepcopy(self.test.kwargs_source)
        kwargs_lens_light = copy.deepcopy(self.test.kwargs_lens_light)
        kwargs_ps = copy.deepcopy(self.test.kwargs_ps)
        _, kwargs_source, kwargs_lens_light, kwargs_ps = (
            self.test.singleband0.update_linear_kwargs(
                param,
                self.test.kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
            )
        )
        assert kwargs_source[0]["amp"] == 1
        assert kwargs_lens_light[0]["amp"] == 2
        assert kwargs_ps[0]["point_amp"] == 3
        assert kwargs_ps[1]["source_amp"] == 4

        param = [1, 2, 3, 4, 5]
        kwargs_source = copy.deepcopy(self.test.kwargs_source)
        kwargs_lens_light = copy.deepcopy(self.test.kwargs_lens_light)
        kwargs_ps = copy.deepcopy(self.test.kwargs_ps)
        _, kwargs_source, kwargs_lens_light, kwargs_ps = (
            self.test.singleband1.update_linear_kwargs(
                param,
                self.test.kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
            )
        )
        assert kwargs_source[0]["amp"] == 1
        assert kwargs_lens_light[0]["amp"] == 2
        assert kwargs_ps[0]["point_amp"] == 3
        npt.assert_array_equal(kwargs_ps[1]["point_amp"], [4, 5])

    def test_num_param_linear(self):
        num_param_linear = self.test.singleband0.num_param_linear(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
        )
        num_param_linear_ref = self.test.singleband0_ref.num_param_linear(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
        )
        assert num_param_linear == num_param_linear_ref
        assert num_param_linear == 4

        num_param_linear = self.test.singleband1.num_param_linear(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
        )
        num_param_linear_ref = self.test.singleband1_ref.num_param_linear(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
        )
        assert num_param_linear == num_param_linear_ref
        assert num_param_linear == 5

    def test_linear_response_matrix(self):
        A = self.test.singleband0.linear_response_matrix(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
            None,
            self.test.kwargs_special,
        )
        A_ref = self.test.singleband0_ref.linear_response_matrix(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
            None,
            self.test.kwargs_special,
        )
        npt.assert_allclose(A, A_ref, atol=3e-12, rtol=3e-12)

        A = self.test.singleband1.linear_response_matrix(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
            None,
            self.test.kwargs_special,
        )
        A_ref = self.test.singleband1_ref.linear_response_matrix(
            self.test.kwargs_lens,
            self.test.kwargs_source,
            self.test.kwargs_lens_light,
            self.test.kwargs_ps,
            None,
            self.test.kwargs_special,
        )
        npt.assert_allclose(A, A_ref, atol=3e-12, rtol=3e-12)

    def test_linear_param_from_kwargs(self):
        param = self.test.singleband0.linear_param_from_kwargs(
            self.test.kwargs_source, self.test.kwargs_lens_light, self.test.kwargs_ps
        )
        param_ref = self.test.singleband0_ref.linear_param_from_kwargs(
            self.test.kwargs_source, self.test.kwargs_lens_light, self.test.kwargs_ps
        )
        npt.assert_array_equal(param, param_ref)

        param = self.test.singleband1.linear_param_from_kwargs(
            self.test.kwargs_source, self.test.kwargs_lens_light, self.test.kwargs_ps
        )
        param_ref = self.test.singleband1_ref.linear_param_from_kwargs(
            self.test.kwargs_source, self.test.kwargs_lens_light, self.test.kwargs_ps
        )
        npt.assert_array_equal(param, param_ref)


if __name__ == "__main__":
    pytest.main()
