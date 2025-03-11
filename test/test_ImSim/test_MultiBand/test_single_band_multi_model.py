__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.Util import simulation_util as sim_util, param_util
from lenstronomy.ImSim.MultiBand.single_band_multi_model import (
    SingleBandMultiModel as SingleBandMultiModel_ref,
)
from lenstronomy.Data.imaging_data import ImageData as ImageData_ref
from lenstronomy.Data.psf import PSF

from jaxtronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from jaxtronomy.Data.imaging_data import ImageData


class TestSingleBandMultiModel(object):
    """Tests the SingleBandMultiModel routines."""

    def setup_method(self):
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

        kwargs_data2 = sim_util.data_configure_simple(
            numPix2, deltaPix, exp_time + 30, sigma_bkg + 0.01, inverse=True
        )

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
            "source_amp": [13.23],
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
            linear_solver=False,
        )
        self.singleband1 = SingleBandMultiModel(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=1,
            kwargs_pixelbased=None,
            linear_solver=False,
        )
        self.singleband0_ref = SingleBandMultiModel_ref(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=0,
            kwargs_pixelbased=None,
            linear_solver=False,
        )
        self.singleband1_ref = SingleBandMultiModel_ref(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list,
            band_index=1,
            kwargs_pixelbased=None,
            linear_solver=False,
        )

    def test_raises(self):
        # Linear solver not supported
        npt.assert_raises(
            ValueError,
            SingleBandMultiModel,
            self.multi_band_list,
            self.kwargs_model,
            linear_solver=True,
        )
        npt.assert_raises(
            ValueError,
            self.singleband0.likelihood_data_given_model,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            linear_solver=True,
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
        npt.assert_array_almost_equal(image0, image0_ref, decimal=8)
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
        npt.assert_array_almost_equal(image1, image1_ref, decimal=8)
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
            AssertionError, npt.assert_array_almost_equal, image1, image1_ref, decimal=8
        )
        image1_ref = self.singleband1_ref.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_array_almost_equal(image1, image1_ref, decimal=8)

    def test_source_surface_brightness(self):
        flux0 = self.singleband0.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        flux0_ref = self.singleband0_ref.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        npt.assert_array_almost_equal(flux0, flux0_ref, decimal=8)

        flux1 = self.singleband1.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        flux1_ref = self.singleband1_ref.source_surface_brightness(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
        )
        npt.assert_array_almost_equal(flux1, flux1_ref, decimal=8)

    def test_lens_surface_brightness(self):
        flux0 = self.singleband0.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        flux0_ref = self.singleband0_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_array_almost_equal(flux0, flux0_ref, decimal=8)

        flux1 = self.singleband1.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        flux1_ref = self.singleband1_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_array_almost_equal(flux1, flux1_ref, decimal=8)

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
            linear_solver=False,
        )
        npt.assert_array_almost_equal(likelihood0, likelihood0_ref, decimal=8)

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
            linear_solver=False,
        )
        npt.assert_array_almost_equal(likelihood1, likelihood1_ref, decimal=8)

    def test_error_response(self):
        c_d, error0 = self.singleband0.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        c_d_ref, error0_ref = self.singleband0_ref.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=8)
        npt.assert_array_almost_equal(error0, error0_ref, decimal=8)

        c_d, error1 = self.singleband1.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        c_d_ref, error1_ref = self.singleband1_ref.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=8)
        npt.assert_array_almost_equal(error1, error1_ref, decimal=8)

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


if __name__ == "__main__":
    pytest.main()
