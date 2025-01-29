from jaxtronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood

from lenstronomy.Sampling.Likelihoods.image_likelihood import (
    ImageLikelihood as ImageLikelihood_ref,
)
from lenstronomy.Util import simulation_util as sim_util, param_util
import numpy as np
import numpy.testing as npt
import pytest


class TestImageLikelihood(object):
    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        numPix2 = 120
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )

        kwargs_data2 = sim_util.data_configure_simple(
            numPix2, deltaPix, exp_time + 30, sigma_bkg + 0.01, inverse=True
        )
        kwargs_data["image_data"] = np.ones((numPix, numPix)) * 0.9392
        kwargs_data2["image_data"] = np.ones((numPix, numPix)) * 0.3262

        # Create likelihood masks
        # Likelihood mask 1 will mask out 70 pixels
        likelihood_mask = np.ones((numPix, numPix))
        likelihood_mask[50][::2] -= likelihood_mask[50][::2]
        likelihood_mask[25][::5] -= likelihood_mask[25][::5]

        # Likelihood mask 1 will mask out 83 pixels
        likelihood_mask2 = np.ones((numPix2, numPix2))
        likelihood_mask2[60][::3] -= likelihood_mask2[60][::3]
        likelihood_mask2[30][::2] -= likelihood_mask2[30][::2]
        likelihood_mask_list = [likelihood_mask, likelihood_mask2]

        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        kwargs_psf2 = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 4,
            "pixel_size": deltaPix,
        }

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.03,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.234,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_epl = {
            "theta_E": 3.0,
            "gamma": 1.7,
            "center_x": -0.5,
            "center_y": 1.2,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SIE", "EPL", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_epl, kwargs_shear]

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 31.354,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0.3,
            "center_y": 0.4,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 73.0,
            "R_sersic": 0.7,
            "n_sersic": 7,
            "center_x": 0.1,
            "center_y": -0.3,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]

        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]

        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
        }
        kwargs_numerics2 = {
            "supersampling_factor": 3,
            "supersampling_convolution": False,
        }

        self.multi_band_list = [
            [kwargs_data, kwargs_psf, kwargs_numerics],
            [kwargs_data2, kwargs_psf2, kwargs_numerics2],
        ]

        # band 0 involves the SIE + SHEAR lens models, a SERSIC lens light model, and a SERSIC_ELLIPSE source model
        # band 1 involves the EPL + SHEAR lens models, the same SERSIC lens light model, and the same SERSIC_ELLIPSE source model
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "index_lens_model_list": [[0, 2], [1, 2]],
            # The next line is not needed if the models are the same for both bands
            "index_source_light_model_list": [[0], [0]],
        }

        self.image_likelihood = ImageLikelihood(
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            image_likelihood_mask_list=likelihood_mask_list,
        )
        self.image_likelihood_ref = ImageLikelihood_ref(
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            image_likelihood_mask_list=likelihood_mask_list,
            linear_solver=False,
        )

    def test_raises(self):
        npt.assert_raises(
            ValueError,
            ImageLikelihood,
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            source_marg=True,
        )
        npt.assert_raises(
            ValueError,
            ImageLikelihood,
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            linear_solver=True,
        )
        npt.assert_raises(
            ValueError,
            ImageLikelihood,
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            check_positive_flux=True,
        )
        npt.assert_raises(
            ValueError,
            ImageLikelihood,
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            kwargs_pixelbased={"error": 1},
        )
        npt.assert_raises(
            ValueError,
            ImageLikelihood,
            self.multi_band_list,
            "single-band",
            self.kwargs_model,
            linear_prior=1,
        )

    def test_logL(self):
        logL, _ = self.image_likelihood.logL(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        logL_ref, _ = self.image_likelihood_ref.logL(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

        self.kwargs_lens[2]["gamma1"] = 0.05

        logL, _ = self.image_likelihood.logL(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        logL_ref, _ = self.image_likelihood_ref.logL(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        npt.assert_almost_equal(logL, logL_ref, decimal=8)

    def test_num_data(self):
        assert self.image_likelihood.num_data == self.image_likelihood_ref.num_data
        assert self.image_likelihood.num_data == 100 * 100 - 70


if __name__ == "__main__":
    pytest.main()
