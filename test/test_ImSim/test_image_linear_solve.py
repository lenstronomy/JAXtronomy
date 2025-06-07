__author__ = "sibirrer"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import numpy.testing as npt

from jaxtronomy.ImSim.image_linear_solve import ImageLinearFit
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.PointSource.point_source import PointSource
from jaxtronomy.Data.imaging_data import ImageData

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit as ImageLinearFit_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from lenstronomy.LightModel.light_model import LightModel as LightModel_ref
from lenstronomy.PointSource.point_source import PointSource as PointSource_ref
from lenstronomy.Data.imaging_data import ImageData as ImageData_ref

import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.psf import PSF


class TestImageLinearFit(object):
    def setup_method(self):
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )
        self.kwargs_data = kwargs_data
        data_class = ImageData(**kwargs_data)
        data_class_ref = ImageData_ref(**kwargs_data)

        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        psf_class = PSF(**kwargs_psf)

        kwargs_sis = {"theta_E": 1.0, "center_x": 0, "center_y": 0}

        lens_model_list = ["SIS"]
        self.kwargs_lens = [kwargs_sis]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        lens_model_class_ref = LensModel_ref(lens_model_list=lens_model_list)

        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse1 = {
            "amp": 1.4,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_sersic_ellipse2 = {
            "amp": 2.1,
            "R_sersic": 0.3,
            "n_sersic": 6,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        lens_light_model_class_ref = LightModel_ref(
            light_model_list=lens_light_model_list
        )

        source_model_list = ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse1, kwargs_sersic_ellipse2]
        source_model_class = LightModel(light_model_list=source_model_list)
        source_model_class_ref = LightModel_ref(light_model_list=source_model_list)

        self.kwargs_ps = [
            {"ra_image": [0.01], "dec_image": [0.0], "source_amp": 1.2},
            {"ra_image": [0.01, 1.1, -0.4], "dec_image": [0.0, -0.3, 0.2], "point_amp": [1.3, 2.3, 1.3]}
        ] 

        point_source_class = PointSource(
            point_source_type_list=["LENSED_POSITION", "LENSED_POSITION"], fixed_magnification_list=[True, False]
        )
        point_source_class_ref = PointSource_ref(
            point_source_type_list=["LENSED_POSITION", "LENSED_POSITION"], fixed_magnification_list=[True, False]
        )
        kwargs_numerics = {
            "supersampling_factor": 2,
            "supersampling_convolution": False,
        }

        self.imageLinearFit = ImageLinearFit(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.imageLinearFit_ref = ImageLinearFit_ref(
            data_class_ref,
            psf_class,
            lens_model_class_ref,
            source_model_class_ref,
            lens_light_model_class_ref,
            point_source_class_ref,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            self.imageLinearFit,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        data_class.update_data(image_sim)
        data_class_ref.update_data(image_sim)

    def test_init(self):
        source_model_class = LightModel(["SHAPELETS"])
        lens_light_model_class = LightModel(["SERSIC", "SHAPELETS"])
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
            "truncation": 5,
            "pixel_size": 0.05,
        }
        psf_class = PSF(**kwargs_psf)
        data_class = ImageData(**self.kwargs_data)

        npt.assert_raises(
            ValueError,
            ImageLinearFit,
            data_class=data_class,
            psf_class=psf_class,
            source_model_class=source_model_class,
        )
        npt.assert_raises(
            ValueError,
            ImageLinearFit,
            data_class=data_class,
            psf_class=psf_class,
            lens_light_model_class=lens_light_model_class,
        )

        data_class = ImageData(
            likelihood_method="interferometry_natwt", **self.kwargs_data
        )
        npt.assert_raises(
            ValueError, ImageLinearFit, psf_class=psf_class, data_class=data_class
        )

    def test_likelihood_data_given_model(self):
        logL, param = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
            check_positive_flux=False,
        )
        logL_ref, param_ref = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
            check_positive_flux=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        logL, param = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
            check_positive_flux=True,
        )
        logL_ref, param_ref = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
            check_positive_flux=True,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

        logL, param = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=True,
            check_positive_flux=False,
        )
        logL_ref, param_ref = self.imageLinearFit.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=True,
            check_positive_flux=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(param, param_ref, atol=1e-12, rtol=1e-12)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageLinearFit.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=False,
        )
        model_ref, error_map_ref, cov_param_ref, param_ref = (
            self.imageLinearFit_ref.image_linear_solve(
                self.kwargs_lens,
                self.kwargs_source,
                self.kwargs_lens_light,
                self.kwargs_ps,
                inv_bool=False,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-11, rtol=1e-11)
        npt.assert_allclose(error_map, error_map_ref, atol=1e-11, rtol=1e-11)
        npt.assert_allclose(param, param_ref, atol=2e-9, rtol=2e-9)
        assert cov_param is None

        chi2_reduced = self.imageLinearFit.reduced_chi2(model, error_map)
        chi2_reduced_ref = self.imageLinearFit_ref.reduced_chi2(model, error_map)
        npt.assert_allclose(chi2_reduced, chi2_reduced_ref, atol=1e-11, rtol=1e-11)

        x = np.tile(np.linspace(-1, 1, 50), 50)
        y = np.repeat(np.linspace(-1, 1, 50), 50)
        error_map = self.imageLinearFit.error_map_source(
            self.kwargs_source, x, y, cov_param
        )
        error_map_ref = self.imageLinearFit_ref.error_map_source(
            self.kwargs_source, x, y, cov_param
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-11, rtol=1e-11)

        model, error_map, cov_param, param = self.imageLinearFit.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=True,
        )
        model_ref, error_map_ref, cov_param_ref, param_ref = (
            self.imageLinearFit_ref.image_linear_solve(
                self.kwargs_lens,
                self.kwargs_source,
                self.kwargs_lens_light,
                self.kwargs_ps,
                inv_bool=True,
            )
        )
        npt.assert_allclose(model, model_ref, atol=1e-11, rtol=1e-11)
        npt.assert_allclose(error_map, error_map_ref, atol=1e-11, rtol=1e-11)
        npt.assert_allclose(cov_param, cov_param_ref, atol=1e-10, rtol=1e-10)
        npt.assert_allclose(param, param_ref, atol=2e-9, rtol=2e-9)

        chi2_reduced = self.imageLinearFit.reduced_chi2(model, error_map)
        chi2_reduced_ref = self.imageLinearFit_ref.reduced_chi2(model, error_map)
        npt.assert_allclose(chi2_reduced, chi2_reduced_ref, atol=1e-11, rtol=1e-11)

        error_map = self.imageLinearFit.error_map_source(
            self.kwargs_source, x, y, cov_param
        )
        error_map_ref = self.imageLinearFit_ref.error_map_source(
            self.kwargs_source, x, y, cov_param
        )
        npt.assert_allclose(error_map, error_map_ref, atol=1e-11, rtol=1e-11)

    def test_num_param_linear(self):
        num_param_linear = self.imageLinearFit.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        num_param_linear_ref = self.imageLinearFit_ref.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        assert num_param_linear == num_param_linear_ref == 7

    def test_linear_response_matrix(self):
        A = self.imageLinearFit.linear_response_matrix(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        A_ref = self.imageLinearFit_ref.linear_response_matrix(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        npt.assert_allclose(A, A_ref, atol=3e-8, rtol=3e-8)

    def test_linear_param_from_kwargs(self):
        param = self.imageLinearFit.linear_param_from_kwargs(
            self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        param_ref = self.imageLinearFit_ref.linear_param_from_kwargs(
            self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        npt.assert_array_equal(param, param_ref)

    def test_update_linear_kwargs(self):
        num = self.imageLinearFit.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        param = np.ones(num) * 10
        (
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
        ) = self.imageLinearFit.update_linear_kwargs(
            param,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
        )
        assert kwargs_source[0]["amp"] == 10
        assert kwargs_source[1]["amp"] == 10
        assert kwargs_lens_light[0]["amp"] == 10
        assert kwargs_ps[0]["source_amp"] == 10
        npt.assert_array_equal(kwargs_ps[1]["point_amp"], np.ones(3)*10)

    def test_error_response(self):
        C_D_response, model_error = self.imageLinearFit.error_response(
            kwargs_lens=self.kwargs_lens, kwargs_ps=self.kwargs_ps, kwargs_special=None
        )
        C_D_response_ref, model_error_ref = self.imageLinearFit.error_response(
            kwargs_lens=self.kwargs_lens, kwargs_ps=self.kwargs_ps, kwargs_special=None
        )
        npt.assert_allclose(C_D_response, C_D_response_ref, atol=1e-11, rtol=1e-11)
        npt.assert_allclose(model_error, model_error_ref, atol=1e-10, rtol=1e-10)

    def test_point_source_linear_response_set(self):
        # without kwargs_special
        (
            ra_pos,
            dec_pos,
            amp,
            num_point,
        ) = self.imageLinearFit.point_source_linear_response_set(
            self.kwargs_ps, self.kwargs_lens
        )
        (
            ra_pos_ref,
            dec_pos_ref,
            amp_ref,
            num_point_ref,
        ) = self.imageLinearFit_ref.point_source_linear_response_set(
            self.kwargs_ps, self.kwargs_lens
        )

        for i in range(len(ra_pos)):
            print(f"testing param {i}")
            npt.assert_allclose(ra_pos[i], ra_pos_ref[i], atol=1e-12, rtol=1e-12)
            npt.assert_allclose(dec_pos[i], dec_pos_ref[i], atol=1e-12, rtol=1e-12)
            npt.assert_allclose(amp[i], amp_ref[i], atol=1e-12, rtol=1e-12)
        assert num_point == num_point_ref

        # with kwargs_special
        kwargs_special = {"delta_x_image": [0.1, 0.1], "delta_y_image": [-0.1, -0.1]}
        (
            ra_pos,
            dec_pos,
            amp,
            num_point,
        ) = self.imageLinearFit.point_source_linear_response_set(
            self.kwargs_ps, self.kwargs_lens, kwargs_special
        )
        (
            ra_pos_ref,
            dec_pos_ref,
            amp_ref,
            num_point_ref,
        ) = self.imageLinearFit_ref.point_source_linear_response_set(
            self.kwargs_ps, self.kwargs_lens, kwargs_special
        )

        for i in range(len(ra_pos)):
            print(f"testing param {i}")
            npt.assert_allclose(ra_pos[i], ra_pos_ref[i], atol=1e-12, rtol=1e-12)
            npt.assert_allclose(dec_pos[i], dec_pos_ref[i], atol=1e-12, rtol=1e-12)
            npt.assert_allclose(amp[i], amp_ref[i], atol=1e-12, rtol=1e-12)
        assert num_point == num_point_ref            

    def test_check_positive_flux(self):
        self.kwargs_source[1]['amp'] = -1.1
        pos_bool = self.imageLinearFit.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        pos_bool_ref = self.imageLinearFit_ref.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        assert pos_bool == pos_bool_ref
        assert pos_bool == False

        self.kwargs_lens_light[0]['amp'] = -1.1
        self.kwargs_source[1]['amp'] = 1.1
        pos_bool = self.imageLinearFit.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        pos_bool_ref = self.imageLinearFit_ref.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        assert pos_bool == pos_bool_ref
        assert pos_bool == False

        self.kwargs_ps[1]['point_amp'] = [1.1, 1.1, -1.1]
        self.kwargs_lens_light[0]['amp'] = 1.1
        pos_bool = self.imageLinearFit.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        pos_bool_ref = self.imageLinearFit_ref.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        assert pos_bool == pos_bool_ref
        assert pos_bool == False

        self.kwargs_ps[1]['point_amp'] = [1.1, 1.1, 1.1]
        pos_bool = self.imageLinearFit.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        pos_bool_ref = self.imageLinearFit_ref.check_positive_flux(self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps)
        assert pos_bool == pos_bool_ref
        assert pos_bool == True