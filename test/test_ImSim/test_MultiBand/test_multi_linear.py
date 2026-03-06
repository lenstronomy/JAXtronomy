import numpy.testing as npt
import pytest
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from jaxtronomy.ImSim.MultiBand.multi_linear import MultiLinear
from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear as MultiLinear_ref

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util


class TestImageModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        psf_class = PSF(**kwargs_psf)
        # 'EXTERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["EPL", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
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
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [
            {"ra_image": [0.1], "dec_image": [-0.3], "source_amp": [3.1]}
        ]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(
            point_source_type_list=["LENSED_POSITION"], fixed_magnification_list=[True]
        )
        kwargs_numerics = {"supersampling_factor": 2, "supersampling_convolution": True}
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim

        multi_band_list = [
            [kwargs_data, kwargs_psf, kwargs_numerics],
            [kwargs_data, kwargs_psf, kwargs_numerics],
        ]
        self.multi_band_list = multi_band_list
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "point_source_model_list": ["LENSED_POSITION"],
            "fixed_magnification_list": [True],
        }
        self.kwargs_model = kwargs_model
        self.imageModel = MultiLinear(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list=None,
            compute_bool=[True, False],
        )
        self.imageModel_ref = MultiLinear_ref(
            multi_band_list,
            kwargs_model,
            likelihood_mask_list=None,
            compute_bool=[True, False],
        )

    def test_init(self):
        npt.assert_raises(
            ValueError,
            MultiLinear,
            self.multi_band_list,
            self.kwargs_model,
            likelihood_mask_list=None,
            compute_bool=[True, False, False],
        )
        assert self.imageModel.num_bands == self.imageModel_ref.num_bands
        assert (
            self.imageModel.num_response_list == self.imageModel_ref.num_response_list
        )
        assert (
            self.imageModel.num_data_evaluate == self.imageModel_ref.num_data_evaluate
        )
        num_param_linear = self.imageModel.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        num_param_linear_ref = self.imageModel_ref.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        assert num_param_linear == num_param_linear_ref

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=False,
        )

        model_ref, error_map_ref, cov_param_ref, param_ref = (
            self.imageModel_ref.image_linear_solve(
                self.kwargs_lens,
                self.kwargs_source,
                self.kwargs_lens_light,
                self.kwargs_ps,
                inv_bool=False,
            )
        )
        assert len(model) == len(model_ref)
        assert len(error_map) == len(error_map_ref)
        assert len(cov_param) == len(cov_param_ref)
        assert len(param) == len(param_ref)

        for i in range(len(model)):
            print(i)
            if model[i] is None and model_ref[i] is None:
                pass
            else:
                npt.assert_allclose(model[i], model_ref[i], atol=1e-7, rtol=1e-7)

            if error_map[i] is None and error_map_ref[i] is None:
                pass
            else:
                npt.assert_allclose(
                    error_map[i], error_map_ref[i], atol=1e-7, rtol=1e-7
                )

            if cov_param[i] is None and cov_param_ref[i] is None:
                pass
            else:
                npt.assert_allclose(
                    cov_param[i], cov_param_ref[i], atol=1e-7, rtol=1e-7
                )

            if param[i] is None and param_ref[i] is None:
                pass
            else:
                npt.assert_allclose(param[i], param_ref[i], atol=1e-6, rtol=1e-6)

        residuals = self.imageModel.reduced_residuals(model, error_map)
        residuals_ref = self.imageModel_ref.reduced_residuals(model_ref, error_map_ref)
        assert len(residuals) == len(residuals_ref)
        for i in range(len(residuals)):
            npt.assert_allclose(residuals[i], residuals_ref[i], atol=1e-5, rtol=1e-5)

    def test_likelihood_data_given_model(self):
        logL, param = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
        )

        logL_ref, param_ref = self.imageModel_ref.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)
        assert len(param) == len(param_ref)

        for i in range(len(param)):
            if param[i] is None and param_ref[i] is None:
                pass
            else:
                npt.assert_allclose(param[i], param_ref[i], atol=1e-6, rtol=1e-6)

    def test_update_linear_kwargs(self):
        num_param_linear = self.imageModel.num_param_linear(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        param = np.ones(int(num_param_linear)) * 10
        kwargs = self.imageModel.update_linear_kwargs(
            [param * 2, param],
            0,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        kwargs_ref = self.imageModel_ref.update_linear_kwargs(
            [param * 2, param],
            0,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert kwargs_ref == kwargs


if __name__ == "__main__":
    pytest.main()
