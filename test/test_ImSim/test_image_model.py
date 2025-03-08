__author__ = "sibirrer"

from jax import grad, config
import numpy.testing as npt
import numpy as np
import pytest

import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.psf import PSF

from lenstronomy.Data.imaging_data import ImageData as ImageData_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from lenstronomy.LightModel.light_model import LightModel as LightModel_ref
from lenstronomy.ImSim.image_model import ImageModel as ImageModel_ref
from lenstronomy.PointSource.point_source import PointSource as PointSource_ref

from jaxtronomy.Data.imaging_data import ImageData
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.ImSim.image_model import ImageModel
from jaxtronomy.PointSource.point_source import PointSource

config.update("jax_enable_x64", True)


class TestImageModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )
        self.data_class = ImageData(**kwargs_data)
        self.data_class_ref = ImageData_ref(**kwargs_data)

        # PSF specification
        kernel = np.zeros((17, 17))
        kernel[5:-5, 5:-5] = 1
        kernel[7:-7, 7:-7] = 3
        kernel[9, 9] = 7
        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": kernel,
            "psf_variance_map": np.ones_like(kernel) * kernel**2,
        }
        self.psf_class_gaussian = PSF(**kwargs_psf)

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
        kwargs_spemd2 = {
            "theta_E": 1.8,
            "center_x": 0.1,
            "center_y": -0.4,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SIE", "SHEAR"]
        lens_model_class_ref = LensModel_ref(lens_model_list=lens_model_list)
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        self.kwargs_lens2 = [kwargs_spemd2, kwargs_shear]
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
            "amp": 23.0,
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
        lens_light_model_class_ref = LightModel_ref(
            light_model_list=lens_light_model_list
        )

        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        source_model_class_ref = LightModel_ref(light_model_list=source_model_list)

        point_source_type_list = ["UNLENSED", "LENSED_POSITION", "LENSED_POSITION"]
        fixed_magnification_list = [None, True, False]
        kwargs_unlensed = {
            "ra_image": [5.342],
            "dec_image": [2.8743],
            "point_amp": [18.23],
        }
        kwargs_lensed_fixed_mag = {
            "ra_image": [1.342],
            "dec_image": [3.8743],
            "source_amp": [13.23],
        }
        kwargs_lensed = {
            "ra_image": [3.342, -2.23498],
            "dec_image": [-0.8743, 4.2384],
            "point_amp": [19.23, 18.543],
        }
        self.kwargs_ps = [kwargs_unlensed, kwargs_lensed_fixed_mag, kwargs_lensed]
        point_source_class = PointSource(
            lens_model=None,
            point_source_type_list=point_source_type_list,
            fixed_magnification_list=fixed_magnification_list,
        )
        point_source_class_ref = PointSource_ref(
            lens_model=None,
            point_source_type_list=point_source_type_list,
            fixed_magnification_list=fixed_magnification_list,
        )

        self.kwargs_special = {
            "delta_x_image": [0.334],
            "delta_y_image": [-1.3287],
        }

        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
        }

        # Create likelihood mask
        likelihood_mask = np.ones((numPix, numPix))
        likelihood_mask[50][::2] -= likelihood_mask[50][::2]
        likelihood_mask[25][::3] -= likelihood_mask[25][::3]
        self.likelihood_mask = likelihood_mask

        # Create 2 class instances with likelihood mask and w/o point source
        self.imageModel = ImageModel(
            self.data_class,
            self.psf_class_gaussian,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
            likelihood_mask=likelihood_mask,
        )
        self.imageModel_ref = ImageModel_ref(
            self.data_class_ref,
            self.psf_class_gaussian,
            lens_model_class_ref,
            source_model_class_ref,
            lens_light_model_class_ref,
            kwargs_numerics=kwargs_numerics,
            likelihood_mask=likelihood_mask,
        )

        # Create 2 class instances without likelihood mask and with point source
        self.imageModel_nomask = ImageModel(
            self.data_class,
            self.psf_class_gaussian,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class=point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.imageModel_nomask_ref = ImageModel_ref(
            self.data_class_ref,
            self.psf_class_gaussian,
            lens_model_class_ref,
            source_model_class_ref,
            lens_light_model_class_ref,
            point_source_class=point_source_class_ref,
            kwargs_numerics=kwargs_numerics,
        )

        image_sim = sim_util.simulate_simple(
            self.imageModel_ref,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            no_noise=True,
        )
        self.data_class.update_data(image_sim)
        self.data_class_ref.update_data(image_sim)

    def test_init(self):
        # pixelbased solver not supported
        npt.assert_raises(
            ValueError,
            ImageModel,
            self.data_class,
            self.psf_class_gaussian,
            kwargs_pixelbased={"not supported": 1},
        )

        del self.data_class.flux_scaling
        imageModel = ImageModel(
            self.data_class,
            self.psf_class_gaussian,
        )
        assert imageModel._flux_scaling == 1
        assert imageModel._pb is None
        assert imageModel._pb_1d is None

        # primary beam not supported
        self.data_class._pb = np.ones((100, 100))
        npt.assert_raises(
            ValueError, ImageModel, self.data_class, self.psf_class_gaussian
        )

    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        logL_ref = self.imageModel_ref.likelihood_data_given_model(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )
        npt.assert_array_almost_equal(logL, logL_ref, decimal=8)

        logL = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens2, self.kwargs_source, self.kwargs_lens_light
        )
        logL_ref = self.imageModel_ref.likelihood_data_given_model(
            self.kwargs_lens2, self.kwargs_source, self.kwargs_lens_light
        )
        npt.assert_array_almost_equal(logL, logL_ref, decimal=8)

        logL = self.imageModel_nomask.likelihood_data_given_model(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        logL_ref = self.imageModel_nomask_ref.likelihood_data_given_model(
            self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps
        )
        npt.assert_array_almost_equal(logL, logL_ref, decimal=8)

        logL = self.imageModel_nomask.likelihood_data_given_model(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        logL_ref = self.imageModel_nomask_ref.likelihood_data_given_model(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_array_almost_equal(logL, logL_ref, decimal=8)

    def test_autodifferentiate_likelihood(self):
        # NOTE: For autodifferentiation to work, input values need to be floats, not ints
        for key, value in self.kwargs_source[0].items():
            self.kwargs_source[0][key] = float(value)
        for key, value in self.kwargs_lens_light[0].items():
            self.kwargs_lens_light[0][key] = float(value)

        # differentiates with respect to the 0th argument by default
        grad_log_func = grad(self.imageModel_nomask.likelihood_data_given_model)
        grad_log = grad_log_func(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert len(grad_log[0]) == len(self.kwargs_lens2[0])
        assert len(grad_log[1]) == len(self.kwargs_lens2[1])

        # kwargs_lens has ints so this will result in an error
        npt.assert_raises(
            TypeError,
            grad_log_func,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
        )

        # differentiates with respect to kwargs_source
        grad_log_func = grad(
            self.imageModel_nomask.likelihood_data_given_model, argnums=1
        )
        grad_log = grad_log_func(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert len(grad_log[0]) == len(self.kwargs_source[0])

        # differentiates with respect to kwargs_lens_light
        grad_log_func = grad(
            self.imageModel_nomask.likelihood_data_given_model, argnums=2
        )
        grad_log = grad_log_func(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert len(grad_log[0]) == len(self.kwargs_lens_light[0])

        # differentiates with respect to kwargs_ps
        grad_log_func = grad(
            self.imageModel_nomask.likelihood_data_given_model, argnums=3
        )
        grad_log = grad_log_func(
            self.kwargs_lens2,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        assert len(grad_log[2]) == len(self.kwargs_ps[2])

    def test_source_surface_brightness(self):
        flux = self.imageModel.source_surface_brightness(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        flux_ref = self.imageModel_ref.source_surface_brightness(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

        flux = self.imageModel_nomask.source_surface_brightness(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        flux_ref = self.imageModel_nomask_ref.source_surface_brightness(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

        empty_imageModel = ImageModel(self.data_class, self.psf_class_gaussian)
        zero_flux = empty_imageModel.source_surface_brightness(kwargs_source={})
        npt.assert_array_equal(zero_flux, np.zeros_like(self.data_class.data))

    def test_source_surface_brightness_analytical(self):

        flux = self.imageModel._source_surface_brightness_analytical(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        flux_ref = self.imageModel_ref._source_surface_brightness_analytical(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

        flux = self.imageModel._source_surface_brightness_analytical_numerics(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        flux_ref = self.imageModel_ref._source_surface_brightness_analytical_numerics(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=7)

        flux = self.imageModel._source_surface_brightness_analytical_numerics(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
            de_lensed=True,
        )
        flux_ref = self.imageModel_ref._source_surface_brightness_analytical_numerics(
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
            de_lensed=True,
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

    def test_lens_surface_brightness(self):
        flux = self.imageModel.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        flux_ref = self.imageModel_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

        flux = self.imageModel_nomask.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        flux_ref = self.imageModel_nomask_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

    def test_point_source(self):
        flux = self.imageModel_nomask.point_source(
            self.kwargs_ps,
            self.kwargs_lens,
            self.kwargs_special,
        )
        flux_ref = self.imageModel_nomask_ref.point_source(
            self.kwargs_ps,
            self.kwargs_lens,
            self.kwargs_special,
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

        flux = self.imageModel_nomask.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, k=2
        )
        flux_ref = self.imageModel_nomask_ref.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, k=2
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

        flux = self.imageModel_nomask.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, unconvolved=True
        )
        flux_ref = self.imageModel_nomask_ref.point_source(
            self.kwargs_ps, self.kwargs_lens, self.kwargs_special, unconvolved=True
        )
        npt.assert_allclose(flux, flux_ref, atol=1e-8, rtol=1e-8)

    def test_image(self):
        image = self.imageModel.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        image_ref = self.imageModel_ref.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_array_almost_equal(image, image_ref, decimal=8)

        image = self.imageModel.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        image_ref = self.imageModel_ref.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        npt.assert_array_almost_equal(image, image_ref, decimal=8)

        image = self.imageModel_nomask.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
        )
        image_ref = self.imageModel_nomask_ref.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
        )
        npt.assert_array_almost_equal(image, image_ref, decimal=8)

        image = self.imageModel_nomask.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        image_ref = self.imageModel_nomask_ref.image(
            kwargs_lens=self.kwargs_lens2,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_special,
        )
        npt.assert_array_almost_equal(image, image_ref, decimal=8)

        image = self.imageModel_nomask.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            unconvolved=True,
        )
        image_ref = self.imageModel_nomask_ref.image(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            unconvolved=True,
        )
        npt.assert_array_almost_equal(image, image_ref, decimal=8)

    def test_data_response(self):
        npt.assert_array_almost_equal(
            self.imageModel.data_response, self.imageModel_ref.data_response
        )
        npt.assert_array_almost_equal(
            self.imageModel_nomask.data_response,
            self.imageModel_nomask_ref.data_response,
        )

    def test_error(self):
        # error response
        cd_response, model_error = self.imageModel_nomask.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        cd_response_ref, model_error_ref = self.imageModel_nomask_ref.error_response(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_array_almost_equal(cd_response, cd_response_ref, decimal=8)
        npt.assert_array_almost_equal(model_error, model_error_ref, decimal=8)

        # error map psf
        error = self.imageModel_nomask._error_map_psf(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        error_ref = self.imageModel_nomask_ref._error_map_psf(
            self.kwargs_lens, self.kwargs_ps, self.kwargs_special
        )
        npt.assert_array_almost_equal(error, error_ref, decimal=8)

    def test_reduced_residuals(self):
        model = self.data_class.data
        residuals = self.imageModel.reduced_residuals(model)
        residuals_ref = self.imageModel_ref.reduced_residuals(model)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

        error_map = np.ones_like(model) * 0.5
        residuals = self.imageModel.reduced_residuals(model, error_map)
        residuals_ref = self.imageModel_ref.reduced_residuals(model, error_map)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

    def test_reduced_chi2(self):
        model = self.data_class.data
        chi2 = self.imageModel.reduced_chi2(model)
        chi2_ref = self.imageModel_ref.reduced_chi2(model)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)

        error_map = np.ones_like(model) * 0.5
        chi2 = self.imageModel.reduced_chi2(model, error_map)
        chi2_ref = self.imageModel_ref.reduced_chi2(model, error_map)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)

    def test_image2array_masked(self):
        image = self.data_class.data
        array = self.imageModel.image2array_masked(image)
        array_ref = self.imageModel_ref.image2array_masked(image)
        npt.assert_array_almost_equal(array, array_ref, decimal=8)

    def test_array_masked2image(self):
        image_0 = self.data_class.data
        array = self.imageModel.image2array_masked(image_0)
        array_ref = self.imageModel_ref.image2array_masked(image_0)

        image = self.imageModel.array_masked2image(array)
        image_ref = self.imageModel_ref.array_masked2image(array_ref)
        npt.assert_array_almost_equal(image, image_ref, decimal=8)
        npt.assert_array_almost_equal(image, image_0 * self.likelihood_mask, decimal=8)

    def test_raises(self):
        # check positive flux not supported in jaxtronomy
        npt.assert_raises(
            ValueError,
            self.imageModel.likelihood_data_given_model,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            check_positive_flux=True,
        )

        # Update psf and update data not supported
        npt.assert_raises(
            ValueError, self.imageModel.update_psf, self.psf_class_gaussian
        )
        npt.assert_raises(ValueError, self.imageModel.update_data, self.data_class)

        # extinction not supported
        npt.assert_raises(
            ValueError,
            self.imageModel._source_surface_brightness_analytical_numerics,
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
            kwargs_extinction={"incorrect": 0},
        )


if __name__ == "__main__":
    pytest.main()
