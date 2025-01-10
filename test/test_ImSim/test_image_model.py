__author__ = "sibirrer"

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

from jaxtronomy.Data.imaging_data import ImageData
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.ImSim.image_model import ImageModel


class TestImageModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        supersampling_factor = 3
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )
        self.data_class = ImageData(**kwargs_data)
        self.data_class_ref = ImageData_ref(**kwargs_data)
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        self.psf_class_gaussian = PSF(**kwargs_psf)
        kernel = self.psf_class_gaussian.kernel_point_source
        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": kernel,
            "psf_error_map": np.ones_like(kernel) * 0.001 * kernel**2,
        }
        self.psf_class = PSF(**kwargs_psf)

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

        lens_model_list = ["SIE", "SHEAR"]
        lens_model_class_ref = LensModel_ref(lens_model_list=lens_model_list)
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
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
        lens_light_model_class_ref = LightModel_ref(
            light_model_list=lens_light_model_list
        )

        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        source_model_class_ref = LightModel_ref(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": supersampling_factor,
            "supersampling_convolution": True,
        }
        self.imageModel = ImageModel(
            self.data_class,
            self.psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.imageModel_ref = ImageModel_ref(
            self.data_class_ref,
            self.psf_class,
            lens_model_class_ref,
            source_model_class_ref,
            lens_light_model_class_ref,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            self.imageModel_ref,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
        )
        self.data_class.update_data(image_sim)
        self.data_class_ref.update_data(image_sim)

    def test_init(self):
        npt.assert_raises(
            ValueError,
            ImageModel,
            self.data_class,
            self.psf_class,
            kwargs_pixelbased={},
        )

        del self.data_class.flux_scaling
        imageModel = ImageModel(
            self.data_class,
            self.psf_class,
        )
        assert imageModel._flux_scaling == 1
        assert imageModel._pb is None
        assert imageModel._pb_1d is None

        self.data_class._pb = np.ones((100, 100))
        npt.assert_raises(ValueError, ImageModel, self.data_class, self.psf_class)

    def test_update_psf(self):
        assert self.imageModel.PSF.psf_type == "PIXEL"
        self.imageModel.update_psf(self.psf_class_gaussian)
        assert self.imageModel.PSF.psf_type == "GAUSSIAN"

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

        empty_imageModel = ImageModel(self.data_class, self.psf_class)
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
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

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

        npt.assert_raises(
            ValueError,
            self.imageModel._source_surface_brightness_analytical_numerics,
            kwargs_source=self.kwargs_source,
            kwargs_lens=self.kwargs_lens,
            kwargs_extinction={"incorrect": 0},
        )

    def test_lens_surface_brightness(self):
        flux = self.imageModel.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        flux_ref = self.imageModel_ref.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_lens_light
        )
        npt.assert_array_almost_equal(flux, flux_ref, decimal=8)

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
        npt.assert_raises(
            ValueError,
            self.imageModel.image,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            point_source_add=True,
        )


if __name__ == "__main__":
    pytest.main()
