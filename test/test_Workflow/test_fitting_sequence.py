__author__ = "sibirrer"

import copy

import pytest
import numpy.testing as npt
import numpy as np
from lenstronomy.Util import simulation_util as sim_util, util

from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource

from jaxtronomy.Workflow.fitting_sequence import FittingSequence
from jaxtronomy.ImSim.image_model import ImageModel
from jaxtronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestFittingSequence(object):
    """Test the fitting sequences."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        self.kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**self.kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)
        self.kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": psf_gaussian.kernel_point_source,
        }
        psf_class = PSF(**self.kwargs_psf)
        # 'EXTERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angle (in radian)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_model_list = ["EPL", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        kwargs_sersic = {
            "amp": 21.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {
            "amp": 13.0,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [
            {"ra_source": 0.0, "dec_source": 0.0, "source_amp": 1.0}
        ]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = []  # ["SOURCE_POSITION"]
        point_source_class = PointSource(
            point_source_type_list=point_source_list,  # fixed_magnification_list=[True]
        )
        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
            "compute_mode": "regular",
            # "point_source_supersampling_factor": 1,
        }
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
            # self.kwargs_psf,
            point_source_add=False,
        )

        data_class.update_data(image_sim)
        self.data_class = data_class
        self.psf_class = psf_class
        self.kwargs_data["image_data"] = image_sim
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": point_source_list,
            "fixed_magnification_list": [False],
            "index_lens_model_list": [[0, 1]],
            "point_source_frame_list": [[0]],
        }
        self.kwargs_numerics = kwargs_numerics

        # num_source_model = len(source_model_list)

        self.kwargs_constraints = {
            # "num_point_source_list": [4],
            # "image_plane_source_list": [False] * num_source_model,
            "linear_solver": False
        }

        self.kwargs_likelihood = {
            # This is false by default anyways
            "check_positive_flux": False,
            "source_marg": False,
        }

        lens_sigma = [
            {
                "theta_E": 0.1,
                "gamma": 0.1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.1,
                "center_y": 0.1,
            },
            {"gamma1": 0.1, "gamma2": 0.1},
        ]
        lens_lower = [
            {
                "theta_E": 0.0,
                "gamma": 1.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            },
            {"gamma1": -0.3, "gamma2": -0.3},
        ]
        lens_upper = [
            {
                "theta_E": 10.0,
                "gamma": 2.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            },
            {"gamma1": 0.3, "gamma2": 0.3},
        ]
        source_sigma = [
            {
                "amp": 0.1,
                "R_sersic": 0.05,
                "n_sersic": 0.5,
                "center_x": 0.1,
                "center_y": 0.1,
                "e1": 0.1,
                "e2": 0.1,
            }
        ]
        source_lower = [
            {
                "amp": 0,
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        source_upper = [
            {
                "amp": 100,
                "R_sersic": 10,
                "n_sersic": 5.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        lens_light_sigma = [
            {
                "amp": 0.1,
                "R_sersic": 0.05,
                "n_sersic": 0.5,
                "center_x": 0.1,
                "center_y": 0.1,
            }
        ]
        lens_light_lower = [
            {
                "amp": 0,
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -2,
                "center_y": -2,
            }
        ]
        lens_light_upper = [
            {"amp": 100, "R_sersic": 10, "n_sersic": 5.5, "center_x": 2, "center_y": 2}
        ]
        ps_sigma = [{"ra_source": 1, "dec_source": 1, "point_amp": 1}]

        lens_param = (
            self.kwargs_lens,
            lens_sigma,
            [{}, {"ra_0": 0, "dec_0": 0}],
            lens_lower,
            lens_upper,
        )
        source_param = (
            self.kwargs_source,
            source_sigma,
            [{}],
            source_lower,
            source_upper,
        )
        lens_light_param = (
            self.kwargs_lens_light,
            lens_light_sigma,
            [{"center_x": 0}],
            lens_light_lower,
            lens_light_upper,
        )
        ps_param = self.kwargs_ps, ps_sigma, [{}], self.kwargs_ps, self.kwargs_ps

        self.kwargs_params = {
            "lens_model": lens_param,
            "source_model": source_param,
            "lens_light_model": lens_light_param,
            "point_source_model": ps_param,
            # 'special': special_param
        }
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        self.kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
        }

    def test_simulationAPI_image(self):
        npt.assert_almost_equal(self.data_class.data[4, 4], 0.1, decimal=0)

    def test_simulationAPI_psf(self):
        npt.assert_almost_equal(
            np.sum(self.psf_class.kernel_point_source), 1, decimal=6
        )

    def test_fitting_sequence(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )

        kwargs_result = fittingSequence.best_fit(bijective=False)
        lens_temp = kwargs_result["kwargs_lens"]
        npt.assert_almost_equal(
            lens_temp[0]["theta_E"], self.kwargs_lens[0]["theta_E"], decimal=2
        )

        logL = fittingSequence.best_fit_likelihood()
        print(logL, "test")
        # print(lens_temp, source_temp, lens_light_temp, ps_temp, special_temp)
        assert logL < 0
        bic = fittingSequence.bic
        assert bic > 0
        # npt.assert_almost_equal(bic, 20000000220.29376, decimal=-4)

        # npt.assert_almost_equal(logL, -10000000061.792593, decimal=-4)

        n_p = 2
        n_i = 2
        fitting_list = []

        kwargs_pso = {"sigma_scale": 1, "n_particles": n_p, "n_iterations": n_i}
        fitting_list.append(["PSO", kwargs_pso])
        kwargs_align = {"delta_shift": 0.2, "n_particles": 2, "n_iterations": 2}
        fitting_list.append(["align_images", kwargs_align])
        kwargs_psf_iter = {
            "num_iter": 2,
            "psf_iter_factor": 0.5,
            "stacking_method": "mean",
            "new_procedure": False,
        }
        # fitting_list.append(["psf_iteration", kwargs_psf_iter])
        fitting_list.append(["restart", None])
        fitting_list.append(["fix_not_computed", {"free_bands": [True]}])
        n_sersic_overwrite = 4
        kwargs_update = {
            "lens_light_add_fixed": [[0, ["n_sersic"], [n_sersic_overwrite]]],
            "lens_light_remove_fixed": [[0, ["center_x"]]],
            "change_source_lower_limit": [[0, ["n_sersic"], [0.1]]],
            "change_source_upper_limit": [[0, ["n_sersic"], [10]]],
        }
        fitting_list.append(["update_settings", kwargs_update])

        chain_list = fittingSequence.fit_sequence(fitting_list)
        (
            lens_fixed,
            source_fixed,
            lens_light_fixed,
            ps_fixed,
            special_fixed,
            extinction_fixed,
            tracer_source_fixed,
        ) = fittingSequence._updateManager.fixed_kwargs
        kwargs_result = fittingSequence.best_fit(bijective=False)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"],
            self.kwargs_lens[0]["theta_E"],
            decimal=1,
        )
        npt.assert_almost_equal(
            fittingSequence._updateManager._lens_light_fixed[0]["n_sersic"],
            n_sersic_overwrite,
            decimal=8,
        )
        npt.assert_almost_equal(lens_light_fixed[0]["n_sersic"], 4, decimal=-1)
        assert fittingSequence._updateManager._lower_kwargs[1][0]["n_sersic"] == 0.1
        assert fittingSequence._updateManager._upper_kwargs[1][0]["n_sersic"] == 10

        # test 'set_param_value' fitting sequence
        fitting_list = [
            ["set_param_value", {"lens": [[1, ["gamma1"], [0.013]]]}],
            ["set_param_value", {"lens_light": [[0, ["center_x"], [0.009]]]}],
            ["set_param_value", {"source": [[0, ["n_sersic"], [2.993]]]}],
            # ["set_param_value", {"ps": [[0, ["ra_source"], [0.007]]]}],
        ]

        fittingSequence.fit_sequence(fitting_list)

        kwargs_set = fittingSequence._updateManager.parameter_state
        assert kwargs_set["kwargs_lens"][1]["gamma1"] == 0.013
        assert kwargs_set["kwargs_lens_light"][0]["center_x"] == 0.009
        assert kwargs_set["kwargs_source"][0]["n_sersic"] == 2.993
        # assert kwargs_set["kwargs_ps"][0]["ra_source"] == 0.007

        from unittest import TestCase

        t = TestCase()
        with t.assertRaises(ValueError):
            fitting_list_two = []
            fitting_list_two.append(["fake_mcmc_method", kwargs_pso])
            fittingSequence.fit_sequence(fitting_list_two)

        with t.assertRaises(ValueError):
            # should raise a value error for n_walkers = walkerRatio = None
            fitting_list_three = []
            kwargs_test = {"n_burn": 10, "n_run": 10}
            fitting_list_three.append(["emcee", kwargs_test])
            fittingSequence.fit_sequence(fitting_list_three)

        fitting_list4 = [["psf_iteration", kwargs_psf_iter]]
        with t.assertRaises(ValueError):
            fittingSequence.fit_sequence(fitting_list4)
        fitting_list5 = [["calibrate_images", {}]]
        with t.assertRaises(ValueError):
            fittingSequence.fit_sequence(fitting_list5)

    def test_cobaya(self):
        np.random.seed(42)

        # make a basic lens model to fit
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification
        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)

        # make a lens
        lens_model_list = ["SIS"]
        kwargs_lens = [{"theta_E": 1.5, "center_x": 0.0, "center_y": 0.0}]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # make a source
        source_model_list = ["SERSIC"]
        kwargs_source = [
            {
                "amp": 1.0,
                "R_sersic": 0.3,
                "n_sersic": 3.0,
                "center_x": 0.1,
                "center_y": 0.1,
            }
        ]
        source_model_class = LightModel(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            data_class,
            psf_gaussian,
            lens_model_class,
            source_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            kwargs_lens,
            kwargs_source,
            point_source_add=False,
            no_noise=True,
        )

        data_class.update_data(image_sim)

        kwargs_data["image_data"] = image_sim

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
        }

        lens_fixed = [{"center_x": 0.0, "center_y": 0.0}]
        lens_sigma = [{"theta_E": 0.1, "center_x": 0.1, "center_y": 0.1}]
        lens_lower = [{"theta_E": 0.1, "center_x": -10, "center_y": -10}]
        lens_upper = [{"theta_E": 3.0, "center_x": 10, "center_y": 10}]

        source_fixed = [{"amp": 1.0}]
        source_sigma = [
            {
                "R_sersic": 0.01,
                "n_sersic": 0.01,
                "center_x": 0.01,
                "center_y": 0.01,
            }
        ]
        source_lower = [
            {
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -1,
                "center_y": -1,
            }
        ]
        source_upper = [
            {
                "R_sersic": 1.0,
                "n_sersic": 6.0,
                "center_x": 1,
                "center_y": 1,
            }
        ]

        lens_param = [kwargs_lens, lens_sigma, lens_fixed, lens_lower, lens_upper]
        source_param = [
            kwargs_source,
            source_sigma,
            source_fixed,
            source_lower,
            source_upper,
        ]

        kwargs_params = {"lens_model": lens_param, "source_model": source_param}

        multi_band_list = [[kwargs_data, kwargs_psf_gaussian, kwargs_numerics]]

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
        }

        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        kwargs_cobaya = {
            "proposal_widths": [0.001, 0.001, 0.001, 0.001, 0.001],
            "Rminus1_stop": 100,
            "max_tries": 1000,
            "force_overwrite": True,
        }

        chain_list = fittingSequence.fit_sequence([["Cobaya", kwargs_cobaya]])
        assert fittingSequence.kwargs_fixed == (
            lens_fixed,
            source_fixed,
            [],
            [],
            {},
            [],
            [],
        )

    def test_zeus(self):
        np.random.seed(42)
        # we make a very basic lens+source model to feed to check zeus can be run through fitting sequence
        # we don't use the kwargs defined in setup() as those are modified during the tests; using unique kwargs here is safer

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)

        # make a lens
        lens_model_list = ["EPL"]
        kwargs_epl = {
            "theta_E": 0.6,
            "gamma": 2.6,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": 0.1,
        }
        kwargs_lens = [kwargs_epl]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # make a source
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": 0.1,
        }
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            data_class,
            psf_gaussian,
            lens_model_class,
            source_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            kwargs_lens,
            kwargs_source,
            point_source_add=False,
            no_noise=True,
        )

        data_class.update_data(image_sim)

        kwargs_data["image_data"] = image_sim

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
        }

        lens_fixed = [{}]
        lens_sigma = [
            {
                "theta_E": 0.1,
                "gamma": 0.1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.1,
                "center_y": 0.1,
            }
        ]
        lens_lower = [
            {
                "theta_E": 0.0,
                "gamma": 1.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        lens_upper = [
            {
                "theta_E": 10.0,
                "gamma": 2.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        source_fixed = [{"amp": 1.0}]
        source_sigma = [
            {
                "R_sersic": 0.05,
                "n_sersic": 0.5,
                "center_x": 0.1,
                "center_y": 0.1,
                "e1": 0.1,
                "e2": 0.1,
            }
        ]
        source_lower = [
            {
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        source_upper = [
            {
                "R_sersic": 10,
                "n_sersic": 5.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        lens_param = [kwargs_lens, lens_sigma, lens_fixed, lens_lower, lens_upper]
        source_param = [
            kwargs_source,
            source_sigma,
            source_fixed,
            source_lower,
            source_upper,
        ]

        kwargs_params = {"lens_model": lens_param, "source_model": source_param}

        multi_band_list = [[kwargs_data, kwargs_psf_gaussian, kwargs_numerics]]

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
        }

        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []

        kwargs_zeus = {
            "n_burn": 2,
            "n_run": 2,
            "walkerRatio": 4,
            "backend_filename": "test_mcmc_zeus.h5",
        }

        fitting_list.append(["zeus", kwargs_zeus])

        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_multinest(self):
        # Nested sampler tests
        # further decrease the parameter space for nested samplers to run faster

        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        fitting_list = []
        kwargs_update = {
            # "ps_add_fixed": [[0, ["ra_source", "dec_source"], [0, 0]]],
            "lens_light_add_fixed": [
                [0, ["n_sersic", "R_sersic", "center_x", "center_y"], [4, 0.1, 0, 0]]
            ],
            "source_add_fixed": [
                [
                    0,
                    ["R_sersic", "e1", "e2", "center_x", "center_y"],
                    [0.6, 0.1, 0.1, 0, 0],
                ]
            ],
            "lens_add_fixed": [
                [
                    0,
                    ["gamma", "theta_E", "e1", "e2", "center_x", "center_y"],
                    [1.8, 1.0, 0.1, 0.1, 0, 0],
                ],
                [1, ["gamma1", "gamma2"], [0.01, 0.01]],
            ],
            "change_source_lower_limit": [[0, ["n_sersic"], [2.9]]],
            "change_source_upper_limit": [[0, ["n_sersic"], [3.1]]],
        }
        fitting_list.append(["update_settings", kwargs_update])
        kwargs_multinest = {
            "kwargs_run": {
                "n_live_points": 10,
                "evidence_tolerance": 0.5,
                "sampling_efficiency": 0.8,  # 1 for posterior-only, 0 for evidence-only
                "importance_nested_sampling": False,
                "multimodal": True,
                "const_efficiency_mode": False,  # reduce sampling_efficiency to 5% when True
            },
            "remove_output_dir": True,
        }

        fitting_list.append(["MultiNest", kwargs_multinest])

        chain_list2 = fittingSequence.fit_sequence(fitting_list)
        kwargs_fixed = fittingSequence._updateManager.fixed_kwargs
        npt.assert_almost_equal(kwargs_fixed[0][1]["gamma1"], 0.01, decimal=2)
        assert fittingSequence._updateManager._lower_kwargs[1][0]["n_sersic"] == 2.9
        assert fittingSequence._updateManager._upper_kwargs[1][0]["n_sersic"] == 3.1

        kwargs_test = {"kwargs_lens": 1}
        fittingSequence.update_state(kwargs_test)
        kwargs_out = fittingSequence.best_fit(bijective=True)
        assert kwargs_out["kwargs_lens"] == 1

    def test_dynesty(self):
        np.random.seed(42)
        kwargs_params = copy.deepcopy(self.kwargs_params)
        kwargs_params["lens_model"][0][0]["theta_E"] += 0.01
        kwargs_params["lens_model"][0][0]["gamma"] += 0.01
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []
        kwargs_dynesty = {
            "kwargs_run": {
                "dlogz_init": 0.01,
                "nlive_init": 20,
                "nlive_batch": 20,
                "maxbatch": 1,
            },
        }

        fitting_list.append(["nested_sampling", kwargs_dynesty])

        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_nautilus(self):
        np.random.seed(42)
        kwargs_params = copy.deepcopy(self.kwargs_params)
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []
        kwargs_nautilus = {
            "prior_type": "uniform",
            "verbose": True,
            "f_live": 1.0,
            "n_eff": 0.0,
            "n_live": 2,
            "seed": 42,
        }

        fitting_list.append(["Nautilus", kwargs_nautilus])
        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_dypolychord(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        fitting_list = []
        kwargs_dypolychord = {
            "kwargs_run": {
                "ninit": 8,
                "nlive_const": 10,
                #'seed_increment': 1,
                "resume_dyn_run": False,
                #'init_step': 10,
            },
            "polychord_settings": {
                "seed": 1,
                #'num_repeats': 20
            },
            "dypolychord_dynamic_goal": 0.8,  # 1 for posterior-only, 0 for evidence-only
            "remove_output_dir": True,
        }

        fitting_list.append(["dyPolyChord", kwargs_dypolychord])

        kwargs_dypolychord2 = copy.deepcopy(kwargs_dypolychord)
        kwargs_dypolychord2["kwargs_run"]["resume_dyn_run"] = True
        kwargs_dypolychord2["prior_type"] = "gaussian"
        fitting_list.append(["dyPolyChord", kwargs_dypolychord2])
        npt.assert_raises(ValueError, fittingSequence.fit_sequence, fitting_list)

        kwargs_dypolychord2["kwargs_run"]["resume_dyn_run"] = False
        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_minimizer(self):
        n_p = 2
        n_i = 2

        fitting_list = []

        kwargs_simplex = {"n_iterations": n_i, "method": "Nelder-Mead"}
        fitting_list.append(["SIMPLEX", kwargs_simplex])
        kwargs_simplex = {"n_iterations": n_i, "method": "Powell"}
        fitting_list.append(["SIMPLEX", kwargs_simplex])
        kwargs_pso = {"sigma_scale": 1, "n_particles": n_p, "n_iterations": n_i}
        fitting_list.append(["PSO", kwargs_pso])
        kwargs_mcmc = {"sigma_scale": 1, "n_burn": 1, "n_run": 1, "n_walkers": 10}
        fitting_list.append(["emcee", kwargs_mcmc])
        kwargs_mcmc["re_use_samples"] = True
        # Change the number of parameters from 1 to 2; this should raise an error
        kwargs_mcmc["init_samples"] = np.array(
            [[np.random.normal(1, 0.001)] * 2 for i in range(100)]
        )
        fitting_list.append(["MCMC", kwargs_mcmc])

        def custom_likelihood(kwargs_lens, **kwargs):
            theta_E = kwargs_lens[0]["theta_E"]
            return -((theta_E - 1.0) ** 2) / 0.1**2 / 2

        kwargs_likelihood = {"custom_logL_addition": custom_likelihood}

        kwargs_data_joint = {
            "multi_band_list": [],
            "multi_band_type": "single-band",
        }
        kwargs_model = {"lens_model_list": ["SIS"]}
        lens_param = (
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"theta_E": 0.1, "center_x": 0.1, "center_y": 0.1}],
            [{"center_x": 0, "center_y": 0}],
            [{"theta_E": 0, "center_x": -10, "center_y": -10}],
            [{"theta_E": 10, "center_x": 10, "center_y": 10}],
        )

        kwargs_params = {"lens_model": lens_param}
        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            self.kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
        )
        npt.assert_raises(ValueError, fittingSequence.fit_sequence, fitting_list)
        kwargs_mcmc["init_samples"] = None

        args = fittingSequence.param_class.kwargs2args(
            kwargs_lens=[{"theta_E": 1, "center_x": 0, "center_y": 0}]
        )
        kwargs_result = fittingSequence.param_class.args2kwargs(args)
        print(kwargs_result)
        print(args, "test args")
        chain_list = fittingSequence.fit_sequence(fitting_list)
        kwargs_result = fittingSequence.best_fit(bijective=False)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"], 1, decimal=2
        )

    def test_jaxopt_minimizer(self):
        # data specifics
        background_rms = 0.005  #  background noise per pixel
        exp_time = 100.0  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 60  #  cutout pixel size per axis
        pixel_scale = 0.11  #  pixel size in arcsec (area per pixel = pixel_scale**2)
        fwhm = 0.5  # full width at half maximum of PSF

        # lensing quantities
        lens_model_list = ["EPL", "SHEAR"]
        kwargs_epl = {
            "theta_E": 0.66,
            "gamma": 1.7,
            "e1": 0.07,
            "e2": -0.03,
            "center_x": 0.05,
            "center_y": 0.1,
        }  # parameters of the deflector lens model
        kwargs_shear = {
            "gamma1": 0.0,
            "gamma2": -0.05,
        }  # shear values to the source plane

        kwargs_lens = [kwargs_epl, kwargs_shear]

        lens_model_class = LensModel(lens_model_list)

        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {
            "amp": 16.0,
            "R_sersic": 0.1,
            "n_sersic": 1.0,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.1,
            "center_y": 0.0,
        }
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_source = [kwargs_sersic]

        source_model_class = LightModel(source_model_list)

        kwargs_sersic_lens = {
            "amp": 16.0,
            "R_sersic": 0.6,
            "n_sersic": 2.0,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.05,
            "center_y": 0.0,
        }

        lens_light_model_list = ["SERSIC_ELLIPSE"]
        kwargs_lens_light = [kwargs_sersic_lens]

        lens_light_model_class = LightModel(lens_light_model_list)
        # generate the coordinate grid and image properties (we only read out the relevant lines we need)
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
            util.make_grid_with_coordtransform(
                numPix=numPix,
                deltapix=pixel_scale,
                center_ra=0,
                center_dec=0,
                subgrid_res=1,
                inverse=False,
            )
        )

        kwargs_data = {
            "background_rms": background_rms,  # rms of background noise
            "exposure_time": exp_time,  # exposure time (or a map per pixel)
            "ra_at_xy_0": ra_at_xy_0,  # RA at (0,0) pixel
            "dec_at_xy_0": dec_at_xy_0,  # DEC at (0,0) pixel
            "transform_pix2angle": Mpix2coord,  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular un
            "image_data": np.zeros(
                (numPix, numPix)
            ),  # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is cre
        }

        data_class = ImageData(**kwargs_data)
        # generate the psf variables
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "pixel_size": pixel_scale,
            "fwhm": fwhm,
        }
        psf_class = PSF(**kwargs_psf_gaussian)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
            "convolution_type": "fft_static",
        }

        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
        )

        # generate image
        image_model = imageModel.image(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=None,
            point_source_add=False,
        )
        kwargs_data["image_data"] = image_model

        data_class.update_data(image_model)

        fixed_lens = []
        kwargs_lens_init = []
        kwargs_lens_sigma = []
        kwargs_lower_lens = []
        kwargs_upper_lens = []

        fixed_lens.append({})
        kwargs_lens_init.append(
            {
                "center_x": 0.1,
                "center_y": 0.2,
                "e1": 0.13,
                "e2": -0.06,
                "theta_E": 0.6,
            }
        )
        kwargs_lens_sigma.append(
            {"theta_E": 0.1, "e1": 0.1, "e2": 0.1, "center_x": 0.1, "center_y": 0.1}
        )
        kwargs_lower_lens.append(
            {
                "theta_E": 0.01,
                "e1": -0.5,
                "e2": -0.5,
                "center_x": -10.0,
                "center_y": -10.0,
            }
        )
        kwargs_upper_lens.append(
            {"theta_E": 10.0, "e1": 0.5, "e2": 0.5, "center_x": 10.0, "center_y": 10.0}
        )

        kwargs_lens_init[0]["gamma"] = 2.1
        kwargs_lens_sigma[0]["gamma"] = 0.1
        kwargs_lower_lens[0]["gamma"] = 1.5
        kwargs_upper_lens[0]["gamma"] = 2.5

        fixed_lens.append({"ra_0": 0, "dec_0": 0})
        kwargs_lens_init.append({"gamma1": 0.1, "gamma2": 0.04})
        kwargs_lens_sigma.append({"gamma1": 0.01, "gamma2": 0.01})
        kwargs_lower_lens.append({"gamma1": -0.2, "gamma2": -0.2})
        kwargs_upper_lens.append({"gamma1": 0.2, "gamma2": 0.2})

        lens_params = [
            kwargs_lens_init,
            kwargs_lens_sigma,
            fixed_lens,
            kwargs_lower_lens,
            kwargs_upper_lens,
        ]

        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []

        fixed_source.append({})
        kwargs_source_init.append(
            {
                "R_sersic": 0.2,
                "n_sersic": 2.5,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.3,
                "center_y": 0.0,
                "amp": 26.0,
            }
        )
        kwargs_source_sigma.append(
            {
                "n_sersic": 1,
                "R_sersic": 0.01,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.2,
                "center_y": 0.2,
                "amp": 10,
            }
        )
        kwargs_lower_source.append(
            {
                "e1": -0.5,
                "e2": -0.5,
                "R_sersic": 0.001,
                "n_sersic": 0.5,
                "center_x": -10.0,
                "center_y": -10.0,
                "amp": 0.0,
            }
        )
        kwargs_upper_source.append(
            {
                "e1": 0.5,
                "e2": 0.5,
                "R_sersic": 10,
                "n_sersic": 5.0,
                "center_x": 10,
                "center_y": 10,
                "amp": 100,
            }
        )

        source_params = [
            kwargs_source_init,
            kwargs_source_sigma,
            fixed_source,
            kwargs_lower_source,
            kwargs_upper_source,
        ]

        fixed_lens_light = []
        kwargs_lens_light_init = []
        kwargs_lens_light_sigma = []
        kwargs_lower_lens_light = []
        kwargs_upper_lens_light = []

        fixed_lens_light.append({})
        kwargs_lens_light_init.append(
            {
                "R_sersic": 0.5,
                "n_sersic": 2.0,
                "e1": 0.1,
                "e2": 0.3,
                "center_x": 0.1,
                "center_y": 0.0,
                "amp": 7.0,
            }
        )
        kwargs_lens_light_sigma.append(
            {
                "n_sersic": 0.01,
                "R_sersic": 0.03,
                "e1": 0.5,
                "e2": 0.5,
                "center_x": 0.01,
                "center_y": 0.01,
                "amp": 10,
            }
        )
        kwargs_lower_lens_light.append(
            {
                "e1": -0.5,
                "e2": -0.5,
                "R_sersic": 0.001,
                "n_sersic": 0.5,
                "center_x": -10.0,
                "center_y": -10.0,
                "amp": 0.0,
            }
        )
        kwargs_upper_lens_light.append(
            {
                "e1": 0.5,
                "e2": 0.5,
                "R_sersic": 10.0,
                "n_sersic": 5.0,
                "center_x": 10.0,
                "center_y": 10.0,
                "amp": 100.0,
            }
        )

        lens_light_params = [
            kwargs_lens_light_init,
            kwargs_lens_light_sigma,
            fixed_lens_light,
            kwargs_lower_lens_light,
            kwargs_upper_lens_light,
        ]

        kwargs_params = {
            "lens_model": lens_params,
            "source_model": source_params,
            "lens_light_model": lens_light_params,
        }

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        multi_band_list = [[kwargs_data, kwargs_psf_gaussian, kwargs_numerics]]

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
        }

        fitting_seq = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        # options are BFGS and TNC
        # Other options such as Nelder-Mead, Powell, CG, Newton-CG, L-BFGS-B, COBYLA,
        # SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov
        # either do not work yet or do not perform as well as BFGS and TNC
        jaxopt_kwargs = {
            "method": "BFGS",
            "maxiter": 300,
            "num_chains": 5,
            "tolerance": 1e-5,
            "sigma_scale": 1,
            "rng_int": 1,
        }
        fitting_kwargs_list_jaxopt = [["Jaxopt", jaxopt_kwargs]]
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list_jaxopt)
        fitting_type, args_history, logL_history, kwargs_result = chain_list[0]

        assert fitting_type == "Jaxopt"
        assert len(args_history) == len(logL_history)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"], 0.66, decimal=1
        )
        npt.assert_almost_equal(logL_history[-1], 0, decimal=8)


if __name__ == "__main__":
    pytest.main()
