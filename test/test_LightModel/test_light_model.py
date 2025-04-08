__author__ = "sibirrer"

from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.LightModel.light_model_base import _JAXXED_MODELS
from lenstronomy.LightModel.light_model import LightModel as LightModel_ref

import numpy as np
import numpy.testing as npt
import pytest
import unittest


class TestLightModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.light_model_list = [
            "SERSIC",
            "SERSIC_ELLIPSE",
            "CORE_SERSIC",
        ]
        e1, e2 = 0.1, -0.3
        self.kwargs = [
            {
                "amp": 1,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC_ELLIPSE'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "Rb": 0.1,
                "gamma": 2.0,
                "n_sersic": 1,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            },
            # 'CORE_SERSIC'
        ]

        self.lightModel = LightModel(light_model_list=self.light_model_list)
        self.lightModel_ref = LightModel_ref(
            light_model_list=self.light_model_list,
        )
        test_sersic_ellipse_qphi = LightModel(["SERSIC_ELLIPSE_Q_PHI"])

    def test_import_profiles(self):
        lightModel = LightModel(light_model_list=_JAXXED_MODELS)

    def test_check_parameters(self):
        lightModel = LightModel(light_model_list=["GAUSSIAN"])
        kwargs_list = [{"amp": 0, "sigma": 1, "center_x": 0, "center_y": 0}]

        lightModel.check_parameters(kwargs_list)
        kwargs_list_add = [
            {"amp": 0, "sigma": 1, "center_x": 0, "center_y": 0, "bad": 1}
        ]
        kwargs_list_remove = [{"amp": 0, "center_x": 0, "center_y": 0}]
        kwargs_list_too_long = [
            {"amp": 0, "sigma": 1, "center_x": 0, "center_y": 0},
            {},
        ]

        npt.assert_raises(ValueError, lightModel.check_parameters, kwargs_list_add)
        npt.assert_raises(ValueError, lightModel.check_parameters, kwargs_list_remove)
        npt.assert_raises(ValueError, lightModel.check_parameters, kwargs_list_too_long)

    def test_surface_brightness(self):
        x = 1.0
        y = 1.3
        output = self.lightModel.surface_brightness(x, y, self.kwargs)
        output_ref = self.lightModel_ref.surface_brightness(x, y, self.kwargs)
        npt.assert_almost_equal(output, output_ref, decimal=6)

    def test_surface_brightness_array(self):
        x = [1, 3]
        y = [2, 1.34]
        output = self.lightModel.surface_brightness(x, y, self.kwargs)
        output_ref = self.lightModel_ref.surface_brightness(x, y, self.kwargs)
        npt.assert_array_almost_equal(output, output_ref, decimal=6)

    def test_functions_split(self):
        output = self.lightModel.functions_split(x=1.0, y=1.0, kwargs_list=self.kwargs)
        output_ref = self.lightModel_ref.functions_split(
            x=1.0, y=1.0, kwargs_list=self.kwargs
        )
        npt.assert_array_almost_equal(output[0], output_ref[0], decimal=6)
        assert output[1] == output_ref[1]

    def test_param_name_list(self):
        param_name_list = self.lightModel.param_name_list
        assert len(self.light_model_list) == len(param_name_list)

    def test_param_name_list_latex(self):
        param_name_list = self.lightModel.param_name_list_latex
        assert len(self.light_model_list) == len(param_name_list)

    def test_num_param_linear(self):
        num = self.lightModel.num_param_linear(self.kwargs, list_return=False)
        assert num == 3

        num_list = self.lightModel.num_param_linear(self.kwargs, list_return=True)
        assert num_list == [1, 1, 1]

    def test_update_linear(self):
        response, n = self.lightModel.functions_split(1, 1, self.kwargs)
        param = np.ones(n) * 2
        kwargs_out, i = self.lightModel.update_linear(
            param, i=0, kwargs_list=self.kwargs
        )
        assert i == n
        assert kwargs_out[0]["amp"] == 2

    def test_total_flux(self):
        light_model_list = ["SERSIC", "SERSIC_ELLIPSE", "MULTI_GAUSSIAN"]
        kwargs_list = [
            {
                "amp": 1.1234,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC'
            {
                "amp": 1.345,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "e1": 0.1,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC_ELLIPSE'
            {
                "amp": [1.3894, 32.298324, 21.23498],
                "sigma": [0.5, 1.5, 2],
                "center_x": 0.234,
                "center_y": -1.98342,
            },  # MULTI_GAUSSIAN
        ]
        lightModel = LightModel(light_model_list=light_model_list)
        lightModel_ref = LightModel_ref(light_model_list=light_model_list)
        total_flux_list = lightModel.total_flux(kwargs_list)
        total_flux_list_ref = lightModel_ref.total_flux(kwargs_list)
        npt.assert_array_almost_equal(
            np.array(total_flux_list), np.array(total_flux_list_ref), decimal=5
        )

        lightModel = LightModel(light_model_list=light_model_list)
        lightModel_ref = LightModel_ref(light_model_list=light_model_list)
        total_flux_list = lightModel.total_flux(kwargs_list, norm=True)
        total_flux_list_ref = lightModel_ref.total_flux(kwargs_list, norm=True)
        npt.assert_array_almost_equal(
            np.array(total_flux_list), np.array(total_flux_list_ref)
        )


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["WRONG"])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["SERSIC"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.functions_split(x=0, y=0, kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["SERSIC"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.num_param_linear(kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["SERSIC"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.update_linear(param=[1], i=0, kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["SERSIC"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.total_flux(kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lightmodel = LightModel(
                light_model_list=["SERSIC"], source_redshift_list=[1, 3]
            )


if __name__ == "__main__":
    pytest.main()
