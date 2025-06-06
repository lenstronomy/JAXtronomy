import jax

jax.config.update("jax_enable_x64", True)

import numpy as np, numpy.testing as npt

from jaxtronomy.LightModel.linear_basis import LinearBasis
from lenstronomy.LightModel.linear_basis import LinearBasis as LinearBasis_ref


class TestLinearBasis(object):
    def setup_method(self):
        n_max = 3

        self.linear_basis = LinearBasis(
            light_model_list=["SERSIC", "MULTI_GAUSSIAN", "SHAPELETS"],
            profile_kwargs_list=[{}, {}, {"n_max": n_max}],
        )
        self.linear_basis_ref = LinearBasis_ref(
            light_model_list=["SERSIC", "MULTI_GAUSSIAN", "SHAPELETS"]
        )

        kwargs_sersic = {
            "R_sersic": 0.7,
            "n_sersic": 5,
            "center_x": -0.1,
            "center_y": 0.3,
        }
        kwargs_multi_gaussian = {
            "sigma": [2, 3, 3],
            "center_x": -1.1,
            "center_y": 0.3,
        }
        kwargs_shapelets = {
            "n_max": n_max,
            "beta": 4.01,
            "center_x": -0.1,
            "center_y": -0.2,
        }
        self.kwargs_list = [kwargs_sersic, kwargs_multi_gaussian, kwargs_shapelets]

    def test_functions_split(self):
        x = np.tile(np.linspace(-5, 5, 50), 50)
        y = np.repeat(np.linspace(-5, 5, 50), 50)
        response, n = self.linear_basis.functions_split(x, y, self.kwargs_list)
        response_ref, n_ref = self.linear_basis_ref.functions_split(
            x, y, self.kwargs_list
        )
        npt.assert_allclose(response, response_ref, atol=1e-12, rtol=1e-12)
        assert n == n_ref == 14

    def test_num_param_linear(self):
        param_list = self.linear_basis.num_param_linear_list(self.kwargs_list)
        param_list_ref = self.linear_basis_ref.num_param_linear_list(self.kwargs_list)
        npt.assert_array_equal(param_list, param_list_ref)

        assert (
            self.linear_basis.num_param_linear(self.kwargs_list)
            == self.linear_basis_ref.num_param_linear(self.kwargs_list)
            == 14
        )

        param_list = self.linear_basis.num_param_linear(self.kwargs_list, list_return=True)
        param_list_ref = self.linear_basis_ref.num_param_linear(self.kwargs_list, list_return=True)
        npt.assert_array_equal(param_list, param_list_ref)

    def test_update_linear(self):
        param = np.arange(1, 15, 1)
        i = 0
        kwargs_list, i_new = self.linear_basis.update_linear(param, i, self.kwargs_list)
        kwargs_list_ref, i_new_ref = self.linear_basis_ref.update_linear(
            param, i, self.kwargs_list
        )
        assert i_new == i_new_ref == 14
        npt.assert_array_equal(kwargs_list[0]["amp"], kwargs_list_ref[0]["amp"])
        npt.assert_array_equal(kwargs_list[1]["amp"], kwargs_list_ref[1]["amp"])
        npt.assert_array_equal(kwargs_list[2]["amp"], kwargs_list_ref[2]["amp"])

    def test_add_fixed_linear(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        kwargs_list = [{"amp": 0.5}, {}]
        kwargs_fixed_list = linear_basis.add_fixed_linear(kwargs_list)
        assert kwargs_fixed_list[1]["amp"] == 1

    def test_linear_param_from_kwargs(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        kwargs_list = [{"amp": 0.5}, {"amp": -1}]
        param = linear_basis.linear_param_from_kwargs(kwargs_list)
        assert param[0] == kwargs_list[0]["amp"]
        assert param[1] == kwargs_list[1]["amp"]

    def test_check_positive_flux_profile(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        linear_basis_ref = LinearBasis_ref(
            light_model_list=["SERSIC", "SERSIC_ELLIPSE"]
        )
        kwargs_list = [{"amp": 0.5}, {"amp": -1}]
        pos_bool = linear_basis.check_positive_flux_profile(kwargs_list)
        pos_bool_ref = linear_basis_ref.check_positive_flux_profile(kwargs_list)
        assert pos_bool == pos_bool_ref
        assert pos_bool == False

        kwargs_list = [{"amp": 0.5}, {"amp": 1}]
        pos_bool = linear_basis.check_positive_flux_profile(kwargs_list)
        pos_bool_ref = linear_basis_ref.check_positive_flux_profile(kwargs_list)
        assert pos_bool == pos_bool_ref
        assert pos_bool == True
