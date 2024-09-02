from jaxtronomy.LightModel.linear_basis import LinearBasis


class TestLinearBasis(object):
    def setup_method(self):
        pass

    def test_linear_param_from_kwargs(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        kwargs_list = [{"amp": 0.5}, {"amp": -1}]
        param = linear_basis.linear_param_from_kwargs(kwargs_list)
        assert param[0] == kwargs_list[0]["amp"]
        assert param[1] == kwargs_list[1]["amp"]

    def test_add_fixed_linear(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        kwargs_list = [{"amp": 0.5}, {}]
        kwargs_fixed_list = linear_basis.add_fixed_linear(kwargs_list)
        assert kwargs_fixed_list[1]["amp"] == 1

    def test_check_positive_flux_profile(self):
        linear_basis = LinearBasis(light_model_list=["SERSIC", "SERSIC_ELLIPSE"])
        kwargs_list = [{"amp": 0.5}, {"amp": -1}]
        pos_bool = linear_basis.check_positive_flux_profile(kwargs_list)
        assert pos_bool == False

        kwargs_list = [{"amp": 0.5}, {"amp": 1}]
        pos_bool = linear_basis.check_positive_flux_profile(kwargs_list)
        assert pos_bool == True
