__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
from jaxtronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.single_plane import SinglePlane as SinglePlane_ref
from jaxtronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.sis import SIS as SIS_ref

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # 64-bit floats

try:
    import fastell4py

    bool_test = True
except:
    bool_test = False


class TestLensModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.lensModel = SinglePlane(["GAUSSIAN"])
        self.lensModel_ref = SinglePlane_ref(["GAUSSIAN"])

        self.kwargs = [
            {
                "amp": 1.0,
                "sigma_x": 2.0,
                "sigma_y": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]

    def test_potential(self):
        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=self.kwargs)
        output_ref = self.lensModel_ref.potential(x=1.0, y=1.0, kwargs=self.kwargs)
        assert output == 0.77880078307140488 / (8 * jnp.pi)
        assert output == output_ref

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1.0, y=1.0, kwargs=self.kwargs)
        output1_ref, output2_ref = self.lensModel_ref.alpha(
            x=1.0, y=1.0, kwargs=self.kwargs
        )
        assert output1 == -0.19470019576785122 / (8 * jnp.pi)
        assert output2 == -0.19470019576785122 / (8 * jnp.pi)
        assert output1 == output1_ref

        assert output1 == output1_ref
        assert output2 == output2_ref

    def test_hessian(self):
        f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(
            x=1.0, y=1.0, kwargs=self.kwargs
        )

        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.lensModel_ref.hessian(
        x=1.0, y=1.0, kwargs=self.kwargs
        )

        npt.assert_almost_equal(f_xx, -0.00581, decimal=6)
        npt.assert_almost_equal(f_xy, 0.001937, decimal=6)
        npt.assert_almost_equal(f_yx, 0.001937, decimal=6)
        npt.assert_almost_equal(f_yy, -0.00581, decimal=6)

        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=6)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=6)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=6)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=6)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1.0, y=1.0, kwargs=self.kwargs)
        delta_x_ref, delta_y_ref = self.lensModel_ref.ray_shooting(
            x=1.0, y=1.0, kwargs=self.kwargs
        )

        assert delta_x == 1 + 0.19470019576785122 / (8 * jnp.pi)
        assert delta_y == 1 + 0.19470019576785122 / (8 * jnp.pi)

        assert delta_x == delta_x_ref
        assert delta_y == delta_y_ref

    def test_mass_2d(self):
        lensModel = SinglePlane(["GAUSSIAN_KAPPA"])
        lensModel_ref = SinglePlane_ref(["GAUSSIAN_KAPPA"])
        kwargs = [{"amp": 1.0, "sigma": 2.0, "center_x": 0.0, "center_y": 0.0}]
        output = lensModel.mass_2d(r=1, kwargs=kwargs)
        output_ref = lensModel_ref.mass_2d(r=1, kwargs=kwargs)
        npt.assert_almost_equal(output, 0.11750309741540453, decimal=9)
        npt.assert_almost_equal(output, output_ref, decimal=9)

    def test_density(self):
        theta_E = 1
        r = 1
        lensModel = SinglePlane(lens_model_list=["SIS"])
        lensModel_ref = SinglePlane_ref(lens_model_list=["SIS"])

        density = lensModel.density(r=r, kwargs=[{"theta_E": theta_E}])
        density_ref = lensModel_ref.density(r=r, kwargs=[{"theta_E": theta_E}])

        sis = SIS()
        sis_ref = SIS_ref()

        density_model = sis.density_lens(r=r, theta_E=theta_E)
        density_model_ref = sis_ref.density_lens(r=r, theta_E=theta_E)

        npt.assert_almost_equal(density, density_model, decimal=8)
        npt.assert_almost_equal(density, density_ref, decimal=8)
        npt.assert_almost_equal(density_model, density_model_ref, decimal=8)

    def test_bool_list(self):
        lensModel = SinglePlane(["SPEP", "SHEAR"])
        lensModel_ref = SinglePlane_ref(["SPEP", "SHEAR"])

        kwargs = [
            {
                "theta_E": 1,
                "gamma": 2,
                "e1": 0.1,
                "e2": -0.1,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.01, "gamma2": -0.02},
        ]

        alphax_1, alphay_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_ref, alphay_1_ref = lensModel_ref.alpha(1, 1, kwargs, k=0)

        alphax_1_list, alphay_1_list = lensModel.alpha(1, 1, kwargs, k=[0])
        alphax_1_list_ref, alphay_1_list_ref = lensModel_ref.alpha(1, 1, kwargs, k=[0])

        npt.assert_almost_equal(alphax_1, alphax_1_list, decimal=5)
        npt.assert_almost_equal(alphay_1, alphay_1_list, decimal=5)

        npt.assert_almost_equal(alphax_1, alphax_1_ref)
        npt.assert_almost_equal(alphay_1, alphay_1_ref)

        npt.assert_almost_equal(alphax_1_list, alphax_1_list_ref)
        npt.assert_almost_equal(alphay_1_list, alphay_1_list_ref)

        alphax_1_1, alphay_1_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_2, alphay_1_2 = lensModel.alpha(1, 1, kwargs, k=1)
        alphax_full, alphay_full = lensModel.alpha(1, 1, kwargs, k=None)

        alphax_1_1_ref, alphay_1_1_ref = lensModel_ref.alpha(1, 1, kwargs, k=0)
        alphax_1_2_ref, alphay_1_2_ref = lensModel_ref.alpha(1, 1, kwargs, k=1)
        alphax_full_ref, alphay_full_ref = lensModel_ref.alpha(1, 1, kwargs, k=None)

        npt.assert_almost_equal(
            alphax_1_1_ref + alphax_1_2_ref, alphax_full_ref, decimal=5
        )
        npt.assert_almost_equal(
            alphay_1_1_ref + alphay_1_2_ref, alphay_full_ref, decimal=5
        )

        npt.assert_almost_equal(alphax_1_1, alphax_1_1_ref)
        npt.assert_almost_equal(alphax_1_2, alphax_1_2_ref)
        npt.assert_almost_equal(alphay_1_1, alphay_1_1_ref)
        npt.assert_almost_equal(alphay_1_2, alphay_1_2_ref)
        npt.assert_almost_equal(alphax_full, alphax_full_ref)
        npt.assert_almost_equal(alphay_full, alphay_full_ref)

    def test_init(self):
        lens_model_list = [
            "TNFW",
            "TRIPLE_CHAMELEON",
            "SHEAR_GAMMA_PSI",
            "CURVED_ARC_CONST",
            "NFW_MC",
            "NFW_MC_ELLIPSE",
            "ARC_PERT",
            "MULTIPOLE",
            "CURVED_ARC_SPP",
        ]
        lensModel = SinglePlane(lens_model_list=lens_model_list)
        assert lensModel.func_list[0].param_names[0] == "Rs"


class TestRaise(unittest.TestCase):
    def test_raise(self):
        """Check whether raises occurs if fastell4py is not installed.

        :return:
        """
        if bool_test is False:
            with self.assertRaises(ImportError):
                SinglePlane(lens_model_list=["PEMD"])
            with self.assertRaises(ImportError):
                SinglePlane(lens_model_list=["SPEMD"])
        else:
            SinglePlane(lens_model_list=["PEMD", "SPEMD"])


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
