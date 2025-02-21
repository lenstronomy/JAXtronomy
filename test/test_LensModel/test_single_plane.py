__author__ = "sibirrer"

import numpy.testing as npt
import pytest
from jaxtronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.single_plane import SinglePlane as SinglePlane_ref
from jaxtronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.sis import SIS as SIS_ref
from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS

import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # 64-bit floats

try:
    import fastell4py

    bool_test = True
except:
    bool_test = False


class TestSinglePlane(object):
    """Tests the Single Plane lens model routines."""

    def setup_method(self):
        self.lensModel = SinglePlane(["EPL", "SHEAR"])
        self.lensModel_ref = SinglePlane_ref(["EPL", "SHEAR"])

        self.kwargs = [
            {
                "theta_E": 1.7,
                "e1": -0.3,
                "e2": 0.3,
                "center_x": 0.3,
                "center_y": 0.1,
                "gamma": 1.73,
            },
            {
                "gamma1": 0.1,
                "gamma2": 0.3,
            },
        ]

    def test_fermat_potential(self):
        output = self.lensModel.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs, k=0
        )
        output_ref = self.lensModel_ref.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs, k=0
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

        output = self.lensModel.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs
        )
        output_ref = self.lensModel_ref.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

        output = self.lensModel.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs, k=(0, 1)
        )
        output_ref = self.lensModel_ref.fermat_potential(
            x_image=1.0, y_image=1.0, kwargs_lens=self.kwargs, k=(0, 1)
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

        output = self.lensModel.fermat_potential(
            x_image=1.0,
            y_image=1.0,
            kwargs_lens=self.kwargs,
            x_source=3.1,
            y_source=2.7,
            k=(0, 1),
        )
        output_ref = self.lensModel_ref.fermat_potential(
            x_image=1.0,
            y_image=1.0,
            kwargs_lens=self.kwargs,
            x_source=3.1,
            y_source=2.7,
            k=(0, 1),
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

    def test_potential(self):
        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=self.kwargs, k=0)
        output_ref = self.lensModel_ref.potential(x=1.0, y=1.0, kwargs=self.kwargs, k=0)
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=self.kwargs)
        output_ref = self.lensModel_ref.potential(x=1.0, y=1.0, kwargs=self.kwargs)
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=self.kwargs, k=(0, 1))
        output_ref = self.lensModel_ref.potential(
            x=1.0, y=1.0, kwargs=self.kwargs, k=(0, 1)
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=8)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1.0, y=1.0, kwargs=self.kwargs)
        output1_ref, output2_ref = self.lensModel_ref.alpha(
            x=1.0, y=1.0, kwargs=self.kwargs
        )
        npt.assert_array_almost_equal(output1, output1_ref, decimal=8)
        npt.assert_array_almost_equal(output2, output2_ref, decimal=8)

    def test_hessian(self):
        f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(
            x=1.0, y=1.0, kwargs=self.kwargs, k=(0, 1)
        )

        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.lensModel_ref.hessian(
            x=1.0, y=1.0, kwargs=self.kwargs, k=(0, 1)
        )

        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=6)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=6)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=6)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=6)

        f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(
            x=1.0, y=1.0, kwargs=self.kwargs, k=0
        )

        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.lensModel_ref.hessian(
            x=1.0, y=1.0, kwargs=self.kwargs, k=0
        )

        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=6)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=6)
        npt.assert_almost_equal(f_yx, f_yx_ref, decimal=6)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=6)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1.0, y=1.0, kwargs=self.kwargs)
        delta_x_ref, delta_y_ref = self.lensModel_ref.ray_shooting(
            x=1.0, y=1.0, kwargs=self.kwargs
        )

        npt.assert_array_almost_equal(delta_x, delta_x_ref, decimal=8)
        npt.assert_array_almost_equal(delta_y, delta_y_ref, decimal=8)

    def test_mass_2d(self):
        lensModel = SinglePlane(["SIE"])
        lensModel_ref = SinglePlane_ref(["SIE"])

        kwargs = [
            {
                "theta_E": 1.7,
                "e1": -0.3,
                "e2": 0.3,
                "center_x": 0.3,
                "center_y": 0.1,
            }
        ]
        output = lensModel.mass_2d(r=1, kwargs=kwargs)
        output_ref = lensModel_ref.mass_2d(r=1, kwargs=kwargs)
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
        lensModel = SinglePlane(["EPL", "SHEAR"])
        lensModel_ref = SinglePlane_ref(["EPL", "SHEAR"])

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

        alphax_1_list, alphay_1_list = lensModel.alpha(1, 1, kwargs, k=(0, 1))
        alphax_1_list_ref, alphay_1_list_ref = lensModel_ref.alpha(
            1, 1, kwargs, k=(0, 1)
        )

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
            "NFW_MC_ELLIPSE_POTENTIAL",
            "ARC_PERT",
            "MULTIPOLE",
            "CURVED_ARC_SPP",
        ]
        npt.assert_raises(ValueError, SinglePlane, lens_model_list)

    def test_profile_list_base(self):
        # this tests the giant elif statement in profile_list_base

        lensModel = SinglePlane(lens_model_list=_JAXXED_MODELS)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        """Check whether raises occurs if fastell4py is not installed.

        :return:
        """
        # NOTE: These profiles are not yet implemented in JAXtronomy
        # if bool_test is False:
        #     with self.assertRaises(ImportError):
        #         SinglePlane(lens_model_list=["PEMD"])
        #     with self.assertRaises(ImportError):
        #         SinglePlane(lens_model_list=["SPEMD"])
        # else:
        #     SinglePlane(lens_model_list=["PEMD", "SPEMD"])

        with self.assertRaises(ValueError):
            SinglePlane(lens_model_list=["MY_FAKE_PROFILE"])


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
