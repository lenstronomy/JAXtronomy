__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
from jaxtronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref

from lenstronomy.Util.util import make_grid
import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # 64-bit floats


class TestLensModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.kwargs = [
            {
                "theta_E": 1.0,
                "gamma": 2,
                "center_x": 0.0,
                "center_y": 0.0,
                "e1": 0,
                "e2": 0,
            }
        ]
        self.lensModel = LensModel(["EPL"])
        self.lensModel_ref = LensModel_ref(["EPL"])
        self.x = np.array([-1.5, -0.3, 1.1, 1.3, 2.7])
        self.y = np.array([-1.1, -0.6, 0.7, 1.2, 1.9])

    def test_init(self):
        lens_model_list = [  # NH: removing non-jaxxed profiles for now
            "PJAFFE",
            "PJAFFE_ELLIPSE_POTENTIAL",
        ]

        lensModel = LensModel(lens_model_list)
        assert len(lensModel.lens_model_list) == len(lens_model_list)

        lens_model_list = ["NFW"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        x, y = 0.2, 1
        kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
        value = lensModel.potential(self.x, self.y, kwargs)
        value_ref = lensModel_ref.potential(self.x, self.y, kwargs)
        npt.assert_array_almost_equal(value, value_ref, decimal=6)

        lens_model_list = ["SIS", "SIS"]
        lensModel = LensModel(lens_model_list, decouple_multi_plane=True)

        lens_model_list = ["SIS", "LOS"]
        lensModel = LensModel(lens_model_list)

        lensModel.info()

    def test_check_parameters(self):
        lens_model = LensModel(lens_model_list=["SIS"])
        # check_parameters
        kwargs_list = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        lens_model.check_parameters(kwargs_list)
        kwargs_list_add = [
            {"theta_E": 1.0, "center_x": 0, "center_y": 0, "not_a_parameter": 1}
        ]
        kwargs_list_remove = [{"center_x": 0, "center_y": 0}]
        kwargs_list_too_long = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}, {}]
        npt.assert_raises(ValueError, lens_model.check_parameters, kwargs_list_add)
        npt.assert_raises(ValueError, lens_model.check_parameters, kwargs_list_remove)
        npt.assert_raises(ValueError, lens_model.check_parameters, kwargs_list_too_long)

    def test_kappa(self):
        lensModel = LensModel(lens_model_list=["CONVERGENCE"])
        lensModel_ref = LensModel_ref(lens_model_list=["CONVERGENCE"])
        kappa_ext = 0.5
        kwargs = [{"kappa": kappa_ext}]
        output = lensModel.kappa(self.x, self.y, kwargs=kwargs)
        output_ref = lensModel_ref.kappa(self.x, self.y, kwargs=kwargs)
        npt.assert_array_almost_equal(
            output, np.ones_like(output) * kappa_ext, decimal=6
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=6)

    def test_potential(self):
        output = self.lensModel.potential(self.x, self.y, kwargs=self.kwargs)
        output_ref = self.lensModel_ref.potential(self.x, self.y, kwargs=self.kwargs)
        npt.assert_array_almost_equal(output, output_ref, decimal=6)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(self.x, self.y, kwargs=self.kwargs)
        output1_ref, output2_ref = self.lensModel_ref.alpha(
            self.x, self.y, kwargs=self.kwargs
        )
        npt.assert_array_almost_equal(output1, output1_ref, decimal=6)
        npt.assert_array_almost_equal(output2, output2_ref, decimal=6)

        output1_diff, output2_diff = self.lensModel.alpha(
            self.x, self.y, kwargs=self.kwargs, diff=0.00001
        )
        npt.assert_array_almost_equal(output1_diff, output1, decimal=5)
        npt.assert_array_almost_equal(output2_diff, output2, decimal=5)

    def test_gamma(self):
        lensModel = LensModel(lens_model_list=["SHEAR"])
        lensModel_ref = LensModel_ref(lens_model_list=["SHEAR"])
        gamma1, gamma2 = 0.1, -0.1
        kwargs = [{"gamma1": gamma1, "gamma2": gamma2}]
        e1_out, e2_out = lensModel.gamma(self.x, self.y, kwargs=kwargs)
        e1_out_ref, e2_out_ref = lensModel_ref.gamma(self.x, self.y, kwargs=kwargs)
        npt.assert_array_almost_equal(e1_out, np.ones_like(e1_out) * gamma1, decimal=6)
        npt.assert_array_almost_equal(e2_out, np.ones_like(e2_out) * gamma2, decimal=6)
        npt.assert_array_almost_equal(e1_out, e1_out_ref, decimal=6)
        npt.assert_array_almost_equal(e2_out, e2_out_ref, decimal=6)

    def test_magnification(self):
        output = self.lensModel.magnification(self.x, self.y, kwargs=self.kwargs)
        output_ref = self.lensModel_ref.magnification(
            self.x, self.y, kwargs=self.kwargs
        )
        npt.assert_array_almost_equal(output, output_ref, decimal=6)

    def test_flexion(self):
        f_xxx, f_xxy, f_xyy, f_yyy = self.lensModel.flexion(
            self.x, self.y, kwargs=self.kwargs, hessian_diff=True
        )
        f_xxx_ref, f_xxy_ref, f_xyy_ref, f_yyy_ref = self.lensModel_ref.flexion(
            self.x, self.y, kwargs=self.kwargs, hessian_diff=True
        )
        npt.assert_array_almost_equal(f_xxx, f_xxx_ref, decimal=6)
        npt.assert_array_almost_equal(f_xxy, f_xxy_ref, decimal=6)
        npt.assert_array_almost_equal(f_xyy, f_xyy_ref, decimal=6)
        npt.assert_array_almost_equal(f_yyy, f_yyy_ref, decimal=6)

        f_xxx, f_xxy, f_xyy, f_yyy = self.lensModel.flexion(
            self.x, self.y, kwargs=self.kwargs, hessian_diff=False
        )
        f_xxx_ref, f_xxy_ref, f_xyy_ref, f_yyy_ref = self.lensModel.flexion(
            self.x, self.y, kwargs=self.kwargs, hessian_diff=False
        )
        npt.assert_array_almost_equal(f_xxx, f_xxx_ref, decimal=6)
        npt.assert_array_almost_equal(f_xxy, f_xxy_ref, decimal=6)
        npt.assert_array_almost_equal(f_xyy, f_xyy_ref, decimal=6)
        npt.assert_array_almost_equal(f_yyy, f_yyy_ref, decimal=6)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(
            self.x, self.y, kwargs=self.kwargs
        )
        delta_x_ref, delta_y_ref = self.lensModel_ref.ray_shooting(
            self.x, self.y, kwargs=self.kwargs
        )
        npt.assert_array_almost_equal(delta_x, delta_x_ref, decimal=6)
        npt.assert_array_almost_equal(delta_y, delta_y_ref, decimal=6)

    # def test_arrival_time(self):
    #    z_lens = 0.5
    #    z_source = 1.5
    #    x_image, y_image = 1.0, 0.0
    #    lensModel_mp = LensModel(
    #        lens_model_list=["SIS"],
    #        multi_plane=True,
    #        lens_redshift_list=[z_lens],
    #        z_source=z_source,
    #    )
    #    lensModel_mp_ref = LensModel_ref(
    #        lens_model_list=["SIS"],
    #        multi_plane=True,
    #        lens_redshift_list=[z_lens],
    #        z_source=z_source,
    #    )
    #    kwargs = [
    #        {
    #            "theta_E": 1.0,
    #            "center_x": 0.0,
    #            "center_y": 0.0,
    #        }
    #    ]
    #    arrival_time_mp = lensModel_mp.arrival_time(x_image, y_image, kwargs)
    #    arrival_time_mp_ref = lensModel_mp_ref.arrival_time(x_image, y_image, kwargs)
    #    lensModel_sp = LensModel(
    #        lens_model_list=["SIS"], z_source=z_source, z_lens=z_lens
    #    )
    #    lensModel_sp_ref = LensModel_ref(
    #        lens_model_list=["SIS"], z_source=z_source, z_lens=z_lens
    #    )
    #    arrival_time_sp = lensModel_sp.arrival_time(x_image, y_image, kwargs)
    #    arrival_time_sp_ref = lensModel_sp_ref.arrival_time(x_image, y_image, kwargs)
    #    npt.assert_array_almost_equal(arrival_time_sp, arrival_time_mp, decimal=8)
    #    npt.assert_array_almost_equal(arrival_time_sp, arrival_time_sp_ref, decimal=6)
    #    npt.assert_array_almost_equal(arrival_time_mp, arrival_time_mp_ref, decimal=6)

    def test_fermat_potential(self):
        z_lens = 0.5
        z_source = 1.5
        x_image, y_image = 1.032, -2.0234
        lensModel = LensModel(
            lens_model_list=["SIS"],
            # multi_plane=True,
            # lens_redshift_list=[z_lens],
            # z_lens=z_lens,
            # z_source=z_source,
        )
        lensModel_ref = LensModel_ref(
            lens_model_list=["SIS"],
            # multi_plane=True,
            # lens_redshift_list=[z_lens],
            # z_lens=z_lens,
            # z_source=z_source,
        )
        kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
        fermat_pot = lensModel.fermat_potential(x_image, y_image, kwargs)
        fermat_pot_ref = lensModel_ref.fermat_potential(x_image, y_image, kwargs)
        # arrival_time = lensModel.arrival_time(x_image, y_image, kwargs)
        # arrival_time_from_fermat_pot = lensModel._lensCosmo.time_delay_units(fermat_pot)
        npt.assert_allclose(fermat_pot, fermat_pot_ref, rtol=1e-10, atol=1e-10)

    def test_curl(self):
        # z_lens_list = [0.2, 0.8]
        # z_source = 1.5
        lensModel = LensModel(
            lens_model_list=["SIS", "SIS"],
            # multi_plane=True,
            # lens_redshift_list=z_lens_list,
            # z_source=z_source,
        )
        lensModel_ref = LensModel_ref(
            lens_model_list=["SIS", "SIS"],
            # multi_plane=True,
            # lens_redshift_list=z_lens_list,
            # z_source=z_source,
        )
        kwargs = [
            {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0},
            {"theta_E": 0.1, "center_x": 0.0, "center_y": 0.2},
        ]
        curl = lensModel.curl(x=1.2438, y=1.37485, kwargs=kwargs)
        curl_ref = lensModel_ref.curl(x=1.2438, y=1.37485, kwargs=kwargs)
        npt.assert_allclose(curl, curl_ref, rtol=1e-10, atol=1e-10)

        kwargs = [
            {"theta_E": 1.35, "center_x": -0.348, "center_y": 0.102},
            {"theta_E": 1.23, "center_x": 0.329, "center_y": 0.2},
        ]
        curl = lensModel.curl(x=1.2438, y=1.37485, kwargs=kwargs)
        curl_ref = lensModel_ref.curl(x=1.2438, y=1.37485, kwargs=kwargs)
        npt.assert_allclose(curl, curl_ref, rtol=1e-10, atol=1e-10)

    def test_hessian_differentials(self):
        """Routine to test the private numerical differentials, both cross and square
        methods in the infinitesimal regime."""
        lens_model = LensModel(lens_model_list=["SIS"])
        kwargs = [{"theta_E": 1, "center_x": 0.01, "center_y": 0}]
        x, y = make_grid(numPix=10, deltapix=0.2)
        diff = 0.0000001
        f_xx_sq, f_xy_sq, f_yx_sq, f_yy_sq = lens_model.hessian(
            x, y, kwargs, diff=diff, diff_method="square"
        )
        f_xx_cr, f_xy_cr, f_yx_cr, f_yy_cr = lens_model.hessian(
            x, y, kwargs, diff=diff, diff_method="cross"
        )
        f_xx, f_xy, f_yx, f_yy = lens_model.hessian(x, y, kwargs, diff=None)
        npt.assert_array_almost_equal(f_xx_cr, f_xx, decimal=5)
        npt.assert_array_almost_equal(f_xy_cr, f_xy, decimal=5)
        npt.assert_array_almost_equal(f_yx_cr, f_yx, decimal=5)
        npt.assert_array_almost_equal(f_yy_cr, f_yy, decimal=5)

        npt.assert_array_almost_equal(f_xx_sq, f_xx, decimal=5)
        npt.assert_array_almost_equal(f_xy_sq, f_xy, decimal=5)
        npt.assert_array_almost_equal(f_yx_sq, f_yx, decimal=5)
        npt.assert_array_almost_equal(f_yy_sq, f_yy, decimal=5)

    # def test_hessian_z1z2(self):
    #    z_source = 1.5
    #    lens_model_list = ["SIS"]
    #    kwargs_lens = [{"theta_E": 1}]
    #    redshift_list = [0.5]
    #    lensModel = LensModel(
    #        lens_model_list=lens_model_list,
    #        multi_plane=True,
    #        lens_redshift_list=redshift_list,
    #        z_source=z_source,
    #    )
    #    z1, z2 = 0.5, 1.5
    #    theta_x, theta_y = jnp.linspace(start=-1, stop=1, num=10), jnp.linspace(
    #        start=-1, stop=1, num=10
    #    )

    #    f_xx, f_xy, f_yx, f_yy = lensModel.hessian_z1z2(
    #        z1, z2, theta_x, theta_y, kwargs_lens
    #    )
    #    # Use the method in multi_plane.hessian_z1z2 as a comparison
    #    multi_plane = MultiPlane(
    #        z_source=1.5,
    #        lens_model_list=lens_model_list,
    #        lens_redshift_list=redshift_list,
    #        z_interp_stop=3,
    #        cosmo_interp=False,
    #    )
    #    (
    #        f_xx_expected,
    #        f_xy_expected,
    #        f_yx_expected,
    #        f_yy_expected,
    #    ) = multi_plane.hessian_z1z2(
    #        z1=z1, z2=z2, theta_x=theta_x, theta_y=theta_y, kwargs_lens=kwargs_lens
    #    )
    #    npt.assert_array_almost_equal(f_xx, f_xx_expected, decimal=5)
    #    npt.assert_array_almost_equal(f_xy, f_xy_expected, decimal=5)
    #    npt.assert_array_almost_equal(f_yx, f_yx_expected, decimal=5)
    #    npt.assert_array_almost_equal(f_yy, f_yy_expected, decimal=5)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
            lensModel = LensModel(
                ["NFW"], multi_plane=True, lens_redshift_list=[1], z_source=2
            )
            f_x, f_y = lensModel.alpha(1, 1, kwargs, diff=0.0001)
        with self.assertRaises(ValueError):
            lensModel = LensModel(["NFW"], multi_plane=True, lens_redshift_list=[1])
        # with self.assertRaises(ValueError):
        #    kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
        #    lensModel = LensModel(["NFW"], multi_plane=False)
        #    t_arrival = lensModel.arrival_time(1, 1, kwargs)
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            x_image, y_image = 1.0, 0.0
            lensModel = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                lens_redshift_list=[z_lens],
                z_source=z_source,
            )
            kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
            fermat_pot = lensModel.fermat_potential(x_image, y_image, kwargs)
        with self.assertRaises(ValueError):
            lens_model = LensModel(lens_model_list=["SIS"])
            kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
            lens_model.hessian(0, 0, kwargs, diff=0.001, diff_method="bad")

        with self.assertRaises(ValueError):
            lens_model = LensModel(lens_model_list=["LOS", "LOS_MINIMAL"])
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["EPL", "NFW"], multi_plane=True, z_source=1.0
            )
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["EPL", "NFW"],
                multi_plane=True,
                lens_redshift_list=[0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["LOS", "EPL", "NFW"],
                multi_plane=True,
                z_source=1.0,
                lens_redshift_list=[0.5, 0.5, 0.5],
            )
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["LOS_MINIMAL", "SIS", "GAUSSIAN"],
                multi_plane=True,
                z_source=1.0,
                lens_redshift_list=[0.5, 0.5, 0.5],
            )

    # def test_hessian_z1z2_raise(self):
    #    lensModel = LensModel(
    #        lens_model_list=["SIS"],
    #        multi_plane=True,
    #        lens_redshift_list=[1],
    #        z_source=2,
    #    )
    #    kwargs = [{"theta_E": 1, "center_x": 0, "center_y": 0}]

    #    # Test when the model is not in multi-plane mode
    #    lensModel_non_multi = LensModel(
    #        lens_model_list=["SIS"],
    #        multi_plane=False,
    #        lens_redshift_list=[1],
    #        z_source=2,
    #    )
    #    with self.assertRaises(ValueError):
    #        lensModel_non_multi.hessian_z1z2(0.5, 1.5, 1, 1, kwargs)

    #    # Test when z1 >= z2
    #    with self.assertRaises(ValueError):
    #        lensModel.hessian_z1z2(1.5, 1.5, 1, 1, kwargs)
    #    with self.assertRaises(ValueError):
    #        lensModel.hessian_z1z2(2.0, 1.5, 1, 1, kwargs)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
