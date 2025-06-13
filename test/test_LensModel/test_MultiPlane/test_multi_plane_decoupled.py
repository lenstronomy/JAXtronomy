__author__ = "dangilman"

import copy

from jax import config

config.update("jax_enable_x64", True)
import numpy.testing as npt
from jaxtronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_lens_model as setup_lens_model_ref,
    setup_grids as setup_grids_ref,
    coordinates_and_deflections as coordinates_and_deflections_ref,
    setup_raytracing_lensmodels as setup_raytracing_lensmodels_ref,
)
from jaxtronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_lens_model,
    setup_grids,
    coordinates_and_deflections,
    setup_raytracing_lensmodels,
    decoupled_multiplane_class_setup,
)
from copy import deepcopy
import numpy as np
import pytest


def test_multiplane_decoupled_util_imports():
    assert setup_lens_model == setup_lens_model_ref
    assert setup_grids == setup_grids_ref
    assert coordinates_and_deflections == coordinates_and_deflections_ref
    assert setup_raytracing_lensmodels == setup_raytracing_lensmodels_ref


class TestMultiPlaneDecoupled(object):

    def setup_method(self):
        self.zlens = 0.5
        self.z_source = 2.0
        self.kwargs_lens_true = [
            {
                "theta_E": 0.7,
                "center_x": 0.0,
                "center_y": -0.0,
                "e1": 0.2,
                "e2": -0.1,
                "gamma": 2.0,
            },
            {"theta_E": 0.2, "center_x": 0.0, "center_y": -0.4},
            {"theta_E": 0.2, "center_x": 0.6, "center_y": 0.3},
            {"theta_E": 0.15, "center_x": -0.6, "center_y": -1.0},
            {"gamma1": 0.1, "gamma2": -0.2},
        ]
        self.lens_model_list = ["EPL", "SIS", "SIS", "SIS", "SHEAR"]
        self.index_lens_split = [0, 1]
        self.lens_redshift_list = [self.zlens, self.zlens, 0.25, 1.0, 1.5]
        self.lens_model_true = LensModel_ref(
            self.lens_model_list,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source=self.z_source,
        )
        self.cosmo = self.lens_model_true.cosmo

        (
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.z_source,
            self.z_split,
            self.cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, self.index_lens_split
        )

        self.Td = self.cosmo_bkg.T_xy(0, self.zlens)
        self.Ts = self.cosmo_bkg.T_xy(0, self.z_source)
        self.Tds = self.cosmo_bkg.T_xy(self.zlens, self.z_source)
        self.reduced_to_phys = self.cosmo_bkg.d_xy(
            0, self.z_source
        ) / self.cosmo_bkg.d_xy(self.zlens, self.z_source)

        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES
        self.x_image = np.array([1.0, 0.5, -0.5, -1.0])
        self.y_image = np.array([-0.4, 0.5, 1.0, 0.7])
        self.x_point = self.x_image[1]
        self.y_point = self.y_image[1]
        self._setup_point()
        self._setup_grid()
        # self._setup_multiple_images()

    def _setup_point(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES

        (
            self.x0_point,
            self.y0_point,
            self.alphax_foreground_point,
            self.alphay_foreground_point,
            self.alphax_background_point,
            self.alphay_background_point,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.x_image[1],
            self.y_image[1],
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_point = decoupled_multiplane_class_setup(
            self.lens_model_free,
            self.x0_point,
            self.y0_point,
            self.alphax_foreground_point,
            self.alphay_foreground_point,
            self.alphax_background_point,
            self.alphay_background_point,
            self.z_split,
            coordinate_type="POINT",
        )

    def _setup_grid(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES
        grid_size = 2.5
        grid_resolution = 0.005
        self.grid_x, self.grid_y, self.interp_points_grid, self.npix_grid = setup_grids(
            grid_size, grid_resolution, 0.0, 0.0
        )
        (
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.grid_x,
            self.grid_y,
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_grid = decoupled_multiplane_class_setup(
            self.lens_model_free,
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
            self.z_split,
            coordinate_type="GRID",
            interp_points=self.interp_points_grid,
        )

    def _setup_multiple_images(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES

        (
            self.x0_MI,
            self.y0_MI,
            self.alphax_foreground_MI,
            self.alphay_foreground_MI,
            self.alphax_background_MI,
            self.alphay_background_MI,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.x_image,
            self.y_image,
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_MI = decoupled_multiplane_class_setup(
            self.lens_model_free,
            self.x0_MI,
            self.y0_MI,
            self.alphax_foreground_MI,
            self.alphay_foreground_MI,
            self.alphax_background_MI,
            self.alphay_background_MI,
            self.z_split,
            coordinate_type="MULTIPLE_IMAGES",
            x_image=self.x_image,
            y_image=self.y_image,
        )

    def test_raises(self):
        npt.assert_raises(
            ValueError,
            decoupled_multiplane_class_setup,
            self.lens_model_free,
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
            self.z_split,
            coordinate_type="GRID",
            interp_points=self.interp_points_grid,
            bounds_error=True,
        )
        npt.assert_raises(
            Exception,
            decoupled_multiplane_class_setup,
            self.lens_model_free,
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
            self.z_split,
            coordinate_type="invalid",
            interp_points=self.interp_points_grid,
        )

    def test_point_deflection_model(self):
        lens_model_decoupled = LensModel(**self.kwargs_multiplane_model_point)
        lens_model_decoupled_ref = LensModel_ref(**self.kwargs_multiplane_model_point)
        beta_x, beta_y = lens_model_decoupled.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        beta_x_ref, beta_y_ref = lens_model_decoupled_ref.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(beta_x, beta_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(beta_y, beta_y_ref, atol=1e-12, rtol=1e-12)

        alpha_x, alpha_y = lens_model_decoupled.alpha(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        alpha_x_ref, alpha_y_ref = lens_model_decoupled_ref.alpha(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(alpha_x, alpha_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(alpha_y, alpha_y_ref, atol=1e-12, rtol=1e-12)

        f_xx, f_xy, f_yx, f_yy = lens_model_decoupled.hessian(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = lens_model_decoupled_ref.hessian(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_xy, f_xy_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_yx, f_yx_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_yy, f_yy_ref, atol=5e-7, rtol=5e-7)

        npt.assert_raises(Exception, lens_model_decoupled.lens_model.geo_shapiro_delay)
        npt.assert_raises(
            Exception, lens_model_decoupled.lens_model.ray_shooting_partial_comoving
        )

    def test_grid_deflection_model(self):
        lens_model_decoupled = LensModel(**self.kwargs_multiplane_model_grid)
        lens_model_decoupled_ref = LensModel_ref(**self.kwargs_multiplane_model_grid)
        beta_x, beta_y = lens_model_decoupled.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        beta_x_ref, beta_y_ref = lens_model_decoupled_ref.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(beta_x, beta_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(beta_y, beta_y_ref, atol=1e-12, rtol=1e-12)

        alpha_x, alpha_y = lens_model_decoupled.alpha(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        alpha_x_ref, alpha_y_ref = lens_model_decoupled_ref.alpha(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(alpha_x, alpha_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(alpha_y, alpha_y_ref, atol=1e-12, rtol=1e-12)

        f_xx, f_xy, f_yx, f_yy = lens_model_decoupled.hessian(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = lens_model_decoupled_ref.hessian(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_xy, f_xy_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_yx, f_yx_ref, atol=5e-7, rtol=5e-7)
        npt.assert_allclose(f_yy, f_yy_ref, atol=5e-7, rtol=5e-7)

    def test_multiple_image_deflection_model(self):
        npt.assert_raises(Exception, self._setup_multiple_images)

    def test_change_cosmology(self):
        from astropy.cosmology import FlatwCDM

        cosmo = FlatwCDM(H0=67, Om0=0.3, w0=-0.8)
        cosmo_new = FlatwCDM(H0=73, Om0=0.3, w0=-1)

        z_lens = 0.5
        z_source_convention = 2
        print(self.kwargs_multiplane_model_grid)
        kwargs_multiplane_model_grid_ = copy.deepcopy(self.kwargs_multiplane_model_grid)
        lens_model_list = self.kwargs_multiplane_model_grid["lens_model_list"]
        kwargs_multiplane_model_grid_.pop("cosmo")
        kwargs_multiplane_model_grid_.pop("lens_model_list")
        lens_model = LensModel(
            lens_model_list=lens_model_list,
            z_lens=z_lens,
            # lens_redshift_list=[z_lens],
            z_source_convention=z_source_convention,
            # z_source=z_source_new,
            # multi_plane=True,
            cosmo=cosmo,
            # kwargs_multiplane_model=kwargs_multiplane_model_grid_,
            # decouple_multi_plane=True,
            **kwargs_multiplane_model_grid_,
        )
        npt.assert_raises(Exception, lens_model.update_cosmology, cosmo=cosmo_new)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
