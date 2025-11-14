import jax

jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import numpy as np
import pytest

from astropy.cosmology import FlatwCDM
from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane as MultiPlane_ref
from jaxtronomy.LensModel.MultiPlane.multi_plane import MultiPlane
from lenstronomy.LensModel.MultiPlane.multi_plane_base import MultiPlaneBase


class TestMultiPlane(object):
    """Tests the MultiPlane routines."""

    def setup_method(self):
        z_source = 3.5
        lens_model_list = ["NFW", "NIE", "NFW", "NFW", "NFW"]
        redshift_list = [0.5, 1.1, 1.1, 1.5, 1.3]
        cosmo = FlatwCDM(H0=70, Om0=0.3, w0=-0.8)
        self.multiplane = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_interp_stop=4,
            cosmo_interp=True,
            distance_ratio_sampling=False,
            z_lens_convention=0.5,
            cosmo=cosmo,
            cosmology_sampling=False,
        )

        self.multiplane_ref = MultiPlane_ref(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_interp_stop=4,
            cosmo_interp=True,
            distance_ratio_sampling=False,
            z_lens_convention=0.5,
            cosmo=cosmo,
            cosmology_sampling=False,
        )

        kwargs_nfw1 = {"Rs": 1.3, "alpha_Rs": 2.18, "center_x": 0.1, "center_y": -2.1}
        kwargs_nfw2 = {"Rs": 1.4, "alpha_Rs": 3.18, "center_x": -0.11, "center_y": 1.1}
        kwargs_nfw3 = {"Rs": 1.5, "alpha_Rs": 1.11, "center_x": -0.13, "center_y": 1.2}
        kwargs_nfw4 = {"Rs": 1.1, "alpha_Rs": 2.12, "center_x": 0.21, "center_y": -2.2}
        kwargs_nie1 = {"theta_E": 1.5, "e1": 0.1, "e2": 0.2, "s_scale": 3.1}
        kwargs_nie2 = {"theta_E": 3.5, "e1": 0.3, "e2": -0.2, "s_scale": 1.1}
        self.kwargs_lens = [
            kwargs_nfw1,
            kwargs_nie1,
            kwargs_nfw2,
            kwargs_nfw3,
            kwargs_nfw4,
        ]
        self.kwargs_lens2 = [
            kwargs_nfw1,
            kwargs_nie2,
            kwargs_nfw2,
            kwargs_nfw3,
            kwargs_nfw4,
        ]

    def test_init(self):
        assert self.multiplane.z_source == self.multiplane_ref.z_source
        assert (
            self.multiplane.z_source_convention
            == self.multiplane_ref.z_source_convention
        )
        assert (
            self.multiplane.z_lens_convention == self.multiplane_ref.z_lens_convention
        )
        assert self.multiplane.T_ij_start == self.multiplane_ref.T_ij_start
        assert self.multiplane.T_ij_stop == self.multiplane_ref.T_ij_stop

        npt.assert_array_equal(
            self.multiplane.multi_plane_base.T_z_list,
            self.multiplane_ref.multi_plane_base.T_z_list,
        )
        npt.assert_array_equal(
            self.multiplane.multi_plane_base.T_ij_list,
            self.multiplane_ref.multi_plane_base.T_ij_list,
        )
        npt.assert_array_equal(
            self.multiplane.multi_plane_base.z_source_convention,
            self.multiplane_ref.multi_plane_base.z_source_convention,
        )
        npt.assert_array_equal(
            self.multiplane.multi_plane_base.sorted_redshift_index,
            self.multiplane_ref.multi_plane_base.sorted_redshift_index,
        )

        self.multiplane.model_info()

        T_ij_start, T_ij_end = self.multiplane.transverse_distance_start_stop(
            z_start=0, z_stop=self.multiplane.z_source, include_z_start=False
        )
        T_ij_start_ref, T_ij_end_ref = (
            self.multiplane_ref.transverse_distance_start_stop(
                z_start=0, z_stop=self.multiplane_ref.z_source, include_z_start=False
            )
        )
        npt.assert_allclose(T_ij_start, T_ij_start_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(T_ij_end, T_ij_end_ref, atol=1e-8, rtol=1e-8)

        z_source = 3.5
        lens_model_list = ["NFW", "NIE", "NFW", "NFW", "NFW"]
        redshift_list = [0.5, 1.1, 1.1, 1.5, 1.3]

        multi_plane_base = MultiPlaneBase(
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_source_convention=z_source,
        )

        # Incorrect cosmology_model
        with pytest.raises(ValueError):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                z_interp_stop=4,
                cosmo_interp=False,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                cosmo=None,
                cosmology_sampling=True,
                cosmology_model="stringTheory",
            )
        # Incompatible redshift list and lens model list
        with pytest.raises(ValueError):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=[1.1, 1.1],
                z_interp_stop=4,
                cosmo_interp=False,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                cosmo=None,
                cosmology_sampling=True,
                cosmology_model="FlatLambdaCDM",
            )
        # z_interp_stop smaller than z_source
        with pytest.raises(ValueError):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                z_interp_stop=3,
                cosmo_interp=False,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                cosmo=None,
                cosmology_sampling=False,
                cosmology_model="FlatLambdaCDM",
            )
        # z_source_convention smaller than z_lens
        with pytest.raises(ValueError):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                cosmo_interp=False,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                z_source_convention=0.1,
                cosmo=None,
                cosmology_sampling=False,
                cosmology_model="FlatLambdaCDM",
            )
        # warn that both cosmo and cosmology_model are provided
        with pytest.warns(UserWarning):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                z_interp_stop=4,
                cosmo_interp=False,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                cosmo=FlatwCDM(H0=70, Om0=0.3, w0=-0.8),
                cosmology_sampling=False,
                cosmology_model="FlatwCDM",
            )
        # warn that cosmo_interp and cosmology_sampling are True
        with pytest.warns(UserWarning):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                z_interp_stop=4,
                cosmo_interp=True,
                distance_ratio_sampling=False,
                z_lens_convention=0.5,
                cosmo=None,
                cosmology_sampling=True,
                cosmology_model="FlatLambdaCDM",
            )
        # warn that distance_ratio_sampliing and cosmology_sampling are True
        with pytest.warns(UserWarning):
            MultiPlane(
                z_source=z_source,
                lens_model_list=lens_model_list,
                lens_redshift_list=redshift_list,
                z_interp_stop=4,
                cosmo_interp=False,
                distance_ratio_sampling=True,
                z_lens_convention=0.5,
                cosmo=None,
                cosmology_sampling=True,
                cosmology_model="FlatLambdaCDM",
            )

        # empty lens model list
        multiplane = MultiPlane(
            z_source=0.1,
            lens_model_list=[],
            lens_redshift_list=[],
        )
        assert len(multiplane.multi_plane_base._sorted_redshift_index) == 0

    def test_alpha(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        alpha_x, alpha_y = self.multiplane.alpha(x, y, self.kwargs_lens)
        alpha_x_ref, alpha_y_ref = self.multiplane_ref.alpha(x, y, self.kwargs_lens)
        npt.assert_allclose(alpha_x, alpha_x_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(alpha_y, alpha_y_ref, atol=1e-8, rtol=1e-8)

    def test_ray_shooting(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        f_x, f_y = self.multiplane.ray_shooting(x, y, self.kwargs_lens)
        f_x_ref, f_y_ref = self.multiplane_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-8, rtol=1e-8)

        f_x, f_y, _, _ = self.multiplane.multi_plane_base.ray_shooting_partial_comoving(
            np.zeros_like(x),
            np.zeros_like(y),
            x,
            y,
            z_start=0,
            z_stop=self.multiplane.z_source,
            kwargs_lens=self.kwargs_lens,
            include_z_start=True,
            T_ij_start=None,
            T_ij_end=self.multiplane.T_ij_stop,
        )
        f_x_ref, f_y_ref, _, _ = (
            self.multiplane_ref.multi_plane_base.ray_shooting_partial_comoving(
                np.zeros_like(x),
                np.zeros_like(y),
                x,
                y,
                z_start=0,
                z_stop=self.multiplane_ref.z_source,
                kwargs_lens=self.kwargs_lens,
                include_z_start=True,
                T_ij_start=None,
                T_ij_end=self.multiplane_ref.T_ij_stop,
            )
        )
        npt.assert_allclose(f_x, f_x_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-8, rtol=1e-8)

    def test_hessian(self):
        # NOTE: This function has significant numerical difference from lenstronomy whenever diff < 0.0001
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        f_xx, f_xy, f_yx, f_yy = self.multiplane.hessian(
            x, y, self.kwargs_lens, diff=0.0001
        )
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.multiplane_ref.hessian(
            x, y, self.kwargs_lens, diff=0.0001
        )
        npt.assert_allclose(f_xx, f_xx_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_xy, f_xy_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_yx, f_yx_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_yy, f_yy_ref, atol=1e-8, rtol=1e-8)

    def test_raises(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        # k must be None in MultiPlane lensing
        with pytest.raises(ValueError):
            self.multiplane.ray_shooting(x, y, self.kwargs_lens, k=1)

        # z_start must be 0 OR T_ij_start must be given
        with pytest.raises(ValueError):
            self.multiplane.multi_plane_base.ray_shooting_partial_comoving(
                np.zeros_like(x),
                np.zeros_like(x),
                x,
                y,
                z_start=1,
                z_stop=5,
                kwargs_lens=self.kwargs_lens,
                T_ij_start=None,
                T_ij_end=self.multiplane._T_ij_stop,
            )

        # T_ij_end must be supplied
        with pytest.raises(ValueError):
            self.multiplane.multi_plane_base.ray_shooting_partial_comoving(
                np.zeros_like(x),
                np.zeros_like(x),
                x,
                y,
                z_start=0,
                z_stop=5,
                kwargs_lens=self.kwargs_lens,
                T_ij_start=self.multiplane._T_ij_start,
                T_ij_end=None,
            )

        # updating source redshift not allowed in jaxtronomy
        with pytest.raises(Exception):
            self.multiplane.update_source_redshift(1)

        # updating cosmology not allowed in jaxtronomy
        with pytest.raises(Exception):
            self.multiplane.set_background_cosmo(FlatwCDM(H0=71, Om0=0.3, w0=-0.8))
        with pytest.raises(Exception):
            self.multiplane.multi_plane_base.set_background_cosmo(
                FlatwCDM(H0=71, Om0=0.3, w0=-0.8)
            )

        # set static and set dynamic are not supported in jaxtronomy
        with pytest.raises(Exception):
            self.multiplane.set_static(self.kwargs_lens)
        with pytest.raises(Exception):
            self.multiplane.set_dynamic()


class TestMultiPlane2(TestMultiPlane):
    """Tests the MultiPlane routines again, this time with observed_convention_index
    set."""

    def setup_method(self):
        z_source = 3.5
        lens_model_list = ["NFW", "SIS", "NFW", "NFW", "NFW"]
        redshift_list = [0.5, 1.1, 1.1, 1.5, 1.3]
        cosmo = FlatwCDM(H0=70, Om0=0.3, w0=-0.8)
        self.multiplane = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            observed_convention_index=[2, 3],
            cosmo_interp=True,
            distance_ratio_sampling=False,
            z_lens_convention=0.5,
            cosmo=cosmo,
            cosmology_sampling=False,
        )

        self.multiplane_ref = MultiPlane_ref(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            observed_convention_index=[2, 3],
            cosmo_interp=True,
            distance_ratio_sampling=False,
            z_lens_convention=0.5,
            cosmo=cosmo,
            cosmology_sampling=False,
        )

        kwargs_nfw1 = {"Rs": 1.3, "alpha_Rs": 2.18, "center_x": 0.1, "center_y": -2.1}
        kwargs_nfw2 = {"Rs": 1.4, "alpha_Rs": 3.18, "center_x": -0.11, "center_y": 1.1}
        kwargs_nfw3 = {"Rs": 1.5, "alpha_Rs": 1.11, "center_x": -0.13, "center_y": 1.2}
        kwargs_nfw4 = {"Rs": 1.1, "alpha_Rs": 2.12, "center_x": 0.21, "center_y": -2.2}
        kwargs_sis = {"theta_E": 1.5}
        self.kwargs_lens = [
            kwargs_nfw1,
            kwargs_sis,
            kwargs_nfw2,
            kwargs_nfw3,
            kwargs_nfw4,
        ]

    def test_init(self):
        kwargs_lens = self.multiplane.observed2flat_convention(self.kwargs_lens)
        kwargs_lens_ref = self.multiplane_ref.observed2flat_convention(self.kwargs_lens)

        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        f_x, f_y = self.multiplane.ray_shooting(x, y, kwargs_lens)
        f_x_ref, f_y_ref = self.multiplane_ref.ray_shooting(x, y, kwargs_lens_ref)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-8, rtol=1e-8)

    def test_raises(self):
        pass

    def test_set_static_dynamic(self):
        pass


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
