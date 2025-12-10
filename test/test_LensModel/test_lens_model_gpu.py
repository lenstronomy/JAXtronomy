import jax

jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from jaxtronomy.LensModel.lens_model_gpu import LensModelGPU


class TestLensModelGPU(object):
    """Tests the LensModelGPU routines."""

    def setup_method(self):
        self.z_source = 3.5
        self.lens_model_list = ["EPL", "TNFW", "TNFW", "SHEAR"]
        unique_lens_model_list = list(set(self.lens_model_list))
        self.redshift_list = [1.1, 0.5, 1.3, 1.5]
        self.lens_model_multi = LensModelGPU(
            unique_lens_model_list=unique_lens_model_list,
            multi_plane=True,
        )

        self.lens_model_single = LensModelGPU(
            unique_lens_model_list=unique_lens_model_list,
            multi_plane=False,
        )

        self.lens_model_multi_ref = LensModel_ref(
            multi_plane=True,
            z_source=self.z_source,
            lens_model_list=self.lens_model_list,
            lens_redshift_list=self.redshift_list,
        )
        self.lens_model_single_ref = LensModel_ref(
            multi_plane=False,
            lens_model_list=self.lens_model_list,
        )

        kwargs_tnfw1 = {"Rs": 1.3, "alpha_Rs": 2.18, "r_trunc": 1.5, "center_x": 0.1, "center_y": -2.1}
        kwargs_tnfw2 = {"Rs": 1.4, "alpha_Rs": 3.18, "r_trunc": 1.5, "center_x": -0.11, "center_y": 1.1}
        kwargs_epl = {"theta_E": 1.5, "gamma": 1.7, "e1": 0.1, "e2": 0.2}
        kwargs_shear = {"gamma1": 1.1, "gamma2": 0.3}
        self.kwargs_lens = [
            kwargs_epl,
            kwargs_tnfw1,
            kwargs_tnfw2,
            kwargs_shear,
        ]

    def test_ray_shooting_multi(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        ray_shooting_kwargs = self.lens_model_multi.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            z_source=self.z_source,
            lens_redshift_list=self.redshift_list,
        )
        f_x, f_y = self.lens_model_multi.ray_shooting(x, y, ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.lens_model_multi_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        ray_shooting_kwargs = self.lens_model_multi.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            z_source=self.z_source,
            lens_redshift_list=self.redshift_list,
            num_deflectors=6,
        )
        f_x, f_y = self.lens_model_multi.ray_shooting(x, y, ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.lens_model_multi_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_ray_shooting_single(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        ray_shooting_kwargs = self.lens_model_single.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
        )
        f_x, f_y = self.lens_model_single.ray_shooting(x, y, ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.lens_model_single_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        ray_shooting_kwargs = self.lens_model_single.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            num_deflectors=6,
        )
        f_x, f_y = self.lens_model_single.ray_shooting(x, y, ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.lens_model_single_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_raises(self):
        # must provide z_source
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.lens_model_multi.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens,
                z_source=None,
                lens_redshift_list=self.redshift_list,
            )
        # must provide redshift list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.lens_model_multi.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens,
                z_source=self.z_source,
                lens_redshift_list=None,
            )