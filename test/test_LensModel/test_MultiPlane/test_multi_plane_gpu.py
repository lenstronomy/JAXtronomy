import jax

jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane as MultiPlane_ref
from jaxtronomy.LensModel.MultiPlane.multi_plane_gpu import MultiPlaneGPU


class TestMultiPlaneGPU(object):
    """Tests the MultiPlaneGPU routines."""

    def setup_method(self):
        self.z_source = 3.5
        self.lens_model_list = ["NFW", "NIE", "NFW", "NFW", "NFW"]
        unique_lens_model_list = list(set(self.lens_model_list))
        self.redshift_list = [0.5, 1.1, 1.1, 1.5, 1.3]
        self.multiplane = MultiPlaneGPU(
            unique_lens_model_list=unique_lens_model_list,
        )

        self.multiplane_ref = MultiPlane_ref(
            z_source=self.z_source,
            lens_model_list=self.lens_model_list,
            lens_redshift_list=self.redshift_list,
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

    def test_ray_shooting(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            z_source=self.z_source,
            lens_redshift_list=self.redshift_list,
        )
        f_x, f_y = self.multiplane.ray_shooting(x, y, **ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.multiplane_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

        ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens2,
            z_source=self.z_source,
            lens_redshift_list=self.redshift_list,
            num_deflectors=6,
        )
        f_x, f_y = self.multiplane.ray_shooting(x, y, **ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.multiplane_ref.ray_shooting(x, y, self.kwargs_lens2)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_raises(self):
        # too many lens models, doesnt match with kwargs_lens or redshift list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list + ["NFW"],
                kwargs_lens=self.kwargs_lens,
                z_source=self.z_source,
                lens_redshift_list=self.redshift_list,
            )
        # too many kwargs_lens, doesnt match with redshift list or lens model list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens + self.kwargs_lens2,
                z_source=self.z_source,
                lens_redshift_list=self.redshift_list,
            )
        # too many redshifts, doesnt match with kwargs_lens or lens model list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens,
                z_source=self.z_source,
                lens_redshift_list=self.redshift_list + [1.3],
            )
        # provided num_deflectors is smaller than the number of deflectors in lens model list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.multiplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens,
                z_source=self.z_source,
                lens_redshift_list=self.redshift_list,
                num_deflectors=3,
            )
