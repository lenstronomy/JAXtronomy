import pytest
import jax, jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import unittest

from jaxtronomy.PointSource.point_source import PointSource
from jaxtronomy.LensModel.lens_model import LensModel

from lenstronomy.PointSource.point_source import PointSource as PointSource_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref

jax.config.update("jax_enable_x64", True)


class TestPointSource(object):
    def setup_method(self):

        lens_model_list = ["EPL", "SIS"]
        point_source_type_list = ["LENSED_POSITION", "UNLENSED", "LENSED_POSITION"]

        lensmodel = LensModel(lens_model_list=lens_model_list)
        self.ps = PointSource(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel,
            fixed_magnification_list=[False, False, True],
        )

        lensmodel_ref = LensModel_ref(lens_model_list=lens_model_list)
        self.ps_ref = PointSource_ref(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel_ref,
            fixed_magnification_list=[False, False, True],
        )

        kwargs_ps1 = {
            "ra_image": [0.1, 0.3, -0.298],
            "dec_image": [-0.3, 0.2, 0.1293],
            "point_amp": [1.3, 1.4, 0.2387],
        }

        kwargs_ps2 = {
            "ra_image": [0.5, 0.1, 0.2198, 1.38234],
            "dec_image": [-0.31, 0.232, -0.23487, 0.2347],
            "point_amp": [1.3534, 1.4345, 0.434, 2.3429],
        }

        kwargs_ps3 = {
            "ra_image": [0.4321, 0.233, 0.345],
            "dec_image": [-0.123, 0.243, 0.389],
            "source_amp": 1.5,
        }
        self.kwargs_ps = [kwargs_ps1, kwargs_ps2, kwargs_ps3]

        kwargs_epl = {
            "theta_E": 1.3,
            "gamma": 1.7,
            "e1": 0.13,
            "e2": 0.01,
        }
        kwargs_sis = {
            "theta_E": 1.5,
            "center_x": -0.11,
            "center_y": -0.03,
        }

        self.kwargs_lens = [kwargs_epl, kwargs_sis]

    def test_k_list(self):
        k_list = self.ps.k_list(k=2)
        k_list_ref = self.ps_ref.k_list(k=2)
        assert k_list == k_list_ref

    def test_source_position(self):
        x_source_list, y_source_list = self.ps.source_position(
            self.kwargs_ps, self.kwargs_lens
        )
        x_source_list_ref, y_source_list_ref = self.ps_ref.source_position(
            self.kwargs_ps, self.kwargs_lens
        )
        for i in range(len(x_source_list)):
            print(f"testing point_source_type_list {i}")
            npt.assert_allclose(
                x_source_list[i], x_source_list_ref[i], atol=1e-8, rtol=1e-8
            )
            npt.assert_allclose(
                y_source_list[i], y_source_list_ref[i], atol=1e-8, rtol=1e-8
            )

    def test_image_position(self):
        x_image_list, y_image_list = self.ps.image_position(
            self.kwargs_ps, self.kwargs_lens
        )
        x_image_list_ref, y_image_list_ref = self.ps_ref.image_position(
            self.kwargs_ps, self.kwargs_lens
        )
        for i in range(len(x_image_list)):
            print(f"testing point_source_type_list {i}")
            npt.assert_allclose(
                x_image_list[i], x_image_list_ref[i], atol=1e-8, rtol=1e-8
            )
            npt.assert_allclose(
                y_image_list[i], y_image_list_ref[i], atol=1e-8, rtol=1e-8
            )

    def test_point_source_list(self):
        ra_array, dec_array, amp_array = self.ps.point_source_list(
            self.kwargs_ps, self.kwargs_lens
        )
        ra_array_ref, dec_array_ref, amp_array_ref = self.ps_ref.point_source_list(
            self.kwargs_ps, self.kwargs_lens
        )

        # In lenstronomy, images with 0 amplitude are automatically removed
        # We have to do this manually in jaxtronomy
        ra_array = ra_array[jnp.where(amp_array != 0)]
        dec_array = dec_array[jnp.where(amp_array != 0)]
        amp_array = amp_array[jnp.where(amp_array != 0)]

        npt.assert_allclose(ra_array, ra_array_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(dec_array, dec_array_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(amp_array, amp_array_ref, atol=1e-8, rtol=1e-8)

        ra_array, dec_array, amp_array = self.ps.point_source_list(
            self.kwargs_ps, self.kwargs_lens, with_amp=False
        )
        ra_array_ref, dec_array_ref, amp_array_ref = self.ps_ref.point_source_list(
            self.kwargs_ps, self.kwargs_lens, with_amp=False
        )
        ra_array = ra_array[jnp.where(amp_array != 0)]
        dec_array = dec_array[jnp.where(amp_array != 0)]
        amp_array = amp_array[jnp.where(amp_array != 0)]

        npt.assert_allclose(ra_array, ra_array_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(dec_array, dec_array_ref, atol=1e-8, rtol=1e-8)
        npt.assert_allclose(amp_array, amp_array_ref, atol=1e-8, rtol=1e-8)

    def test_image_amplitude(self):
        amp_list = self.ps.image_amplitude(self.kwargs_ps, self.kwargs_lens)
        amp_list_ref = self.ps_ref.image_amplitude(self.kwargs_ps, self.kwargs_lens)
        for i in range(len(amp_list)):
            print(f"testing point_source_type_list {i}")
            npt.assert_allclose(amp_list[i], amp_list_ref[i], atol=1e-8, rtol=1e-8)
            assert np.array(amp_list[i]).ndim == 1

    def test_source_amplitude(self):
        amp_list = self.ps.source_amplitude(self.kwargs_ps, self.kwargs_lens)
        amp_list_ref = self.ps_ref.source_amplitude(self.kwargs_ps, self.kwargs_lens)
        for i in range(len(amp_list)):
            print(f"testing point_source_type_list {i}")
            npt.assert_allclose(amp_list[i], amp_list_ref[i], atol=1e-8, rtol=1e-8)

    def test_check_image_position(self):
        within_tolerance = self.ps.check_image_positions(
            self.kwargs_ps, self.kwargs_lens
        )
        within_tolerance_ref = self.ps_ref.check_image_positions(
            self.kwargs_ps, self.kwargs_lens
        )
        assert within_tolerance == within_tolerance_ref

        within_tolerance = self.ps.check_image_positions(
            self.kwargs_ps, self.kwargs_lens, tolerance=30
        )
        within_tolerance_ref = self.ps_ref.check_image_positions(
            self.kwargs_ps, self.kwargs_lens, tolerance=30
        )
        assert within_tolerance == within_tolerance_ref

    def test_set_amplitudes(self):
        amp_list = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]]
        kwargs_new = self.ps.set_amplitudes(amp_list, self.kwargs_ps)
        kwargs_new_ref = self.ps_ref.set_amplitudes(amp_list, self.kwargs_ps)
        assert kwargs_new == kwargs_new_ref
        npt.assert_array_equal(amp_list[0], kwargs_new[0]["point_amp"])
        npt.assert_array_equal(amp_list[1], kwargs_new[1]["point_amp"])
        npt.assert_array_equal(amp_list[2], kwargs_new[2]["source_amp"])

    def test_raises(self):
        npt.assert_raises(
            ValueError, PointSource, ["LENSED_POSITION"], index_lens_model_list=[[0]]
        )
        npt.assert_raises(ValueError, PointSource, [], save_cache=True)
        npt.assert_raises(ValueError, PointSource, ["SOURCE_POSITION"])
        npt.assert_raises(ValueError, PointSource, ["invalid_model"])

        ps = PointSource([])
        npt.assert_raises(ValueError, ps.update_search_window, 1, 1, 1)
        npt.assert_raises(ValueError, ps.update_lens_model, LensModel([]))


# Same tests as before but this time with set index_lens_model_list and point_source_frame_list
# Different setup method but inherits all the tests from above
class TestPointSourcewithFrames(TestPointSource):
    def setup_method(self):
        lens_model_list = ["EPL", "SIS"]
        point_source_type_list = ["LENSED_POSITION", "UNLENSED", "LENSED_POSITION"]

        lensmodel = LensModel(lens_model_list=lens_model_list)
        self.ps = PointSource(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel,
            fixed_magnification_list=[False, False, True],
            index_lens_model_list=[[0], [1]],
            point_source_frame_list=[[0, 0, 0], None, [1, 1, 1]],
        )

        lensmodel_ref = LensModel_ref(lens_model_list=lens_model_list)
        self.ps_ref = PointSource_ref(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel_ref,
            fixed_magnification_list=[False, False, True],
            index_lens_model_list=[[0], [1]],
            point_source_frame_list=[[0, 0, 0], None, [1, 1, 1]],
        )

        kwargs_ps1 = {
            "ra_image": [0.1, 0.3, -0.298],
            "dec_image": [-0.3, 0.2, 0.1293],
            "point_amp": [1.3, 1.4, 0.2387],
        }

        kwargs_ps2 = {
            "ra_image": [0.5, 0.1, 0.2198, 1.38234],
            "dec_image": [-0.31, 0.232, -0.23487, 0.2347],
            "point_amp": [1.3534, 1.4345, 0.434, 2.3429],
        }

        kwargs_ps3 = {
            "ra_image": [0.4321, 0.233, 0.345],
            "dec_image": [-0.123, 0.243, 0.389],
            "source_amp": [1.5, 0.23, 16.1234],
        }
        self.kwargs_ps = [kwargs_ps1, kwargs_ps2, kwargs_ps3]

        kwargs_epl = {
            "theta_E": 1.3,
            "gamma": 1.7,
            "e1": 0.13,
            "e2": 0.01,
        }
        kwargs_sis = {
            "theta_E": 1.5,
            "center_x": -0.11,
            "center_y": -0.03,
        }

        self.kwargs_lens = [kwargs_epl, kwargs_sis]


# Same tests as before but this time with set flux_from_point_source_list
# Different setup method but inherits all the tests from above
class TestPointSourcewithFluxList(TestPointSource):
    def setup_method(self):
        lens_model_list = ["EPL", "SIS"]
        point_source_type_list = ["LENSED_POSITION", "UNLENSED", "LENSED_POSITION"]

        lensmodel = LensModel(lens_model_list=lens_model_list)
        self.ps = PointSource(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel,
            fixed_magnification_list=[False, False, True],
            flux_from_point_source_list=[False, True, True],
        )

        lensmodel_ref = LensModel_ref(lens_model_list=lens_model_list)
        self.ps_ref = PointSource_ref(
            point_source_type_list=point_source_type_list,
            lens_model=lensmodel_ref,
            fixed_magnification_list=[False, False, True],
            flux_from_point_source_list=[False, True, True],
        )

        kwargs_ps1 = {
            "ra_image": [0.1, 0.3, -0.298],
            "dec_image": [-0.3, 0.2, 0.1293],
            "point_amp": [1.3, 1.4, 0.2387],
        }

        kwargs_ps2 = {
            "ra_image": [0.5, 0.1, 0.2198, 1.38234],
            "dec_image": [-0.31, 0.232, -0.23487, 0.2347],
            "point_amp": [1.3534, 1.4345, 0.434, 2.3429],
        }

        kwargs_ps3 = {
            "ra_image": [0.4321, 0.233, 0.345],
            "dec_image": [-0.123, 0.243, 0.389],
            "source_amp": [1.5, 0.23, 16.1234],
        }
        self.kwargs_ps = [kwargs_ps1, kwargs_ps2, kwargs_ps3]

        kwargs_epl = {
            "theta_E": 1.3,
            "gamma": 1.7,
            "e1": 0.13,
            "e2": 0.01,
        }
        kwargs_sis = {
            "theta_E": 1.5,
            "center_x": -0.11,
            "center_y": -0.03,
        }

        self.kwargs_lens = [kwargs_epl, kwargs_sis]

    def test_set_amplitudes(self):
        amp_list = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]]
        kwargs_new = self.ps.set_amplitudes(amp_list, self.kwargs_ps)
        kwargs_new_ref = self.ps_ref.set_amplitudes(amp_list, self.kwargs_ps)
        assert kwargs_new == kwargs_new_ref

        # The amplitudes are not updated for models where flux_from_point_source is False
        npt.assert_array_equal(
            self.kwargs_ps[0]["point_amp"], kwargs_new[0]["point_amp"]
        )

        # These amplitudes are updated
        npt.assert_array_equal(amp_list[1], kwargs_new[1]["point_amp"])
        npt.assert_array_equal(amp_list[2], kwargs_new[2]["source_amp"])


if __name__ == "__main__":
    pytest.main()
