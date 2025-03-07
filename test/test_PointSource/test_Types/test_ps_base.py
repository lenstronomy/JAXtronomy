import pytest
import numpy.testing as npt

from jaxtronomy.LensModel.Profiles.sis import SIS
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.PointSource.Types.base_ps import PSBase

from jaxtronomy.PointSource.Types.base_ps import _expand_to_array
from jaxtronomy.PointSource.Types.base_ps import _shrink_array

from lenstronomy.PointSource.Types.base_ps import (
    _expand_to_array as _expand_to_array_ref,
)
from lenstronomy.PointSource.Types.base_ps import _shrink_array as _shrink_array_ref


class TestPSBase(object):
    def setup_method(self):
        self.ps = PSBase(
            lens_model=LensModel(lens_model_list=["SIS"]),
            fixed_magnification=False,
            additional_images=False,
        )
        self.kwargs_ps = {"image_amp": 1, "ra_image": 0.1, "dec_image": 0.1}

    def test_init(self):
        assert self.ps._fixed_magnification == False
        assert self.ps.additional_images == False
        assert type(self.ps._lens_model.lens_model.func_list[0]) == SIS
        assert type(self.ps._solver.lensModel.lens_model.func_list[0]) == SIS
        assert self.ps.k_list is None
        assert self.ps._redshift is None

        # 2 bands, each with 2 different lens models
        index_lens_model_list = [[0, 1], [2, 3]]

        # 4 images, each assigned to one of the two bands
        point_source_frame_list = [0, 1, 1, 0]
        ps = PSBase(
            fixed_magnification=True,
            index_lens_model_list=index_lens_model_list,
            point_source_frame_list=point_source_frame_list,
        )
        assert ps._lens_model is None
        assert ps._solver is None
        assert ps._fixed_magnification == True
        assert ps.k_list == [(0, 1), (2, 3), (2, 3), (0, 1)]

        npt.assert_raises(ValueError, PSBase, additional_images=True)
        npt.assert_raises(ValueError, PSBase, redshift=3)

    def test_image_position(self):
        npt.assert_raises(ValueError, self.ps.image_position, self.kwargs_ps)

    def test_source_position(self):
        npt.assert_raises(ValueError, self.ps.source_position, self.kwargs_ps)

    def test_image_amplitude(self):
        npt.assert_raises(ValueError, self.ps.image_amplitude, self.kwargs_ps)

    def test_source_amplitude(self):
        npt.assert_raises(ValueError, self.ps.source_amplitude, self.kwargs_ps)

    def test_update_lens_model(self):
        npt.assert_raises(ValueError, self.ps.update_lens_model, LensModel([]))


class TestUtil(object):
    def setup_method(self):
        pass

    def test_expand_to_array(self):
        array = 1
        num = 3
        array_out = _expand_to_array(array, num)
        array_out_ref = _expand_to_array_ref(array, num)
        npt.assert_array_equal(array_out, [1, 1, 1])
        npt.assert_array_equal(array_out, array_out_ref)

        array = [1]
        num = 3
        array_out = _expand_to_array(array, num)
        array_out_ref = _expand_to_array_ref(array, num)
        npt.assert_array_equal(array_out, [1, 0, 0])
        npt.assert_array_equal(array_out, array_out_ref)

        array = [1, 2, 2]
        num = 3
        array_out = _expand_to_array(array, num)
        array_out_ref = _expand_to_array_ref(array, num)
        npt.assert_array_equal(array_out, [1, 2, 2])
        npt.assert_array_equal(array_out, array_out_ref)

    def test_shrink_array(self):
        array = [1, 2, 3]
        num = 2
        array_out = _shrink_array(array, num)
        array_out_ref = _shrink_array_ref(array, num)
        npt.assert_array_equal(array_out, [1, 2])
        npt.assert_array_equal(array_out, array_out_ref)

        array = 1
        num = 3
        array_out = _shrink_array(array, num)
        array_out_ref = _shrink_array_ref(array, num)
        assert array_out == array
        npt.assert_array_equal(array_out, array_out_ref)

        # Ensure that jaxtronomy behaves the same as lenstronomy
        array = [1]
        num = 2
        npt.assert_raises(ValueError, _shrink_array, array, num)
        npt.assert_raises(ValueError, _shrink_array_ref, array, num)


if __name__ == "__main__":
    pytest.main()
