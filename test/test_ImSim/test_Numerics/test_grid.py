__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
from jaxtronomy.ImSim.Numerics.grid import RegularGrid
from lenstronomy.ImSim.Numerics.grid import RegularGrid as RegularGrid_ref

import pytest


class TestRegularGrid_Supersample4(object):
    def setup_method(self):
        self._deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        self.nx, self.ny = 11, 11
        self._supersampling_factor = 4
        self.regular_grid = RegularGrid(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
        )
        self.regular_grid_ref = RegularGrid_ref(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
        )

    def test_init(self):
        deltaPix = self.regular_grid.grid_points_spacing
        assert deltaPix == self._deltaPix / self._supersampling_factor

        nx, ny = self.regular_grid.num_grid_points_axes
        assert nx == self.nx * self._supersampling_factor
        assert ny == self.ny * self._supersampling_factor

        ssf = self.regular_grid.supersampling_factor
        assert ssf == self._supersampling_factor

        npt.assert_array_almost_equal(
            self.regular_grid.coordinates_evaluate,
            self.regular_grid_ref.coordinates_evaluate,
            decimal=8,
        )

    def test_flux_array2image_low_high(self):
        flux_array = np.linspace(
            0, 3, self.nx * self.ny * self._supersampling_factor**2
        )
        image_low_res, image_high_res = self.regular_grid.flux_array2image_low_high(
            flux_array
        )
        image_low_res_ref, image_high_res_ref = (
            self.regular_grid_ref.flux_array2image_low_high(flux_array)
        )
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=6)
        npt.assert_array_almost_equal(image_high_res, image_high_res_ref, decimal=6)


class TestRegularGrid_Supersample1(object):
    def setup_method(self):
        self._deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        self.nx, self.ny = 11, 11
        self._supersampling_factor = 1

        self.flux_evaluate_indexes = np.ones((self.nx, self.ny))
        self.flux_evaluate_indexes[1::2, 0::3] = 0
        self.regular_grid = RegularGrid(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
            flux_evaluate_indexes=self.flux_evaluate_indexes,
        )
        self.regular_grid_ref = RegularGrid_ref(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
            flux_evaluate_indexes=self.flux_evaluate_indexes.astype(bool),
        )

    def test_init(self):
        deltaPix = self.regular_grid.grid_points_spacing
        assert deltaPix == self._deltaPix / self._supersampling_factor

        nx, ny = self.regular_grid.num_grid_points_axes
        assert nx == self.nx * self._supersampling_factor
        assert ny == self.ny * self._supersampling_factor

        ssf = self.regular_grid.supersampling_factor
        assert ssf == self._supersampling_factor

        npt.assert_array_almost_equal(
            self.regular_grid.coordinates_evaluate,
            self.regular_grid_ref.coordinates_evaluate,
            decimal=8,
        )
        npt.assert_array_equal(
            self.regular_grid._compute_indexes, self.regular_grid_ref._compute_indexes
        )

    def test_flux_array2image_low_high(self):
        flux_array = np.linspace(0, 3, np.count_nonzero(self.flux_evaluate_indexes))
        image_low_res, image_high_res = self.regular_grid.flux_array2image_low_high(
            flux_array
        )
        image_low_res_ref, image_high_res_ref = (
            self.regular_grid_ref.flux_array2image_low_high(flux_array)
        )
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=6)
        assert image_high_res is None


if __name__ == "__main__":
    pytest.main()
