__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
from jaxtronomy.ImSim.Numerics.grid import RegularGrid, AdaptiveGrid
from lenstronomy.ImSim.Numerics.grid import (
    RegularGrid as RegularGrid_ref,
    AdaptiveGrid as AdaptiveGrid_ref,
)

import pytest


class TestAdaptiveGrid(object):
    def setup_method(self):
        self._delta_pix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._delta_pix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        self.nx, self.ny = 11, 11
        self._supersampling_indexes = np.zeros((self.nx, self.ny), dtype=bool)
        self._supersampling_indexes[3:7][2:4] = 1
        self._supersampling_factor = 3

        self.adaptive_grid = AdaptiveGrid(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_indexes=self._supersampling_indexes,
            supersampling_factor=self._supersampling_factor,
        )
        self.adaptive_grid_ref = AdaptiveGrid_ref(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_indexes=self._supersampling_indexes,
            supersampling_factor=self._supersampling_factor,
        )

    def test_init(self):
        ssf = self.adaptive_grid.supersampling_factor
        assert ssf == self._supersampling_factor

        npt.assert_array_almost_equal(
            self.adaptive_grid._high_res_coordinates,
            self.adaptive_grid_ref._high_res_coordinates,
            decimal=12,
        )

        npt.assert_array_almost_equal(
            self.adaptive_grid.coordinates_evaluate,
            self.adaptive_grid_ref.coordinates_evaluate,
            decimal=12,
        )

        assert self.adaptive_grid._num_low_res == self.adaptive_grid_ref._num_low_res

        npt.assert_array_equal(
            self.adaptive_grid._low_res_indexes1d,
            self.adaptive_grid_ref._low_res_indexes1d,
        )
        npt.assert_array_equal(
            self.adaptive_grid._high_res_indexes1d,
            self.adaptive_grid_ref._high_res_indexes1d,
        )

    def test_flux_array2image_low_high(self):
        flux_array = np.linspace(0, 10, len(self.adaptive_grid.coordinates_evaluate[0]))
        image_low_res, image_high_res = self.adaptive_grid.flux_array2image_low_high(
            flux_array
        )
        image_low_res_ref, image_high_res_ref = (
            self.adaptive_grid_ref.flux_array2image_low_high(flux_array)
        )
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=12)

        # high res image in jaxtronomy is implemented differently than lenstronomy
        # jaxtronomy also includes pixel values for the non-supersampled indices
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_ij = image_high_res[
                    i :: self._supersampling_factor, j :: self._supersampling_factor
                ]
                image_ij_ref = image_high_res_ref[
                    i :: self._supersampling_factor, j :: self._supersampling_factor
                ]

                # adds the non supersampled pixel values to the lenstronomy version
                image_ij_ref += image_low_res_ref * np.invert(
                    self._supersampling_indexes
                )

                npt.assert_array_almost_equal(image_ij, image_ij_ref, decimal=12)

    def test_merge_low_high_res(self):
        flux_array = np.linspace(0, 10, len(self.adaptive_grid.coordinates_evaluate[0]))
        low_res_values = flux_array[0 : self.adaptive_grid._num_low_res]
        high_res_values = flux_array[self.adaptive_grid._num_low_res :]

        image_low_res = self.adaptive_grid._merge_low_high_res(
            low_res_values, high_res_values
        )
        image_low_res_ref = self.adaptive_grid_ref._merge_low_high_res(
            low_res_values, high_res_values
        )
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=12)

    def test_high_res_image(self):
        flux_array = np.linspace(0, 10, len(self.adaptive_grid.coordinates_evaluate[0]))
        low_res_values = flux_array[0 : self.adaptive_grid._num_low_res]
        high_res_values = flux_array[self.adaptive_grid._num_low_res :]
        image_high_res = self.adaptive_grid._high_res_image(
            low_res_values, high_res_values
        )
        image_high_res_ref = self.adaptive_grid_ref._high_res_image(high_res_values)

        # high res image in jaxtronomy is implemented differently than lenstronomy
        # jaxtronomy also includes pixel values for the non-supersampled indices
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_ij = image_high_res[
                    i :: self._supersampling_factor, j :: self._supersampling_factor
                ]
                image_ij_ref = image_high_res_ref[
                    i :: self._supersampling_factor, j :: self._supersampling_factor
                ]

                # adds the non supersampled pixel values to the lenstronomy version
                other_pixels = np.zeros_like(image_ij_ref).flatten()
                other_pixels[self.adaptive_grid._low_res_indexes1d] = low_res_values
                image_ij_ref += other_pixels.reshape(image_ij_ref.shape)
                npt.assert_array_almost_equal(image_ij, image_ij_ref, decimal=12)

    def test_average_subgrid(self):
        flux_array = np.linspace(0, 10, len(self.adaptive_grid.coordinates_evaluate[0]))
        low_res_values = flux_array[0 : self.adaptive_grid._num_low_res]
        high_res_values = flux_array[self.adaptive_grid._num_low_res :]
        image_high_res = self.adaptive_grid._high_res_image(
            low_res_values, high_res_values
        )

        result = self.adaptive_grid._average_subgrid(image_high_res)
        result_ref = self.adaptive_grid_ref._average_subgrid(image_high_res)
        npt.assert_array_almost_equal(result, result_ref, decimal=12)

    def test_array2image_subset(self):
        flux_array = np.linspace(
            3.3, 14.57, len(self.adaptive_grid.coordinates_evaluate[0])
        )
        low_res_values = flux_array[0 : self.adaptive_grid._num_low_res]

        for i in range(self._supersampling_factor):
            high_res_values = flux_array[self.adaptive_grid._num_low_res :]
            high_res_values = high_res_values[i :: self._supersampling_factor**2]

            # high res image in jaxtronomy is implemented differently than lenstronomy
            # jaxtronomy also includes pixel values for the non-supersampled indices
            image = self.adaptive_grid._array2image_subset(
                low_res_values, high_res_values
            )
            image_ref = self.adaptive_grid_ref._array2image_subset(high_res_values)

            # adds the non supersampled pixel values to the lenstronomy version
            shape = image_ref.shape
            image_ref = image_ref.flatten()
            image_ref[self.adaptive_grid_ref._low_res_indexes1d] = low_res_values
            image_ref = image_ref.reshape(shape)
            npt.assert_array_almost_equal(image, image_ref, decimal=12)


class TestRegularGrid_Supersample4(object):
    def setup_method(self):
        self._delta_pix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._delta_pix
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
        delta_pix = self.regular_grid.grid_points_spacing
        assert delta_pix == self._delta_pix / self._supersampling_factor

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
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=12)
        npt.assert_array_almost_equal(image_high_res, image_high_res_ref, decimal=12)


class TestRegularGrid_Supersample1(object):
    def setup_method(self):
        self._delta_pix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._delta_pix
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
        delta_pix = self.regular_grid.grid_points_spacing
        assert delta_pix == self._delta_pix / self._supersampling_factor

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
        npt.assert_array_almost_equal(image_low_res, image_low_res_ref, decimal=12)
        assert image_high_res is None


if __name__ == "__main__":
    pytest.main()
