__author__ = "sibirrer"

import numpy as np
from lenstronomy.ImSim.Numerics.grid import RegularGrid

import pytest

class TestRegularGrid(object):
    def setup_method(self):
        self._deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        nx, ny = 11, 11
        self._supersampling_factor = 4
        self.nx, self.ny = nx, ny
        self._regular_grid = RegularGrid(
            nx,
            ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
        )

    def test_grid_points_spacing(self):
        deltaPix = self._regular_grid.grid_points_spacing
        assert deltaPix == self._deltaPix / self._supersampling_factor

    def test_num_grid_points_axes(self):
        nx, ny = self._regular_grid.num_grid_points_axes
        assert nx == self.nx * self._supersampling_factor
        assert ny == self.ny * self._supersampling_factor

    def test_supersampling_factor(self):
        ssf = self._regular_grid.supersampling_factor
        assert ssf == self._supersampling_factor


if __name__ == "__main__":
    pytest.main()
