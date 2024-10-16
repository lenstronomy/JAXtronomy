__author__ = "sibirrer"

import numpy as np
from lenstronomy.ImSim.Numerics.grid import RegularGrid

import pytest

class TestRegularGrid(object):
    def test_init(self):
        self._deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        self.nx, self.ny = 11, 11
        self._supersampling_factor = 4
        regular_grid = RegularGrid(
            self.nx,
            self.ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
        )

        deltaPix = regular_grid.grid_points_spacing
        assert deltaPix == self._deltaPix / self._supersampling_factor

        nx, ny = regular_grid.num_grid_points_axes
        assert nx == self.nx * self._supersampling_factor
        assert ny == self.ny * self._supersampling_factor

        ssf = regular_grid.supersampling_factor
        assert ssf == self._supersampling_factor


if __name__ == "__main__":
    pytest.main()
