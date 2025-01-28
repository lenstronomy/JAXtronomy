from functools import partial
from jax import jit, numpy as jnp
import numpy as np

from lenstronomy.Util import util as util_lenstronomy
from jaxtronomy.Util import util
from jaxtronomy.Util import image_util
from lenstronomy.Data.coord_transforms import Coordinates1D

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class RegularGrid(Coordinates1D):
    """Manages a super-sampled grid on the partial image."""

    # NOTE: JIT-compiled functions need to be recompiled each time a new instance
    # of RegularGrid is created

    def __init__(
        self,
        nx,
        ny,
        transform_pix2angle,
        ra_at_xy_0,
        dec_at_xy_0,
        supersampling_factor=1,
        flux_evaluate_indexes=None,
    ):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        :param supersampling_factor: int, factor (per axis) of super-sampling
        :param flux_evaluate_indexes: bool array of shape nx x ny, corresponding to pixels being evaluated
         (for both low and high res). Default is None, replaced by setting all pixels to being evaluated.
         This cannot be a jnp array; must be a np array
        """
        super(RegularGrid, self).__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self._supersampling_factor = supersampling_factor
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)

        # This cannot be a jnp array
        if flux_evaluate_indexes is None:
            flux_evaluate_indexes = np.ones_like(self._x_grid)
        else:
            flux_evaluate_indexes = util_lenstronomy.image2array(flux_evaluate_indexes)
        # This cannot be a jnp array
        self._compute_indexes = self._subgrid_index(
            flux_evaluate_indexes, self._supersampling_factor, self._nx, self._ny
        ).astype(bool)

        x_grid_sub, y_grid_sub = util.make_subgrid(
            self._x_grid, self._y_grid, self._supersampling_factor
        )
        self._ra_subgrid = x_grid_sub[self._compute_indexes]
        self._dec_subgrid = y_grid_sub[self._compute_indexes]

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._ra_subgrid, self._dec_subgrid

    @property
    def grid_points_spacing(self):
        """Effective spacing between coordinate points, after supersampling.

        :return: sqrt(pixel_area)/supersampling_factor.
        """
        return self.pixel_width / self._supersampling_factor

    @property
    def num_grid_points_axes(self):
        """Effective number of points along each axes, after supersampling.

        :return: number of pixels per axis, nx*supersampling_factor
            ny*supersampling_factor
        """
        return (
            self._nx * self._supersampling_factor,
            self._ny * self._supersampling_factor,
        )

    @property
    def supersampling_factor(self):
        """
        :return: factor (per axis) of super-sampling relative to a pixel
        """
        return self._supersampling_factor

    @partial(jit, static_argnums=0)
    def flux_array2image_low_high(self, flux_array, **kwargs):
        """

        :param flux_array: 1d array of low and high resolution flux values corresponding
            to the coordinates_evaluate order
        :return: 2d array, 2d array, corresponding to (partial) images in low and high
            resolution (to be convolved)
        """
        image = self._array2image(flux_array)
        if self._supersampling_factor > 1:
            image_high_res = image
            image_low_res = image_util.re_size(image, self._supersampling_factor)
        else:
            image_high_res = None
            image_low_res = image
        return image_low_res, image_high_res

    @staticmethod
    def _subgrid_index(idex_mask, subgrid_res, nx, ny):
        """

        :param idex_mask: 1d array of mask of data
        :param subgrid_res: subgrid resolution
        :return: 1d array of equivalent mask in subgrid resolution
        """
        idex_sub = np.repeat(idex_mask, subgrid_res, axis=0)
        idex_sub = util_lenstronomy.array2image(idex_sub, nx=nx, ny=ny * subgrid_res)
        idex_sub = np.repeat(idex_sub, subgrid_res, axis=0)
        idex_sub = util_lenstronomy.image2array(idex_sub)
        return idex_sub

    @partial(jit, static_argnums=(0))
    def _array2image(self, array):
        """Maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask
        indices.

        :param array: 1d array
        :return:
        """
        nx, ny = (
            self._nx * self._supersampling_factor,
            self._ny * self._supersampling_factor,
        )
        grid1d = jnp.zeros((nx * ny))
        grid1d = grid1d.at[self._compute_indexes].set(array)
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d
