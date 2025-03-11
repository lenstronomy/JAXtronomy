__author__ = "sibirrer"

from functools import partial
from jax import jit, lax, numpy as jnp, scipy
import numpy as np


@partial(jit, static_argnums=4)
def add_layer2image(grid2d, x_pos, y_pos, kernel, order=1):
    """Adds a kernel on the grid2d image at position x_pos, y_pos with an interpolated
    subgrid pixel shift of order=order.

    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :param order: interpolation order for sub-pixel shift of the kernel to be added
    :return: image with added layer, cut to original size
    """
    if order > 1:
        raise ValueError(f"interpolation order > 1 is not supported in jaxtronomy")
    k_rows, k_cols = jnp.shape(kernel)

    if k_rows % 2 == 0 or k_cols % 2 == 0:
        raise ValueError("kernel dimensions must be odd")

    n_row, n_col = jnp.shape(grid2d)

    # Create a coordinate grid where the origin is placed at the point source
    # shifted left and up by the kernel radius
    xrange = jnp.arange(n_col) + k_cols // 2 - x_pos
    yrange = jnp.arange(n_row) + k_rows // 2 - y_pos
    x_grid, y_grid = jnp.meshgrid(xrange, yrange)

    # Maps kernel onto coordinate grid and add original image
    # Row indices are given by the y_grid and column indices are given by the x_grid
    return (
        scipy.ndimage.map_coordinates(kernel, coordinates=[y_grid, x_grid], order=order)
        + grid2d
    )


@jit
def add_layer2image_int(grid2d, x_pos, y_pos, kernel):
    """Adds a kernel on the grid2d image at position x_pos, y_pos at integer positions
    of pixel.

    :param grid2d: 2d pixel grid (i.e. image)
    :param x_pos: x-position center (pixel coordinate) of the layer to be added
    :param y_pos: y-position center (pixel coordinate) of the layer to be added
    :param kernel: the layer to be added to the image
    :return: image with added layer
    """

    n_row, n_col = jnp.shape(grid2d)
    k_rows, k_cols = jnp.shape(kernel)

    if k_rows % 2 == 0 or k_cols % 2 == 0:
        raise ValueError("kernel dimensions must be odd")

    x_int = (jnp.round(x_pos)).astype(int)
    y_int = (jnp.round(y_pos)).astype(int)

    # Create a coordinate grid where the origin is placed at the point source
    # shifted left and up by the kernel radius
    xrange = jnp.arange(n_col) + k_cols // 2 - x_int
    yrange = jnp.arange(n_row) + k_rows // 2 - y_int
    x_grid, y_grid = jnp.meshgrid(xrange, yrange)

    # Maps kernel onto coordinate grid and add original image
    # Row indices are given by the y_grid and column indices are given by the x_grid
    return (
        scipy.ndimage.map_coordinates(kernel, coordinates=[y_grid, x_grid], order=0)
        + grid2d
    )


@partial(jit, static_argnums=1)
def re_size(image, factor=1):
    """Re-sizes image with nx x ny to nx/factor x ny/factor.

    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError("scaling factor in re-sizing %s < 1" % factor)
    elif factor == 1:
        return image
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx / f) == nx / f and int(ny / f) == ny / f:
        small = image.reshape([int(nx / f), f, int(ny / f), f]).mean(3).mean(1)
        return small
    else:
        raise ValueError(
            "scaling with factor %s is not possible with grid size %s, %s" % (f, nx, ny)
        )
