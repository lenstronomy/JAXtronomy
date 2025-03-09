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
        scipy.ndimage.map_coordinates(kernel, coordinates=[y_grid, x_grid], order=1)
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


@jit
def shift(input, shift):
    """Equivalent to scipy.ndimage.shift in the specific case where:

    - the input array is 2 dimensional
    - shift = [shift_row, shift_col] with shift_row and shift_col between -1 and 1
    - the order of spline interpolation is 1
    - mode is "constant" with cval = 0.0

    :param input: 2 dimensional array to be shifted
    :param shift: float, amount to shift the rows and columns by. Can also be a
        tuple or list of floats, i.e. shift = [shift_row, shift_col]
    :param shift_row: float between -1 and 1, amount to shift the rows up or down by
    :param shift_col: float between -1 and 1, amount to shift the columns left or right by
    """
    old_array = jnp.array(input)
    if old_array.ndim != 2:
        raise ValueError("This function only supports 2 dimensional input arrays")

    new_array = jnp.zeros_like(old_array)
    shift = jnp.array(shift)
    if shift.shape == ():
        shift_row = shift
        shift_col = shift
    elif shift.shape == (2,):
        shift_row = shift[0]
        shift_col = shift[1]
    else:
        raise ValueError(
            "shift must be either a float or a list/tuple/array of floats with length 2"
        )

    # If the shift value is 0, apply no shift
    def no_shift(new_array, old_array, shift):
        return old_array

    # Shifts the rows of the array downwards if shift_row > 0
    def shift_down(new_array, old_array, shift):
        num_rows = len(old_array)

        def body_fun(i, new_array):
            row_index = num_rows - i
            value = (
                old_array.at[row_index].get() * (1.0 - shift)
                + old_array.at[row_index - 1].get() * shift
            )
            new_array = new_array.at[row_index].set(value)
            return new_array

        return lax.fori_loop(1, num_rows, body_fun, new_array)

    # Shifts the rows of the array upwards if shift_row < 0
    def shift_up(new_array, old_array, shift):
        num_rows = len(old_array)

        def body_fun(row_index, new_array):
            value = (
                old_array.at[row_index].get() * (1.0 + shift)
                - old_array.at[row_index + 1].get() * shift
            )
            new_array = new_array.at[row_index].set(value)
            return new_array

        return lax.fori_loop(0, num_rows - 1, body_fun, new_array)

    # Use the JAX implementation of switch-cases
    func_list = [no_shift, shift_down, shift_up]
    case_col = jnp.where(shift_col == 0, 0, jnp.where(shift_col > 0, 1, 2))
    case_row = jnp.where(shift_row == 0, 0, jnp.where(shift_row > 0, 1, 2))

    # Shift the rows and save the result as the old array
    old_array = lax.switch(case_row, func_list, new_array, old_array, shift_row)
    new_array = jnp.zeros_like(old_array)

    # Shifting columns is equivalent to taking the transpose and shifting rows
    return lax.switch(case_col, func_list, new_array, old_array.T, shift_col).T
