__author__ = "Simon Birrer"
"""This file contains standard routines.

TODO: Import jaxified versions of other lenstronomy.Util.util.py
      functions (when necessary)
"""

from functools import partial
from jax import jit, numpy as jnp
import numpy as np

from lenstronomy.Util.util import make_grid, local_minima_2d, selectBest
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
@partial(jit, static_argnums=(1,2))
def array2image(array, nx=0, ny=0):
    """Returns the information contained in a 1d array into an n*n 2d array (only works
    when length of array is n**2, or nx and ny are provided)

    :param array: image values
    :type array: array of size n**2
    :returns: 2d array
    :raises: ValueError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(jnp.size(array)))
        if n**2 != jnp.size(array):
            raise ValueError(
                "length of input array given as %s is not square of integer number!"
                % (jnp.size(array))
            )
        nx, ny = n, n
    image = jnp.reshape(array, (nx, ny))
    return image

@export
@jit
def displaceAbs(x, y, sourcePos_x, sourcePos_y):
    """Calculates a grid of distances to the observer in angel.

    :param x: cartesian coordinates
    :type x: numpy array
    :param y: cartesian coordinates
    :type y: numpy array
    :param sourcePos_x: source position
    :type sourcePos_x: float
    :param sourcePos_y: source position
    :type sourcePos_y: float
    :returns: array of displacement
    :raises: AttributeError, KeyError
    """
    x_mapped = x - sourcePos_x
    y_mapped = y - sourcePos_y
    absmapped = jnp.sqrt(x_mapped**2 + y_mapped**2)
    return absmapped

@export
@jit
def image2array(image):
    """Returns the information contained in a 2d array into an n*n 1d array.

    :param image: image values
    :type image: array of size (n,n)
    :returns: 1d array
    :raises: AttributeError, KeyError
    """
    imgh = jnp.reshape(image, jnp.size(image))  # change the shape to be 1d
    return imgh

@export
@partial(jit, static_argnums=2)
def make_subgrid(ra_coord, dec_coord, subgrid_res=2):
    """Return a grid with subgrid resolution.

    :param ra_coord:
    :param dec_coord:
    :param subgrid_res:
    :return:
    """
    ra_array = array2image(ra_coord)
    dec_array = array2image(dec_coord)
    n = jnp.size(ra_array, axis=0)
    d_ra_x = ra_array.at[0,1].get() - ra_array.at[0,0].get()
    d_ra_y = ra_array.at[1,0].get() - ra_array.at[0,0].get()
    d_dec_x = dec_array.at[0,1].get() - dec_array.at[0,0].get()
    d_dec_y = dec_array.at[1,0].get() - dec_array.at[0,0].get()

    ra_array_new = jnp.zeros((n * subgrid_res, n * subgrid_res))
    dec_array_new = jnp.zeros((n * subgrid_res, n * subgrid_res))
    for i in range(0, subgrid_res):
        for j in range(0, subgrid_res):
            ra_array_new = ra_array_new.at[i::subgrid_res, j::subgrid_res].set(
                ra_array
                + d_ra_x * (-1 / 2.0 + 1 / (2.0 * subgrid_res) + j / float(subgrid_res))
                + d_ra_y * (-1 / 2.0 + 1 / (2.0 * subgrid_res) + i / float(subgrid_res))
            )
            dec_array_new = dec_array_new.at[i::subgrid_res, j::subgrid_res].set(
                dec_array
                + d_dec_x
                * (-1 / 2.0 + 1 / (2.0 * subgrid_res) + j / float(subgrid_res))
                + d_dec_y
                * (-1 / 2.0 + 1 / (2.0 * subgrid_res) + i / float(subgrid_res))
            )
    ra_coords_sub = image2array(ra_array_new)
    dec_coords_sub = image2array(dec_array_new)
    return ra_coords_sub, dec_coords_sub

@export
@jit
def rotate(xcoords, ycoords, angle):
    """
    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    return xcoords * jnp.cos(angle) + ycoords * jnp.sin(angle), -xcoords * jnp.sin(
        angle
    ) + ycoords * jnp.cos(angle)