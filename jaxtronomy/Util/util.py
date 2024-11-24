__author__ = "Simon Birrer"
"""This file contains standard routines.

TODO: Import jaxified versions of other lenstronomy.Util.util.py
      functions (when necessary)
"""

from functools import partial
from jax import jit, numpy as jnp
import numpy as np

from lenstronomy.Util.util import local_minima_2d, make_grid, make_subgrid, selectBest
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
@partial(jit, static_argnums=(1, 2))
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
