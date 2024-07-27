__author__ = "Simon Birrer"
"""This file contains standard routines.

Copy pasted from Lenstronomy.
Only the rotate function has been jaxified.
TODO: Import jaxified version of everything else
      (when applicable) and write test functions
"""

import numpy as np
import itertools
from lenstronomy.Util.package_util import exporter
import jax.numpy as jnp
from jax import jit
from functools import partial
from lenstronomy.Util.util import make_grid, local_minima_2d

export, __all__ = exporter()


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


# NOTE: This code will recompile each time nx or ny changes value
# This function is best used if the number of pixels stays the same
@export
@partial(jit, static_argnums=(0,1))
def grid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0):
    """Return a grid in x and y coordinates that satisfy the coordinate system.

    :param nx: number of pixels in x-axis
    :param ny: number of pixels in y-axis
    :param Mpix2coord: transformation matrix (2x2) of pixels into coordinate
        displacements
    :param ra_at_xy_0: RA coordinate at (x,y) = (0,0)
    :param dec_at_xy_0: DEC coordinate at (x,y) = (0,0)
    :return: RA coordinate grid, DEC coordinate grid
    """
    a = jnp.arange(nx)
    b = jnp.arange(ny)
    matrix = jnp.dstack(jnp.meshgrid(a, b)).reshape(-1, 2)
    x_grid = matrix[:, 0]
    y_grid = matrix[:, 1]
    ra_grid = x_grid * Mpix2coord[0, 0] + y_grid * Mpix2coord[0, 1] + ra_at_xy_0
    dec_grid = x_grid * Mpix2coord[1, 0] + y_grid * Mpix2coord[1, 1] + dec_at_xy_0
    return ra_grid, dec_grid