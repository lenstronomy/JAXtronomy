__author__ = "Simon Birrer"
"""This file contains standard routines.

Copy pasted from Lenstronomy.
Only the rotate function has been jaxified.
TODO: Import jaxified version of everything else
      (when applicable) and write test functions
"""

from lenstronomy.Util.package_util import exporter
import jax.numpy as jnp
from jax import jit
from lenstronomy.Util.util import make_grid, local_minima_2d, selectBest

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
