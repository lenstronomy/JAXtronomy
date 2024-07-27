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