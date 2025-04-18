__author__ = "sibirrer"
# this file contains a class to make a gaussian

from jax import jit, numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["GaussianPotential"]


class GaussianPotential(LensProfileBase):
    """This class contains functions to evaluate a Gaussian potential and calculates its
    derivative and hessian matrix."""

    param_names = ["amp", "sigma_x", "sigma_y", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma_x": 0,
        "sigma_y": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "sigma_x": 100,
        "sigma_y": 100,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    @jit
    def function(x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns Gaussian.

        :param x: x position
        :param y: y position
        :param amp: amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in the x direction
        :param sigma_y: standard deviation of Gaussian in the y direction
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        c = amp / (2 * jnp.pi * sigma_x * sigma_y)
        delta_x = x - center_x
        delta_y = y - center_y
        exponent = -((delta_x / sigma_x) ** 2 + (delta_y / sigma_y) ** 2) / 2.0
        return c * jnp.exp(exponent)

    @staticmethod
    @jit
    def derivatives(x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function.

        :param x: x position
        :param y: y position
        :param amp: amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in the x direction
        :param sigma_y: standard deviation of Gaussian in the y direction
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        f_ = GaussianPotential.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        return f_ * (center_x - x) / sigma_x**2, f_ * (center_y - y) / sigma_y**2

    @staticmethod
    @jit
    def hessian(x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: x position
        :param y: y position
        :param amp: amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in the x direction
        :param sigma_y: standard deviation of Gaussian in the y direction
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        f_ = GaussianPotential.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        f_xx = f_ * ((-1.0 / sigma_x**2) + (center_x - x) ** 2 / sigma_x**4)
        f_yy = f_ * ((-1.0 / sigma_y**2) + (center_y - y) ** 2 / sigma_y**4)
        f_xy = f_ * (center_x - x) / sigma_x**2 * (center_y - y) / sigma_y**2
        return f_xx, f_xy, f_xy, f_yy
