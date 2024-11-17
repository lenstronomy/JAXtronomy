__author__ = "sibirrer"
# this file contains a class to make a gaussian

import jax
from jax import jit, lax, numpy as jnp, tree_util
import jax.scipy.special
from jaxtronomy.LensModel.Profiles.gaussian_potential import GaussianPotential
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

jax.config.update("jax_enable_x64", True)

__all__ = ["Gaussian"]

GAUSSIAN_INSTANCE = GaussianPotential()

class Gaussian(LensProfileBase):
    """This class contains functions to evaluate a Gaussian convergence and calculates
    its derivative and hessian matrix."""

    param_names = ["amp", "sigma", "center_x", "center_y"]
    lower_limit_default = {"amp": 0, "sigma": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"amp": 100, "sigma": 100, "center_x": 100, "center_y": 100}

    def __init__(self):
        self.ds = 0.00001
        super(LensProfileBase, self).__init__()

    def _tree_flatten(self):
        children = ()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns potential for a Gaussian convergence."""
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        c = 1.0 / (2 * sigma_x * sigma_y)
        num_int = Gaussian._num_integral(r, c)
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        amp2d = amp_density / (jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2))
        amp2d *= 2 * 1.0 / (2 * c)
        return num_int * amp2d

    @staticmethod
    @jit
    def _num_integral(r, c):
        """Numerical integral of (1-e^{-c*x^2})/x dx from 0 to r
        calculated using a trapezoidal sum with 1000 steps.
        If r is a 1D array of size n, then there are n integrals
        which are computed in parallel.

        :param r: radius
        :param c: 1/2sigma^2
        :return: Array with the same shape as r containing the result
         for each integral
        """
        def trapezoidal_sum(i, val):
            sum = val
            dx = r/num_iterations
            x_left = dx * i
            x_right = dx * (i + 1)
            y_left = jnp.where(x_left == 0, 0, (1 - jnp.exp(-c * x_left**2))/x_left)
            y_right = (1 - jnp.exp(-c * x_right**2))/x_right
            sum += (y_left + y_right)/2 * dx
            return sum

        num_iterations = 200
        sum = jnp.zeros_like(r, dtype=float)
        return lax.fori_loop(0, num_iterations, trapezoidal_sum, sum)

    @jit
    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function."""
        x_ = x - center_x
        y_ = y - center_y
        R = jnp.sqrt(x_**2 + y_**2)
        R = jnp.where(R <= self.ds, self.ds, R)
        alpha = self.alpha_abs(R, amp, sigma)
        return alpha / R * x_, alpha / R * y_

    @jit
    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
        d^f/dy^2."""
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        r = jnp.where(r < self.ds, self.ds, r)
        d_alpha_dr = -self.d_alpha_dr(r, amp, sigma_x, sigma_y)
        alpha = self.alpha_abs(r, amp, sigma)

        f_xx = -(d_alpha_dr / r + alpha / r**2) * x_**2 / r + alpha / r
        f_yy = -(d_alpha_dr / r + alpha / r**2) * y_**2 / r + alpha / r
        f_xy = -(d_alpha_dr / r + alpha / r**2) * x_ * y_ / r
        return f_xx, f_xy, f_xy, f_yy

    @jit
    def density(self, r, amp, sigma):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        return GAUSSIAN_INSTANCE.function(r, 0, amp, sigma_x, sigma_y)

    @jit
    def density_2d(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = self._amp3d_to_2d(amp, sigma_x, sigma_y)
        return GAUSSIAN_INSTANCE.function(x, y, amp2d, sigma_x, sigma_y, center_x, center_y)

    @jit
    def mass_2d(self, R, amp, sigma):
        """

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = amp / (jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2))
        c = 1.0 / (2 * sigma_x * sigma_y)
        return amp2d * 2 * jnp.pi * 1.0 / (2 * c) * (1.0 - jnp.exp(-c * R**2))

    @jit
    def mass_2d_lens(self, R, amp, sigma):
        """

        :param R:
        :param amp:
        :param sigma: 
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_2d(R, amp_density, sigma)

    @jit
    def alpha_abs(self, R, amp, sigma):
        """Absolute value of the deflection.

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        alpha = self.mass_2d(R, amp_density, sigma) / jnp.pi / R
        return alpha

    @jit
    def d_alpha_dr(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        c = 1.0 / (2 * sigma_x * sigma_y)
        A = self._amp2d_to_3d(amp, sigma_x, sigma_y) * jnp.sqrt(
            2 / jnp.pi * sigma_x * sigma_y
        )
        return 1.0 / R**2 * (-1 + (1 + 2 * c * R**2) * jnp.exp(-c * R**2)) * A

    @jit
    def mass_3d(self, R, amp, sigma):
        """

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        A = amp / (2 * jnp.pi * sigma_x * sigma_y)
        c = 1.0 / (2 * sigma_x * sigma_y)
        result = (
            1.0
            / (2 * c)
            * (
                -R * jnp.exp(-c * R**2)
                + jax.scipy.special.erf(jnp.sqrt(c) * R) * jnp.sqrt(jnp.pi / (4 * c))
            )
        )
        return result * A * 4 * jnp.pi

    @jit
    def mass_3d_lens(self, R, amp, sigma):
        """

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_3d(R, amp_density, sigma)

    @staticmethod
    @jit
    def _amp3d_to_2d(amp, sigma_x, sigma_y):
        """Converts 3d density into 2d density parameter.

        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return amp * jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2)

    @staticmethod
    @jit
    def _amp2d_to_3d(amp, sigma_x, sigma_y):
        """Converts 3d density into 2d density parameter.

        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return amp / (jnp.sqrt(jnp.pi) * jnp.sqrt(sigma_x * sigma_y * 2))
    

tree_util.register_pytree_node(Gaussian, Gaussian._tree_flatten, Gaussian._tree_unflatten)

