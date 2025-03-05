__author__ = "sibirrer"
# this file contains a class to make a gaussian

import jax
from jax import jit, lax, numpy as jnp, tree_util
import jax.scipy.special
import numpy as np

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

    ds = 0.00001

    @staticmethod
    @jit
    def function(x, y, amp, sigma, center_x=0, center_y=0):
        """Returns potential for a Gaussian convergence.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        c = 1.0 / (2 * sigma_x * sigma_y)
        num_int = Gaussian._num_integral(r, c)
        amp_density = Gaussian._amp2d_to_3d(amp, sigma_x, sigma_y)
        amp2d = amp_density / (np.sqrt(2.0 * np.pi) * jnp.sqrt(sigma_x * sigma_y))
        amp2d *= 2 * 1.0 / (2 * c)
        return num_int * amp2d

    @staticmethod
    @jit
    def _num_integral(r, c):
        """Numerical integral of (1-e^{-c*x^2})/x dx from 0 to r calculated using
        Weddle's rule on 100 subintervals.

        If r is an array of size n, then there are n integrals which are computed
        vectorially. This differs from lenstronomy's implementation, where r can only be
        a scalar.

        :param r: array-like, radius
        :param c: float, 1/2sigma^2
        :return: Array with the same shape as r containing the result for each integral
        """
        r = jnp.array(r)
        r_shape = r.shape
        r = jnp.ravel(r)

        subinterval_widths = r / 100.0
        coeffs = np.array([1, 5, 1, 6, 1, 5, 1], dtype=float) / 20

        def weddles_rule(i, sum):
            """Computes the integral of f_x over the i-th subinterval using Weddle's
            rule.

            See https://mathworld.wolfram.com/WeddlesRule.html for details.
            """
            x = (jnp.ones((7, len(r))) * subinterval_widths).T * (
                jnp.linspace(0.0, 1.0, 7) + i
            )
            # This function has a removable discontinuity at x = 0
            f_x = (1.0 - jnp.exp(-c * x**2)) / x
            f_x = jnp.where(x == 0, 0, f_x)

            sum += subinterval_widths * jnp.sum(f_x * coeffs, axis=1)
            return sum

        sum = jnp.zeros_like(r, dtype=float)
        return lax.fori_loop(0, 100, weddles_rule, sum).reshape(r_shape)

    @staticmethod
    @jit
    def derivatives(x, y, amp, sigma, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        R = jnp.where(R <= Gaussian.ds, Gaussian.ds, R)
        alpha = Gaussian.alpha_abs(R, amp, sigma)
        return alpha / R * x_, alpha / R * y_

    @staticmethod
    @jit
    def hessian(x, y, amp, sigma, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: x position
        :param y: y position
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        sigma_x, sigma_y = sigma, sigma
        r = jnp.where(r < Gaussian.ds, Gaussian.ds, r)
        d_alpha_dr = -Gaussian.d_alpha_dr(r, amp, sigma_x, sigma_y)
        alpha = Gaussian.alpha_abs(r, amp, sigma)

        f_xx = -(d_alpha_dr / r + alpha / r**2) * x_**2 / r + alpha / r
        f_yy = -(d_alpha_dr / r + alpha / r**2) * y_**2 / r + alpha / r
        f_xy = -(d_alpha_dr / r + alpha / r**2) * x_ * y_ / r
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def density(r, amp, sigma):
        """3d mass density as a function of radius r.

        :param r: radius
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        return GAUSSIAN_INSTANCE.function(r, 0, amp, sigma_x, sigma_y)

    @staticmethod
    @jit
    def density_2d(x, y, amp, sigma, center_x=0, center_y=0):
        """Projected 2d density at position (x,y)

        :param x: x position
        :param y: y position
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        :param center_x: x position of the center of the lens
        :param center_y: y position of the center of the lens
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = Gaussian._amp3d_to_2d(amp, sigma_x, sigma_y)
        return GAUSSIAN_INSTANCE.function(
            x, y, amp2d, sigma_x, sigma_y, center_x, center_y
        )

    @staticmethod
    @jit
    def mass_2d(R, amp, sigma):
        """Mass enclosed in a circle of radius R when projected into 2d.

        :param R: projected radius
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp2d = amp / (np.sqrt(np.pi) * jnp.sqrt(sigma_x * sigma_y * 2))
        c = 1.0 / (2 * sigma_x * sigma_y)
        return amp2d * 2 * np.pi * 1.0 / (2 * c) * (1.0 - jnp.exp(-c * R**2))

    @staticmethod
    @jit
    def mass_2d_lens(R, amp, sigma):
        """Mass enclosed in a circle of radius R when projected into 2d.

        :param R: projected radius
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = Gaussian._amp2d_to_3d(amp, sigma_x, sigma_y)
        return Gaussian.mass_2d(R, amp_density, sigma)

    @staticmethod
    @jit
    def alpha_abs(R, amp, sigma):
        """Absolute value of the deflection.

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = Gaussian._amp2d_to_3d(amp, sigma_x, sigma_y)
        alpha = Gaussian.mass_2d(R, amp_density, sigma) / np.pi / R
        return alpha

    @staticmethod
    @jit
    def d_alpha_dr(R, amp, sigma_x, sigma_y):
        """Derivative of deflection angle w.r.t r.

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        c = 1.0 / (2.0 * sigma_x * sigma_y)
        A = Gaussian._amp2d_to_3d(amp, sigma_x, sigma_y) * (
            np.sqrt(2.0 / np.pi) * jnp.sqrt(sigma_x * sigma_y)
        )
        return 1.0 / R**2 * (-1.0 + (1.0 + 2.0 * c * R**2) * jnp.exp(-c * R**2)) * A

    @staticmethod
    @jit
    def mass_3d(R, amp, sigma):
        """Mass enclosed within a 3D sphere of projected radius R given a lens
        parameterization with angular units. The input parameter amp is the 3d
        amplitude.

        :param R: radius projected into 2d
        :param amp: 3d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        A = amp / (2 * np.pi * sigma_x * sigma_y)
        c = 1.0 / (2 * sigma_x * sigma_y)
        result = (
            1.0
            / (2 * c)
            * (
                -R * jnp.exp(-c * R**2)
                + jax.scipy.special.erf(jnp.sqrt(c) * R)
                * np.sqrt(np.pi / 4.0)
                * jnp.sqrt(1.0 / c)
            )
        )
        return result * A * 4 * np.pi

    @staticmethod
    @jit
    def mass_3d_lens(R, amp, sigma):
        """Mass enclosed within a 3D sphere of projected radius R given a lens
        parameterization with angular units. The input parameters are identical as for
        the derivatives definition. (optional definition)

        :param R: radius projected into 2d
        :param amp: 2d amplitude of Gaussian
        :param sigma: standard deviation of Gaussian
        """
        sigma_x, sigma_y = sigma, sigma
        amp_density = Gaussian._amp2d_to_3d(amp, sigma_x, sigma_y)
        return Gaussian.mass_3d(R, amp_density, sigma)

    @staticmethod
    @jit
    def _amp3d_to_2d(amp, sigma_x, sigma_y):
        """Converts 3d density into 2d density parameter.

        :param amp: 3d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        return amp * np.sqrt(2 * np.pi) * jnp.sqrt(sigma_x * sigma_y)

    @staticmethod
    @jit
    def _amp2d_to_3d(amp, sigma_x, sigma_y):
        """Converts 2d density into 3d density parameter.

        :param amp: 2d amplitude of Gaussian
        :param sigma_x: standard deviation of Gaussian in x direction
        :param sigma_y: standard deviation of Gaussian in y direction
        """
        return amp / (np.sqrt(2 * np.pi) * jnp.sqrt(sigma_x * sigma_y))
