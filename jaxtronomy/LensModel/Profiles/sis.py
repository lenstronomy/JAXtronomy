__author__ = "sibirrer"

from functools import partial
from jax import config, jit, numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

config.update("jax_enable_x64", True)  # 64-bit floats

__all__ = ["SIS"]


class SIS(LensProfileBase):
    """This class contains the function and the derivatives of the Singular Isothermal
    Sphere.

    .. math::
        \\kappa(x, y) = \\frac{1}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{x^2 + y^2}} \\right)

    with :math:`\\theta_{E}` is the Einstein radius,
    """

    param_names = ["theta_E", "center_x", "center_y"]
    lower_limit_default = {"theta_E": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"theta_E": 100, "center_x": 100, "center_y": 100}

    _epsilon = 0.000001

    @staticmethod
    @jit
    def function(x, y, theta_E, center_x=0, center_y=0):
        x_shift = x - center_x
        y_shift = y - center_y
        f_ = theta_E * jnp.sqrt(x_shift * x_shift + y_shift * y_shift)
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, theta_E, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function."""
        x_shift = x - center_x
        y_shift = y - center_y
        R = jnp.sqrt(x_shift * x_shift + y_shift * y_shift)
        R = jnp.where(R < SIS._epsilon, SIS._epsilon, R)
        a = theta_E / jnp.maximum(R, SIS._epsilon)
        f_x = a * x_shift
        f_y = a * y_shift
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, theta_E, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
        d^f/dy^2."""
        x_shift = x - center_x
        y_shift = y - center_y
        R = (x_shift * x_shift + y_shift * y_shift) ** (3.0 / 2)
        R = jnp.where(R < SIS._epsilon, SIS._epsilon, R)
        prefac = theta_E / jnp.maximum(SIS._epsilon, R)
        f_xx = y_shift * y_shift * prefac
        f_yy = x_shift * x_shift * prefac
        f_xy = -x_shift * y_shift * prefac
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def rho2theta(rho0):
        """Converts 3d density into 2d projected density parameter :param rho0:

        :return:
        """
        theta_E = jnp.pi * 2 * rho0
        return theta_E

    @staticmethod
    @jit
    def theta2rho(theta_E):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter :param theta_E: Einstein radius :return:"""
        fac1 = jnp.pi * 2
        rho0 = theta_E / fac1
        return rho0

    @staticmethod
    @jit
    def mass_3d(r, rho0):
        """Mass enclosed a 3d sphere or radius r :param r: radius in angular units
        :param rho0: density at angle=1 :return: mass in angular units."""
        mass_3d = 4 * jnp.pi * rho0 * r
        return mass_3d

    @staticmethod
    @jit
    def mass_3d_lens(r, theta_E):
        """Mass enclosed a 3d sphere or radius r given a lens parameterization with
        angular units.

        :param r: radius in angular units
        :param theta_E: Einstein radius
        :return: mass in angular units
        """
        rho0 = SIS.theta2rho(theta_E)
        return SIS.mass_3d(r, rho0)

    @staticmethod
    @jit
    def mass_2d(r, rho0):
        """Mass enclosed projected 2d sphere of radius r :param r:

        :param rho0:
        :return:
        """
        alpha = 2 * rho0 * jnp.pi**2
        mass_2d = alpha * r
        return mass_2d

    @staticmethod
    @jit
    def mass_2d_lens(r, theta_E):
        """

        :param r: radius
        :param theta_E: Einstein radius
        :return: mass within a radius in projection
        """
        rho0 = SIS.theta2rho(theta_E)
        return SIS.mass_2d(r, rho0)

    @staticmethod
    @jit
    def grav_pot(x, y, rho0, center_x=0, center_y=0):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units) :param
        x:

        :param y:
        :param rho0:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        mass_3d = SIS.mass_3d(r, rho0)
        pot = mass_3d / r
        return pot

    @staticmethod
    @jit
    def density(r, rho0):
        """Computes the density :param r: radius in angles :param rho0: density at
        angle=1 :return: density at r."""
        rho = rho0 / r**2
        return rho

    @staticmethod
    @jit
    def density_lens(r, theta_E):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in projected in units of angles (i.e. arc seconds) results in the
        convergence quantity.

        :param r: 3d radius
        :param theta_E: Einstein radius
        :return: density(r)
        """
        rho0 = SIS.theta2rho(theta_E)
        return SIS.density(r, rho0)

    @staticmethod
    @jit
    def density_2d(x, y, rho0, center_x=0, center_y=0):
        """Projected density :param x:

        :param y:
        :param rho0:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        sigma = jnp.pi * rho0 / r
        return sigma
