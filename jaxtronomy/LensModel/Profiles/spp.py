__author__ = "sibirrer"

from jax import jit
import jax.numpy as jnp
from jax.scipy import special

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["SPP"]


class SPP(LensProfileBase):
    """Class for circular power-law mass distribution."""

    param_names = ["theta_E", "gamma", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    @jit
    def function(x, y, theta_E, gamma, center_x=0, center_y=0):
        """
        :param x: set of x-coordinates
        :type x: array of size (n)
        :param y: set of y-coordinates
        :type y: array of size (n)
        :param theta_E: Einstein radius of lens
        :type theta_E: float.
        :param gamma: power law slope of mass profile
        :type gamma: <2 float
        :returns:  function
        :raises: AttributeError, KeyError
        """
        gamma = SPP._gamma_limit(gamma)

        x_ = x - center_x
        y_ = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3

        p2 = x_**2 + y_**2
        s2 = 0.0  # softening
        return 2 * E**2 / eta**2 * ((p2 + s2) / E**2) ** (eta / 2)

    @staticmethod
    @jit
    def derivatives(x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        gamma = SPP._gamma_limit(gamma)

        xt1 = x - center_x
        xt2 = y - center_y

        r2 = xt1 * xt1 + xt2 * xt2
        a = jnp.maximum(r2, 0.000001)
        r = jnp.sqrt(a)
        alpha = theta_E * (r2 / theta_E**2) ** (1 - gamma / 2.0)
        fac = alpha / r
        f_x = fac * xt1
        f_y = fac * xt2
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        gamma = SPP._gamma_limit(gamma)
        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3.0

        P2 = xt1**2 + xt2**2
        a = jnp.where(P2 < 0.000001, 0.000001, P2)

        kappa = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta - 2) * (xt1**2 + xt2**2) / a + (1 + 1))
        )
        gamma1 = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta / 2 - 1) * (2 * xt1**2 - 2 * xt2**2) / a)
        )
        gamma2 = (
            4 * xt1 * xt2 * (1.0 / 2 - 1 / eta) * (a / E**2) ** (eta / 2 - 2) / E**2
        )

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def rho2theta(rho0, gamma):
        """Converts 3d density into 2d projected density parameter.

        :param rho0:
        :param gamma:
        :return:
        """
        fac = (
            jnp.sqrt(jnp.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
            * rho0
        )

        # fac = theta_E**(gamma - 1)
        theta_E = fac ** (1.0 / (gamma - 1))
        return theta_E

    @staticmethod
    @jit
    def theta2rho(theta_E, gamma):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter.

        :param theta_E:
        :param gamma:
        :return:
        """
        fac1 = (
            jnp.sqrt(jnp.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
        )
        fac2 = theta_E ** (gamma - 1)
        rho0 = fac2 / fac1
        return rho0

    @staticmethod
    @jit
    def mass_3d(r, rho0, gamma):
        """Mass enclosed a 3d sphere or radius r.

        :param r:
        :param rho0:
        :param gamma:
        :return:
        """
        mass_3d = 4 * jnp.pi * rho0 / (-gamma + 3) * r ** (-gamma + 3)
        return mass_3d

    @staticmethod
    @jit
    def mass_3d_lens(r, theta_E, gamma):
        """

        :param r:
        :param theta_E:
        :param gamma:
        :return:
        """
        rho0 = SPP.theta2rho(theta_E, gamma)
        return SPP.mass_3d(r, rho0, gamma)

    @staticmethod
    @jit
    def mass_2d(r, rho0, gamma):
        """Mass enclosed projected 2d sphere of radius r.

        :param r:
        :param rho0:
        :param gamma:
        :return:
        """
        alpha = (
            jnp.sqrt(jnp.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (2 - gamma)
            / (3 - gamma)
            * 2
            * rho0
        )
        mass_2d = alpha * r * jnp.pi
        return mass_2d

    @staticmethod
    @jit
    def mass_2d_lens(r, theta_E, gamma):
        """

        :param r: projected radius
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :return: 2d projected radius enclosed
        """
        rho0 = SPP.theta2rho(theta_E, gamma)
        return SPP.mass_2d(r, rho0, gamma)

    @staticmethod
    @jit
    def grav_pot(x, y, rho0, gamma, center_x=0, center_y=0):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units)

        :param x:
        :param y:
        :param rho0:
        :param gamma:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        mass_3d = SPP.mass_3d(r, rho0, gamma)
        pot = mass_3d / r
        return pot

    @staticmethod
    @jit
    def density(r, rho0, gamma):
        """Computes the density.

        :param r:
        :param rho0:
        :param gamma:
        :return:
        """
        rho = rho0 / r**gamma
        return rho

    @staticmethod
    @jit
    def density_lens(r, theta_E, gamma):
        """Computes the density at 3d radius r given lens model parameterization.

        The integral in projected in units of angles (i.e. arc seconds) results in the
        convergence quantity.
        """
        rho0 = SPP.theta2rho(theta_E, gamma)
        return SPP.density(r, rho0, gamma)

    @staticmethod
    @jit
    def density_2d(x, y, rho0, gamma, center_x=0, center_y=0):
        """Projected density.

        :param x:
        :param y:
        :param rho0:
        :param gamma:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        sigma = (
            jnp.sqrt(jnp.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (1 - gamma)
            * rho0
        )
        return sigma

    @staticmethod
    @jit
    def _gamma_limit(gamma):
        """Limits the power-law slope to certain bounds.

        :param gamma: power-law slope
        :return: bounded power-law slopte
        """
        return gamma
