import numpy as np
from jax import jit, numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["PseudoJaffe"]


class PseudoJaffe(LensProfileBase):
    """
    class to compute the DUAL PSEUDO ISOTHERMAL MASS DISTRIBUTION
    based on Eliasdottir (2007) https://arxiv.org/pdf/0710.5636.pdf Appendix A

    Module name: 'PJAFFE';

    An alternative name is dPIED (in the elliptical scenario)

    This profile is for the spherical case. For an elliptical version, use
    "PJAFFE_ELLIPSE" (ellipticitly in the potential)
    # TODO: add/revise name once ellipticity in the mass is available

    The 3D density distribution is

    .. math::
        \\rho(r) = \\frac{\\rho_0}{(1+r^2/Ra^2)(1+r^2/Rs^2)}

    with :math:`Rs > Ra`.

    The projected density is

    .. math::
        \\Sigma(R) = \\Sigma_0 \\frac{Ra Rs}{Rs-Ra}\\left(\\frac{1}{\\sqrt{Ra^2+R^2}} - \\frac{1}{\\sqrt{Rs^2+R^2}} \\right)

    with

    .. math::
        \\Sigma_0 = \\pi \\rho_0 \\frac{Ra Rs}{Rs + Ra}

    In the lensing parameterization,

    .. math::
        \\sigma_0 = \\frac{\\Sigma_0}{\\Sigma_{\\rm crit}}

    """

    param_names = ["sigma0", "Ra", "Rs", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "center_x": 100,
        "center_y": 100,
    }

    # Define this as class variable instead of instance variable to avoid recompiling
    # upon creating multiple instances of PseudoJaffe
    _s = 0.0001

    @staticmethod
    @jit
    def density(r, rho0, Ra, Rs):
        """Computes the density.

        :param r: radial distance from the center (in 3D)
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: density at r
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        rho = rho0 / ((1 + (r / Ra) ** 2) * (1 + (r / Rs) ** 2))
        return rho

    @staticmethod
    @jit
    def density_2d(x, y, rho0, Ra, Rs, center_x=0, center_y=0):
        """Projected density.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :param center_x: center of profile
        :param center_y: center of profile
        :return: projected density
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        sigma0 = PseudoJaffe.rho2sigma(rho0, Ra, Rs)
        sigma = (
            sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (1 / jnp.sqrt(Ra**2 + r**2) - 1 / jnp.sqrt(Rs**2 + r**2))
        )
        return sigma

    @staticmethod
    @jit
    def mass_3d(r, rho0, Ra, Rs):
        """Mass enclosed a 3d sphere or radius r.

        :param r: radial distance from the center (in 3D)
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: M(<r)
        """
        m_3d = (
            4
            * np.pi
            * rho0
            * Ra**2
            * Rs**2
            / (Rs**2 - Ra**2)
            * (Rs * jnp.arctan(r / Rs) - Ra * jnp.arctan(r / Ra))
        )
        return m_3d

    @staticmethod
    @jit
    def mass_3d_lens(r, sigma0, Ra, Rs):
        """Mass enclosed a 3d sphere or radius r given a lens parameterization with
        angular units.

        :param r: radial distance from the center (in 3D)
        :param sigma0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: M(<r) in angular units (modulo critical mass density)
        """
        rho0 = PseudoJaffe.sigma2rho(sigma0, Ra, Rs)
        return PseudoJaffe.mass_3d(r, rho0, Ra, Rs)

    @staticmethod
    @jit
    def mass_2d(r, rho0, Ra, Rs):
        """Mass enclosed projected 2d sphere of radius r.

        :param r: radial distance from the center in projection
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: Sigma(<r)
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        sigma0 = PseudoJaffe.rho2sigma(rho0, Ra, Rs)
        m_2d = (
            2
            * np.pi
            * sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (jnp.sqrt(Ra**2 + r**2) - Ra - jnp.sqrt(Rs**2 + r**2) + Rs)
        )
        return m_2d

    @staticmethod
    @jit
    def mass_tot(rho0, Ra, Rs):
        """Total mass within the profile.

        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: total mass
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        sigma0 = PseudoJaffe.rho2sigma(rho0, Ra, Rs)
        m_tot = 2 * np.pi * sigma0 * Ra * Rs
        return m_tot

    @staticmethod
    @jit
    def grav_pot(r, rho0, Ra, Rs):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units)

        :param r: radial distance from the center (in 3D)
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        pot = (
            4
            * np.pi
            * rho0
            * Ra**2
            * Rs**2
            / (Rs**2 - Ra**2)
            * (
                Rs / r * jnp.arctan(r / Rs)
                - Ra / r * jnp.arctan(r / Ra)
                + 1.0 / 2 * jnp.log((Rs**2 + r**2) / (Ra**2 + r**2))
            )
        )
        return pot

    @staticmethod
    @jit
    def function(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """Lensing potential.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param sigma0: sigma0/sigma_crit (see class documentation above)
        :param Ra: core radius (see class documentation above)
        :param Rs: transition radius from logarithmic slope -2 to -4 (see class
            documentation above)
        :param center_x: center of profile
        :param center_y: center of profile
        :return: lensing potential
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        f_ = (
            -2
            * sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (
                jnp.sqrt(Rs**2 + r**2)
                - jnp.sqrt(Ra**2 + r**2)
                + Ra * jnp.log(Ra + jnp.sqrt(Ra**2 + r**2))
                - Rs * jnp.log(Rs + jnp.sqrt(Rs**2 + r**2))
            )
        )
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """Deflection angles.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param sigma0: sigma0/sigma_crit (see class documentation above)
        :param Ra: core radius (see class documentation above)
        :param Rs: transition radius from logarithmic slope -2 to -4 (see class
            documentation above)
        :param center_x: center of profile
        :param center_y: center of profile
        :return: f_x, f_y
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        r = jnp.where(r < PseudoJaffe._s, PseudoJaffe._s, r)

        # There is a 0/0 here if Ra = Rs which can be avoided by taking the limit as Ra -> Rs
        factor1 = PseudoJaffe._f_A20(r / Ra, r / Rs) / (Rs - Ra)
        factor2 = (
            r / (Rs + jnp.sqrt(Rs**2 + r**2)) ** 2 * (1 + Rs / jnp.sqrt(Rs**2 + r**2))
        )
        factor = jnp.where(Ra == Rs, factor2, factor1)

        alpha_r = 2 * sigma0 * Ra * Rs * factor
        f_x = alpha_r * x_ / r
        f_y = alpha_r * y_ / r
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """Hessian of lensing potential.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param sigma0: sigma0/sigma_crit (see class documentation above)
        :param Ra: core radius (see class documentation above)
        :param Rs: transition radius from logarithmic slope -2 to -4 (see class
            documentation above)
        :param center_x: center of profile
        :param center_y: center of profile
        :return: f_xx, f_xy, f_yx, f_yy
        """
        Ra, Rs = PseudoJaffe._sort_ra_rs(Ra, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        r = jnp.where(r < PseudoJaffe._s, PseudoJaffe._s, r)
        gamma = (
            sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (
                2
                * (
                    1.0 / (Ra + jnp.sqrt(Ra**2 + r**2))
                    - 1.0 / (Rs + jnp.sqrt(Rs**2 + r**2))
                )
                - (1 / jnp.sqrt(Ra**2 + r**2) - 1 / jnp.sqrt(Rs**2 + r**2))
            )
        )
        kappa = (
            sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (1 / jnp.sqrt(Ra**2 + r**2) - 1 / jnp.sqrt(Rs**2 + r**2))
        )
        sin_2phi = -2 * x_ * y_ / r**2
        cos_2phi = (y_**2 - x_**2) / r**2
        gamma1 = cos_2phi * gamma
        gamma2 = sin_2phi * gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def _f_A20(r_a, r_s):
        """Equation A20 in Eliasdottir (2007)

        :param r_a: r/Ra
        :param r_s: r/Rs
        :return: f(R/a, R/s)
        """
        return r_a / (1 + jnp.sqrt(1 + r_a**2)) - r_s / (1 + jnp.sqrt(1 + r_s**2))

    @staticmethod
    @jit
    def rho2sigma(rho0, Ra, Rs):
        """Converts 3d density into 2d projected density parameter, Equation A4 in
        Eliasdottir (2007)

        :param rho0: density normalization
        :param Ra: core radius (see class documentation above)
        :param Rs: transition radius from logarithmic slope -2 to -4 (see class
            documentation above)
        :return: projected density normalization
        """
        return np.pi * rho0 * Ra * Rs / (Rs + Ra)

    @staticmethod
    @jit
    def sigma2rho(sigma0, Ra, Rs):
        """Inverse of rho2sigma()

        :param sigma0: projected density normalization
        :param Ra: core radius (see class documentation above)
        :param Rs: transition radius from logarithmic slope -2 to -4 (see class
            documentation above)
        :return: 3D density normalization
        """
        return (Rs + Ra) / Ra / Rs / np.pi * sigma0

    @staticmethod
    @jit
    def _sort_ra_rs(Ra, Rs):
        """Sorts Ra and Rs to make sure Rs > Ra.

        :param Ra:
        :param Rs:
        :return: Ra, Rs in conventions used
        """
        # makes sure these parameters do not go below some small values
        Ra = jnp.where(Ra < 1e-4, 1e-4, Ra)
        Rs = jnp.where(Rs < 1e-4, 1e-4, Rs)
        # Autodifferentiation works with swaps
        Ra, Rs = jnp.where(Rs < Ra, Rs, Ra), jnp.where(Rs < Ra, Ra, Rs)
        return Ra, Rs
