__author__ = "sibirrer"

# this file contains a class to compute the truncated Navaro-Frank-White function (Baltz et al 2009)in mass/kappa space
# the potential therefore is its integral

from jax import config, jit, numpy as jnp, vmap
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from jaxtronomy.LensModel.Profiles.nfw import NFW
from functools import partial

config.update("jax_enable_x64", True)

__all__ = ["TNFW"]


class TNFW(LensProfileBase):
    """This class contains functions concerning the truncated NFW profile with a
    truncation function (r_trunc^2)*(r^2+r_trunc^2)

    density equation is:

    .. math::
        \\rho(r) = \\frac{r_\\text{trunc}^2}{r^2+r_\\text{trunc}^2}\\frac{\\rho_0(\\alpha_{R_s})}{r/R_s(1+r/R_s)^2}

    relation are: R_200 = c * Rs
    """

    profile_name = "TNFW"
    param_names = ["Rs", "alpha_Rs", "r_trunc", "center_x", "center_y"]
    lower_limit_default = {
        "Rs": 0,
        "alpha_Rs": 0,
        "r_trunc": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "Rs": 100,
        "alpha_Rs": 10,
        "r_trunc": 100,
        "center_x": 100,
        "center_y": 100,
    }

    _s = 0.001

    @staticmethod
    @jit
    def function(x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param r_trunc: truncation radius
        :param center_x: center of halo
        :param center_y: center of halo
        :return: lensing potential
        """
        rho0_input = TNFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        R = jnp.maximum(R, TNFW._s * Rs)
        f_ = TNFW.tnfw_potential(R, Rs, rho0_input, r_trunc)

        return f_

    @staticmethod
    @jit
    def derivatives(x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function (integral of TNFW), which are the
        deflection angles.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_trunc: truncation radius (angular units)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_input = TNFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        R = jnp.maximum(R, TNFW._s * Rs)
        f_x, f_y = TNFW.tnfw_alpha(R, Rs, rho0_input, r_trunc, x_, y_)
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):
        """Returns d^2f/dx^2, d^2f/dxdy, d^2f/dydx, d^2f/dy^2 of the TNFW potential f.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_trunc: truncation radius (angular units)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """

        rho0_input = TNFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        R = jnp.maximum(R, TNFW._s * Rs)

        kappa = TNFW.density_2d(x_, y_, Rs, rho0_input, r_trunc)
        gamma1, gamma2 = TNFW.tnfw_gamma(R, Rs, rho0_input, r_trunc, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def density(r, Rs, rho0, r_trunc):
        """Three dimensional truncated NFW profile.

        :param r: radius of interest
        :type r: float/numpy array
        :param Rs: scale radius
        :type Rs: float > 0
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: rho(r) density
        """
        return (
            (r_trunc**2 * (r_trunc**2 + r**2) ** -1)
            * rho0
            / (r / Rs * (1 + r / Rs) ** 2)
        )

    @staticmethod
    @jit
    def density_2d(x, y, Rs, rho0, r_trunc, center_x=0, center_y=0):
        """Projected two dimensional NFW profile (kappa*Sigma_crit)

        :param R: projected radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: Epsilon(R) projected density at radius R
        """
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        x = R / Rs
        tau = r_trunc / Rs
        Fx = TNFW._F(x, tau)
        return 2 * rho0 * Rs * Fx

    @staticmethod
    @jit
    def mass_2d(R, Rs, rho0, r_trunc):
        """Analytic solution of the projection integral (convergence)

        :param R: projected radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :param r_trunc: truncation radius (angular units)
        :return: mass enclosed 2d projected cylinder
        """

        x = R / Rs
        x = jnp.maximum(x, TNFW._s)
        tau = r_trunc / Rs
        gx = TNFW._g(x, tau)
        m_2d = 4 * rho0 * Rs * R**2 * gx / x**2 * jnp.pi
        return m_2d

    @staticmethod
    @jit
    def mass_3d(r, Rs, rho0, r_trunc):
        """Mass enclosed a 3d sphere or radius r.

        :param r: 3d radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :param r_trunc: truncation radius (angular units)
        :return: M(<r)
        """

        x = r / Rs
        x = jnp.maximum(x, TNFW._s)
        func = (
            r_trunc**2
            * (
                -2 * x * (1 + r_trunc**2)
                + 4 * (1 + x) * r_trunc * jnp.arctan(x / r_trunc)
                - 2 * (1 + x) * (-1 + r_trunc**2) * jnp.log(Rs)
                + 2 * (1 + x) * (-1 + r_trunc**2) * jnp.log(Rs * (1 + x))
                + 2 * (1 + x) * (-1 + r_trunc**2) * jnp.log(Rs * r_trunc)
                - (1 + x) * (-1 + r_trunc**2) * jnp.log(Rs**2 * (x**2 + r_trunc**2))
            )
        ) / (2.0 * (1 + x) * (1 + r_trunc**2) ** 2)

        m_3d = 4 * jnp.pi * Rs**3 * rho0 * func
        return m_3d

    @staticmethod
    @jit
    def tnfw_potential(R, Rs, rho0, r_trunc):
        """Lensing potential of truncated NFW profile.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: lensing potential
        """
        x = R / Rs
        x = jnp.maximum(x, TNFW._s)
        tau = r_trunc / Rs
        hx = TNFW._h(x, tau)
        return 2 * rho0 * Rs**3 * hx

    @staticmethod
    @jit
    def tnfw_alpha(R, Rs, rho0, r_trunc, ax_x, ax_y):
        """Deflection angle of TNFW profile along the projection to coordinate axis.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return:
        """
        x = R / Rs
        x = jnp.maximum(x, TNFW._s)
        tau = r_trunc / Rs
        gx = TNFW._g(x, tau)
        a = 4 * rho0 * Rs * gx / x**2
        return a * ax_x, a * ax_y

    @staticmethod
    @jit
    def tnfw_gamma(R, Rs, rho0, r_trunc, ax_x, ax_y):
        """Shear gamma of TNFW profile (times Sigma_crit) along the projection to
        coordinate 'axis'.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return:
        """
        x = R / Rs
        x = jnp.maximum(x, TNFW._s)
        tau = r_trunc / Rs
        gx = TNFW._g(x, tau)
        Fx = TNFW._F(x, tau)
        a = 2 * rho0 * Rs * (2 * gx / x**2 - Fx)
        return a * (ax_y**2 - ax_x**2) / R**2, -a * 2 * (ax_x * ax_y) / R**2

    @staticmethod
    @jit
    def _L(x, tau):
        """Logarithm that appears frequently.

        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """
        x = jnp.maximum(x, TNFW._s)
        return jnp.log(x * (tau + jnp.sqrt(tau**2 + x**2)) ** -1)

    @staticmethod
    @jit
    def F(x):
        """Classic NFW function in terms of arctanh and arctan.

        :param x: r/Rs
        :return:
        """

        x = jnp.maximum(x, TNFW._s)
        result = jnp.where(
            x < 1, (1 - x**2) ** -0.5 * jnp.arctanh((1 - x**2) ** 0.5), x
        )
        result = jnp.where(
            x > 1, (x**2 - 1) ** -0.5 * jnp.arctan((x**2 - 1) ** 0.5), result
        )
        return result

    @staticmethod
    @jit
    def _F(X, tau):
        """Analytic solution of the projection integral (convergence)

        :param X: R/Rs
        :type X: float >0
        """
        t2 = tau**2
        X = jnp.maximum(X, TNFW._s)
        _F = TNFW.F(X)
        a = t2 * (t2 + 1) ** -2
        b = jnp.ones_like(X)
        b = jnp.where(X == 1, (t2 + 1) * 1.0 / 3, b)
        b = jnp.where(X != 1, (t2 + 1) * (X**2 - 1) ** -1 * (1 - _F), b)

        c = 2 * _F
        d = -jnp.pi * (t2 + X**2) ** -0.5
        e = (t2 - 1) * (tau * (t2 + X**2) ** 0.5) ** -1 * TNFW._L(X, tau)
        result = a * (b + c + d + e)

        return result

    @staticmethod
    @jit
    def _g(x, tau):
        """Analytic solution of integral for NFW profile to compute deflection angel and
        gamma.

        :param x: R/Rs
        :type x: float >0
        """
        x = jnp.maximum(x, TNFW._s)
        return (
            tau**2
            / (tau**2 + 1) ** 2
            * (
                (tau**2 + 1 + 2 * (x**2 - 1)) * TNFW.F(x)
                + tau * jnp.pi
                + (tau**2 - 1) * jnp.log(tau)
                + jnp.sqrt(tau**2 + x**2)
                * (-jnp.pi + TNFW._L(x, tau) * (tau**2 - 1) / tau)
            )
        )

    @staticmethod
    @jit
    def _cos_function(x):
        out = jnp.ones_like(x)
        out = jnp.where(x < 1, -jnp.arccosh(1 / x) ** 2, out)
        out = jnp.where(x >= 1, jnp.arccos(1 / x) ** 2, out)

        return out

    @staticmethod
    @jit
    def _h(x, tau):
        """Expression for the integral to compute potential.

        :param x: R/Rs
        :param tau: r_trunc/Rs
        :type x: float >0
        """
        x = jnp.maximum(x, TNFW._s)

        u = x**2
        t2 = tau**2
        Lx = TNFW._L(x, tau)
        Fx = TNFW.F(x)

        return (t2 + 1) ** -2 * (
            2
            * t2
            * jnp.pi
            * (tau - (t2 + u) ** 0.5 + tau * jnp.log(tau + (t2 + u) ** 0.5))
            + 2 * (t2 - 1) * tau * (t2 + u) ** 0.5 * Lx
            + t2 * (t2 - 1) * Lx**2
            + 4 * t2 * (u - 1) * Fx
            + t2 * (t2 - 1) * TNFW._cos_function(x)
            + t2 * ((t2 - 1) * jnp.log(tau) - t2 - 1) * jnp.log(u)
            - t2
            * (
                (t2 - 1) * jnp.log(tau) * jnp.log(4 * tau)
                + 2 * jnp.log(0.5 * tau)
                - 2 * tau * (tau - jnp.pi) * jnp.log(tau * 2)
            )
        )

    @staticmethod
    @jit
    def alpha2rho0(alpha_Rs, Rs):
        """Convert angle at Rs into rho0; neglects the truncation.

        :param alpha_Rs: deflection angle at RS
        :param Rs: scale radius
        :return: density normalization (characteristic density)
        """
        return NFW.alpha2rho0(alpha_Rs, Rs)

    @staticmethod
    @jit
    def rho02alpha(rho0, Rs):
        """Convert rho0 to angle at Rs; neglects the truncation.

        :param rho0: density normalization (characteristic density)
        :param Rs: scale radius
        :return: deflection angle at RS
        """
        return NFW.rho02alpha(rho0, Rs)
