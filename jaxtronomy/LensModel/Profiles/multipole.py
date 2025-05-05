__author__ = "lynevdv"

from jax import config, jit, numpy as jnp, lax

config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy

import jaxtronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Multipole", "EllipticalMultipole"]


class Multipole(LensProfileBase):
    """
    This class contains a CIRCULAR multipole contribution (for 1 component with m>=2)
    This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf, Equation B12
    Only the q=1 case (ie., circular symmetry) makes this definition consistent with interpretation of multipoles
    as a deformation of the isophotes with an order m symmetry (eg., disky/boxy in the m=4 case).

    m : int, multipole order, m>=1
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """

    param_names = ["m", "a_m", "phi_m", "center_x", "center_y", "r_E"]
    lower_limit_default = {
        "m": 1,
        "a_m": 0,
        "phi_m": -jnp.pi,
        "center_x": -100,
        "center_y": -100,
        "r_E": 0,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": jnp.pi,
        "center_x": 100,
        "center_y": 100,
        "r_E": 100,
    }

    @staticmethod
    @jit
    def function(x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Lensing potential of multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for the m=1, Einstein radius by default)
        :return: lensing potential
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        def m_equal_1(r, phi, a_m, phi_m):
            r = jnp.maximum(r, 0.000001)
            return r * jnp.log(r / r_E) * a_m / 2 * jnp.cos(phi - phi_m)

        def m_not_1(r, phi, a_m, phi_m):
            return r * a_m / (1 - m**2) * jnp.cos(m * (phi - phi_m))

        return lax.cond(m == 1, m_equal_1, m_not_1, r, phi, a_m, phi_m)

    @staticmethod
    @jit
    def derivatives(x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Deflection of a multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
        Equation B12

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for the m=1, Einstein radius by default)
        :return: deflection angles alpha_x, alpha_y
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        def m_equal_1(r, phi, a_m, phi_m):
            r = jnp.maximum(r, 0.000001)
            f_x = (
                a_m
                / 2
                * (
                    jnp.cos(phi_m) * jnp.log(r / r_E)
                    + jnp.cos(phi - phi_m) * jnp.cos(phi)
                )
            )
            f_y = (
                a_m
                / 2
                * (
                    jnp.sin(phi_m) * jnp.log(r / r_E)
                    + jnp.cos(phi - phi_m) * jnp.sin(phi)
                )
            )
            return f_x, f_y

        def m_not_1(r, phi, a_m, phi_m):
            f_x = jnp.cos(phi) * a_m / (1 - m**2) * jnp.cos(
                m * (phi - phi_m)
            ) + jnp.sin(phi) * m * a_m / (1 - m**2) * jnp.sin(m * (phi - phi_m))
            f_y = jnp.sin(phi) * a_m / (1 - m**2) * jnp.cos(
                m * (phi - phi_m)
            ) - jnp.cos(phi) * m * a_m / (1 - m**2) * jnp.sin(m * (phi - phi_m))
            return f_x, f_y

        return lax.cond(m == 1, m_equal_1, m_not_1, r, phi, a_m, phi_m)

    @staticmethod
    @jit
    def hessian(x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Hessian of a multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (not used for Hessian)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = jnp.maximum(r, 0.000001)

        def m_equal_1(r, phi, a_m, phi_m):
            f_xx = (
                a_m
                / (2 * r)
                * (
                    2 * jnp.cos(phi_m) * jnp.cos(phi)
                    - jnp.cos(phi - phi_m) * jnp.cos(2 * phi)
                )
            )
            f_yy = (
                a_m
                / (2 * r)
                * (
                    2 * jnp.sin(phi_m) * jnp.sin(phi)
                    + jnp.cos(phi - phi_m) * jnp.cos(2 * phi)
                )
            )
            f_xy = (
                a_m
                / (2 * r)
                * (jnp.sin(phi + phi_m) - jnp.cos(phi - phi_m) * jnp.sin(2 * phi))
            )
            return f_xx, f_xy, f_xy, f_yy

        def m_not_1(r, phi, a_m, phi_m):
            f_xx = 1.0 / r * jnp.sin(phi) ** 2 * a_m * jnp.cos(m * (phi - phi_m))
            f_yy = 1.0 / r * jnp.cos(phi) ** 2 * a_m * jnp.cos(m * (phi - phi_m))
            f_xy = (
                -1.0
                / r
                * a_m
                * jnp.cos(phi)
                * jnp.sin(phi)
                * jnp.cos(m * (phi - phi_m))
            )
            return f_xx, f_xy, f_xy, f_yy

        return lax.cond(m == 1, m_equal_1, m_not_1, r, phi, a_m, phi_m)


class EllipticalMultipole(LensProfileBase):
    """This class contains a multipole contribution that encode deviations from the
    elliptical isodensity contours of a SIE with any axis ratio q.

    This uses the definitions from Paugnat & Gilman (2025): "Elliptical multipoles for gravitational lenses"

    m : int, multipole order, (m=1, m=3 or m=4)
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    q : axis ratio of the reference ellipses
    """

    param_names = ["m", "a_m", "phi_m", "q", "center_x", "center_y", "r_E"]
    lower_limit_default = {
        "m": 1,
        "a_m": 0,
        "phi_m": -jnp.pi,
        "q": 0.001,
        "center_x": -100,
        "center_y": -100,
        "r_E": 0,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": jnp.pi,
        "q": 1,
        "center_x": 100,
        "center_y": 100,
        "r_E": 100,
    }

    @staticmethod
    @jit
    def function(x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Lensing potential of multipole contribution (for 1 component with m=1, m=3 or
        m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for odd m, Einstein radius by
            default)
        :return: lensing potential
        """

        case = jnp.where(m == 1, 0, 4)
        case = jnp.where(m == 3, 1, case)
        case = jnp.where(m == 4, 2, case)

        # avoid numerical instability when q is too close to 1 by taking circular multipole solution
        case = jnp.where(jnp.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8, 3, case)

        def m_equal_1(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            f_ = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * _potential_m1_1(r, phi, q, r_E)
                    - (1 / q)
                    * jnp.sin(m * phi_m)
                    * _potential_m1_1(r, phi + jnp.pi / 2, 1 / q, r_E)
                )
            )
            return f_

        def m_equal_3(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            f_ = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * _potential_m3_1(r, phi, q, r_E)
                    + (1 / q)
                    * jnp.sin(m * phi_m)
                    * _potential_m3_1(r, phi + jnp.pi / 2, 1 / q, r_E)
                )
            )
            return f_

        def m_equal_4(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            f_ = (
                a_m
                * jnp.sqrt(q)
                * r
                * (
                    _F_m4_1(phi, q=q) * jnp.cos(m * phi_m)
                    + _F_m4_2(phi, q=q) * jnp.sin(m * phi_m)
                )
            )
            return f_

        # Raising runtime errors is expensive in JAX so we just return 1e18
        def error(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            return jnp.ones_like(x) * 1e18

        func_list = [m_equal_1, m_equal_3, m_equal_4, Multipole.function, error]

        return lax.switch(case, func_list, x, y, m, a_m, phi_m, center_x, center_y, r_E)

    @staticmethod
    @jit
    def derivatives(x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Deflection of a multipole contribution (for 1 component with m=1, m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for odd m, Einstein radius by
            default)
        :return: deflection angles alpha_x, alpha_y
        """

        case = jnp.where(m == 1, 0, 4)
        case = jnp.where(m == 3, 1, case)
        case = jnp.where(m == 4, 2, case)

        # avoid numerical instability when q is too close to 1 by taking circular multipole solution
        case = jnp.where(jnp.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8, 3, case)

        def m_equal_1(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            alpha_x_1, alpha_y_1 = _alpha_m1_1(r, phi, q, r_E)
            alpha_x_2, alpha_y_2 = _alpha_m1_1(r, phi + jnp.pi / 2, 1 / q, r_E)
            f_x = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * alpha_x_1
                    - (1 / q) * jnp.sin(m * phi_m) * alpha_y_2
                )
            )
            f_y = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * alpha_y_1
                    + (1 / q) * jnp.sin(m * phi_m) * alpha_x_2
                )
            )
            return f_x, f_y

        def m_equal_3(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            alpha_x_1, alpha_y_1 = _alpha_m3_1(r, phi, q, r_E)
            alpha_x_2, alpha_y_2 = _alpha_m3_1(r, phi + jnp.pi / 2, 1 / q, r_E)
            f_x = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * alpha_x_1
                    + (1 / q) * jnp.sin(m * phi_m) * alpha_y_2
                )
            )
            f_y = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * alpha_y_1
                    - (1 / q) * jnp.sin(m * phi_m) * alpha_x_2
                )
            )
            return f_x, f_y

        def m_equal_4(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            F_m4 = _F_m4_1(phi, q=q) * jnp.cos(m * phi_m) + _F_m4_2(phi, q=q) * jnp.sin(
                m * phi_m
            )
            F_m4_prime = _F_m4_1_derivative(phi, q=q) * jnp.cos(
                m * phi_m
            ) + _F_m4_2_derivative(phi, q=q) * jnp.sin(m * phi_m)
            f_x = a_m * jnp.sqrt(q) * (F_m4 * jnp.cos(phi) - F_m4_prime * jnp.sin(phi))
            f_y = a_m * jnp.sqrt(q) * (F_m4 * jnp.sin(phi) + F_m4_prime * jnp.cos(phi))
            return f_x, f_y

        # Raising runtime errors is expensive in JAX so we just return 1e18
        def error(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            return jnp.ones_like(x) * 1e18, jnp.ones_like(x) * 1e18

        func_list = [m_equal_1, m_equal_3, m_equal_4, Multipole.derivatives, error]

        return lax.switch(case, func_list, x, y, m, a_m, phi_m, center_x, center_y, r_E)

    @staticmethod
    @jit
    def hessian(x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Hessian of a multipole contribution (for 1 component with m=1, m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (not used for Hessian)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        case = jnp.where(m == 1, 0, 4)
        case = jnp.where(m == 3, 1, case)
        case = jnp.where(m % 2 == 0, 2, case)

        # avoid numerical instability when q is too close to 1 by taking circular multipole solution
        case = jnp.where(jnp.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8, 3, case)

        def m_equal_1(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            d2psi_dx2_1, d2psi_dy2_1, d2psi_dxdy_1 = _hessian_m1_1(r, phi, q)
            d2psi_dx2_2, d2psi_dy2_2, d2psi_dxdy_2 = _hessian_m1_1(
                r, phi + jnp.pi / 2, 1 / q
            )
            f_xx = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dx2_1
                    - (1 / q) * jnp.sin(m * phi_m) * d2psi_dy2_2
                )
            )
            f_yy = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dy2_1
                    - (1 / q) * jnp.sin(m * phi_m) * d2psi_dx2_2
                )
            )
            f_xy = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dxdy_1
                    + (1 / q) * jnp.sin(m * phi_m) * d2psi_dxdy_2
                )
            )
            return f_xx, f_xy, f_xy, f_yy

        def m_equal_3(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            d2psi_dx2_1, d2psi_dy2_1, d2psi_dxdy_1 = _hessian_m3_1(r, phi, q)
            d2psi_dx2_2, d2psi_dy2_2, d2psi_dxdy_2 = _hessian_m3_1(
                r, phi + jnp.pi / 2, 1 / q
            )
            f_xx = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dx2_1
                    + (1 / q) * jnp.sin(m * phi_m) * d2psi_dy2_2
                )
            )
            f_yy = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dy2_1
                    + (1 / q) * jnp.sin(m * phi_m) * d2psi_dx2_2
                )
            )
            f_xy = (
                a_m
                * jnp.sqrt(q)
                * (
                    jnp.cos(m * phi_m) * d2psi_dxdy_1
                    - (1 / q) * jnp.sin(m * phi_m) * d2psi_dxdy_2
                )
            )
            return f_xx, f_xy, f_xy, f_yy

        def m_equal_4(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
            r = jnp.maximum(r, 0.000001)
            phi_ell = jnp.angle(q * r * jnp.cos(phi) + 1j * r * jnp.sin(phi))
            R = jnp.sqrt(q * (r * jnp.cos(phi)) ** 2 + (r * jnp.sin(phi)) ** 2 / q)

            delta_r = a_m * jnp.cos(m * (phi_ell - phi_m)) * r / R
            f_xx = jnp.sin(phi) ** 2 * delta_r / r
            f_yy = jnp.cos(phi) ** 2 * delta_r / r
            f_xy = -jnp.sin(phi) * jnp.cos(phi) * delta_r / r
            return f_xx, f_xy, f_xy, f_yy

        # Raising runtime errors is expensive in JAX so we just return 1e18
        def error(x, y, m, a_m, phi_m, center_x, center_y, r_E):
            return (
                jnp.ones_like(x) * 1e18,
                jnp.ones_like(x) * 1e18,
                jnp.ones_like(x) * 1e18,
                jnp.ones_like(x) * 1e18,
            )

        func_list = [m_equal_1, m_equal_3, m_equal_4, Multipole.hessian, error]

        return lax.switch(case, func_list, x, y, m, a_m, phi_m, center_x, center_y, r_E)


@jit
def _phi_ell(phi, q):
    return (
        phi
        - jnp.arctan2(jnp.sin(phi), jnp.cos(phi))
        + jnp.arctan2(jnp.sin(phi), q * jnp.cos(phi))
    )


@jit
def _G_m_1(m, phi, q):
    return jnp.cos(m * jnp.arctan2(jnp.sin(phi), q * jnp.cos(phi))) / jnp.sqrt(
        q**2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2
    )


@jit
def _F_m1_1_hat(phi, q):
    term1 = jnp.cos(phi) * (
        q * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        - (jnp.log(2) * (1 + q) / 2 - (1 - q**2) * (1 + jnp.log(2) / 4))
    )
    term2 = 2 * jnp.sin(phi) * (phi - _phi_ell(phi, q))
    return -(term1 + term2) / (2 * (1 - q**2))


@jit
def _F_m1_1_hat_derivative(phi, q):
    term1 = -jnp.cos(phi) * q * 2 * (q**2 - 1) * jnp.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)
    ) + jnp.sin(phi) * (
        -q * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        + jnp.log(2) * (1 + q) / 2
        - (1 - q**2) * (1 + jnp.log(2) / 4)
    )
    term2 = 2 * jnp.cos(phi) * (phi - _phi_ell(phi, q)) + 2 * jnp.sin(phi) * (
        1 - q / (q**2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2)
    )
    return -(term1 + term2) / (2 * (1 - q**2))


@jit
def _potential_m1_1(r, phi, q, r_E):
    lambda_m1 = 2 / (1 + q)
    return r * _F_m1_1_hat(phi, q) + lambda_m1 / 2 * r * jnp.log(r / r_E) * jnp.cos(phi)


@jit
def _alpha_m1_1(r, phi, q, r_E):
    lambda_m1 = 2 / (1 + q)
    f_phi = _F_m1_1_hat(phi, q)
    df_dphi = _F_m1_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * jnp.cos(phi)
        - df_dphi * jnp.sin(phi)
        + lambda_m1 / 2 * (jnp.log(r / r_E) + jnp.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * jnp.sin(phi)
        + df_dphi * jnp.cos(phi)
        + lambda_m1 / 2 * jnp.cos(phi) * jnp.sin(phi)
    )
    return alpha_x, alpha_y


@jit
def _hessian_m1_1(r, phi, q):
    lambda_m1 = 2 / (1 + q)
    G_m1_1 = _G_m_1(1, phi, q)
    d2psi_dx2 = (jnp.sin(phi) ** 2 * G_m1_1 + lambda_m1 / 2 * jnp.cos(phi)) / r
    d2psi_dy2 = (jnp.cos(phi) ** 2 * G_m1_1 - lambda_m1 / 2 * jnp.cos(phi)) / r
    d2psi_dxdy = (
        -jnp.cos(phi) * jnp.sin(phi) * G_m1_1 + lambda_m1 / 2 * jnp.sin(phi)
    ) / r
    return d2psi_dx2, d2psi_dy2, d2psi_dxdy


@jit
def _A_3_1(q):
    return (
        jnp.log(2) * (1 + q) ** 2
        - 2 * (1 - q) * (1 + q) ** 2 * (1 + jnp.log(2) / 4)
        + (1 - q**2) ** 2 / 4
    )


@jit
def _F_m3_1_hat(phi, q):
    term1 = jnp.cos(phi) * (
        q * (3 + q**2) * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) - _A_3_1(q)
    )
    term2 = 2 * jnp.sin(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q))
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


@jit
def _F_m3_1_hat_derivative(phi, q):
    term1 = -jnp.cos(phi) * q * (3 + q**2) * 2 * (q**2 - 1) * jnp.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)
    ) + jnp.sin(phi) * (
        -q * (3 + q**2) * jnp.log(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) + _A_3_1(q)
    )
    term2 = 2 * jnp.cos(phi) * (1 + 3 * q**2) * (
        phi - _phi_ell(phi, q)
    ) + 2 * jnp.sin(phi) * (1 + 3 * q**2) * (
        1 - q / (q**2 * jnp.cos(phi) ** 2 + jnp.sin(phi) ** 2)
    )
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


@jit
def _potential_m3_1(r, phi, q, r_E):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    return r * _F_m3_1_hat(phi, q) + lambda_m3 / 2 * r * jnp.log(r / r_E) * jnp.cos(phi)


@jit
def _alpha_m3_1(r, phi, q, r_E):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    f_phi = _F_m3_1_hat(phi, q)
    df_dphi = _F_m3_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * jnp.cos(phi)
        - df_dphi * jnp.sin(phi)
        + lambda_m3 / 2 * (jnp.log(r / r_E) + jnp.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * jnp.sin(phi)
        + df_dphi * jnp.cos(phi)
        + lambda_m3 / 2 * jnp.cos(phi) * jnp.sin(phi)
    )
    return alpha_x, alpha_y


@jit
def _hessian_m3_1(r, phi, q):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    G_m3_1 = _G_m_1(3, phi, q)
    d2psi_dx2 = (jnp.sin(phi) ** 2 * G_m3_1 + lambda_m3 / 2 * jnp.cos(phi)) / r
    d2psi_dy2 = (jnp.cos(phi) ** 2 * G_m3_1 - lambda_m3 / 2 * jnp.cos(phi)) / r
    d2psi_dxdy = (
        -jnp.cos(phi) * jnp.sin(phi) * G_m3_1 + lambda_m3 / 2 * jnp.sin(phi)
    ) / r
    return d2psi_dx2, d2psi_dy2, d2psi_dxdy


@jit
def _F_m4_1(phi, q):
    term1 = (
        -4
        * jnp.sqrt(2)
        * (1 + 4 * q**2 + q**4 + (q**4 - 1) * jnp.cos(2 * phi))
        / ((3 * (1 - q**2) ** 2) * jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)))
    )
    term2 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * jnp.arctan(
            (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
            / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        )
    )
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.sin(phi)
        * jnp.log(
            jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
            + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


@jit
def _F_m4_1_derivative(phi, q):
    term1 = (
        -4
        * jnp.sqrt(2)
        * (1 + q**4 + (q**4 - 1) * jnp.cos(2 * phi))
        * jnp.sin(2 * phi)
        / (3 * (1 - q**2) * (1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) ** (3 / 2))
    )
    term2 = (
        -(1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * (
            jnp.sin(phi)
            * jnp.arctan(
                (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
                / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
            )
            + jnp.sqrt(2 * (1 - q**2))
            * jnp.sin(2 * phi)
            / (2 * jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)))
        )
    )
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * (
            jnp.log(
                jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
                + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
            )
            + jnp.sqrt(1 - q**2)
            / q
            * jnp.sin(phi)
            / jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


@jit
def _F_m4_2(phi, q):
    term1 = (
        -4
        * jnp.sqrt(2)
        * q
        / (3 * (1 - q**2))
        * jnp.sin(2 * phi)
        / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
    )
    term2 = (
        -4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * jnp.sin(phi)
        * jnp.arctan(
            (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
            / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        )
    )
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * jnp.cos(phi)
        * jnp.log(
            jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
            + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


@jit
def _F_m4_2_derivative(phi, q):
    term1 = (
        -8
        * jnp.sqrt(2)
        * q
        / (6 * (1 - q**2))
        * (
            -(1 - q**2)
            * jnp.sin(2 * phi) ** 2
            / (1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)) ** (3 / 2)
            + 2 * jnp.cos(2 * phi) / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
        )
    )
    term2 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -jnp.cos(phi)
            * jnp.arctan(
                (jnp.sqrt(2 * (1 - q**2)) * jnp.cos(phi))
                / jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi))
            )
            + 2
            * jnp.sqrt(2 * (1 - q**2))
            * jnp.sin(phi) ** 2
            / (2 * jnp.sqrt(1 + q**2 + (q**2 - 1) * jnp.cos(2 * phi)))
        )
    )
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -jnp.sin(phi)
            * jnp.log(
                jnp.sqrt(1 - q**2) * jnp.sin(phi) / q
                + jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
            )
            + jnp.sqrt(1 - q**2)
            / q
            * jnp.cos(phi) ** 2
            / jnp.sqrt(1 + (1 - q**2) / q**2 * jnp.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3
