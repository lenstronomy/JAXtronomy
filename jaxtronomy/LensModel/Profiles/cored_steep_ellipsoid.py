__author__ = "sibirrer"

from jax import config, jit, lax, tree_util
import jax.numpy as jnp

from jaxtronomy.Util import param_util
from jaxtronomy.Util import util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

config.update("jax_enable_x64", True)

__all__ = [
    "CSE",
    "CSEMajorAxis",
    "CSEMajorAxisSet",
    "CSEProductAvg",
    "CSEProductAvgSet",
]


class CSE(LensProfileBase):
    """
    Cored steep ellipsoid (CSE)
    :param axis: 'major' or 'product_avg' ; whether to evaluate corresponding to r= major axis or r= sqrt(ab)
    source:
    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{x^2 + \\frac{y^2}{q^2}}

    """

    param_names = ["a", "s", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "a": -1000,
        "s": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "a": 1000,
        "s": 10000,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": -100,
        "center_y": -100,
    }

    def __init__(self, axis="product_avg"):
        if axis != "major" and axis != "product_avg":
            raise ValueError(
                "axis must be set to 'major' or 'product_avg'. Input is %s ." % axis
            )
        self.axis = axis
        super(CSE, self).__init__()

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value
    def _tree_flatten(self):
        children = ()
        aux_data = {"axis": self.axis}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: lensing potential
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        case = jnp.where(self.axis == "major", 0, 1)
        func = [CSEMajorAxis.function, CSEProductAvg.function]

        f_ = lax.switch(case, func, x__, y__, a, s, q)

        return f_

    @jit
    def derivatives(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: deflection in x- and y-direction
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        case = jnp.where(self.axis == "major", 0, 1)
        func = [CSEMajorAxis.derivatives, CSEProductAvg.derivatives]

        f__x, f__y = lax.switch(case, func, x__, y__, a, s, q)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        return f_x, f_y

    @jit
    def hessian(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        case = jnp.where(self.axis == "major", 0, 1)
        func = [CSEMajorAxis.hessian, CSEProductAvg.hessian]

        f__xx, f__xy, _, f__yy = lax.switch(case, func, x__, y__, a, s, q)

        # rotate back
        kappa = 1.0 / 2 * (f__xx + f__yy)
        gamma1__ = 1.0 / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = jnp.cos(2 * phi_q) * gamma1__ - jnp.sin(2 * phi_q) * gamma2__
        gamma2 = jnp.sin(2 * phi_q) * gamma1__ + jnp.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return f_xx, f_xy, f_xy, f_yy


class CSEMajorAxis(LensProfileBase):
    """
    Cored steep ellipsoid (CSE) along the major axis
    source:
    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{x^2 + \\frac{y^2}{q^2}}

    """

    param_names = ["a", "s", "q", "center_x", "center_y"]
    lower_limit_default = {
        "a": -1000,
        "s": 0,
        "q": 0.001,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "a": 1000,
        "s": 10000,
        "q": 0.99999,
        "center_x": -100,
        "center_y": -100,
    }

    @staticmethod
    @jit
    def function(x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: lensing potential
        """

        # potential calculation
        psi = jnp.sqrt(q**2 * (s**2 + x**2) + y**2)
        Phi = (psi + s) ** 2 + (1 - q**2) * x**2
        phi = q / (2 * s) * jnp.log(Phi) - q / s * jnp.log((1 + q) * s)
        return a * phi

    @staticmethod
    @jit
    def derivatives(x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """

        psi = jnp.sqrt(q**2 * (s**2 + x**2) + y**2)
        Phi = (psi + s) ** 2 + (1 - q**2) * x**2
        f_x = q * x * (psi + q**2 * s) / (s * psi * Phi)
        f_y = q * y * (psi + s) / (s * psi * Phi)

        return a * f_x, a * f_y

    @staticmethod
    @jit
    def hessian(x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """

        # equations 21-23 in Oguri 2021
        psi = jnp.sqrt(q**2 * (s**2 + x**2) + y**2)
        Phi = (psi + s) ** 2 + (1 - q**2) * x**2
        f_xx = (
            q
            / (s * Phi)
            * (
                1
                + q**2 * s * (q**2 * s**2 + y**2) / psi**3
                - 2 * x**2 * (psi + q**2 * s) ** 2 / (psi**2 * Phi)
            )
        )
        f_yy = (
            q
            / (s * Phi)
            * (
                1
                + q**2 * s * (s**2 + x**2) / psi**3
                - 2 * y**2 * (psi + s) ** 2 / (psi**2 * Phi)
            )
        )
        f_xy = (
            -q
            * x
            * y
            / (s * Phi)
            * (q**2 * s / psi**3 + 2 * (psi + q**2 * s) * (psi + s) / (psi**2 * Phi))
        )

        return a * f_xx, a * f_xy, a * f_xy, a * f_yy


class CSEMajorAxisSet(LensProfileBase):
    """A set of CSE profiles along a joint center and axis."""

    def __init__(self):
        super(CSEMajorAxisSet, self).__init__()

    @staticmethod
    @jit
    def function(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: lensing potential
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_ = jnp.zeros_like(x)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_ = val
            f_ += CSEMajorAxis.function(x, y, a_list.at[i].get(), s_list.at[i].get(), q)
            return x, y, a_list, s_list, q, f_

        _, _, _, _, _, f_ = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_)
        )
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(y)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_x, f_y = val
            f_x_, f_y_ = CSEMajorAxis.derivatives(
                x, y, a_list.at[i].get(), s_list.at[i].get(), q
            )
            f_x += f_x_
            f_y += f_y_
            return x, y, a_list, s_list, q, f_x, f_y

        _, _, _, _, _, f_x, f_y = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_x, f_y)
        )

        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_xx, f_xy, f_yy = jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_xx, f_xy, f_yy = val
            f_xx_, f_xy_, _, f_yy_ = CSEMajorAxis.hessian(
                x, y, a_list.at[i].get(), s_list.at[i].get(), q
            )
            f_xx += f_xx_
            f_xy += f_xy_
            f_yy += f_yy_
            return x, y, a_list, s_list, q, f_xx, f_xy, f_yy

        _, _, _, _, _, f_xx, f_xy, f_yy = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_xx, f_xy, f_yy)
        )
        return f_xx, f_xy, f_xy, f_yy


class CSEProductAvg(LensProfileBase):
    """Cored steep ellipsoid (CSE) evaluated at the product-averaged radius sqrt(ab),
    such that mass is not changed when increasing ellipticity.

    Same as CSEMajorAxis but evaluated at r=sqrt(q)*r_original

    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{qx^2 + \\frac{y^2}{q}}
    """

    param_names = ["a", "s", "q", "center_x", "center_y"]
    lower_limit_default = {
        "a": -1000,
        "s": 0,
        "q": 0.001,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "a": 1000,
        "s": 10000,
        "q": 0.99999,
        "center_x": -100,
        "center_y": -100,
    }

    def __init__(self):
        super(CSEProductAvg, self).__init__()

    @staticmethod
    @jit
    def _convert2prodavg(x, y, a, s, q):
        """Converts coordinates and re-normalizes major-axis parameterization to instead
        be wrt.

        product-averaged
        """
        a = a / q
        x = x * jnp.sqrt(q)
        y = y * jnp.sqrt(q)
        return x, y, a, s, q

    @staticmethod
    @jit
    def function(x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: lensing potential
        """
        x, y, a, s, q = CSEProductAvg._convert2prodavg(x, y, a, s, q)
        return CSEMajorAxis.function(x, y, a, s, q)

    @staticmethod
    @jit
    def derivatives(x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        x, y, a, s, q = CSEProductAvg._convert2prodavg(x, y, a, s, q)
        af_x, af_y = CSEMajorAxis.derivatives(x, y, a, s, q)
        # extra sqrt(q) factor from taking derivative of transformed coordinate
        return jnp.sqrt(q) * af_x, jnp.sqrt(q) * af_y

    @staticmethod
    @jit
    def hessian(x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        x, y, a, s, q = CSEProductAvg._convert2prodavg(x, y, a, s, q)
        af_xx, af_xy, af_xy, af_yy = CSEMajorAxis.hessian(x, y, a, s, q)
        # two sqrt(q) factors from taking derivatives of transformed coordinate
        return q * af_xx, q * af_xy, q * af_xy, q * af_yy


class CSEProductAvgSet(LensProfileBase):
    """A set of CSE profiles along a joint center and axis."""

    def __init__(self):
        super(CSEProductAvgSet, self).__init__()

    @staticmethod
    @jit
    def function(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: lensing potential
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_ = jnp.zeros_like(x)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_ = val
            f_ += CSEProductAvg.function(
                x, y, a_list.at[i].get(), s_list.at[i].get(), q
            )
            return x, y, a_list, s_list, q, f_

        _, _, _, _, _, f_ = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_)
        )
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(y)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_x, f_y = val
            f_x_, f_y_ = CSEProductAvg.derivatives(
                x, y, a_list.at[i].get(), s_list.at[i].get(), q
            )
            f_x += f_x_
            f_y += f_y_
            return x, y, a_list, s_list, q, f_x, f_y

        _, _, _, _, _, f_x, f_y = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_x, f_y)
        )
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list or array of lensing strength
        :param s_list: list or array of core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        x = x.astype(float)
        y = y.astype(float)
        a_list = jnp.asarray(a_list)
        s_list = jnp.asarray(s_list)
        f_xx, f_xy, f_yy = jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

        def body_fun(i, val):
            x, y, a_list, s_list, q, f_xx, f_xy, f_yy = val
            f_xx_, f_xy_, _, f_yy_ = CSEProductAvg.hessian(
                x, y, a_list.at[i].get(), s_list.at[i].get(), q
            )
            f_xx += f_xx_
            f_xy += f_xy_
            f_yy += f_yy_
            return x, y, a_list, s_list, q, f_xx, f_xy, f_yy

        _, _, _, _, _, f_xx, f_xy, f_yy = lax.fori_loop(
            0, len(a_list), body_fun, (x, y, a_list, s_list, q, f_xx, f_xy, f_yy)
        )
        return f_xx, f_xy, f_xy, f_yy


tree_util.register_pytree_node(CSE, CSE._tree_flatten, CSE._tree_unflatten)
