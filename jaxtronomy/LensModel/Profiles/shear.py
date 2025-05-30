__author__ = "sibirrer"

import jaxtronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from jaxtronomy.LensModel.Profiles.convergence import Convergence
import jax.numpy as jnp
from jax import jit

__all__ = ["Shear", "ShearGammaPsi", "ShearReduced"]


class Shear(LensProfileBase):
    """Class for external shear gamma1, gamma2 expression."""

    param_names = ["gamma1", "gamma2", "ra_0", "dec_0"]
    lower_limit_default = {"gamma1": -0.5, "gamma2": -0.5, "ra_0": -100, "dec_0": -100}
    upper_limit_default = {"gamma1": 0.5, "gamma2": 0.5, "ra_0": 100, "dec_0": 100}

    @staticmethod
    @jit
    def function(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: lensing potential
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 1 / 2.0 * (gamma1 * x_ * x_ + 2 * gamma2 * x_ * y_ - gamma1 * y_ * y_)
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: deflection angles
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = gamma1 * x_ + gamma2 * y_
        f_y = +gamma2 * x_ - gamma1 * y_
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: f_xx, f_xy, f_yx, f_yy
        """
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy


class ShearGammaPsi(LensProfileBase):
    """
    class to model a shear field with shear strength and direction. The translation ot the cartesian shear distortions
    is as follow:

    .. math::
        \\gamma_1 = \\gamma_{ext} \\cos(2 \\phi_{ext})
        \\gamma_2 = \\gamma_{ext} \\sin(2 \\phi_{ext})

    """

    param_names = ["gamma_ext", "psi_ext", "ra_0", "dec_0"]
    lower_limit_default = {
        "gamma_ext": 0,
        "psi_ext": -jnp.pi,
        "ra_0": -100,
        "dec_0": -100,
    }
    upper_limit_default = {"gamma_ext": 1, "psi_ext": jnp.pi, "ra_0": 100, "dec_0": 100}

    def __init__(self):
        super(ShearGammaPsi, self).__init__()

    @staticmethod
    @jit
    def function(x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma_ext: shear strength
        :param psi_ext: shear angle (radian)
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return:
        """
        # change to polar coordinate
        r, phi = param_util.cart2polar(x - ra_0, y - dec_0)
        f_ = 1.0 / 2 * gamma_ext * r**2 * jnp.cos(2 * (phi - psi_ext))
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        # rotation angle
        gamma1, gamma2 = param_util.shear_polar2cartesian(psi_ext, gamma_ext)
        return Shear.derivatives(x, y, gamma1, gamma2, ra_0, dec_0)

    @staticmethod
    @jit
    def hessian(x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        gamma1, gamma2 = param_util.shear_polar2cartesian(psi_ext, gamma_ext)
        return Shear.hessian(x, y, gamma1, gamma2, ra_0, dec_0)


class ShearReduced(LensProfileBase):
    """Reduced shear distortions :math:`\\gamma' = \\gamma / (1-\\kappa)`. This
    distortion keeps the magnification as unity and, thus, does not change the size of
    apparent objects. To keep the magnification at unity, it requires.

    .. math::
        (1-\\kappa)^2) - \\gamma_1^2 - \\gamma_2^ = 1

    Thus, for given pair of reduced shear :math:`(\\gamma'_1, \\gamma'_2)`, an additional convergence term is calculated
    and added to the lensing distortions.
    """

    param_names = ["gamma1", "gamma2", "ra_0", "dec_0"]
    lower_limit_default = {"gamma1": -0.5, "gamma2": -0.5, "ra_0": -100, "dec_0": -100}
    upper_limit_default = {"gamma1": 0.5, "gamma2": 0.5, "ra_0": 100, "dec_0": 100}

    def __init__(self):
        super(ShearReduced, self).__init__()

    @staticmethod
    @jit
    def kappa_reduced(gamma1, gamma2):
        """Compute convergence such that magnification is unity.

        :param gamma1: reduced shear
        :param gamma2: reduced shear
        :return: kappa
        """
        kappa = 1 - 1.0 / jnp.sqrt(1 - gamma1**2 - gamma2**2)
        gamma1_ = (1 - kappa) * gamma1
        gamma2_ = (1 - kappa) * gamma2
        return kappa, gamma1_, gamma2_

    @staticmethod
    @jit
    def function(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: lensing potential
        """
        kappa, gamma1_, gamma2_ = ShearReduced.kappa_reduced(gamma1, gamma2)
        f_shear = Shear.function(x, y, gamma1_, gamma2_, ra_0, dec_0)
        f_kappa = Convergence.function(x, y, kappa, ra_0, dec_0)
        return f_shear + f_kappa

    @staticmethod
    @jit
    def derivatives(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: deflection angles
        """
        kappa, gamma1_, gamma2_ = ShearReduced.kappa_reduced(gamma1, gamma2)
        f_x_shear, f_y_shear = Shear.derivatives(x, y, gamma1_, gamma2_, ra_0, dec_0)
        f_x_kappa, f_y_kappa = Convergence.derivatives(x, y, kappa, ra_0, dec_0)
        return f_x_shear + f_x_kappa, f_y_shear + f_y_kappa

    @staticmethod
    @jit
    def hessian(x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: f_xx, f_xy, f_yx, f_yy
        """
        kappa, gamma1_, gamma2_ = ShearReduced.kappa_reduced(gamma1, gamma2)
        f_xx_g, f_xy_g, f_yx_g, f_yy_g = Shear.hessian(
            x, y, gamma1_, gamma2_, ra_0, dec_0
        )
        f_xx_k, f_xy_k, f_yx_k, f_yy_k = Convergence.hessian(x, y, kappa, ra_0, dec_0)
        f_xx = f_xx_g + f_xx_k
        f_yy = f_yy_g + f_yy_k
        f_xy = f_xy_g + f_xy_k
        return f_xx, f_xy, f_xy, f_yy
