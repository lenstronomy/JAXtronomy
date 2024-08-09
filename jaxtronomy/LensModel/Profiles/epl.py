# Copy pasted from lenstronomy's epl profile but with all instances of numpy
# replaced by jnp, and scipy.special.hyp2f1 replaced with a jaxified version


__author__ = "ntessore"

import jax
from jax import jit, tree_util
import jax.numpy as jnp

from jaxtronomy.Util.hyp2f1_util import hyp2f1_series as hyp2f1
import jaxtronomy.Util.util as util
import jaxtronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.spp import SPP

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy

__all__ = ["EPL", "EPLMajorAxis", "EPLQPhi"]


class EPL(LensProfileBase):
    """Elliptical Power Law mass profile.

    .. math::
        \\kappa(x, y) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)^{\\gamma-1}

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`\\gamma` is the negative power-law slope of the 3D mass distributions,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate system aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 - e*\\cos(2*\\phi)}} \\right)^{\\gamma-1}

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.

    The mathematical form of the calculation is presented by Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819.
    The current implementation is using hyperbolic functions. The paper presents an iterative calculation scheme,
    converging in few iterations to high precision and accuracy.

    A (faster) implementation of the same model using numba is accessible as 'EPL_NUMBA' with the iterative calculation
    scheme. An alternative implementation of the same model using a fortran code FASTELL is implemented as 'PEMD'
    profile.
    """

    param_names = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.epl_major_axis = EPLMajorAxis()
        self.spp = SPP()
        super(EPL, self).__init__()

    def param_conv(self, theta_E, gamma, e1, e2):
        """Converts parameters as defined in this class to the parameters used in the
        EPLMajorAxis() class.

        :param theta_E: Einstein radius as defined in the profile class
        :param gamma: negative power-law slope
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: b, t, q, phi_G
        """
        if self._static is True:
            return self._b_static, self._t_static, self._q_static, self._phi_G_static
        return self._param_conv(theta_E, gamma, e1, e2)

    @staticmethod
    def _param_conv(theta_E, gamma, e1, e2):
        """Convert parameters from :math:`R = \\sqrt{q x^2 + y^2/q}` to :math:`R =
        \\sqrt{q^2 x^2 + y^2}`

        :param gamma: power law slope
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: critical radius b, slope t, axis ratio q, orientation angle phi_G
        """
        t = gamma - 1
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        b = theta_E * jnp.sqrt(q)
        return b, t, q, phi_G

    def set_static(self, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        (
            self._b_static,
            self._t_static,
            self._q_static,
            self._phi_G_static,
        ) = self._param_conv(theta_E, gamma, e1, e2)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, "_b_static"):
            del self._b_static
        if hasattr(self, "_t_static"):
            del self._t_static
        if hasattr(self, "_phi_G_static"):
            del self._phi_G_static
        if hasattr(self, "_q_static"):
            del self._q_static

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.epl_major_axis.function(x__, y__, b, t, q)
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.epl_major_axis.derivatives(x__, y__, b, t, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """

        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__xy, f__yx, f__yy = self.epl_major_axis.hessian(x__, y__, b, t, q)
        # rotate back
        kappa = 1.0 / 2 * (f__xx + f__yy)
        gamma1__ = 1.0 / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = jnp.cos(2 * phi_G) * gamma1__ - jnp.sin(2 * phi_G) * gamma2__
        gamma2 = +jnp.sin(2 * phi_G) * gamma1__ + jnp.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    def mass_3d_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the spherical power-law mass enclosed (with SPP routine)

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r.
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.density_lens(r, theta_E, gamma)


class EPLMajorAxis(LensProfileBase):
    """This class contains the function and the derivatives of the elliptical power law.

    .. math::
        \\kappa = (2-t)/2 * \\left[\\frac{b}{\\sqrt{q^2 x^2 + y^2}}\\right]^t

    where with :math:`t = \\gamma - 1` (from EPL class) being the projected power-law slope of the convergence profile,
    critical radius b, axis ratio q.

    Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819
    """

    param_names = ["b", "t", "q", "center_x", "center_y"]

    def __init__(self):
        super(EPLMajorAxis, self).__init__()

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (in this case, no recompiling is ever done)
    def _tree_flatten(self):
        children = ()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(self, x, y, b, t, q):
        """Returns the lensing potential.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: lensing potential
        """
        # deflection from method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # deflection potential, eq. (15)
        psi = (x * alpha_x + y * alpha_y) / (2 - t)

        return psi

    @staticmethod
    @jit
    def derivatives(x, y, b, t, q):
        """Returns the deflection angles.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: f_x, f_y
        """
        # elliptical radius, eq. (5)
        Z = q * x + y * 1j
        R = jnp.abs(Z)
        R = jnp.maximum(R, 0.000000001)
        f = (1.0 - q) / (1.0 + q)

        # nmax is the number of iterations required to calculate hyp2f1
        # In order to maintain high accuracy, the number of iterations
        # depends on how close -f exp{2 i phi} is to the unit circle
        nmax = jnp.where(
            f > 0.97,
            1000,
            jnp.where(
                f > 0.95,
                300,
                jnp.where(
                    f > 0.9, 175, jnp.where(f > 0.71, 100, jnp.where(f > 0.3, 50, 20))
                ),
            ),
        )

        # angular dependency with extra factor of R, eq. (23)
        R_omega = Z * hyp2f1(1, t / 2, 2 - t / 2, -f * Z / jnp.conj(Z), nmax)

        # deflection, eq. (22)
        alpha = 2 / (1 + q) * (b / R) ** t * R_omega

        # return real and imaginary part
        alpha_real = jnp.nan_to_num(alpha.real, posinf=10**10, neginf=-(10**10))
        alpha_imag = jnp.nan_to_num(alpha.imag, posinf=10**10, neginf=-(10**10))

        return alpha_real, alpha_imag

    @jit
    def hessian(self, x, y, b, t, q):
        """Hessian matrix of the lensing potential.

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: f_xx, f_yy, f_xy
        """
        R = jnp.hypot(q * x, y)
        R = jnp.maximum(R, 0.00000001)
        r = jnp.hypot(x, y)

        cos, sin = x / r, y / r
        cos2, sin2 = cos * cos * 2 - 1, sin * cos * 2

        # convergence, eq. (2)
        kappa = (2 - t) / 2 * (b / R) ** t
        kappa = jnp.nan_to_num(kappa, posinf=10**10, neginf=-(10**10))

        # deflection via method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # shear, eq. (17), corrected version from arXiv/corrigendum
        gamma_1 = (1 - t) * (alpha_x * cos - alpha_y * sin) / r - kappa * cos2
        gamma_2 = (1 - t) * (alpha_y * cos + alpha_x * sin) / r - kappa * sin2
        gamma_1 = jnp.nan_to_num(gamma_1, posinf=10**10, neginf=-(10**10))
        gamma_2 = jnp.nan_to_num(gamma_2, posinf=10**10, neginf=-(10**10))

        # second derivatives from convergence and shear
        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2

        return f_xx, f_xy, f_xy, f_yy


class EPLQPhi(LensProfileBase):
    """Class to model a EPL sampling over q and phi instead of e1 and e2."""

    param_names = ["theta_E", "gamma", "q", "phi", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "q": 0,
        "phi": -jnp.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "q": 1,
        "phi": jnp.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self._EPL = EPL()
        super(EPLQPhi, self).__init__()

    def function(self, x, y, theta_E, gamma, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return self._EPL.function(x, y, theta_E, gamma, e1, e2, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return self._EPL.derivatives(x, y, theta_E, gamma, e1, e2, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, q, phi, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return self._EPL.hessian(x, y, theta_E, gamma, e1, e2, center_x, center_y)

    def mass_3d_lens(self, r, theta_E, gamma, q=None, phi=None):
        """Computes the spherical power-law mass enclosed (with SPP routine).

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param q: axis ratio (not used)
        :param phi: position angle (not used)
        :return: mass enclosed a 3D radius r.
        """
        return self._EPL.mass_3d_lens(r, theta_E, gamma)

    def density_lens(self, r, theta_E, gamma, q=None, phi=None):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param q: axis ratio (not used)
        :param phi: position angle (not used)
        :return: mass enclosed a 3D radius r
        """
        return self._EPL.density_lens(r, theta_E, gamma)


tree_util.register_pytree_node(
    EPLMajorAxis, EPLMajorAxis._tree_flatten, EPLMajorAxis._tree_unflatten
)
