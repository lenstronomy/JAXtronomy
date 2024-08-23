__author__ = "sibirrer"

from jax import jit, tree_util
import jax.numpy as jnp
import jaxtronomy.Util.util as util
import jaxtronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["NIE", "NIEMajorAxis"]


class NIE(LensProfileBase):
    """Non-singular isothermal ellipsoid (NIE)

    .. math::
     \\kappa = \\theta_E/2 \\left[s^2_{scale} + qx^2 + y^2/q]−1/2
    """

    param_names = ["theta_E", "e1", "e2", "s_scale", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "e1": -0.5,
        "e2": -0.5,
        "s_scale": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 10,
        "e1": 0.5,
        "e2": 0.5,
        "s_scale": 100,
        "center_x": 100,
        "center_y": 100,
    }

    # These static self variables are not used until self.set_static is called
    # However these need to be here for the JAX to correctly keep track of them
    def __init__(self, b=0, s=0, q=0, phi=0, static=False):
        self._static = static
        self._b_static = b
        self._s_static = s
        self._q_static = q
        self._phi_G_static = phi

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (but there's no need to recompile if it changes to a previous value)
    def _tree_flatten(self):
        children = (self._b_static, self._s_static, self._q_static, self._phi_G_static)
        aux_data = {"static": self._static}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = NIEMajorAxis.function(x__, y__, b, s, q)
        # rotate back
        return f_

    @jit
    def derivatives(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = NIEMajorAxis.derivatives(x__, y__, b, s, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    @jit
    def hessian(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__xy, _, f__yy = NIEMajorAxis.hessian(x__, y__, b, s, q)
        # rotate back
        kappa = 1.0 / 2 * (f__xx + f__yy)
        gamma1__ = 1.0 / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = jnp.cos(2 * phi_G) * gamma1__ - jnp.sin(2 * phi_G) * gamma2__
        gamma2 = jnp.sin(2 * phi_G) * gamma1__ + jnp.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @jit
    def density_lens(self, r, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """3d mass density at 3d radius r. This function assumes spherical
        symmetry/ignoring the eccentricity.

        :param r: 3d radius
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: 3d mass density at 3d radius r
        """
        # kappa=1/2 at Einstein radius
        rho0 = 1 / 2 * theta_E / jnp.pi
        return rho0 / (r**2 + s_scale**2)

    @jit
    def mass_3d_lens(self, r, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """Mass enclosed a 3d radius r. This function assumes spherical
        symmetry/ignoring the eccentricity.

        :param r: 3d radius
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: 3d mass density at 3d radius r
        """
        rho0 = 1 / 2 * theta_E / jnp.pi
        return rho0 * 4 * jnp.pi * (r - s_scale * jnp.arctan(r / s_scale))

    @jit
    def param_conv(self, theta_E, e1, e2, s_scale):
        if self._static is True:
            return self._b_static, self._s_static, self._q_static, self._phi_G_static
        return self._param_conv(theta_E, e1, e2, s_scale)

    @jit
    def _param_conv(self, theta_E, e1, e2, s_scale):
        """
        convert parameters from 2*kappa = bIE [s2IE + r2(1 − e *cos(2*phi)]−1/2 to
        2*kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
        see expressions after Equation 8 in Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf

        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :return: critical radius b, smoothing scale s, axis ratio q, orientation angle phi_G
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E_conv = self._theta_E_prod_average2major_axis(theta_E, q)
        b = theta_E_conv * jnp.sqrt((1 + q**2) / 2)
        s = s_scale / jnp.sqrt(q)
        # s = s_scale * jnp.sqrt((1 + q**2) / (2*q**2))
        return b, s, q, phi_G

    # NOTE: Do not jit-decorate this function; it won't work correctly
    def set_static(self, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        (
            self._b_static,
            self._s_static,
            self._q_static,
            self._phi_G_static,
        ) = self._param_conv(theta_E, e1, e2, s_scale)

    # NOTE: Do not jit-decorate this function; it won't work correctly
    def set_dynamic(self):
        """

        :return:
        """
        self._static = False

    @staticmethod
    @jit
    def _theta_E_prod_average2major_axis(theta_E, q):
        """Converts a product averaged Einstein radius (of semi-minor and semi-major
        axis) to a major axis Einstein radius for an Isothermal ellipse. The standard
        lenstronomy conventions are product averaged Einstein radii while other codes
        (such as e.g. gravlens) use the semi-major axis convention.

        .. math::
          \\frac{\\theta_{E, prod ave}}{\\theta_{E, major}} = \\sqrt{(1+q^2) / (2 q) }

        :param theta_E: Einstein radius in lenstronomy conventions (product average of major and minor axes)
        :param q: axis ratio minor/major
        :return: theta_E in convention of kappa= b *(q^2(s^2 + x^2) + y^2􏰉)^{−1/2} (major axis)
        """
        theta_E_major_axis = theta_E / (jnp.sqrt((1.0 + q**2) / (2.0 * q)))
        return theta_E_major_axis


class NIEMajorAxis(LensProfileBase):
    """
    This class contains the function and the derivatives of the non-singular isothermal ellipse.
    See Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf

    .. math::
        \\kappa =  b * (q2(s2 + x2) + y2􏰉)^{−1/2}`

    """

    param_names = ["b", "s", "q", "center_x", "center_y"]

    def __init__(self):
        super(NIEMajorAxis, self).__init__()

    @staticmethod
    @jit
    def function(x, y, b, s, q):
        psi = NIEMajorAxis._psi(x, y, q, s)
        alpha_x, alpha_y = NIEMajorAxis.derivatives(x, y, b, s, q)
        f_ = (
            x * alpha_x
            + y * alpha_y
            - b * s * 1.0 / 2.0 * jnp.log((psi + s) ** 2 + (1.0 - q**2) * x**2)
        )
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, b, s, q):
        """Returns df/dx and df/dy of the function."""
        q = jnp.where(q >= 1, 0.99999999, q)
        psi = NIEMajorAxis._psi(x, y, q, s)
        f_x = (
            b / jnp.sqrt(1.0 - q**2) * jnp.arctan(jnp.sqrt(1.0 - q**2) * x / (psi + s))
        )
        f_y = (
            b
            / jnp.sqrt(1.0 - q**2)
            * jnp.arctanh(jnp.sqrt(1.0 - q**2) * y / (psi + q**2 * s))
        )
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, b, s, q):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
        d^f/dy^2."""
        alpha_ra, alpha_dec = NIEMajorAxis.derivatives(x, y, b, s, q)
        diff = 0.0000000001
        alpha_ra_dx, alpha_dec_dx = NIEMajorAxis.derivatives(x + diff, y, b, s, q)
        alpha_ra_dy, alpha_dec_dy = NIEMajorAxis.derivatives(x, y + diff, b, s, q)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_xy, f_yx, f_yy

    @staticmethod
    @jit
    def kappa(x, y, b, s, q):
        """convergence.

        :param x: major axis coordinate
        :param y: minor axis coordinate
        :param b: normalization
        :param s: smoothing scale
        :param q: axis ratio
        :return: convergence
        """
        kappa = b / 2.0 * (q**2 * (s**2 + x**2) + y**2) ** (-1.0 / 2)
        return kappa

    @staticmethod
    @jit
    def _psi(x, y, q, s):
        """Expression after equation (8) in Keeton&Kochanek 1998.

        :param x: semi-major axis coordinate
        :param y: semi-minor axis coordinate
        :param q: axis ratio minor/major
        :param s: smoothing scale in major axis direction
        :return: phi
        """
        return jnp.sqrt(q**2 * (s**2 + x**2) + y**2)


tree_util.register_pytree_node(NIE, NIE._tree_flatten, NIE._tree_unflatten)
