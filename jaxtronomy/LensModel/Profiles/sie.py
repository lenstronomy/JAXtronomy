from jax import jit, tree_util
import jax.numpy as jnp
from jaxtronomy.LensModel.Profiles.nie import NIE
from jaxtronomy.LensModel.Profiles.epl import EPL
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


__all__ = ["SIE"]

NIE_INSTANCE = NIE()
EPL_INSTANCE = EPL()


class SIE(LensProfileBase):
    """Class for singular isothermal ellipsoid (SIS with ellipticity)

    .. math::
        \\kappa(x, y) = \\frac{1}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate system aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{1}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 − e*\\cos(2*\\phi)}} \\right)

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.
    """

    param_names = ["theta_E", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, s_scale=0.0000000001, gamma=2, NIE=True):
        """

        :param NIE: bool, if True, is using the NIE analytic model. Otherwise it uses EPL
        """
        self._s_scale = s_scale
        self._gamma = gamma
        self._nie = NIE
        super(SIE, self).__init__()

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (but there's no need to recompile if it changes to a previous value)
    def _tree_flatten(self):
        children = (self._s_scale, self._gamma)
        aux_data = {"NIE": self._nie}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angular coordinates)
        :param y: y-coordinate (angular coordinates)
        :param theta_E: Einstein radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: centroid
        :param center_y: centroid
        :return:
        """
        if self._nie:
            return NIE_INSTANCE.function(
                x, y, theta_E, e1, e2, self._s_scale, center_x, center_y
            )
        else:
            return EPL_INSTANCE.function(
                x, y, theta_E, self._gamma, e1, e2, center_x, center_y
            )

    @jit
    def derivatives(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angular coordinates)
        :param y: y-coordinate (angular coordinates)
        :param theta_E: Einstein radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: centroid
        :param center_y: centroid
        :return:
        """
        if self._nie:
            return NIE_INSTANCE.derivatives(
                x, y, theta_E, e1, e2, self._s_scale, center_x, center_y
            )
        else:
            return EPL_INSTANCE.derivatives(
                x, y, theta_E, self._gamma, e1, e2, center_x, center_y
            )

    @jit
    def hessian(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angular coordinates)
        :param y: y-coordinate (angular coordinates)
        :param theta_E: Einstein radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: centroid
        :param center_y: centroid
        :return:
        """
        if self._nie:
            return NIE_INSTANCE.hessian(
                x, y, theta_E, e1, e2, self._s_scale, center_x, center_y
            )
        else:
            return EPL_INSTANCE.hessian(
                x, y, theta_E, self._gamma, e1, e2, center_x, center_y
            )

    @staticmethod
    @jit
    def theta2rho(theta_E):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter.

        :param theta_E:
        :return:
        """
        fac1 = jnp.pi * 2
        rho0 = theta_E / fac1
        return rho0

    @staticmethod
    @jit
    def mass_3d(r, rho0, e1=0, e2=0):
        """Mass enclosed a 3d sphere or radius r.

        :param r: radius in angular units
        :param rho0: density at angle=1
        :return: mass in angular units
        """
        mass_3d = 4 * jnp.pi * rho0 * r
        return mass_3d

    @jit
    def mass_3d_lens(self, r, theta_E, e1=0, e2=0):
        """Mass enclosed a 3d sphere or radius r given a lens parameterization with
        angular units.

        :param r: radius in angular units
        :param theta_E: Einstein radius
        :return: mass in angular units
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_3d(r, rho0)

    @jit
    def mass_2d(self, r, rho0, e1=0, e2=0):
        """Mass enclosed projected 2d sphere of radius r.

        :param r:
        :param rho0:
        :param e1:
        :param e2:
        :return:
        """
        alpha = 2 * rho0 * jnp.pi**2
        mass_2d = alpha * r
        return mass_2d

    @jit
    def mass_2d_lens(self, r, theta_E, e1=0, e2=0):
        """

        :param r:
        :param theta_E:
        :param e1:
        :param e2:
        :return:
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_2d(r, rho0)

    @jit
    def grav_pot(self, x, y, rho0, e1=0, e2=0, center_x=0, center_y=0):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units)

        :param x:
        :param y:
        :param rho0:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        mass_3d = self.mass_3d(r, rho0)
        pot = mass_3d / r
        return pot

    @jit
    def density_lens(self, r, theta_E, e1=0, e2=0):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: radius in angles
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: density
        """
        rho0 = self.theta2rho(theta_E)
        return self.density(r, rho0)

    @staticmethod
    @jit
    def density(r, rho0, e1=0, e2=0):
        """Computes the density.

        :param r: radius in angles
        :param rho0: density at angle=1
        :return: density at r
        """
        rho = rho0 / r**2
        return rho

    @staticmethod
    @jit
    def density_2d(x, y, rho0, e1=0, e2=0, center_x=0, center_y=0):
        """Projected density.

        :param x:
        :param y:
        :param rho0:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        sigma = jnp.pi * rho0 / r
        return sigma


tree_util.register_pytree_node(SIE, SIE._tree_flatten, SIE._tree_unflatten)
