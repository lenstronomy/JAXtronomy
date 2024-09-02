__author__ = "sibirrer"

#  this file contains a class to make a Sersic profile

from jax import jit, tree_util
import jax.numpy as jnp

from jaxtronomy.LensModel.Profiles.sersic_utils import SersicUtil
import jaxtronomy.Util.param_util as param_util


class SersicElliptic(SersicUtil):
    """This class contains functions to evaluate an elliptical Sersic function.

    .. math::

        I(R) = I_{\\rm e} \\exp \\left( -b_n \\left[(R/R_{\\rm Sersic})^{\\frac{1}{n}}-1\\right]\\right)

    with :math:`I_0 = amp`,
    :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`
    and
    with :math:`b_{n}\\approx 1.999n-0.327`
    """

    param_names = ["amp", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (but there's no need to recompile if it changes to a previous value)
    def _tree_flatten(self):
        children = (self._smoothing,)
        aux_data = {"sersic_major_axis": self._sersic_major_axis}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        n_sersic,
        e1,
        e2,
        center_x=0,
        center_y=0,
        max_R_frac=1000.0,
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter e1
        :param e2: eccentricity parameter e2
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        R_sersic = jnp.maximum(0, R_sersic)
        R = self.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


class SersicElliptic_qPhi(SersicUtil):
    """This class is the same as SersicElliptic except sampling over q and phi instead
    of e1 and e2."""

    param_names = ["amp", "R_sersic", "n_sersic", "q", "phi", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "q": 0,
        "phi": -jnp.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "q": 1.0,
        "phi": jnp.pi,
        "center_x": 100,
        "center_y": 100,
    }

    # --------------------------------------------------------------------------------
    # The following two methods are required to allow the JAX compiler to recognize
    # this class. Methods involving the self variable can be jit-decorated.
    # Class methods will need to be recompiled each time a variable in the aux_data
    # changes to a new value (but there's no need to recompile if it changes to a previous value)
    def _tree_flatten(self):
        children = (self._smoothing,)
        aux_data = {"sersic_major_axis": self._sersic_major_axis}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # --------------------------------------------------------------------------------

    @jit
    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        n_sersic,
        q,
        phi,
        center_x=0,
        center_y=0,
        max_R_frac=100.0,
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param n_sersic: Sersic index
        :param q: axis ratio
        :param phi: position angle (radians)
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        R_sersic = jnp.maximum(0, R_sersic)
        R = self.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


tree_util.register_pytree_node(
    SersicElliptic, SersicElliptic._tree_flatten, SersicElliptic._tree_unflatten
)


tree_util.register_pytree_node(
    SersicElliptic_qPhi,
    SersicElliptic_qPhi._tree_flatten,
    SersicElliptic_qPhi._tree_unflatten,
)
