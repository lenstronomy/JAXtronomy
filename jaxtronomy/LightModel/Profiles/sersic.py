__author__ = "sibirrer"

#  this file contains a class to make a Sersic profile

from jax import jit, tree_util

from jaxtronomy.LensModel.Profiles.sersic_utils import SersicUtil


class Sersic(SersicUtil):
    """This class contains functions to evaluate a spherical Sersic function.

    .. math::
        I(R) = I_{\\rm e} \\exp \\left( -b_n \\left[(R/R_{\\rm Sersic})^{\\frac{1}{n}}-1\\right]\\right)

    with :math:`I_0 = amp`
    and
    with :math:`b_{n}\\approx 1.999n-0.327`
    """

    param_names = ["amp", "R_sersic", "n_sersic", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
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
        self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=1000.0
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """
        R = self.get_distance_from_center(
            x, y, e1=0, e2=0, center_x=center_x, center_y=center_y
        )
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


tree_util.register_pytree_node(Sersic, Sersic._tree_flatten, Sersic._tree_unflatten)
