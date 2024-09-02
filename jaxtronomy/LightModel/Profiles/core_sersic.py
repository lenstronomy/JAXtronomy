__author__ = "sibirrer"

#  this file contains a class to make a Sersic profile

from jax import jit, tree_util
import jax.numpy as jnp

from jaxtronomy.LensModel.Profiles.sersic_utils import SersicUtil


class CoreSersic(SersicUtil):
    """This class contains the Core-Sersic function introduced by e.g. Trujillo et al.
    2004.

    .. math::

        I(R) = I' \\left[1 + (R_b/R)^{\\alpha} \\right]^{\\gamma / \\alpha}
        \\exp \\left{ -b_n \\left[(R^{\\alpha} + R_b^{\\alpha})/R_e^{\\alpha}  \\right]^{1 / (n\\alpha)}  \\right}

    with

    .. math::
        I' = I_b 2^{-\\gamma/ \\alpha} \\exp \\left[b_n 2^{1 / (n\\alpha)} (R_b/R_e)^{1/n}  \\right]

    where :math:`I_b` is the intensity at the break radius and :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`.
    """

    param_names = [
        "amp",
        "R_sersic",
        "Rb",
        "n_sersic",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "Rb": 0,
        "n_sersic": 0.5,
        "gamma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "Rb": 100,
        "n_sersic": 8,
        "gamma": 10,
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
        Rb,
        n_sersic,
        gamma,
        e1,
        e2,
        center_x=0,
        center_y=0,
        alpha=3.0,
        max_R_frac=1000.0,
    ):
        """
        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param Rb: "break" core radius
        :param n_sersic: Sersic index
        :param gamma: inner power-law exponent
        :param e1: eccentricity parameter e1
        :param e2: eccentricity parameter e2
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param alpha: sharpness of the transition between the cusp and the outer Sersic profile (float)
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Cored Sersic profile value at (x, y)
        """
        # TODO: max_R_frac not implemented
        R_ = self.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        R = self._R_stable(R_)
        bn = self.b_n(n_sersic)
        result = (
            amp
            * (1 + (Rb / R) ** alpha) ** (gamma / alpha)
            * jnp.exp(
                -bn
                * (
                    ((R**alpha + Rb**alpha) / R_sersic**alpha)
                    ** (1.0 / (alpha * n_sersic))
                    - 1.0
                )
            )
        )
        return jnp.nan_to_num(result)


tree_util.register_pytree_node(
    CoreSersic, CoreSersic._tree_flatten, CoreSersic._tree_unflatten
)
