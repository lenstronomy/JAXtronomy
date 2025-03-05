from jax import jit, tree_util
import jax.numpy as jnp
import jax.scipy.special as special

from jaxtronomy.Util import param_util

__all__ = ["SersicUtil"]


class SersicUtil(object):

    def __init__(self, smoothing=0.0001, sersic_major_axis=False):
        """

        :param smoothing: smoothing scale of the innermost part of the profile (for numerical reasons)
        :param sersic_major_axis: boolean; if True, defines the half-light radius of the Sersic light profile along
         the semi-major axis (which is the Galfit convention)
         if False, uses the product average of semi-major and semi-minor axis as the convention
         (default definition for all light profiles in lenstronomy other than the Sersic profile)
        """
        self._smoothing = smoothing
        self._sersic_major_axis = sersic_major_axis

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
    def k_bn(self, n, Re):
        """Returns normalisation of the sersic profile such that Re is the half light
        radius given n_sersic slope.

        :param n: Sersic index
        :param Re: the desired half light radius
        """

        bn = self.b_n(n)
        k = bn * Re ** (-1.0 / n)
        return k, bn

    @jit
    def k_Re(self, n, k):
        """Returns the half light radius given the n_sersic slope and normalization of
        the sersic profile.

        :param n: Sersic index
        :param k: normalization of the sersic profile
        """

        bn = self.b_n(n)
        Re = (bn / k) ** n
        return Re

    @staticmethod
    @jit
    def b_n(n):
        """B(n) computation. This is the approximation of the exact solution to the
        relation, 2*incomplete_gamma_function(2n; b_n) = Gamma_function(2*n).

        :param n: the sersic index
        :return: b(n)
        """
        bn = 1.9992 * n - 0.3271
        bn = jnp.maximum(
            bn, 0.00001
        )  # make sure bn is strictly positive as a save guard for very low n_sersic
        return bn

    @jit
    def get_distance_from_center(self, x, y, e1, e2, center_x, center_y):
        """Get the distance from the center of Sersic, accounting for orientation and
        axis ratio.

        :param x: position
        :param y: position
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center x of sersic
        :param center_y: center y of sersic
        :return: distance from center of Sersic
        """

        if self._sersic_major_axis:
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            x_shift = x - center_x
            y_shift = y - center_y
            cos_phi = jnp.cos(phi_G)
            sin_phi = jnp.sin(phi_G)
            xt1 = cos_phi * x_shift + sin_phi * y_shift
            xt2 = -sin_phi * x_shift + cos_phi * y_shift
            xt2difq2 = xt2 / (q * q)
            r = jnp.sqrt(xt1 * xt1 + xt2 * xt2difq2)
        else:
            x_, y_ = param_util.transform_e1e2_product_average(
                x, y, e1, e2, center_x, center_y
            )
            r = jnp.sqrt(x_**2 + y_**2)
        return r

    @jit
    def _x_reduced(self, x, y, n_sersic, r_eff, center_x, center_y):
        """Coordinate transform to normalized radius.

        :param x: position
        :param y: position
        :param center_x: position of the center of the source
        :param center_y: position of the center of the source
        :return: transformed normalized radius coordinate
        """
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        r = jnp.where(r < self._smoothing, self._smoothing, r)
        x_reduced = (r / r_eff) ** (1.0 / n_sersic)
        return x_reduced

    @jit
    def _alpha_eff(self, r_eff, n_sersic, k_eff):
        """Deflection angle at r_eff.

        :param r_eff: projected half light radius
        :param n_sersic: Sersic index
        :param k_eff: convergence at half light radius
        :return: Deflection angle at r_eff
        """
        b = self.b_n(n_sersic)
        alpha_eff = (
            n_sersic
            * r_eff
            * k_eff
            * b ** (-2 * n_sersic)
            * jnp.exp(b)
            * special.gamma(2 * n_sersic)
        )
        return -alpha_eff

    @jit
    def alpha_abs(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """Returns the absolute value of the deflection angle.

        :param x: position
        :param y: position
        :param n_sersic: Sersic index
        :param r_eff: projected half light radius
        :param k_eff: convergence at half light radius
        :param center_x: position of the center of the source
        :param center_y: position of the center of the source
        :return: absolute value of deflection angle
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n_sersic, r_eff, center_x, center_y)
        b = self.b_n(n_sersic)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        alpha = 2.0 * a_eff * x_red ** (-n) * (special.gammainc(2 * n, b * x_red))
        return alpha

    @jit
    def d_alpha_dr(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """Returns the derivative of the deflection angle w.r.t radius.

        :param x: position
        :param y: position
        :param n_sersic: Sersic index
        :param r_eff: projected half light radius
        :param k_eff: convergence at half light radius
        :param center_x: position of the center of the source
        :param center_y: position of the center of the source
        :return: derivative of deflection angle w.r.t radius
        """
        _dr = 0.00001
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        alpha = self.alpha_abs(r, 0, n_sersic, r_eff, k_eff)
        alpha_dr = self.alpha_abs(r + _dr, 0, n_sersic, r_eff, k_eff)
        d_alpha_dr = (alpha_dr - alpha) / _dr
        return d_alpha_dr

    def density(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """De-projection of the Sersic profile based on Prugniel & Simien (1997)
        :return:"""
        raise ValueError(
            "not implemented! Use a Multi-Gaussian-component decomposition."
        )

    @jit
    def _total_flux(self, r_eff, I_eff, n_sersic):
        """Computes total flux of a round Sersic profile.

        :param r_eff: projected half light radius
        :param I_eff: surface brightness at r_eff (in same units as r_eff)
        :param n_sersic: Sersic index
        :return: integrated flux to infinity
        """
        bn = self.b_n(n_sersic)
        return (
            I_eff
            * r_eff**2
            * 2
            * jnp.pi
            * n_sersic
            * jnp.exp(bn)
            / bn ** (2 * n_sersic)
            * special.gamma(2 * n_sersic)
        )

    @jit
    def total_flux(self, amp, R_sersic, n_sersic, e1=0, e2=0, **kwargs):
        """Computes analytical integral to compute total flux of the Sersic profile.

        :param amp: amplitude parameter in Sersic function (surface brightness at
            R_sersic
        :param R_sersic: half-light radius in semi-major axis
        :param n_sersic: Sersic index
        :param e1: eccentricity
        :param e2: eccentricity
        :return: Analytic integral of the total flux of the Sersic profile
        """
        # compute product average half-light radius
        if self._sersic_major_axis:
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            # translate semi-major axis R_eff into product averaged definition for circularization
            r_eff = R_sersic * jnp.sqrt(q)
        else:
            r_eff = R_sersic
        return self._total_flux(r_eff=r_eff, I_eff=amp, n_sersic=n_sersic)

    @jit
    def _R_stable(self, R):
        """Floor R_ at self._smoothing for numerical stability.

        :param R: radius
        :return: smoothed and stabilized radius
        """
        return jnp.maximum(self._smoothing, R)

    @jit
    def _r_sersic(
        self, R, R_sersic, n_sersic, max_R_frac=1000.0, alpha=1.0, R_break=0.0
    ):
        """

        :param R: radius (array or float)
        :param R_sersic: Sersic radius (half-light radius)
        :param n_sersic: Sersic index (float)
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: kernel of the Sersic surface brightness at R
        """

        R_ = self._R_stable(R)
        R_sersic_ = self._R_stable(R_sersic)
        bn = self.b_n(n_sersic)
        R_frac = R_ / R_sersic_

        exponent = -bn * (R_frac ** (1.0 / n_sersic) - 1.0)
        result = jnp.where(R_frac <= max_R_frac, jnp.exp(exponent), 0)
        return jnp.nan_to_num(result)


tree_util.register_pytree_node(
    SersicUtil, SersicUtil._tree_flatten, SersicUtil._tree_unflatten
)
