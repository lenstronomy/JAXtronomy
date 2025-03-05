__author__ = "sibirrer"

from functools import partial
from jax import lax, jit, numpy as jnp
import math
import numpy as np

from jaxtronomy.Util.herm_util import eval_hermite, hermval


class Shapelets(object):
    """Class for 2d cartesian Shapelets.

    Sources:
    Refregier 2003: Shapelets: I. A Method for Image Analysis https://arxiv.org/abs/astro-ph/0105178
    Refregier 2003: Shapelets: II. A Method for Weak Lensing Measurements https://arxiv.org/abs/astro-ph/0105179

    For one dimension, the shapelets are defined as

    .. math::
        \\phi_n(x) \\equiv \\left[2^n \\pi^{1/2} n!  \\right]]^{-1/2}H_n(x) e^{-\\frac{x^2}{2}}

    This basis is orthonormal. The dimensional basis function is

    .. math::
        B_n(x;\\beta) \\equiv \\beta^{-1/2} \\phi_n(\\beta^{-1}x)

    which are orthonormal as well.

    The two-dimensional basis function is

    .. math::
        \\phi_{\\bf n}({\\bf x}) \\equiv \\phi_{n1}(x1) \\phi_{n2}(x2)

    where :math:`{\\bf n} \\equiv (n1, n2)` and :math:`{\\bf x} \\equiv (x1, x2)`.

    The dimensional two-dimentional basis function is

    .. math::
        B_{\\bf n}({\\bf x};\\beta) \\equiv \\beta^{-1/2} \\phi_{\\bf n}(\\beta^{-1}{\\bf x}).
    """

    param_names = ["amp", "beta", "n1", "n2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "beta": 0.01,
        "n1": 0,
        "n2": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "beta": 100,
        "n1": 150,
        "n2": 150,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(
        self, interpolation=False, precalc=False, stable_cut=True, cut_scale=5
    ):
        """

        :param interpolation: boolean; False in jaxtronomy
        :param precalc: boolean; if True interprets as input (x, y) as pre-calculated
            normalized shapelets
        :param stable_cut: boolean; if True, sets the values outside of
            :math:`\\sqrt\\left(n_{\\rm max} + 1 \\right) \\beta s_{\\rm cut scale} =
            0`.
        :param cut_scale: float, scaling parameter where to cut the shapelets. This is
            for numerical reasons such that the polynomials in the Hermite function do
            not get unstable.
        """
        # Interpolation is actually slower than calculating the hermite polynomials normally using JAX
        if interpolation:
            raise ValueError(
                "interpolation feature for Shapelets is not supported in jaxtronomy"
            )

        self._precalc = precalc
        self._stable_cut = stable_cut
        self._cut_scale = cut_scale

    @partial(jit, static_argnums=(0, 3))
    def hermval(self, x, n_array, tensor=True):
        """
        computes the Hermit polynomial as numpy.polynomial.hermite.hermval
        difference: for values more than sqrt(n_max + 1) * cut_scale, the value is set to zero
        this should be faster and numerically stable

        :param x: array of values
        :param n_array: 1d list of coeffs in H_n
        :param tensor: see numpy.polynomial.hermite.hermval
        :return: see numpy.polynomial.hermite.hermval
        """
        result = hermval(x, n_array)
        if self._stable_cut:
            n_max = len(n_array)
            x_cut = np.sqrt(n_max + 1) * self._cut_scale
            result = jnp.where(x < x_cut, result, 0)
        return result

    # This function needs to recompile for different values of n1 and n2
    # This is required in order to use reverse-mode autodifferentiation
    # Fortunately, this function compiles quickly
    @partial(jit, static_argnums=(0, 5, 6))
    def function(self, x, y, amp, beta, n1, n2, center_x, center_y):
        """2d cartesian shapelet.

        :param x: x-coordinate
        :param y: y-coordinate
        :param amp: amplitude of shapelet
        :param beta: scale factor of shapelet
        :param n1: x-order of Hermite polynomial
        :param n2: y-order of Hermite polynomial
        :param center_x: center in x
        :param center_y: center in y
        :return: flux surface brightness at (x, y)
        """

        if self._precalc:
            return amp * x[n1] * y[n2]

        x_ = x - center_x
        y_ = y - center_y
        return jnp.nan_to_num(
            amp * self.phi_n(n1, x_ / beta) * self.phi_n(n2, y_ / beta)
        )

    @partial(jit, static_argnums=(0, 1))
    def H_n(self, n, x):
        """Constructs the Hermite polynomial of order n at position x (dimensionless)

        :param n: The n'the basis function.
        :param x: 1-dim position (dimensionless)
        :type x: float or jax.numpy array.
        :returns: array-- H_n(x).
        """
        result = eval_hermite(n, x)
        if self._stable_cut:
            x_cut = np.sqrt(n + 2) * self._cut_scale
            result = jnp.where(x < x_cut, result, 0)
        return result

    @partial(jit, static_argnums=(0, 1))
    def phi_n(self, n, x):
        """Constructs the 1-dim basis function (formula (1) in Refregier et al. 2001)

        :param n: The n'the basis function.
        :type n: int.
        :param x: 1-dim position (dimensionless)
        :type x: float or numpy array.
        :returns: array-- phi_n(x).
        """
        prefactor = 1.0 / np.sqrt(2**n * np.sqrt(np.pi) * math.factorial(n))
        return prefactor * self.H_n(n, x) * jnp.exp(-(x**2) / 2.0)

    @partial(jit, static_argnums=(0, 4))
    def pre_calc(self, x, y, beta, n_order, center_x, center_y):
        """Calculates the phi_n(x) and phi_n(y) for a given x-array and y-array for the
        full order in the polynomials.

        :param x: float or 1d array of x-coordinates
        :param y: float or 1d array of y-coordinates
        :param beta: shapelet scale
        :param n_order: order of shapelets
        :param center_x: shapelet center
        :param center_y: shapelet center
        :return: jnp.arrays of phi_n(x) and phi_n(y)
        """
        if n_order > 170:
            raise ValueError(f"polynomial order {n_order} too large")

        x_ = jnp.atleast_1d(x - center_x) / beta
        y_ = jnp.atleast_1d(y - center_y) / beta

        H_x = jnp.zeros((n_order + 1, len(x_)))
        H_x = H_x.at[0].set(jnp.ones_like(x_, dtype=float))
        H_x = H_x.at[1].set(2 * x_)

        H_y = jnp.zeros((n_order + 1, len(y_)))
        H_y = H_y.at[0].set(jnp.ones_like(y_, dtype=float))
        H_y = H_y.at[1].set(2 * y_)

        prefactor = jnp.zeros(n_order + 1, dtype=float)
        prefactor = prefactor.at[0].set(1.0 / (jnp.pi) ** (1.0 / 4.0))
        prefactor = prefactor.at[1].set(prefactor[0] / jnp.sqrt(2))

        exp_x = jnp.exp(-(x_**2) / 2.0)
        exp_y = jnp.exp(-(y_**2) / 2.0)

        def body_fun(n, val):
            H_x, H_y, prefactor = val

            new_Hx = 2 * x_ * H_x.at[n - 1].get() - 2 * (n - 1) * H_x.at[n - 2].get()
            H_x = H_x.at[n].set(new_Hx)

            new_Hy = 2 * y_ * H_y.at[n - 1].get() - 2 * (n - 1) * H_y.at[n - 2].get()
            H_y = H_y.at[n].set(new_Hy)

            new_prefactor = prefactor.at[n - 1].get() / jnp.sqrt(2 * n)
            prefactor = prefactor.at[n].set(new_prefactor)

            return H_x, H_y, prefactor

        H_x, H_y, prefactor = lax.fori_loop(
            2, n_order + 1, body_fun, (H_x, H_y, prefactor)
        )

        if self._stable_cut:
            cut_threshold = np.sqrt(n_order + 2) * self._cut_scale
            cut_x, cut_y = jnp.where(x_ < cut_threshold, 1, 0), jnp.where(
                y_ < cut_threshold, 1, 0
            )
        else:
            cut_x, cut_y = jnp.ones_like(x_), jnp.ones_like(y_)

        phi_x = ((H_x * cut_x * exp_x).T * prefactor).T
        phi_y = ((H_y * cut_y * exp_y).T * prefactor).T
        return phi_x, phi_y


class ShapeletSet(object):
    """Class to operate on entire shapelet set limited by a maximal polynomial order
    n_max, such that n1 + n2 <= n_max."""

    param_names = ["amp", "n_max", "beta", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "n_max": 1,
        "beta": 0.01,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "n_max": 20,
        "beta": 100,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.shapelets = Shapelets(precalc=True)

    @partial(jit, static_argnums=(0,))
    def function(self, x, y, amp, n_max, beta, center_x=0, center_y=0):
        """NOTE: The number of loops is determined based off of len(amp) instead of n_max
        for autodifferentiation to work. As long as len(amp) equals (n_max + 1) * (n_max + 2) / 2,
        this function will work correctly and identically to lenstronomy. However if there is
        any user error, the behaviour of this function will differ from lenstronomy.
        For performance reasons, we do not include any runtime checks of len(amp) and n_max.

        :param x: float or 1d array of x-coordinates
        :param y: float or 1d array of y-coordinates
        :param amp: 1d array of amplitudes in pre-defined order of shapelet basis functions
            amp[0] corresponds to (n1, n2) = (0, 0), amp[1] corresponds to (n1, n2) = (1, 0)
            amp[2] corresponds to (n1, n2) = (0, 1), amp[3] corresponds to (n1, n2) = (2, 0)
            amp[4] corresponds to (n1, n2) = (1, 1), amp[5] corresponds to (n1, n2) = (0, 2)
            amp[6] corresponds to (n1, n2) = (3, 0), etc
            len(amp) should be equal to (n_max + 1) * (n_max + 2) / 2
        :param beta: shapelet scale
        :param n_max: int, maximum order in Hermite polynomial. Although this argument is unused here,
            inonsistencies with len(amp) can lead to incorrect behaviour in e.g. Lightparam and Param classes.
        :param center_x: shapelet center
        :param center_y: shapelet center
        :return: surface brightness of combined shapelet set
        """
        x_shape = x.shape
        x = jnp.atleast_1d(x)
        f_ = jnp.zeros_like(x)
        amp = jnp.array(amp)

        n_order = int((-3 + np.sqrt(9 + 8 * (len(amp) - 1))) / 2)
        phi_x, phi_y = self.shapelets.pre_calc(x, y, beta, n_order, center_x, center_y)

        n1 = 0
        n2 = 0

        def body_fun(i, val):
            f_, n1, n2 = val
            f_ += amp.at[i].get() * phi_x.at[n1].get() * phi_y.at[n2].get()
            n1, n2 = jnp.where(n1 == 0, n2 + 1, n1 - 1), jnp.where(n1 == 0, 0, n2 + 1)
            return f_, n1, n2

        f_ = lax.fori_loop(0, len(amp), body_fun, (f_, n1, n2))[0]
        f_ = f_.reshape(x_shape)
        return jnp.nan_to_num(f_)
