import jax.numpy as jnp
from jax import jit, lax

from functools import partial


@partial(jit, static_argnums=0)
def eval_hermite(n, x):
    """Equivalent to scipy.special.eval_hermite(n, x).

    :param n: int, order of hermite polynomial to be evaluated
    :param x: array-like, coordinates to evaluate hermite polynomial
    :return: array with same shape as x containing H_n(x)
    """
    x = jnp.array(x, dtype=float)

    prev_H = jnp.ones_like(x)
    H = 2.0 * x

    def body_fun(i, val):
        x, H, prev_H = val
        H, prev_H = 2 * x * H - 2 * (i - 1) * prev_H, H
        return (x, H, prev_H)

    result = lax.fori_loop(2, n + 1, body_fun, (x, H, prev_H))[1]

    result = jnp.where(n == 0, 1.0, result)

    return result


@jit
def hermval(x, c):
    """Equivalent to numpy.polynomial.hermite.hermval when the input c is 1 dimensional.

    :param x: array-like, coordinates to evaluate hermite polynomials
    :param c: array-like, with dimension equal to 1
    :return: array with the same shape as x containing hermval(x)
    """
    x = jnp.array(x, dtype=float)
    x_shape = x.shape
    x = jnp.ravel(x)
    n_array = jnp.array(c, dtype=float)

    H = jnp.zeros((len(n_array), len(x)))
    H = H.at[0].set(jnp.ones_like(x))
    H = H.at[1].set(2.0 * x)

    def body_fun(i, H):
        new_H = 2 * x * H.at[i - 1].get() - 2 * (i - 1) * H.at[i - 2].get()
        H = H.at[i].set(new_H)
        return H

    H = lax.fori_loop(2, len(n_array), body_fun, H)
    return jnp.sum(H.T * n_array, axis=1).reshape(x_shape)
