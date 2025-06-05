__author__ = "sibirrer"

from functools import partial
from jax import config, jit, numpy as jnp

config.update("jax_enable_x64", True)

import numpy as np
import sys

EPSILON = sys.float_info.epsilon


@partial(jit, static_argnums=3)
def get_param_WLS(A, C_D_inv, d, inv_bool=True):
    """Returns the parameter values given.

    :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
    :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, diagonal form
    :param d: data array, 1-d Nd
    :param inv_bool: boolean, whether returning also the inverse matrix or just solve
        the linear system
    :return: 1-d array of parameter values
    """
    A = jnp.asarray(A, dtype=float)
    C_D_inv = jnp.asarray(C_D_inv, dtype=float)
    d = jnp.asarray(d, dtype=float)

    M = A.T.dot(jnp.multiply(C_D_inv, A.T).T)
    stability_check = jnp.linalg.cond(M) < 5 / EPSILON
    if inv_bool:
        M_inv = jnp.where(stability_check, _stable_inv(M), jnp.zeros_like(M))
        R = A.T.dot(jnp.multiply(C_D_inv, d))
        B = M_inv.dot(R)
    else:
        R = A.T.dot(jnp.multiply(C_D_inv, d))
        B = jnp.where(stability_check, _solve_stable(M, R), jnp.zeros(len(A.T)))
        M_inv = None
    image = A.dot(B)
    return B, M_inv, image


@jit
def marginalisation_const(M_inv):
    """Get marginalisation constant 1/2 log(M_beta) for flat priors.

    :param M_inv: 2D covariance matrix
    :return: float
    """

    sign, log_det = jnp.linalg.slogdet(M_inv)
    result = jnp.where(sign == 0, -(10**15), sign * log_det / 2)
    return result


@jit
def marginalization_new(M_inv, d_prior=None):
    """

    :param M_inv: 2D covariance matrix
    :param d_prior: maximum prior length of linear parameters
    :return: log determinant with eigenvalues to be smaller or equal d_prior
    """
    if d_prior is None:
        return marginalisation_const(M_inv)
    v, w = jnp.linalg.eig(M_inv)
    sign_v = jnp.sign(v)
    v_abs = jnp.abs(v)

    v_abs = jnp.where(v_abs > d_prior**2, d_prior**2, v_abs)
    log_det = jnp.sum(jnp.log(v_abs)) * jnp.prod(sign_v)
    m = len(v)
    result = jnp.where(
        jnp.isnan(log_det),
        -(10**15),
        log_det / 2 + m / 2.0 * jnp.log(jnp.pi / 2.0) - m * jnp.log(d_prior),
    )
    return result


@jit
def _stable_inv(m):
    """Stable linear inversion.

    :param m: square matrix to be inverted
    :return: inverse of M (or zeros if non-invertible)
    """
    m_inv = jnp.linalg.inv(m)

    # NOTE: Is this even needed? We already check jnp.linalg.cond < 5/EPSILON
    m_inv = jnp.nan_to_num(m_inv, nan=0, posinf=0, neginf=0)
    return m_inv


@jit
def _solve_stable(m, r):
    """

    :param m: matrix
    :param r: vector
    :return: solution for m x b = r
    """
    b = jnp.linalg.solve(m, r).T

    # NOTE: Is this even needed? We already check jnp.linalg.cond < 5/EPSILON
    invertible = jnp.all(jnp.isfinite(b))
    n = jnp.shape(m)[0]
    b = jnp.where(invertible, b, jnp.zeros(n))
    return b
