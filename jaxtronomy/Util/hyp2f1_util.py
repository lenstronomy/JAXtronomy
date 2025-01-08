# hyp2f1 functions to use with JAX

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma

# TODO: The analytic continuation formula used in hyp2f1_continuation only works
#       whenever b - a is not an integer. Additionally, hyp2f1_near_one only works
#       whenever c - b - a is not an integer. Other implementations are required
#       for these situations.


@jit
def hyp2f1_series(a, b, c, z):
    """This computation uses the well known relation between successive terms in the
    hypergeometric series hyp2f1(z) = sum_i A_i where.

    A_0 = 1
    A_i = z * [(i - 1 + a)(i - 1 + b) / i(i - 1 + c)] A_{i-1}

    This iterative implementation is necessary, since the usual hypergeometric
    function `hyp2f1` provided in `scipy.special` has not yet been implemented
    in an autodifferentiable or JIT-compilable way in JAX.

    This implementation can only be used whenever |z| < 1. Otherwise the
    series representation of 2F1 diverges and analytic continuation must
    be used.
    """
    z = jnp.asarray(z)

    # Set the first term of the series, A_0 = 1
    A_i = 1.0 * jnp.ones_like(z)
    partial_sum = A_i

    def body_fun(i, val):
        partial_sum, A_i = val

        # Currrent term in the series is proportional to the previous
        ratio = (i - 1.0 + a) * (i - 1.0 + b) / (i * (i - 1.0 + c))
        A_i = z * ratio * A_i

        # Adds the current term to the partial sum
        partial_sum += A_i

        return [partial_sum, A_i]

    result = lax.fori_loop(1, 150, body_fun, [partial_sum, A_i])
    return result[0]


# TODO: Implement a version that can handle b - a = integer
@jit
def hyp2f1_continuation(a, b, c, z):
    """
    This implementation is based off of the analytic continuation formulas
    Equations 4.21 and 4.22 with z0 = 1/2 in John Pearson's MSc thesis
    https://www.math.ucla.edu/~mason/research/pearson_final.pdf

    This approach works whenever z is outside of the circle |z-1/2| = 1/2.
    Due to the presence of gamma(b - a) and gamma(a - b), there will always
    be a pole whenever b - a is an integer.
    """
    z = jnp.asarray(z)

    # d0 = 1 and d_{-1} = 0
    prev_da = 1.0
    prev_db = 1.0
    prev_prev_da = 0.0
    prev_prev_db = 0.0

    # sum_1 corresponds to the summation on the top line of equation 4.21
    # sum_2 corresponds to the summation on the bottom line of equation 4.21
    # Gamma function prefactors are multiplied at the end
    # Allows for the input z to be an array of values
    sum_1 = 1.0 * jnp.ones_like(z)
    sum_2 = 1.0 * jnp.ones_like(z)

    # The branch cut for this computation of hyp2f1 is on Re(z) >= 1, Im(z) = 0
    # If z is on the branch cut, take the value above the branch cut
    z = jnp.where(jnp.imag(z) == 0.0, jnp.where(jnp.real(z) >= 1, z + 0.0000001j, z), z)

    def body_fun(j, val):
        prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2 = val

        # ------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the first line of equation 4.21
        # calculates d_j and the corresponding term in the sum
        d_ja = (
            (j + a - 1.0)
            / (j * (j + a - b))
            * (
                ((a + b + 1.0) * 0.5 - c) * prev_da
                + 0.25 * (j + a - 2.0) * prev_prev_da
            )
        )
        sum_1 += d_ja * (z - 0.5) ** (-j)

        # updates d_{j-2} and d_{j-1}
        prev_prev_da = prev_da
        prev_da = d_ja

        # ------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the second line of equation 4.21
        # calculates d_j and the corresponding term in the sum
        d_jb = (
            (j + b - 1.0)
            / (j * (j - a + b))
            * (
                ((a + b + 1.0) * 0.5 - c) * prev_db
                + 0.25 * (j + b - 2.0) * prev_prev_db
            )
        )
        sum_2 += d_jb * (z - 0.5) ** (-j)

        # updates d_{j-2} and d_{j-1}
        prev_prev_db = prev_db
        prev_db = d_jb

        # -----------------------------------------------------------------------------------------------------
        return [prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2]

    result = lax.fori_loop(
        1, 200, body_fun, [prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2]
    )

    # includes the gamma function prefactors in equation 4.21 to compute the final result of 2F1
    final_result = gamma(c) * (
        result[4] * gamma(b - a) / gamma(b) / gamma(c - a) * (0.5 - z) ** (-a)
        + result[5] * gamma(a - b) / gamma(a) / gamma(c - b) * (0.5 - z) ** (-b)
    )
    return final_result


# TODO: Implement a version that can handle c - b - a = integer
#       This can be done using equations 15.3.10 - 15.3.12 in
#       Abramowitz and Stegun
@jit
def hyp2f1_near_one(a, b, c, z):
    """This implementation is based off of equation 15.3.6 in Abramowitz and Stegun.
    This transformation formula allows for a calculation of hyp2f1 for points near.

    z = 1 (where other iterative computation schemes converge slowly) by
    transforming z to 1 - z.

    However, due to the presence of gamma(c - a - b) and gamma(a + b - c),
    whenever c - a - b is an integer, one of the two terms will have
    a pole
    """
    z = jnp.asarray(z)

    # The branch cut for the hypergeometric function is on Re(z) >= 1, Im(z) = 0
    # If z is on the branch cut, take the value above the branch cut
    z = jnp.where(
        jnp.imag(z) == 0.0, jnp.where(jnp.real(z) >= 1.0, z + 0.0000001j, z), z
    )

    term1 = (
        gamma(c)
        * gamma(c - a - b)
        / gamma(c - a)
        / gamma(c - b)
        * hyp2f1_series(a, b, a + b - c + 1.0, 1.0 - z)
    )
    term2 = (
        (1.0 - z) ** (c - a - b)
        * gamma(c)
        * gamma(a + b - c)
        / gamma(a)
        / gamma(b)
        * hyp2f1_series(c - a, c - b, c - a - b + 1.0, 1.0 - z)
    )
    return term1 + term2


@jit
def hyp2f1(a, b, c, z):
    """This function looks at where z is located on the complex plane and chooses the
    appropriate hyp2f1 function to use in the interest of maintaining optimal runtime
    and accuracy.

    If the user already knows which hyp2f1 function to use, this step can be skipped and
    the user can directly call the appropriate hyp2f1 function. If the input z is an
    array with values in different regions on the complex plane, this function MUST be
    used so that the appropriate hyp2f1 function is used for each value.
    """
    z = jnp.asarray(z)
    z_shape = jnp.shape(z)
    z = jnp.atleast_1d(z)

    # Case 0: Whenever |z| < 0.89, hyp2f1_series should be used
    # Case 1: Else if |1 - z| < 0.76, hyp2f1_near_one should be used
    # Case 2: Else, use analytic continuation
    # A visual representation of these regions can be found at
    # https://www.desmos.com/calculator/1sympxth2t
    case = jnp.where(jnp.abs(z) < 0.89, 0, jnp.where(jnp.abs(1.0 - z) < 0.76, 1, 2))
    hyp2f1_func = [hyp2f1_series, hyp2f1_near_one, hyp2f1_continuation]

    # Check each z value and evaluate corresponding hyp2f1
    def body_fun(i, val):
        ith_result = lax.switch(case.at[i].get(), hyp2f1_func, a, b, c, z.at[i].get())
        val = val.at[i].set(ith_result)
        return val

    result = lax.fori_loop(0, jnp.size(z), body_fun, jnp.zeros_like(z))

    # Need to reshape result back to the original shape so that if
    # a scalar z is input, a scalar is returned instead of 1d array
    result = jnp.reshape(result, z_shape)
    return result
