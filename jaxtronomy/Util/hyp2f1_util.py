# hyp2f1 functions to use with JAX

from functools import partial
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gamma


@partial(jit, static_argnums=4)
def hyp2f1_series(a, b, c, z, nmax=50):
    """This computation is based off of the standard series expansion of hyp2f1.
    The recurrence relation between successive terms in the sum is well known:

    A_0 = 1
    A_i = z * [(i - 1 + a)(i - 1 + b) / i(i - 1 + c)] A_{i-1}

    The conditions required for this series to converge are:
    1) |z| < 1
    2) c is not a non-positive integer (i.e. c != 0, -1, -2, ...)
    3) If Re(c - a - b) > 0, then the series converges on |z| = 1 as well (but very slowly)
    """
    z = jnp.array(z, dtype=complex)

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

    result = lax.fori_loop(1, nmax, body_fun, [partial_sum, A_i])
    return result[0]


@partial(jit, static_argnums=4)
def hyp2f1_continuation(a, b, c, z, nmax=50):
    """
    This implementation is based off of Bühring's analytic continuation formula
    Equation 4, from J.L. Lopez and N.M. Temme, “New series expansions of the
    Gauss hypergeometric function”, Adv Comput Math 39, 349-365 (2013).
    https://link.springer.com/content/pdf/10.1007/s10444-012-9283-y.pdf

    The conditions required for this series to converge are:
    1) |z-1/2| > 1/2.
    2) b - a is not an integer
    """
    z = jnp.array(z, dtype=complex)

    # d0 = 1 and d_{-1} = 0
    prev_da = 1.0
    prev_db = 1.0
    prev_prev_da = 0.0
    prev_prev_db = 0.0

    # sum_1 corresponds to the summation on the top line of equation 4.21
    # sum_2 corresponds to the summation on the bottom line of equation 4.21
    # Gamma function prefactors are multiplied at the end
    sum_1 = 1.0 * jnp.ones_like(z)
    sum_2 = 1.0 * jnp.ones_like(z)

    # The branch cut for this computation of hyp2f1 is on Re(z) >= 1, Im(z) = 0
    # If z is on the branch cut, take the value above the branch cut
    z = jnp.where(jnp.imag(z) == 0.0, jnp.where(jnp.real(z) >= 1, z + 0.0000001j, z), z)

    # This is the (z - 0.5) ** (-n) term in the series; we start with n = 1
    z_factor = 1.0 / (z - 0.5)

    def body_fun(n, val):
        z_factor, prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2 = val

        # ------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the first line of equation 4
        # calculates d_n and the corresponding term in the sum
        d_na = (
            (n + a - 1.0)
            / (n * (n + a - b))
            * (
                ((a + b + 1.0) * 0.5 - c) * prev_da
                + 0.25 * (n + a - 2.0) * prev_prev_da
            )
        )
        sum_1 += d_na * z_factor

        # updates d_{n-2} and d_{n-1}
        prev_prev_da = prev_da
        prev_da = d_na

        # ------------------------------------------------------------------------------------------------------
        # This section of the function handles the summation on the second line of equation 4
        # calculates d_n and the corresponding term in the sum
        d_nb = (
            (n + b - 1.0)
            / (n * (n - a + b))
            * (
                ((a + b + 1.0) * 0.5 - c) * prev_db
                + 0.25 * (n + b - 2.0) * prev_prev_db
            )
        )
        sum_2 += d_nb * z_factor

        # updates d_{n-2} and d_{n-1}
        prev_prev_db = prev_db
        prev_db = d_nb

        # updates z_factor
        z_factor = z_factor / (z - 0.5)

        # -----------------------------------------------------------------------------------------------------
        return [z_factor, prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2]

    result = lax.fori_loop(
        1,
        nmax,
        body_fun,
        [z_factor, prev_prev_da, prev_prev_db, prev_da, prev_db, sum_1, sum_2],
    )

    # includes the gamma function prefactors in equation 4 to compute the final result
    final_result = gamma(c) * (
        result[5] * gamma(b - a) / gamma(b) / gamma(c - a) * (0.5 - z) ** (-a)
        + result[6] * gamma(a - b) / gamma(a) / gamma(c - b) * (0.5 - z) ** (-b)
    )
    return final_result


@partial(jit, static_argnums=4)
def hyp2f1_lopez_temme_8(a, b, c, z, nmax=75):
    """
    Equation 8 from J.L. Lopez and N.M. Temme, “New series expansions of the
    Gauss hypergeometric function”, Adv Comput Math 39, 349-365 (2013).
    https://link.springer.com/content/pdf/10.1007/s10444-012-9283-y.pdf"

    This series expansion converges whenever Re(z) < 1, and does not have any
    restrictions on a, b, or c. The downside is that this series converges
    slowly for |z| -> infty, so nmax needs to be higher for an accurate result.
    This series expansion coverges a bit faster than the standard series expansion
    when z is in the unit disk.
    """
    z = jnp.array(z, dtype=complex)

    # This prefactor includes the Pochhammer symbol, n factorial and (z/(z-2))**n in the sum.
    # This is the n=1 prefactor
    sum_prefactor = a * (z / (z - 2.0))

    # The values of hyp2f1 inside the sum are computed iteratively used one of Gauss's
    # contiguous relations, see Eq 15.5.11 https://dlmf.nist.gov/15.5#E11
    prev_prev_F = 1.0
    prev_F = 1.0 - 2.0 * b / c

    # The loop starts at n = 2
    init_sum = 1 + sum_prefactor * prev_F

    def body_fun(n, val):
        sum_prefactor, prev_prev_F, prev_F, sum = val

        sum_prefactor *= (a + n - 1.0) / n * (z / (z - 2.0))
        new_F = ((n - 1.0) * prev_prev_F - (2.0 * b - c) * prev_F) / (c + n - 1.0)
        sum += sum_prefactor * new_F

        prev_prev_F = prev_F
        prev_F = new_F

        return sum_prefactor, prev_prev_F, prev_F, sum

    return (1.0 - z / 2.0) ** (-a) * lax.fori_loop(
        2, nmax, body_fun, (sum_prefactor, prev_prev_F, prev_F, init_sum)
    )[3]


# TODO: Implement a version that can handle c - b - a = integer
#       This can be done using equations 15.3.10 - 15.3.12 in
#       Abramowitz and Stegun
@partial(jit, static_argnums=4)
def hyp2f1_near_one(a, b, c, z, nmax=50):
    """This implementation is based off of equation 15.3.6 in Abramowitz and Stegun.
    This transformation formula allows for a calculation of hyp2f1 for points near.

    z = 1 (where other iterative computation schemes converge slowly) by
    transforming z to 1 - z.

    The conditions required for this series to converge are:
    1) |1-z| < 1
    2) z is not purely real such that z >= 1
    3) c - a - b is not an integer
    """
    z = jnp.array(z, dtype=complex)

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
        * hyp2f1_series(a, b, a + b - c + 1.0, 1.0 - z, nmax)
    )
    term2 = (
        (1.0 - z) ** (c - a - b)
        * gamma(c)
        * gamma(a + b - c)
        / gamma(a)
        / gamma(b)
        * hyp2f1_series(c - a, c - b, c - a - b + 1.0, 1.0 - z, nmax)
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
    z = jnp.array(z, dtype=complex)
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
