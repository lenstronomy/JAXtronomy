"""Routines that manipulate convolution kernels."""

from jax import jit, lax, numpy as jnp


@jit
def estimate_amp(data, x_pos, y_pos, psf_kernel):
    """Estimates the amplitude of a point source located at x_pos, y_pos.

    :param data:
    :param x_pos:
    :param y_pos:
    :param psf_kernel:
    :return:
    """
    numPix_x, numPix_y = jnp.shape(data)
    x_int = (jnp.round(x_pos - 0.49999)).astype(int)
    y_int = (jnp.round(y_pos - 0.49999)).astype(int)
    # TODO: make amplitude estimate not susceptible to rounding effects on which pixels to chose to estimate the amplitude
    conditions = True
    conditions = jnp.where(x_int <= 2, False, conditions)
    conditions = jnp.where(x_int >= numPix_x - 2, False, conditions)
    conditions = jnp.where(y_int <= 2, False, conditions)
    conditions = jnp.where(y_int >= numPix_y - 2, False, conditions)

    # This is the same as np.sum(data[y_int - 2: y_int + 3, x_int - 2: x_int + 3])
    # But we have to do it this way since numpy slicing doesn't work unless the
    # start and end indices are known at compile time
    def body_fun(i, sum):
        row_index = y_int - 2 + i

        def body_fun2(j, sum):
            col_index = x_int - 2 + j
            sum += data.at[row_index, col_index].get()
            return sum

        return lax.fori_loop(0, 5, body_fun2, sum)

    sum = lax.fori_loop(0, 5, body_fun, 0)
    mean_image = jnp.maximum(sum, 0)

    num = len(psf_kernel)
    center = int((num - 0.5) / 2)
    mean_kernel = jnp.sum(psf_kernel[center - 2 : center + 3, center - 2 : center + 3])
    amp_estimated = jnp.where(conditions, mean_image / mean_kernel, 0)

    return amp_estimated
