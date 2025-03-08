import lenstronomy.Util.kernel_util as kernel_util_ref
import jaxtronomy.Util.kernel_util as kernel_util

from jax import config
import numpy as np, numpy.testing as npt
import pytest

config.update("jax_enable_x64", True)


def test_estimate_amp():
    data = np.linspace(1, 50, 50 * 50).reshape((50, 50))
    x_pos = 3.45
    y_pos = 2.98
    psf_kernel = np.ones((5, 5))
    psf_kernel[1:5, 1:5] = 3

    estimated_amp = kernel_util.estimate_amp(data, x_pos, y_pos, psf_kernel)
    estimated_amp_ref = kernel_util_ref.estimate_amp(data, x_pos, y_pos, psf_kernel)
    npt.assert_allclose(estimated_amp, estimated_amp_ref, atol=1e-8, rtol=1e-8)

    x_pos = 13.45
    y_pos = 12.78
    estimated_amp = kernel_util.estimate_amp(data, x_pos, y_pos, psf_kernel)
    estimated_amp_ref = kernel_util_ref.estimate_amp(data, x_pos, y_pos, psf_kernel)
    npt.assert_allclose(estimated_amp, estimated_amp_ref, atol=1e-8, rtol=1e-8)

    x_pos = 33.45
    y_pos = 32.78
    estimated_amp = kernel_util.estimate_amp(data, x_pos, y_pos, psf_kernel)
    estimated_amp_ref = kernel_util_ref.estimate_amp(data, x_pos, y_pos, psf_kernel)
    npt.assert_allclose(estimated_amp, estimated_amp_ref, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    pytest.main()
