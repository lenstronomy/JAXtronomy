import pytest
from scipy.special import hyp2f1 as hyp2f1_ref
import numpy.testing as npt

from jax import config, numpy as jnp

config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy

from jaxtronomy.Util.hyp2f1_util import (
    hyp2f1,
    hyp2f1_series,
    hyp2f1_near_one,
    hyp2f1_continuation,
    hyp2f1_lopez_temme_8,
)


def test_hyp2f1_series():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points inside the circle |z| = 1
    x = jnp.array([0.3, 0.5, -0.6, 0.2])
    y = jnp.array([0.5, 0.1, 0.1, -0.6])
    z = x + y * 1j
    result = hyp2f1_series(a, b, c, z, nmax=75)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-12, rtol=1e-12)

    # Points very close to the boundary of the circle are less accurate
    # Supports list inputs
    z = [0.99 + 0.001j]
    result = hyp2f1_series(a, b, c, z, nmax=200)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-4, rtol=1e-4)

    # Also supports scalar inputs
    z = -0.99 + 0.001j
    result = hyp2f1_series(a, b, c, z, nmax=200)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-4, rtol=1e-4)


def test_hyp2f1_near_one():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points inside the circle |1-z| = 1
    x = jnp.array([0.8, 1.3, 1.1, 0.6])
    y = jnp.array([0.3, 0.1, 0.4, -0.2])
    z = x + y * 1j
    result = hyp2f1_near_one(a, b, c, z, nmax=75)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)

    # Points very close to the boundary of the circle are less accurate
    # Also supports list inputs
    z = [0.05 + 0.001j]
    result = hyp2f1_near_one(a, b, c, z, nmax=200)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-3, rtol=1e-3)

    # Tests to see if the value of z above the branch cut is taken
    # Also supports scalar inputs
    z = 1.1 + 0.0j
    result = hyp2f1_near_one(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-7, rtol=1e-7)


def test_hyp2f1_continuation():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points outside the circle |z-1/2| = 1/2
    x = jnp.array([-0.8, 1.3, 1.1, -1.6])
    y = jnp.array([0.3, 0.1, -0.4, -2.2])
    z = x + y * 1j
    result = hyp2f1_continuation(a, b, c, z, nmax=100)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)

    # Points very close to the boundary of the circle are less accurate
    # Also supports list inputs
    z = [1.03 + 0.001j]
    result = hyp2f1_continuation(a, b, c, z, nmax=200)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-4, rtol=1e-4)

    # Tests to see if the value above the branch cut is taken
    # Also supports scalar inputs
    z = 3.0 + 0.0j
    result = hyp2f1_continuation(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)


def test_hyp2f1_lopez_temme():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points such that Re(z) < 1
    x = jnp.array([0.3, 0.5, -0.6, 0.2, -3.2])
    y = jnp.array([0.5, 2.1, 0.1, -1.6, 1.1])
    z = x + y * 1j
    result = hyp2f1_lopez_temme_8(a, b, c, z, nmax=75)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)

    # Points very close to Re(z) are less accurate
    # Supports list inputs
    z = [0.99 + 0.001j]
    result = hyp2f1_lopez_temme_8(a, b, c, z, nmax=200)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-5, rtol=1e-5)

    # Also supports scalar inputs
    z = -10.99 + 0.001j
    result = hyp2f1_lopez_temme_8(a, b, c, z, nmax=75)
    result_ref = hyp2f1_ref(a, b, c, z)
    npt.assert_allclose(result, result_ref, atol=1e-8, rtol=1e-8)


def test_hyp2f1():
    a, b, c = 0.3, 0.7, 2.2
    # Tests points in every region
    x = jnp.array([-0.8, 1.3, 0.6, -1.6])
    y = jnp.array([0.3, 0.1, 0.6, -2.2])
    z = x + y * 1j
    result = hyp2f1(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(result, result_ref), "hyp2f1 result does not match scipy result"
    assert jnp.shape(result) == jnp.shape(
        z
    ), "shape of output does not match shape of input"

    # Points that are problematic for specific cases should not be
    # problematic here in the general case
    z = jnp.array([1.01 + 0.0001j, 0.05 + 0.001j, 0.99 + 0.001j])
    result = hyp2f1(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(result, result_ref), "hyp2f1 result does not match scipy result"
    assert jnp.shape(result) == jnp.shape(
        z
    ), "shape of output does not match shape of input"

    # Supports list inputs
    z = [0.605 + 0.65j, 0.609 + 0.65j, 0.6072 + 0.0651j]
    result = hyp2f1(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(result, result_ref), "hyp2f1 result does not match scipy result"
    assert jnp.shape(result) == jnp.shape(
        z
    ), "shape of output does not match shape of input"

    # Tests to see if the value above the branch cut is taken
    # Also supports scalar inputs
    z = 3.0 + 0.0j
    result = hyp2f1(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(result, result_ref), "hyp2f1 result does not match scipy result"
    assert jnp.shape(result) == jnp.shape(
        z
    ), "shape of output does not match shape of input"
