import pytest
from scipy.special import hyp2f1 as hyp2f1_ref

import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp
from jaxtronomy.Util.hyp2f1_util import (
    hyp2f1,
    hyp2f1_series,
    hyp2f1_near_one,
    hyp2f1_continuation,
)


def test_hyp2f1_series():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points inside the circle |z| = 1
    x = jnp.array([0.3, 0.5, -0.6, 0.2])
    y = jnp.array([0.5, 0.1, 0.1, -0.6])
    z = x + y * 1j
    result = hyp2f1_series(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref
    ), "hyp2f1_series result does not match scipy result"

    # Points very close to the boundary of the circle are less accurate
    # Supports list inputs
    z = [0.99 + 0.001j]
    result = hyp2f1_series(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref, atol=1e-4
    ), "hyp2f1_series result does not match scipy result"

    # Also supports scalar inputs
    z = -0.99 + 0.001j
    result = hyp2f1_series(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref, atol=1e-4
    ), "hyp2f1_series result does not match scipy result"
    assert result.ndim == 0


def test_hyp2f1_near_one():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points inside the circle |1-z| = 1
    x = jnp.array([0.8, 1.3, 1.1, 0.6])
    y = jnp.array([0.3, 0.1, 0.4, -0.2])
    z = x + y * 1j
    result = hyp2f1_near_one(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref
    ), "hyp2f1_near_one result does not match scipy result"

    # Points very close to the boundary of the circle are less accurate
    # Also supports list inputs
    z = [0.05 + 0.001j]
    result = hyp2f1_near_one(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref, atol=1e-2
    ), "hyp2f1_near_one result does not match scipy result"

    # Tests to see if the value of z above the branch cut is taken
    # Also supports scalar inputs
    z = 1.1 + 0.0j
    result = hyp2f1_near_one(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref
    ), "hyp2f1_near_one result does not match scipy result"


def test_hyp2f1_continuation():
    a, b, c = 0.3, 0.7, 2.2
    # Only tests points outside the circle |z-1/2| = 1/2
    x = jnp.array([-0.8, 1.3, 1.1, -1.6])
    y = jnp.array([0.3, 0.1, -0.4, -2.2])
    z = x + y * 1j
    result = hyp2f1_continuation(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref
    ), "hyp2f1_continuation result does not match scipy result"

    # Points very close to the boundary of the circle are less accurate
    # Also supports list inputs
    z = [1.03 + 0.001j]
    result = hyp2f1_continuation(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref, atol=1e-4
    ), "hyp2f1_continuation result does not match scipy result"

    # Tests to see if the value above the branch cut is taken
    # Also supports scalar inputs
    z = 3.0 + 0.0j
    result = hyp2f1_continuation(a, b, c, z)
    result_ref = hyp2f1_ref(a, b, c, z)
    assert jnp.allclose(
        result, result_ref
    ), "hyp2f1_continuation result does not match scipy result"


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
