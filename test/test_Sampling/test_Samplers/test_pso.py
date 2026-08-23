__author__ = "ahuang314"

from functools import partial
from jax import jit, lax, numpy as jnp, vmap
import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from jaxtronomy.Sampling.Samplers.pso_jit import ParticleSwarmOptimizerJIT


@jit
@vmap
def logL(x):
    # Minimum at x = 0.6
    return -jnp.sum((x - 0.6) ** 2)


class TestPSO(object):
    """Tests the PSO class with a simple logL function."""

    def setup_method(self):
        args_lower = np.array([0.0])
        args_upper = np.array([10.1])
        args = (args_lower, args_upper)
        self.pso = ParticleSwarmOptimizer(logL, *args, particle_count=10)

    def test_run(self):
        # Tests to see if the PSO gets close to the true answer
        final_result, _ = self.pso.optimize(max_iter=100)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)


class TestPSOJIT(object):
    """Tests the PSO JIT class with a simple logL function."""

    def setup_method(self):
        args_lower = np.array([0.0])
        args_upper = np.array([10.1])
        args = (args_lower, args_upper)
        self.pso = ParticleSwarmOptimizerJIT(logL, *args, particle_count=10)

    def test_run(self):
        # Tests to see if the PSO gets close to the true answer
        final_result, _ = self.pso.optimize(max_iter=100)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)

        final_result, _ = self.pso.optimize(max_iter=100, early_stop_tolerance=1e-5)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)


if __name__ == "__main__":
    pytest.main()
