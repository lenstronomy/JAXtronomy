__author__ = "ahuang314"

import numpy as np
import numpy.testing as npt

# --------------------------------------------------------------------------
# Remove these lines when numpyro gets updated for JAX v0.7.0 compatibility
import jax.experimental.pjit
from jax.extend.core.primitives import jit_p

jax.experimental.pjit.pjit_p = jit_p
# --------------------------------------------------------------------------
from numpyro.infer.util import unconstrain_fn
import pytest

from jaxtronomy.Sampling.Samplers.optax import OptaxMinimizer


class TestJaxoptMinimizer(object):
    """Tests two different logL functions."""

    def _logL(self, x):
        # Minimum at x = 0.6
        return -np.sum((x - 0.6) ** 2)

    def _logL2(self, x):
        # Minimum at x = 0.25
        return -np.sum((4 * x - 1.0) ** 4)

    def setup_method(self):
        args_mean = np.array([0.7])
        args_sigma = np.array([0.2])
        args_lower = np.array([0.0])
        args_upper = np.array([0.9])
        args = (args_mean, args_sigma, args_lower, args_upper)
        self.minimizer = OptaxMinimizer(self._logL, *args, maxiter=200)
        self.minimizer2 = OptaxMinimizer(self._logL2, *args, maxiter=500)
        self.args_mean = args_mean

    def test_run(self):
        # Tests to see if the minimizer gets close to the analytical answer
        final_result = self.minimizer.run(num_chains=3, rng_int=0, tol=1e-14)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=6)

        final_result = self.minimizer2.run(num_chains=3, rng_int=0, tol=1e-14)
        npt.assert_array_almost_equal(final_result, [0.25], decimal=2)

    def test_loss(self):
        args_constrained = np.array([0.7])
        args_unconstrained = unconstrain_fn(
            self.minimizer._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": args_constrained},
        )["args"]
        loss = self.minimizer._loss(args_unconstrained)
        npt.assert_almost_equal(loss, 0.01, decimal=8)

        args_constrained = np.array([0.275])
        args_unconstrained = unconstrain_fn(
            self.minimizer2._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": args_constrained},
        )["args"]
        loss = self.minimizer2._loss(args_unconstrained)
        npt.assert_almost_equal(loss, 0.0001, decimal=8)


if __name__ == "__main__":
    pytest.main()
