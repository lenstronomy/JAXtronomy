__author__ = "ahuang314"

import numpy as np
import numpy.testing as npt
from numpyro.infer.util import unconstrain_fn
import pytest

from jaxtronomy.Sampling.Samplers.jaxopt_minimizer import JaxoptMinimizer


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
        self.minimizer = JaxoptMinimizer("BFGS", self._logL, *args, maxiter=200)
        self.minimizer2 = JaxoptMinimizer("BFGS", self._logL2, *args, maxiter=500)
        self.args_mean = args_mean

    def test_run(self):
        # Tests to see if the minimizer gets close to the analytical answer
        (
            best_chain_index,
            multi_chain_param_history,
            multi_chain_logL_history,
        ) = self.minimizer.run(num_chains=3, rng_int=0, tolerance=1e-14)
        final_result = multi_chain_param_history[best_chain_index][-1]
        final_logL = multi_chain_logL_history[best_chain_index][-1]
        npt.assert_array_almost_equal(final_result, [0.6], decimal=6)
        npt.assert_almost_equal(final_logL, 0.0, decimal=8)

        (
            best_chain_index,
            multi_chain_param_history,
            multi_chain_logL_history,
        ) = self.minimizer2.run(num_chains=3, rng_int=0)
        final_result = multi_chain_param_history[best_chain_index][-1]
        final_logL = multi_chain_logL_history[best_chain_index][-1]
        npt.assert_array_almost_equal(final_result, [0.25], decimal=2)
        npt.assert_almost_equal(final_logL, 0.0, decimal=7)

    def test_run_single_chain(self):
        self.minimizer.run_single_chain(self.args_mean)
        npt.assert_array_equal(
            self.args_mean, self.minimizer.single_chain_param_history[0]
        )

        final_result = self.minimizer.single_chain_param_history[-1]
        final_logL = self.minimizer.single_chain_logL_history[-1]
        npt.assert_array_almost_equal(final_result, [0.6], decimal=6)
        npt.assert_almost_equal(final_logL, 0.0, decimal=8)

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

    def test_update_logL_history(self):
        current_parameter = np.array([0.5])
        args_unconstrained = unconstrain_fn(
            self.minimizer._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": current_parameter},
        )["args"]
        self.minimizer._update_logL_history(args_unconstrained)
        npt.assert_almost_equal(
            self.minimizer.single_chain_logL_history[-1], -0.01, decimal=8
        )
        npt.assert_array_equal(
            self.minimizer.single_chain_param_history[-1], current_parameter
        )

        current_parameter = np.array([0.225])
        args_unconstrained = unconstrain_fn(
            self.minimizer2._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": current_parameter},
        )["args"]
        self.minimizer2._update_logL_history(args_unconstrained)
        npt.assert_almost_equal(
            self.minimizer2.single_chain_logL_history[-1], -0.0001, decimal=8
        )
        npt.assert_array_almost_equal(
            self.minimizer2.single_chain_param_history[-1],
            current_parameter,
            decimal=8,
        )


if __name__ == "__main__":
    pytest.main()
