__author__ = "ahuang314"

import jax

jax.config.update("jax_enable_x64", True)
from jax import jit
from functools import partial
import optax
import numpyro, numpyro.distributions as dist
from numpyro.infer.util import constrain_fn, unconstrain_fn


class OptaxMinimizer:
    """Gradient descent using Optax's L-BFGS method.
    """
    def __init__(
        self,
        logL_func,
        args_mean,
        args_sigma,
        args_lower,
        args_upper,
        maxiter=500,
    ):
        """
        :param logL_func: callable function, usually the logL function from the likelihood module
        :param args_mean: array of args, to be used as the mean of a normal distribution
            obtained by using Param.kwargs2args
        :param args_sigma: array of args, to be used as the std of a normal distribution
            obtained by using Param.kwargs2args
        :param args_lower: array of args, to be used as the lower bound of a normal distribution
            obtained by using Param.kwargs2args
        :param args_upper: array of args, to be used as the upper bound of a normal distribution
            obtained by using Param.kwargs2args
        :param maxiter: int, maximum number of iterations to perform during minimization of the loss function
        """
        self.logL = logL_func
        self.dist = dist.TruncatedNormal(
            args_mean, args_sigma, low=args_lower, high=args_upper
        )
        self.opt = optax.lbfgs()
        self.maxiter = maxiter
        self.value_and_grad_fun = optax.value_and_grad_from_state(self._loss)

    def run(self, num_chains, tol, rng_int=0):
        """Runs the gradient descent.

        :param num_chains: int, number of chains to run
        :param tol: float, when np.abs(logL) < tol, the gradient descent for that chain is stopped
        :param rng_int: int, used to draw initial parameters from the prior distribution
        """

        init_param_list = self._draw_init_params(num_chains=num_chains, rng_int=rng_int)
        init_param_list = unconstrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": init_param_list},
        )["args"]

        for i in range(len(init_param_list)):
            print(f"Running chain {i+1}")
            final_params, num_iter = self.run_single(
                init_params=init_param_list[i], tol=tol
            )
            final_logL = self._loss(final_params)
            print(f"{num_iter} iterations performed.")
            print("Final logL:", -final_logL)
            if i == 0 or final_logL < best_logL:
                best_logL = final_logL
                best_params = final_params

        best_params = constrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": best_params},
        )["args"]

        return best_params

    @partial(jit, static_argnums=0)
    def run_single(self, init_params, tol):
        """Runs the gradient descent for a single chain.

        :param init_params: 1d array of floats, initial parameters for the loss function
        :param tol: float, when np.abs(logL) < tol, the gradient descent is stopped
        """
        def step(carry):
            params, state = carry
            value, grad = self.value_and_grad_fun(params, state=state)
            updates, state = self.opt.update(
                grad, state, params, value=value, grad=grad, value_fn=self._loss
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = optax.tree.get(state, "count")
            value = optax.tree.get(state, "value")
            return (iter_num == 0) | ((iter_num < self.maxiter) & (value >= tol))

        init_carry = (init_params, self.opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )
        return final_params, optax.tree.get(final_state, "count")

    def _draw_init_params(self, num_chains, rng_int):
        """Draws initial parameters to be passed to the minimizer.

        :param num_chains: int, number of chains to run the minimizer on. Initial
            parameters for each chain are sampled from the user-provided distribution.
            Running more chains takes more time but can help avoid local minima.
        :param rng_int: int, used to seed the JAX RNG
        """
        rng = jax.random.split(jax.random.PRNGKey(rng_int), 1)[0]
        array_of_init_params = numpyro.sample(
            "args", self.dist, rng_key=rng, sample_shape=(num_chains,)
        )
        return array_of_init_params

    def _loss(self, args_unconstrained):
        """Since LikelihoodModule.logL uses parameters in the constrained space while
        the minimizer uses parameters in the unconstrained space, the args need to be
        converted from unconstrained to constrained before calculating logL. Then the
        negative logL is used as the loss function to be minimized.

        :param args_unconstrained: array of args in the unconstrained space
        """

        # Convert uncontrained space parameters to constrained space
        args_constrained = constrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": args_unconstrained},
        )["args"]

        return -self.logL(args_constrained)

    def _numpyro_model(self):
        """This numpyro model is required to convert parameters between the constrained
        and unconstrained space.

        This function does not actually do anything.
        """

        numpyro.sample("args", self.dist)
