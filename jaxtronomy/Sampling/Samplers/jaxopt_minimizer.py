__author__ = "ahuang314"

import jax
import jaxopt
import numpyro, numpyro.distributions as dist
from numpyro.infer.util import constrain_fn, unconstrain_fn


__all__ = ["Jaxopt"]


class JaxoptMinimizer:
    """This class is a wrapper for the jaxopt.ScipyMinimize class.

    This minimizer only works in an unconstrained parameter space (i.e. no hard bounds).
    To get around this, we use numpyro to map from the constrained space to the
    unconstrained space, run the minimizer, then convert the result back into the
    constrained space
    """

    def __init__(
        self,
        method,
        logL_func,
        args_mean,
        args_sigma,
        args_lower,
        args_upper,
        maxiter=500,
    ):
        """
        :param method: string, options are BFGS and TNC. Other options such as Nelder-Mead, Powell, CG, Newton-CG,
            L-BFGS-B, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov have not been tested.
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

        # Create an instance of the minimizer class. The negative logL is used as the loss function to be minimized
        # After each iteration, the logL and param history is updated using the callback function
        self.minimizer = jaxopt.ScipyMinimize(
            fun=self._loss,
            method=method,
            callback=self._update_logL_history,
            maxiter=maxiter,
        )
        self.single_chain_param_history = []
        self.single_chain_logL_history = []

    def run(self, num_chains, rng_int=0, tolerance=0):
        """Runs the minimizer for a certain number of chains.

        :param num_chains: int, number of chains to run the minimizer on. Initial
            parameters for each chain are sampled from the user-provided distribution.
            Running more chains takes more time but can help avoid local minima.
        :param rng_int: int, used to seed the JAX RNG
        :param tolerance: float, only relevant when num_chains > 1. If |logL| <
            tolerance at the end of a chain, the rest of the chains are not run.
        :return: Index of best chain along with the param and logL histories of all
            chains
        """

        # Saves the param and logL histories for all chains
        multi_chain_param_history = []
        multi_chain_logL_history = []

        array_of_init_params = self._draw_init_params(num_chains, rng_int)

        for i in range(0, num_chains):
            print(
                f"Running chain {i+1} with initial parameters: \n",
                array_of_init_params[i],
            )
            print("Initial log likelihood: ", self.logL(array_of_init_params[i]))
            self.run_single_chain(array_of_init_params[i])
            multi_chain_logL_history.append(self.single_chain_logL_history)
            multi_chain_param_history.append(self.single_chain_param_history)
            new_logL = self.single_chain_logL_history[-1]
            print(f"Final log likelihood: ", new_logL)
            print(
                "---------------------------------------------------------------------"
            )

            if i == 0 or new_logL > best_logL:
                best_logL = new_logL
                best_chain_index = i

            if -best_logL < tolerance:
                print(
                    "Tolerance criteria |logL| < tolerance has been met. Remaining chains will be skipped."
                )
                print(
                    "---------------------------------------------------------------------"
                )
                break

        return (
            best_chain_index,
            multi_chain_param_history,
            multi_chain_logL_history,
        )

    def run_single_chain(self, init_args):
        """Runs the minimizer. The initial parameters are assumed to be in constrained
        space. They are converted to the unconstrained space to be passed into the
        minimizer.

        :param init_args: array of initial parameters in the constrained space. Obtained
            by using Param.kwargs2args
        :return: None, just updates class variables containg the single chain param and
            logL histories
        """

        # Reset the parameter and logL histories with initial values only
        self.single_chain_param_history = [
            init_args,
        ]
        self.single_chain_logL_history = [
            self.logL(init_args),
        ]

        # Convert contrained space parameters to unconstrained space
        init_args_unconstrained = unconstrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": init_args},
        )["args"]

        # Run the minimizer
        # After each iteration, the single chain logL and param histories are updated using the callback function
        self.minimizer.run(init_args_unconstrained)

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

    def _update_logL_history(self, current_parameters):
        """This function is automatically called at the end of each iteration during the
        minimization process, with the current parameters as an argument. The parameters
        are converted back into constrained space and the logL is calculated. The
        parameter and logL histories are updated.

        :param current_parameters: current args in unconstrained space
        """

        # Convert uncontrained space parameters to constrained space and update parameter history
        args_constrained = constrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params={"args": current_parameters},
        )["args"]
        self.single_chain_param_history.append(args_constrained)

        # Update logL history
        self.single_chain_logL_history.append(self.logL(args_constrained))

    def _numpyro_model(self):
        """This numpyro model is required to convert parameters between the constrained
        and unconstrained space.

        This function does not actually do anything.
        """

        numpyro.sample("args", self.dist)
