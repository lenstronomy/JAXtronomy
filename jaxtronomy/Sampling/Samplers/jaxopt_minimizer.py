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
        num_init_samples=3,
        maxiter=500,
        tolerance=None,
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
        :param num_init_samples: int, number of initial samples to run the minimizer on.
            Running more initial samples takes more time but can help avoid local minima.
        :param maxiter: int, maximum number of iterations to perform during minimization of the loss function
        :param tolerance: float or None, only relevant when num_init_samples > 1.
            If |logL| < tolerance at the end of a sample, the rest of the samples are not run.
        """
        self.logL = logL_func
        self.dist = dist.TruncatedNormal(
            args_mean, args_sigma, low=args_lower, high=args_upper
        )
        self.num_init_samples = num_init_samples
        if tolerance is None:
            tolerance = 0
        self.tolerance = tolerance

        # Create an instance of the minimizer class. The negative logL is used as the loss function to be minimized
        # After each iteration, the logL and param history is updated using the callback function
        self.minimizer = jaxopt.ScipyMinimize(
            fun=self._loss,
            method=method,
            callback=self._update_logL_history,
            maxiter=maxiter,
        )

        self.single_sample_parameter_history = []
        self.single_sample_logL_history = []
        self.multi_sample_parameter_history = []
        self.multi_sample_logL_history = []

    def run(self, rng_int):
        self.multi_sample_parameter_history = []
        self.multi_sample_logL_history = []

        array_of_init_samples = self._draw_init_samples(rng_int)

        print("Running sample 1 with initial parameters: \n", array_of_init_samples[0])
        print("Initial log likelihood: ", self.logL(array_of_init_samples[0]))
        self.run_single_sample(array_of_init_samples[0])
        best_logL = self.single_sample_logL_history[-1]
        print("Final log likelihood: ", best_logL)
        print("---------------------------------------------------------------------")
        best_sample_index = 0

        for i in range(1, self.num_init_samples):

            if -best_logL < self.tolerance:
                print(
                    "Tolerance criteria |logL| < tolerance has been met. Stopping samples."
                )
                print(
                    "---------------------------------------------------------------------"
                )
                break

            print(
                f"Running sample {i+1} with initial parameters: \n",
                array_of_init_samples[i],
            )
            print("Initial log likelihood: ", self.logL(array_of_init_samples[i]))
            self.run_single_sample(array_of_init_samples[i])
            new_logL = self.single_sample_logL_history[-1]
            print(f"Final log likelihood: ", new_logL)
            print(
                "---------------------------------------------------------------------"
            )

            if new_logL > best_logL:
                best_logL = new_logL
                best_sample_index = i

        return best_sample_index

    def run_single_sample(self, init_args):
        """Runs the minimizer. The initial parameters are assumed to be in constrained
        space. They are converted to the unconstrained space to be passed into the
        minimizer.

        :param init_args: array of initial parameters in the constrained space. Obtained
            by using Param.kwargs2args
        """

        # Reset the parameter and logL histories with initial values only
        self.single_sample_parameter_history = [
            init_args,
        ]
        self.single_sample_logL_history = [
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
        # After each iteration, the single sample logL and param histories are updated using the callback function
        self.minimizer.run(init_args_unconstrained)
        self.multi_sample_logL_history.append(self.single_sample_logL_history)
        self.multi_sample_parameter_history.append(self.single_sample_parameter_history)

    def _draw_init_samples(self, rng_int):
        rng = jax.random.split(jax.random.PRNGKey(rng_int), 1)[0]
        array_of_init_samples = numpyro.sample(
            "args", self.dist, rng_key=rng, sample_shape=(self.num_init_samples,)
        )
        return array_of_init_samples

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
        self.single_sample_parameter_history.append(args_constrained)

        # Update logL history
        self.single_sample_logL_history.append(self.logL(args_constrained))

    def _numpyro_model(self):
        """This numpyro model is required to convert parameters between the constrained
        and unconstrained space.

        This function does not actually do anything.
        """

        numpyro.sample("args", self.dist)
