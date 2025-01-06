__author__ = "ahuang314"

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
        :param method: string, options are BFGS, Nelder-Mead, Powell, CG, BFGS, Newton-CG,
            L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov
        :param logL_func: callable function, usually the logL function from the likelihood module
        :param args_mean: array of args, to be used as the mean of a normal distribution
            obtained by using Param.kwargs2args
        :param args_sigma: array of args, to be used as the std of a normal distribution
            obtained by using Param.kwargs2args
        :param args_lower: array of args, to be used as the lower bound of a normal distribution
            obtained by using Param.kwargs2args
        :param args_upper: array of args, to be used as the upper bound of a normal distribution
            obtained by using Param.kwargs2args
        :param maxiter: int, number of iterations to perform during minimization of the loss function
        """
        self.logL = logL_func
        self.parameter_history = []
        self.logL_history = []
        self.model_args = (args_mean, args_sigma, args_lower, args_upper)

        # Create an instance of the minimizer class. The negative logL is used as the loss function to be minimized
        # After each iteration, the logL and param history is updated using the callback function
        self.minimizer = jaxopt.ScipyMinimize(
            fun=self._loss,
            method=method,
            callback=self._update_logL_history,
            maxiter=maxiter,
        )

    def run_scipy(self, init_args):
        """Runs the minimizer. The initial parameters are assumed to be in constrained
        space. They are converted to the unconstrained space to be passed into the
        minimizer.

        :param init_args: array of initial parameters in the constrained space. Obtained
            by using Param.kwargs2args
        """

        # Reset the parameter and logL histories with initial values only
        self.parameter_history = [
            init_args,
        ]
        self.logL_history = [
            self.logL(init_args),
        ]

        # Convert contrained space parameters to unconstrained space
        init_args_unconstrained = unconstrain_fn(
            self._numpyro_model,
            model_args=self.model_args,
            model_kwargs={},
            params={"args": init_args},
        )["args"]

        # Run the minimizer
        # After each iteration, the logL and param history is updated using the callback function
        self.minimizer.run(init_args_unconstrained)

        # Return the result
        return self.parameter_history[-1], self.logL_history[-1]

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
            model_args=self.model_args,
            model_kwargs={},
            params={"args": args_unconstrained},
        )["args"]

        return -self.logL(args_constrained)

    def _update_logL_history(self, current_parameter):
        """This function is automatically called at the end of each iteration during the
        minimization process, with the current parameter as an argument. This parameter
        is converted back into constrained space and the logL is calculated. The
        parameter and logL histories are updated.

        :param current_parameter: current args in unconstrained space
        """

        # Convert uncontrained space parameters to constrained space and update parameter history
        args_constrained = constrain_fn(
            self._numpyro_model,
            model_args=self.model_args,
            model_kwargs={},
            params={"args": current_parameter},
        )["args"]
        self.parameter_history.append(args_constrained)

        # Update logL history
        self.logL_history.append(self.logL(args_constrained))

    @staticmethod
    def _numpyro_model(args_mean, args_sigma, args_lower, args_upper):
        """This numpyro model is required to convert parameters between the constrained
        and unconstrained space. This function does not actually do anything.

        :param args_mean: array of args, to be used as the mean of a normal distribution
            obtained by using Param.kwargs2args
        :param args_sigma: array of args, to be used as the std of a normal distribution
            obtained by using Param.kwargs2args
        :param args_lower: array of args, to be used as the lower bound of a normal
            distribution obtained by using Param.kwargs2args
        :param args_upper: array of args, to be used as the upper bound of a normal
            distribution obtained by using Param.kwargs2args
        """
        numpyro.sample(
            "args",
            dist.TruncatedNormal(
                args_mean, args_sigma, low=args_lower, high=args_upper
            ),
        )
