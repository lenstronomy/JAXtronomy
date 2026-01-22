import time

import numpy as np
from jaxtronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Sampling.sampler import Sampler as Sampler_lenstronomy
from lenstronomy.Util import sampling_util

from functools import partial
import jax
from jax import lax

__all__ = ["Sampler"]


class Sampler(Sampler_lenstronomy):
    """Inherits samplers from lenstronomy, but modifies the PSO and mcmc to parallelize
    computations across hardware devices."""

    def pso(
        self,
        n_particles,
        n_iterations,
        lower_start=None,
        upper_start=None,
        threadCount=None,
        init_pos=None,
        mpi=False,
        print_key="PSO",
        verbose=True,
    ):
        """Return the best fit for the lens model on catalogue basis with particle swarm
        optimizer.

        :param n_particles: number of particles in the sampling process
        :param n_iterations: number of iterations of the swarm
        :param lower_start: numpy array, lower end parameter of the values of the
            starting particles
        :param upper_start: numpy array, upper end parameter of the values of the
            starting particles
        :param threadCount: number of threads in the computation (only applied if
            mpi=False)
        :param init_pos: numpy array, position of the initial best guess model
        :param mpi: bool, if True, makes instance of MPIPool to allow for MPI execution
        :param print_key: string, prints the process name in the progress bar (optional)
        :param verbose: suppress or turn on print statements
        :return: kwargs_result (of best fit), [lnlikelihood of samples, positions of
            samples, velocity of samples])
        """

        if threadCount is not None:
            raise ValueError(
                "PSO threadCount must be None. In JAXtronomy, parallelization across CPU cores is done automatically."
            )
        if mpi:
            raise ValueError("mpi must be False in JAXtronomy")

        if lower_start is None or upper_start is None:
            lower_start, upper_start = np.array(self.lower_limit), np.array(
                self.upper_limit
            )
            print("PSO initialises its particles with default values")
        else:
            lower_start = np.maximum(lower_start, self.lower_limit)
            upper_start = np.minimum(upper_start, self.upper_limit)

        backend = jax.default_backend()
        if backend == "cpu":
            num_devices = jax.device_count()
            if n_particles % num_devices != 0:
                raise ValueError(
                    f"PSO particle count {n_particles} must be divisible by the number of CPU devices for parallelization. "
                    f"There are {num_devices} cpu devices currently recognized by JAX."
                )
        logL_func = prepare_logL_func(backend=backend, logL_func=self.chain.logL)

        pso = ParticleSwarmOptimizer(logL_func, lower_start, upper_start, n_particles)

        if init_pos is None:
            init_pos = (upper_start - lower_start) / 2 + lower_start

        pso.set_global_best(init_pos, [0] * len(init_pos), self.chain.logL(init_pos))

        time_start = time.time()

        result, [log_likelihood_list, pos_list, vel_list] = pso.optimize(
            n_iterations, verbose=verbose
        )

        kwargs_return = self.chain.param.args2kwargs(result)
        if verbose:
            print(
                pso.global_best.fitness
                * 2
                / (max(self.chain.effective_num_data_points(**kwargs_return), 1)),
                "reduced X^2 of best position",
            )
            print(pso.global_best.fitness, "log likelihood")
            self._print_result(result=result)
            time_end = time.time()
            print(time_end - time_start, "time used for ", print_key)
            print("===================")
        return result, [log_likelihood_list, pos_list, vel_list]

    def mcmc_emcee(
        self,
        n_walkers,
        n_run,
        n_burn,
        mean_start,
        sigma_start,
        mpi=False,
        progress=False,
        threadCount=None,
        initpos=None,
        backend_filename=None,
        start_from_backend=False,
    ):
        """Run MCMC with emcee. For details, please have a look at the documentation of
        the emcee packager.

        :param n_walkers: number of walkers in the emcee process
        :type n_walkers: integer
        :param n_run: number of sampling (after burn-in) of the emcee
        :type n_run: integer
        :param n_burn: number of burn-in iterations (those will not be saved in the output sample)
        :type n_burn: integer
        :param mean_start: mean of the parameter position of the initialising sample
        :type mean_start: numpy array of length the number of parameters
        :param sigma_start: spread of the parameter values (uncorrelated in each dimension) of the initialising sample
        :type sigma_start: numpy array of length the number of parameters
        :param mpi: if True, initializes an MPIPool to allow for MPI execution of the sampler
        :type mpi: bool
        :param progress: if True, prints the progress bar
        :type progress: bool
        :param threadCount: number of threats in multi-processing (not applicable for MPI)
        :type threadCount: integer
        :param initpos: initial walker position to start sampling (optional)
        :type initpos: numpy array of size num param x num walkser
        :param backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
        :type backend_filename: string
        :param start_from_backend: if True, start from the state saved in `backup_filename`.
         Otherwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
        :type start_from_backend: bool
        :return: samples, ln likelihood value of samples
        :rtype: numpy 2d array, numpy 1d array
        """
        if mpi:
            raise ValueError("mpi must be False in JAXtronomy")
        if threadCount is not None:
            raise ValueError(
                "MCMC threadCount must be set to None in JAXtronomy, since parallelization is done automatically."
            )
        if start_from_backend:
            raise ValueError("start_from_backend must be False in JAXtronomy")
        if backend_filename is not None:
            raise ValueError("backend_filename not supported in JAXtronomy")

        import emcee

        num_param, _ = self.chain.param.num_param()
        if initpos is None:
            initpos = sampling_util.sample_ball_truncated(
                mean_start,
                sigma_start,
                self.lower_limit,
                self.upper_limit,
                size=n_walkers,
            )

        n_run_eff = n_burn + n_run

        time_start = time.time()

        backend = jax.default_backend()
        if backend == "cpu":
            num_devices = jax.device_count()
            if n_walkers % (2 * num_devices) != 0:
                raise ValueError(
                    f"Number of MCMC walkers {n_walkers} must be divisible by two times the number of CPU devices for parallelization. "
                    f"There are {num_devices} cpu devices currently recognized by JAX."
                )
        logL_func = prepare_logL_func(backend=backend, logL_func=self.chain.logL)

        sampler = emcee.EnsembleSampler(n_walkers, num_param, logL_func, vectorize=True)

        sampler.run_mcmc(initpos, n_run_eff, progress=progress)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=1)
        print("Computing the MCMC...")
        print("Number of walkers = ", n_walkers)
        print("Burn-in iterations: ", n_burn)
        print("Sampling iterations (in current run):", n_run_eff)
        time_end = time.time()
        print(time_end - time_start, "time taken for MCMC sampling")
        return flat_samples, dist


def prepare_logL_func(backend, logL_func):
    """Parallelizes the logL function for CPU backend, and vectorizes the
    logL function for GPU backend.

    :param backend: string, must be 'cpu' or 'gpu'.
    :param logL_func: callable function that takes a position vector and returns a log likelihood.

    :returns: a callable function that takes a set of position vectors and returns a set of log likelihoods.
    """
    if backend == "cpu":

        mapped_func = partial(lax.map, logL_func)
        pmapped_func = jax.pmap(mapped_func, devices=jax.devices())
        num_devices = jax.device_count()

        def new_logL_func(args):
            args = np.array(args)
            old_shape = args.shape
            new_shape = (num_devices, int(old_shape[0] / num_devices), old_shape[-1])
            result = pmapped_func(args.reshape(new_shape))
            return np.array(result).flatten()

    elif backend == "gpu":
        vmapped_func = jax.jit(jax.vmap(logL_func))

        def new_logL_func(args):
            result = vmapped_func(args)
            return np.array(result).flatten()

    else:
        raise ValueError("backend must be either cpu or gpu")

    return new_logL_func
