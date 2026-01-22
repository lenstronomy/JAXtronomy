from lenstronomy.Sampling.Samplers.pso import (
    ParticleSwarmOptimizer as PSO_lenstronomy,
    Particle,
)

import jax
import numpy as np
from functools import partial

__all__ = ["ParticleSwarmOptimizer"]


class ParticleSwarmOptimizer(PSO_lenstronomy):
    """Optimizer using a swarm of particles. Same as the PSO from lenstronomy, but
    parallelizes computations across CPU cores automatically using JAX. For computation
    on GPU, only one GPU is used due to memory transfer overheads.

    :param func:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param low: array of the lower bound of the parameter space
    :param high: array of the upper bound of the parameter space
    :param particle_count: the number of particles to use.
    :param pool: (optional)
        An alternative method of using the parallelized algorithm. If
        provided, the value of ``threads`` is ignored and the
        object provided by ``pool`` is used for all parallelization. It
        can be any object with a ``map`` method that follows the same
        calling sequence as the built-in ``map`` function.
    """

    def __init__(
        self, func, low, high, particle_count=25,
    ):
        """

        :param func: function to call to return log likelihood
        :type func: python definition
        :param low: lower bound of the parameters
        :type low: numpy array
        :param high: upper bound of the parameters
        :type high: numpy array
        :param particle_count: number of particles in each iteration of the PSO
        :type particle_count: int
        """
        self.low = [l for l in low]
        self.high = [h for h in high]
        self.particleCount = particle_count
        self.pool = None

        self.param_count = len(self.low)
        self.global_best = Particle.create(self.param_count)

        if jax.default_backend() == "cpu":
            mapped_func = partial(jax.lax.map, func)
            pmapped_func = jax.pmap(mapped_func, devices=jax.devices())
            num_devices = jax.device_count()

            def logL_func(position):
                old_shape = position.shape
                new_shape = (
                    num_devices,
                    int(old_shape[0] / num_devices),
                    old_shape[-1],
                )
                return pmapped_func(position.reshape(new_shape)).flatten()

            if particle_count % num_devices != 0:
                raise ValueError(
                    f"PSO particle count {particle_count} must be divisible by the number of CPU devices for parallelization. "
                    f"There are {num_devices} cpu devices currently recognized by JAX."
                )
        else:
            @jax.jit
            def logL_func(position):
                return jax.vmap(func)(position).flatten()

        self.logL_func = logL_func
        self.swarm = self._init_swarm()

    def _get_fitness(self, swarm):
        """Set fitness (probability) of the particles in swarm.

        :param swarm: PSO state
        :type swarm: list of Particle() instances of the swarm
        :return:
        :rtype:
        """
        position = [particle.position for particle in swarm]

        position = np.array(position)
        ln_probability = np.array(self.logL_func(position))

        for i, particle in enumerate(swarm):
            particle.fitness = ln_probability[i]
            particle.position = position[i]
