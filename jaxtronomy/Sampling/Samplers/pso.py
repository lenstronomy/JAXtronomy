from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer as PSO_lenstronomy, Particle

import jax
import numpy as np
from functools import partial

__all__ = ["ParticleSwarmOptimizer"]


class ParticleSwarmOptimizer(PSO_lenstronomy):
    """Optimizer using a swarm of particles.

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
        self, func, low, high, particle_count=25, pool=None, args=None, kwargs=None
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
        :param pool: MPI pool for mapping different processes
        :type pool: None or MPI pool
        :param args: positional arguments to send to `func`. The function
        will be called as `func(x, *args, **kwargs)`.
        :type args: `list`
        :param kwargs: keyword arguments to send to `func`. The function
        will be called as `func(x, *args, **kwargs)`
        :type kwargs: `dict`
        """
        self.low = [l for l in low]
        self.high = [h for h in high]
        self.particleCount = particle_count
        self.pool = pool

        self.param_count = len(self.low)

        self.swarm = self._init_swarm()
        self.global_best = Particle.create(self.param_count)

        mapped_func = partial(jax.lax.map, func)
        self.parallelized_func = jax.pmap(mapped_func, devices=jax.devices())
        self.num_devices = jax.device_count()

        if particle_count % self.num_devices != 0:
            raise ValueError(f"PSO particle count {particle_count} must be divisible by the number of CPU/GPU devices for parallelization. "
                             f"There are {self.num_devices} {jax.default_backend()} devices currently recognized by JAX.")

    def _get_fitness(self, swarm):
        """Set fitness (probability) of the particles in swarm.

        :param swarm: PSO state
        :type swarm: list of Particle() instances of the swarm
        :return:
        :rtype:
        """
        position = [particle.position for particle in swarm]

        position = np.array(position)
        old_shape = position.shape
        new_shape = (self.num_devices, int(old_shape[0]/self.num_devices), old_shape[-1])
        ln_probability = self.parallelized_func(position.reshape(new_shape)).reshape(old_shape[0])

        for i, particle in enumerate(swarm):
            particle.fitness = ln_probability[i]
            particle.position = position[i]