from lenstronomy.Sampling.Samplers.pso import (
    ParticleSwarmOptimizer as PSO_lenstronomy,
    Particle,
)

import numpy as np

__all__ = ["ParticleSwarmOptimizer"]


class ParticleSwarmOptimizer(PSO_lenstronomy):
    """Optimizer using a swarm of particles. Same as the PSO from lenstronomy, but
    the input log likelihood function is assumed to be vectorized.

    :param func: A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that position.
    :param low: array of the lower bound of the parameter space
    :param high: array of the upper bound of the parameter space
    :param particle_count: the number of particles to use.
    """

    def __init__(
        self,
        func,
        low,
        high,
        particle_count=25,
    ):
        """

        :param func: function to call to return log likelihood. Must be a vectorized logL function.
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

        self.logL_func = func
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
