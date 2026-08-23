from lenstronomy.Sampling.Samplers.pso import (
    ParticleSwarmOptimizer as PSO_lenstronomy,
    Particle,
)

import numpy as np

__all__ = ["ParticleSwarmOptimizer"]


class ParticleSwarmOptimizer(PSO_lenstronomy):
    """Optimizer using a swarm of particles. Same as the PSO from lenstronomy, but the
    input log likelihood function is assumed to be parallelized across CPU cores (e.g.
    pmap(f)).

    The PSO algorithm in this class does not happen within JIT, since jit(pmap(f)) is
    generally not recommended and can lead to unwanted side effects (see discussion at
    https://github.com/jax-ml/jax/issues/2926#issuecomment-802411631).
    Thus, this class is unfit
    for running PSO on GPU due to memory transfer overheads between CPU and GPU. To run PSO on GPU,
    use the ParticleSwarmOptimizerJIT class in pso_jit.py.
    """

    def __init__(
        self,
        func,
        low,
        high,
        particle_count=25,
    ):
        """

        :param func: function to call to return log likelihood of swarm. Must take in the position vector
            of the entire swarm.
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

    @property
    def global_best_fitness(self):
        return self.global_best.fitness

    def _get_fitness(self, swarm):
        """Set fitness (probability) of the particles in swarm.

        :param swarm: PSO state
        :type swarm: list of Particle() instances of the swarm
        :return:
        :rtype:
        """
        position = [particle.position for particle in swarm]

        position = np.asarray(position, dtype=float)
        ln_probability = np.asarray(self.logL_func(position), dtype=float)

        for i, particle in enumerate(swarm):
            particle.fitness = ln_probability[i]
            particle.position = position[i]
