import jax
from jax import jit, lax, numpy as jnp
import numpy as np
from functools import partial

__all__ = ["ParticleSwarmOptimizerJIT"]


class ParticleSwarmOptimizerJIT(object):
    """Optimizer using a swarm of particles. Same as the PSO from lenstronomy, but the
    entire computation happens within JIT. This class is intended to be used for GPU so
    that there are no memory transfer overheads between GPU and CPU since the entire
    computation happens on GPU.

    The input log likelihood function is assumed to be vectorized.
    """

    def __init__(
        self,
        func,
        low,
        high,
        particle_count=25,
    ):
        """

        :param func: function to call to return log likelihood of swarm. Must accept
            as argument the position vector of the entire swarm.
        :type func: python definition
        :param low: lower bound of the parameters
        :type low: numpy array
        :param high: upper bound of the parameters
        :type high: numpy array
        :param particle_count: number of particles in each iteration of the PSO
        :type particle_count: int
        """
        self.low = np.array([l for l in low])
        self.high = np.array([h for h in high])
        self.particleCount = particle_count
        self.param_count = len(self.low)
        self.logL_func = func

        self.set_global_best(jnp.zeros(self.param_count), None, -jnp.inf)

    def _init_swarm(self):
        """Initiate the swarm.

        :return: `None`
        """
        swarm_positions = np.zeros((self.particleCount, self.param_count))
        for i in range(self.particleCount):
            swarm_positions[i] = np.random.uniform(
                self.low, self.high, size=self.param_count
            )

        swarm_positions = jnp.array(swarm_positions, dtype=float)
        swarm_velocities = jnp.zeros_like(swarm_positions)
        swarm_fitnesses = self._get_fitness(swarm_positions)

        return swarm_positions, swarm_velocities, swarm_fitnesses

    def set_global_best(self, position, velocity, fitness):
        """Set the global best particle.

        :param position: position of the new global best
        :type position: `list` or `ndarray`
        :param velocity: unused, kept to maintain identical API as lenstronomy
        :param fitness: fitness of the new global best
        :type fitness: `float`
        :return: `None`
        """
        self.global_best_position = jnp.asarray(position, dtype=float)
        self.global_best_fitness = fitness

    # Top level function - should not be jitted
    def optimize(
        self,
        max_iter=1000,
        verbose=True,
        c1=1.193,
        c2=1.193,
        p=0.7,
        m=1e-3,
        n=1e-2,
        early_stop_tolerance=None,
        rng_seed=None,
    ):
        """Run the optimization and return a full list of optimization outputs.
        NOTE: Currently does not return the global best logL, velocity, and position histories.

        :param max_iter: maximum iterations
        :param verbose: if `True`, print a message every 10 iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: float between 0 and 1, determines the percentage of particles to use to
            compute swarm fit and position convergence
        :param m: stop criterion tolerance; average fitness of best p% particles must be within
            m of the global best fitness
        :param n: stop criterion tolerance; positions of best p% of particles must be within
            n euclidean distance of the global best position
        :param early_stop_tolerance: float or None; if set, will terminate the PSO early if
            |2 * global_best_fitness| < early_stop_tolerance. This takes priority over the
            fitness and spatial convergence criteria.
        :param rng_seed: int, rng seed for reproducibility. If None, a random seed is used
        """
        log_likelihood_list = []
        vel_list = []
        pos_list = []

        if rng_seed is None:
            rng_seed = int(np.random.uniform(0, 1) * 1000000)

        init_swarm = self._init_swarm()

        print("starting PSO optimization")
        global_best_position, global_best_fitness = self.run_iterations(
            init_swarm,
            self.global_best_position,
            self.global_best_fitness,
            max_iter,
            c1,
            c2,
            p,
            m,
            n,
            early_stop_tolerance,
            rng_seed,
            verbose,
        )
        self.set_global_best(global_best_position, None, global_best_fitness)

        return np.array(global_best_position), [log_likelihood_list, pos_list, vel_list]

    @partial(jit, static_argnums=(0, 7, 12))
    def run_iterations(
        self,
        init_swarm,
        global_best_position,
        global_best_fitness,
        max_iter=1000,
        c1=1.193,
        c2=1.193,
        p=0.7,
        m=1e-3,
        n=1e-2,
        early_stop_tolerance=None,
        rng_seed=0,
        verbose=True,
    ):
        """Launches the PSO. Yields the complete swarm per iteration.

        :param max_iter: maximum iterations
        :param global_best_position: 1d array of size len(args); the initial values for
            the best particle position
        :param global_best_fitness: float; the initial value for the best particle
            fitness
        :param c1: cognitive weight
        :param c2: social weight
        :param p: float between 0 and 1, determines the percentage of particles to use
            to compute swarm fit and position convergence
        :param m: stop criterion tolerance; average fitness of best p% particles must be
            within m of the global best fitness
        :param n: stop criterion tolerance; positions of best p% of particles must be
            within n euclidean distance of the global best position
        :param early_stop_tolerance: float or None; if set, will terminate the PSO early
            if |2 * global_best_fitness| < early_stop_tolerance. This takes priority
            over the fitness and spatial convergence criteria.
        :param verbose: if True, prints out the iteration number as the loop progresses
        :type verbose: boolean
        """
        key = jax.random.PRNGKey(rng_seed)

        swarm_positions, swarm_velocities, swarm_fitnesses = init_swarm

        # intialize personal bests (these will be updated at the start of every iteration in the loop)
        personal_best_positions = jnp.zeros((self.particleCount, self.param_count))
        personal_best_fitnesses = -jnp.ones(self.particleCount) * jnp.inf

        init_carry = (
            0,
            global_best_position,
            global_best_fitness,
            swarm_positions,
            swarm_velocities,
            swarm_fitnesses,
            personal_best_positions,
            personal_best_fitnesses,
            key,
        )

        # define function to be computed at the start of every iteration
        # determines whether or not to continue the iterations
        def continuing_criteria(carry):
            (
                i,
                global_best_position,
                global_best_fitness,
                swarm_positions,
                swarm_velocities,
                swarm_fitnesses,
                personal_best_positions,
                personal_best_fitnesses,
                key,
            ) = carry

            if early_stop_tolerance is not None:
                stop_early = self._acceptable_convergence(
                    early_stop_tolerance, global_best_fitness
                )
            else:
                stop_early = False

            fit = self._converged_fit(
                p=p,
                m=m,
                global_best_fitness=global_best_fitness,
                personal_best_fitnesses=personal_best_fitnesses,
            )

            space = self._converged_space(
                p=p,
                n=n,
                global_best_position=global_best_position,
                swarm_fitnesses=swarm_fitnesses,
                swarm_positions=swarm_positions,
            )
            converged = fit & space

            # continue if not converged yet and havent reached max iterations
            cont = jnp.logical_not(converged) & (i < max_iter)

            # if stop_early is True, then don't continue
            cont = jnp.where(stop_early, False, cont)
            return cont

        # define function to be iterated over
        def update_swarm(carry):
            (
                i,
                global_best_position,
                global_best_fitness,
                swarm_positions,
                swarm_velocities,
                swarm_fitnesses,
                personal_best_positions,
                personal_best_fitnesses,
                key,
            ) = carry

            if verbose:
                jax.debug.print("iteration {i}", i=i)

            # find the current best particle
            best_particle_index = jnp.argmax(swarm_fitnesses)
            best_particle_fitness = swarm_fitnesses.at[best_particle_index].get()
            best_particle_position = swarm_positions.at[best_particle_index].get()

            # update the global best
            global_best_position = jnp.where(
                best_particle_fitness > global_best_fitness,
                best_particle_position,
                global_best_position,
            )
            global_best_fitness = jnp.where(
                best_particle_fitness > global_best_fitness,
                best_particle_fitness,
                global_best_fitness,
            )

            # update all personal bests
            personal_best_positions = jnp.where(
                swarm_fitnesses > personal_best_fitnesses,
                swarm_positions.T,
                personal_best_positions.T,
            ).T
            personal_best_fitnesses = jnp.where(
                swarm_fitnesses > personal_best_fitnesses,
                swarm_fitnesses,
                personal_best_fitnesses,
            )

            # each particle takes a random step
            key, subkey = jax.random.split(key)
            w = (
                0.5
                + jax.random.uniform(
                    key=subkey,
                    minval=0,
                    maxval=1,
                    shape=(self.particleCount, self.param_count),
                )
                / 2
            )

            part_vel = jnp.multiply(w, swarm_velocities)
            key, subkey = jax.random.split(key)
            cog_vel = c1 * jnp.multiply(
                jax.random.uniform(
                    key=subkey,
                    minval=0,
                    maxval=1,
                    shape=(self.particleCount, self.param_count),
                ),
                (personal_best_positions - swarm_positions),
            )

            key, subkey = jax.random.split(key)
            soc_vel = c2 * jnp.multiply(
                jax.random.uniform(
                    key=subkey,
                    minval=0,
                    maxval=1,
                    shape=(self.particleCount, self.param_count),
                ),
                (global_best_position - swarm_positions),
            )

            swarm_velocities = part_vel + cog_vel + soc_vel
            swarm_positions += swarm_velocities
            swarm_fitnesses = self._get_fitness(swarm_positions)

            return (
                i + 1,
                global_best_position,
                global_best_fitness,
                swarm_positions,
                swarm_velocities,
                swarm_fitnesses,
                personal_best_positions,
                personal_best_fitnesses,
                key,
            )

        # run the loop
        carry = lax.while_loop(continuing_criteria, update_swarm, init_carry)

        # returns the global best position and global best fitness
        return carry[1], carry[2]

    @partial(jit, static_argnums=0)
    def _get_fitness(self, swarm_positions):
        """Set fitness (probability) of the particles in swarm.

        :param swarm_positions: 2d array of shape (n_particles, len(args)) containing
            the position vector of each particle in the swarm
        :type swarm_positions: array
        :return: 1d array of fitnesses of each particle in the swarm
        :rtype: array
        """
        ln_probability = self.logL_func(swarm_positions)
        return ln_probability

    @partial(jit, static_argnums=(0, 1))
    def _converged_fit(self, p, m, global_best_fitness, personal_best_fitnesses):
        """Given the global best fitness and an array of personal best fitnesses of the
        swarm, determine whether or not the average fitness of the best p% of particles
        is within m of the global best fitness.

        :param p: float between 0 and 1, determines what percent of particles to look at
            to compute convergence
        :type p: float
        :param m: tolerance criteria for fitness
        :type m: float
        :param global_best_fitness: float, the current best fitness of the swarm
        :param personal_best_fitnesses: 1d array of floats with size n_particles
            containing the best fitnesses for each particle in the swarm
        :return: whether or not the average fitness of the best p% of particles is
            within m of the global best fitness
        :rtype: bool
        """

        best_sort = jnp.sort(personal_best_fitnesses)[::-1]
        mean_fit = jnp.mean(best_sort[1 : int(self.particleCount * p)])
        return jnp.abs(global_best_fitness - mean_fit) < m

    @partial(jit, static_argnums=(0, 1))
    def _converged_space(
        self, p, n, global_best_position, swarm_fitnesses, swarm_positions
    ):
        """Given the global best position, swarm fitnesses, and swarm positions,
        determine whether or not all of the positions of the best p% of particles is
        within n euclidean-distance of the global best position.

        :param p: float between 0 and 1, determines what percent of particles to look at
            to compute convergence
        :type p: float
        :param n: tolerance criteria for euclidean distance
        :type n: float
        :param global_best_position: 1d array of size len(args), current best position
            of swarm
        :type global_best_position: array
        :param swarm_fitnesses: 1d array of floats with size n_particles containing the
        :return: whether or not the best p% of particles is within n euclidean distance
            of the global best position
        :rtype: bool
        """

        sort_by_fitness_indices = jnp.argsort(swarm_fitnesses)
        sorted_positions = swarm_positions[sort_by_fitness_indices][
            0 : int(self.particleCount * p)
        ]
        diffs = global_best_position - sorted_positions
        max_norm = jnp.max(jnp.linalg.norm(diffs, axis=1))
        return jnp.abs(max_norm) < n

    @partial(jit, static_argnums=0)
    def _acceptable_convergence(self, chi_square_tolerance, global_best_fitness):
        """Checks whether the chi squared is within the chi squared tolerance.

        :param chi_square_tolerance: float, chi square tolerance
        :param global_best_fitness: float, best fitness of the swarm
        :return: whether the chi squared is within the chi squared tolerance
        :rtype: bool
        """

        chi_square = -2 * global_best_fitness
        return chi_square < chi_square_tolerance
