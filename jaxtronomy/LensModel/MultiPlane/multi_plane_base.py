from functools import partial
from jax import jit, numpy as jnp
import numpy as np
from lenstronomy.Cosmo.background import Background
from jaxtronomy.LensModel.profile_list_base import ProfileListBase
import lenstronomy.Util.constants as const

__all__ = ["MultiPlaneBase"]


class MultiPlaneBase(ProfileListBase):
    """Multi-plane lensing class.

    The lens model deflection angles are in units of reduced deflections from the
    specified redshift of the lens to the source redshift of the class instance.
    """

    def __init__(
        self,
        lens_model_list,
        lens_redshift_list,
        z_source_convention,
        cosmo=None,
        cosmo_interp=False,
        z_interp_stop=None,
        num_z_interp=100,
        profile_kwargs_list=None,
    ):
        """
        A description of the recursive multi-plane formalism can be found e.g. here: https://arxiv.org/abs/1312.1536

        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
            models. If None, 'z_source' is used.
        :param cosmo: instance of astropy.cosmology
        :param cosmo_interp: bool, if True, will use interpolated cosmology
        :param z_interp_stop: (only in multi-plane with cosmo_interp=True); maximum redshift for distance interpolation
            This number should be higher or equal the maximum of the source redshift and/or the z_source_convention
        :param num_z_interp: (only in multi-plane with cosmo_interp=True); number of redshift bins for interpolating
            distances
        :param profile_kwargs_list: list of dicts, keyword arguments used to initialize profile classes
            in the same order of the lens_model_list. If any of the profile_kwargs are None, then that
            profile will be initialized using default settings.
        """
        self._lens_model_list = lens_model_list

        if z_interp_stop is None:
            z_interp_stop = z_source_convention
        self._cosmo_bkg = Background(
            cosmo, interp=cosmo_interp, z_stop=z_interp_stop, num_interp=num_z_interp
        )
        self._z_source_convention = z_source_convention
        if len(lens_redshift_list) > 0:
            z_lens_max = np.max(lens_redshift_list)
            if z_lens_max >= z_source_convention:
                raise ValueError(
                    "deflector redshifts higher or equal the source redshift convention (%s >= %s for the "
                    "reduced lens model quantities not allowed (leads to negative reduced deflection "
                    "angles!" % (z_lens_max, z_source_convention)
                )
        if not len(self._lens_model_list) == len(lens_redshift_list):
            raise ValueError(
                "The length of lens_model_list does not correspond to redshift_list"
            )

        self._lens_redshift_list = lens_redshift_list
        super(MultiPlaneBase, self).__init__(
            self._lens_model_list,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention,
            profile_kwargs_list=profile_kwargs_list,
        )

        if len(self._lens_model_list) < 1:
            self._sorted_redshift_index = []
        else:
            self._sorted_redshift_index = self._index_ordering(lens_redshift_list)

        self._T_ij_list = []
        self._T_z_list = []

        self._reduced2physical_factor = []

        self.set_T_zs_and_T_ijs()
        self.set_ddts()

    # This function in called in the init, outside of jit
    def set_T_zs_and_T_ijs(self):
        """Set the transverse angular diameter distances between the observer and the
        lens planes and between the lens planes."""
        z_before = 0
        T_z = 0
        # Sort redshift for vectorized reduced2physical factor calculation
        if len(self._lens_model_list) < 1:
            self._reduced2physical_factor = []
        else:
            z_sort = np.array(self._lens_redshift_list)[self._sorted_redshift_index]
            z_source_array = np.ones(z_sort.shape) * self._z_source_convention
            self._reduced2physical_factor = self._cosmo_bkg.d_xy(
                0, self._z_source_convention
            ) / self._cosmo_bkg.d_xy(z_sort, z_source_array)

        self._T_ij_list = []
        self._T_z_list = []
        for idex in self._sorted_redshift_index:
            z_lens = self._lens_redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                T_z = self._cosmo_bkg.T_xy(0, z_lens)
                delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
            self._T_ij_list.append(delta_T)
            self._T_z_list.append(T_z)
            z_before = z_lens

    # This function is called in the init outside of JIT
    def set_ddts(self):
        """Computes time delay distance (in units of Mpc) from each lens redshift to the
        source."""

        self._D_dt_list = []
        for z_lens in self._lens_redshift_list:
            self._D_dt_list.append(
                self._cosmo_bkg.ddt(z_lens, self._z_source_convention)
            )

    # Updating class variables not allowed
    def set_background_cosmo(self, cosmo):
        """Set the cosmology instance of the background class.

        :param cosmo: instance of astropy.cosmology
        """
        raise Exception("Updating cosmology not allowed; please create a new class")

    @property
    def z_source_convention(self):
        """Redshift of the source to define the reduced deflection angles of the lens
        models."""
        return self._z_source_convention

    @property
    def sorted_redshift_index(self):
        """List of lens indices in the sorted redshift order."""
        return self._sorted_redshift_index

    @property
    def T_z_list(self):
        """List of transverse angular diameter distances between the observer and the
        lens planes."""
        return self._T_z_list

    @property
    def T_ij_list(self):
        """List of transverse angular diameter distances between the lens planes."""
        return self._T_ij_list

    @partial(jit, static_argnums=(0, 5, 6, 8))
    def ray_shooting_partial_comoving(
        self,
        x,
        y,
        alpha_x,
        alpha_y,
        z_start,
        z_stop,
        kwargs_lens,
        include_z_start=False,
        T_ij_start=None,
        T_ij_end=None,
    ):
        """Ray-tracing through parts of the cone, starting with (x,y) co-moving
        distances and angles (alpha_x, alpha_y) at redshift z_start and then backwards
        to redshift z_stop. NOTE: This function recompiles each time a new z_start or z_stop is supplied.

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param include_z_start: bool, if True, includes the computation of the
            deflection angle at the same redshift as the start of the ray-tracing.
            ATTENTION: deflection angles at the same redshift as z_stop will be computed
            always! This can lead to duplications in the computation of deflection
            angles.
        :param T_ij_start: transverse angular distance between the starting redshift to
            the first lens plane to follow.
        :param T_ij_end: transverse angular distance between the last lens plane being
            computed and z_end.
        :return: co-moving position and angles at redshift z_stop
        """
        if z_start != 0 and T_ij_start is None:
            raise ValueError(
                "In jaxtronomy, either z_start must be zero or T_ij_start must be provided. You can use the class function \n"
                "MultiPlaneBase.transverse_distance_start_stop(z_start, z_stop) to compute T_ij_start."
            )
        if T_ij_end is None:
            raise ValueError(
                "T_ij_end must be provided in jaxtronomy. You can use the class function \n"
                "MultiPlaneBase.transverse_distance_start_stop(z_start, z_stop) to compute T_ij_end."
            )
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)

        alpha_x = jnp.array(alpha_x)
        alpha_y = jnp.array(alpha_y)

        # z_lens_last = z_start
        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[idex]

            if (
                self._start_condition(include_z_start, z_lens, z_start)
                and z_lens <= z_stop
            ):
                if i == 0:
                    if T_ij_start is None:
                        # z start is always zero in this case
                        delta_T = self._T_ij_list[0]
                    else:
                        delta_T = T_ij_start
                else:
                    delta_T = self._T_ij_list[i]
                x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
                alpha_x, alpha_y = self._add_deflection(
                    x, y, alpha_x, alpha_y, kwargs_lens, i
                )
        delta_T = T_ij_end
        x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
        return x, y, alpha_x, alpha_y

    # Not implemented in jaxtronomy yet
    # def ray_shooting_partial(
    #    self,
    #    theta_x,
    #    theta_y,
    #    alpha_x,
    #    alpha_y,
    #    z_start,
    #    z_stop,
    #    kwargs_lens,
    #    include_z_start=False,
    #    T_ij_start=None,
    #    T_ij_end=None,
    #    T_z_start=None,
    #    T_z_stop=None,
    # ):
    #    """Ray-tracing through parts of the cone, starting with (x,y) in angular units
    #    as seen on the sky without lensing and angles (alpha_x, alpha_y) as seen at
    #    redshift z_start and then backwards to redshift z_stop.

    #    :param theta_x: angular position on the sky [arcsec]
    #    :param theta_y: angular position on the sky [arcsec]
    #    :param alpha_x: ray angle at z_start [arcsec]
    #    :param alpha_y: ray angle at z_start [arcsec]
    #    :param z_start: redshift of start of computation
    #    :param z_stop: redshift where output is computed
    #    :param kwargs_lens: lens model keyword argument list
    #    :param include_z_start: bool, if True, includes the computation of the
    #        deflection angle at the same redshift as the start of the ray-tracing.
    #        ATTENTION: deflection angles at the same redshift as z_stop will be computed
    #        always! This can lead to duplications in the computation of deflection
    #        angles.
    #    :param T_ij_start: transverse angular distance between the starting redshift to
    #        the first lens plane to follow. If not set, will compute the distance each
    #        time this function gets executed.
    #    :param T_ij_end: transverse angular distance between the last lens plane being
    #        computed and z_end. If not set, will compute the distance each time this
    #        function gets executed.
    #    :param T_z_start: transverse angular distance up to z_start. If not set, will
    #        compute the distance each time this function gets executed.
    #    :param T_z_stop: transverse angular distance up to z_stop. If not set, will
    #        compute the distance each time this function gets executed.
    #    :return: angular position and angles at redshift z_stop
    #    """
    #    if T_z_start is None:
    #        T_z_start = self._cosmo_bkg.T_xy(0, z_start)
    #    x = np.array(theta_x, dtype=float) * T_z_start
    #    y = np.array(theta_y, dtype=float) * T_z_start

    #    x, y, alpha_x, alpha_y = self.ray_shooting_partial_comoving(
    #        x,
    #        y,
    #        alpha_x,
    #        alpha_y,
    #        z_start,
    #        z_stop,
    #        kwargs_lens,
    #        include_z_start=include_z_start,
    #        T_ij_start=T_ij_start,
    #        T_ij_end=T_ij_end,
    #    )
    #    if T_z_stop is None:
    #        T_z_stop = self._cosmo_bkg.T_xy(0, z_stop)
    #    beta_x = x / T_z_stop
    #    beta_y = y / T_z_stop
    #    return beta_x, beta_y, alpha_x, alpha_y

    # This function is called in the init, outside of jit; requires cosmology calculations
    def transverse_distance_start_stop(self, z_start, z_stop, include_z_start=False):
        """Computes the transverse distances (T_ij) that are required by ray-tracing.
        T_ij_start is the distance between the starting redshift and the first deflector
        after z_start. T_ij_end is the distance between the last deflector before z_stop
        and z_stop.

        :param z_start: redshift of the start of the ray-tracing
        :param z_stop: stop of ray-tracing
        :param include_z_start: boolean, if True includes the computation of the
            starting position if the first deflector is at z_start
        :return: T_ij_start, T_ij_end
        """
        z_lens_last = z_start
        first_deflector = True
        T_ij_start = None
        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[idex]
            if (
                self._start_condition(include_z_start, z_lens, z_start)
                and z_lens <= z_stop
            ):
                if first_deflector is True:
                    T_ij_start = self._cosmo_bkg.T_xy(z_start, z_lens)
                    first_deflector = False
                z_lens_last = z_lens
        T_ij_end = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
        return T_ij_start, T_ij_end

    # This function is called outside of jit
    def compute_source_distance(self, z_source):
        """Compute the relevant angular diameter distance to a specific source redshift.

        :param z_source: float, source redshift
        :return: transverse angular distance between z=0 and z_source
        """
        return self._cosmo_bkg.T_xy(0, z_source)

    @partial(jit, static_argnums=(0, 4))
    def geo_shapiro_delay(
        self, theta_x, theta_y, kwargs_lens, z_stop, T_z_stop=None, T_ij_end=None
    ):
        """Geometric and Shapiro (gravitational) light travel time relative to a
        straight path through the coordinate (0,0) Negative sign means earlier arrival
        time. NOTE: This function recompiles each time a new z_stop is supplied.

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens: lens model keyword argument list
        :param z_stop: redshift of the source to stop the backwards ray-tracing
        :param T_z_stop: transverse angular distance from z=0 to z_stop
        :param T_ij_end: transverse angular distance between the last deflector before
            z_stop and z_stop
        :return: dt_geo, dt_shapiro, [days]
        """
        if T_z_stop is None or T_ij_end is None:
            raise ValueError(
                "In jaxtronomy, T_z_stop (transverse angular distance from z=0 to z_stop) and T_ij_end "
                "(transverse angular distance between the last deflector before z_stop and z_stop) must be provided.\n"
                "You can do T_z_stop = MultiPlaneBase.compute_source_distance(z_stop) and "
                "_, T_ij_end = MultiPlaneBase.transverse_distance_start_stop(0, z_stop)."
            )

        dt_grav = jnp.zeros_like(theta_x, dtype=float)
        dt_geo = jnp.zeros_like(theta_x, dtype=float)
        x = jnp.zeros_like(theta_x, dtype=float)
        y = jnp.zeros_like(theta_y, dtype=float)
        alpha_x = jnp.array(theta_x, dtype=float)
        alpha_y = jnp.array(theta_y, dtype=float)

        for i, index in enumerate(self._sorted_redshift_index):
            z_lens = self._lens_redshift_list[index]
            if z_lens <= z_stop:
                T_ij = self._T_ij_list[i]
                x_new, y_new = self._ray_step(x, y, alpha_x, alpha_y, T_ij)
                if i == 0:
                    pass
                elif T_ij > 0:
                    T_j = self._T_z_list[i]
                    T_i = self._T_z_list[i - 1]
                    beta_i_x, beta_i_y = x / T_i, y / T_i
                    beta_j_x, beta_j_y = x_new / T_j, y_new / T_j
                    dt_geo_new = self._geometrical_delay(
                        beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij
                    )
                    dt_geo += dt_geo_new
                x, y = x_new, y_new
                dt_grav_new = self._gravitational_delay(x, y, kwargs_lens, i)
                alpha_x, alpha_y = self._add_deflection(
                    x, y, alpha_x, alpha_y, kwargs_lens, i
                )

                dt_grav += dt_grav_new
        T_ij = T_ij_end
        x_new, y_new = self._ray_step(x, y, alpha_x, alpha_y, T_ij)
        T_j = T_z_stop
        T_i = self._T_z_list[i]
        beta_i_x, beta_i_y = x / T_i, y / T_i
        beta_j_x, beta_j_y = x_new / T_j, y_new / T_j
        dt_geo_new = self._geometrical_delay(
            beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij
        )
        dt_geo += dt_geo_new
        return dt_geo, dt_grav

    # This function is called in the init outside of JIT
    @staticmethod
    def _index_ordering(redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in ascending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        # sort_index = np.argsort(redshift_list[redshift_list < z_source])
        sort_index = np.argsort(redshift_list)
        # if len(sort_index) < 1:
        #    Warning("There is no lens object between observer at z=0 and source at z=%s" % z_source)
        return sort_index

    @partial(jit, static_argnums=(0, 2))
    def _reduced2physical_deflection(self, alpha_reduced, index_lens):
        """alpha_reduced = D_ds/Ds alpha_physical.

        :param alpha_reduced: reduced deflection angle
        :param index_lens: integer, index of the deflector plane
        :return: physical deflection angle
        """
        factor = self._reduced2physical_factor[index_lens]
        return alpha_reduced * factor

    @partial(jit, static_argnums=(0, 4))
    def _gravitational_delay(self, x, y, kwargs_lens, index):
        """

        :param x: co-moving coordinate at the lens plane
        :param y: co-moving coordinate at the lens plane
        :param kwargs_lens: lens model keyword arguments
        :param index: index of the lens model in sorted redshfit convention
        :return: gravitational delay in units of days as seen at z=0
        """
        theta_x, theta_y = self._co_moving2angle(x, y, index)
        k = self._sorted_redshift_index[index]
        potential = self.func_list[k].function(theta_x, theta_y, **kwargs_lens[k])
        D_dt = self._D_dt_list[k]
        delay_days = const.delay_arcsec2days(potential, D_dt)
        return -delay_days

    @staticmethod
    @jit
    def _geometrical_delay(beta_i_x, beta_i_y, beta_j_x, beta_j_y, T_i, T_j, T_ij):
        """

        :param beta_i_x: angle on the sky at plane i
        :param beta_i_y: angle on the sky at plane i
        :param beta_j_x: angle on the sky at plane j
        :param beta_j_y: angle on the sky at plane j
        :param T_i: transverse diameter distance to z_i
        :param T_j: transverse diameter distance to z_j
        :param T_ij: transverse diameter distance from z_i to z_j
        :return: excess delay relative to a straight line
        """
        d_beta_x = beta_j_x - beta_i_x
        d_beta_y = beta_j_y - beta_i_y
        tau_ij = T_i * T_j / T_ij * const.Mpc / const.c / const.day_s * const.arcsec**2
        return tau_ij * (d_beta_x**2 + d_beta_y**2) / 2

    @partial(jit, static_argnums=(0, 3))
    def _co_moving2angle(self, x, y, index):
        """Transforms co-moving distances Mpc into angles on the sky (radian)

        :param x: co-moving distance
        :param y: co-moving distance
        :param index: index of plane
        :return: angles on the sky
        """
        T_z = self._T_z_list[index]
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    @staticmethod
    @jit
    def _ray_step(x, y, alpha_x, alpha_y, delta_T):
        """Ray propagation with small angle approximation The difference to
        _ray_step_add() is that the previous input position (x, y) do NOT get
        overwritten and are still accessible.

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transverse angular diameter distance to the next step
        :return: co-moving position at the next step (backwards)
        """
        x_ = x + alpha_x * delta_T
        y_ = y + alpha_y * delta_T
        return x_, y_

    @staticmethod
    @jit
    def _ray_step_add(x, y, alpha_x, alpha_y, delta_T):
        """Ray propagation with small angle approximation The difference to _ray_step()
        is that the previous input position (x, y) do get overwritten, which is faster.

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transverse angular diameter distance to the next step
        :return: co-moving position at the next step (backwards)
        """
        x += alpha_x * delta_T
        y += alpha_y * delta_T
        return x, y

    @partial(jit, static_argnums=(0, 6))
    def _add_deflection(self, x, y, alpha_x, alpha_y, kwargs_lens, index):
        """Adds the physical deflection angle of a single lens plane to the deflection
        field.

        :param x: co-moving distance at the deflector plane
        :param y: co-moving distance at the deflector plane
        :param alpha_x: physical angle (radian) before the deflector plane
        :param alpha_y: physical angle (radian) before the deflector plane
        :param kwargs_lens: lens model parameter kwargs
        :param index: index of the lens model to be added in sorted redshift list
            convention
        :return: updated physical deflection after deflector plane (in a backwards ray-
            tracing perspective)
        """
        theta_x, theta_y = self._co_moving2angle(x, y, index)
        k = self._sorted_redshift_index[index]
        alpha_x_red, alpha_y_red = self.func_list[k].derivatives(
            theta_x, theta_y, **kwargs_lens[k]
        )
        alpha_x_phys = self._reduced2physical_deflection(alpha_x_red, index)
        alpha_y_phys = self._reduced2physical_deflection(alpha_y_red, index)
        return alpha_x - alpha_x_phys, alpha_y - alpha_y_phys

    # This function should not be jitted as it must return a static result
    @staticmethod
    def _start_condition(inclusive, z_lens, z_start):
        """

        :param inclusive: boolean, if True selects z_lens including z_start, else only selects z_lens > z_start
        :param z_lens: deflector redshift
        :param z_start: starting redshift (lowest redshift)
        :return: boolean of condition
        """

        if inclusive:
            return z_lens >= z_start
        else:
            return z_lens > z_start
