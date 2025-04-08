__author__ = "sibirrer"

import jax
from jax import jit, numpy as jnp
from jaxtronomy.LensModel.profile_list_base import ProfileListBase
from functools import partial

jax.config.update("jax_enable_x64", True)

__all__ = ["SinglePlane"]


class SinglePlane(ProfileListBase):
    """Class to handle an arbitrary list of lens models in a single lensing plane."""

    @partial(jit, static_argnums=(0, 4))
    def ray_shooting(self, x, y, kwargs, k=None):
        """Maps image to source position (inverse deflection).

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: source plane positions corresponding to (x, y) in the image plane
        """

        dx, dy = self.alpha(x, y, kwargs, k=k)
        return x - dx, y - dy

    @partial(jit, static_argnums=(0, 6))
    def fermat_potential(
        self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None
    ):
        """Fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching
            the lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn
            1 in Suyu et al. 2013) as a list
        """

        potential = self.potential(x_image, y_image, kwargs_lens, k=k)
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image, kwargs_lens, k=k)
        geometry = ((x_image - x_source) ** 2 + (y_image - y_source) ** 2) / 2.0
        return geometry - potential

    @partial(jit, static_argnums=(0, 4))
    def potential(self, x, y, kwargs, k=None):
        """Lensing potential.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: lensing potential in units of arcsec^2
        """
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        potential = jnp.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                potential += func.function(x, y, **kwargs[i])
        return potential

    @partial(jit, static_argnums=(0, 4))
    def alpha(self, x, y, kwargs, k=None):
        """Deflection angles.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: deflection angles in units of arcsec
        """
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)

        if isinstance(k, int):
            return self.func_list[k].derivatives(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_x_i, f_y_i = func.derivatives(x, y, **kwargs[i])
                f_x += f_x_i
                f_y += f_y_i

        return f_x, f_y

    @partial(jit, static_argnums=(0, 4))
    def hessian(self, x, y, kwargs, k=None):
        """Hessian matrix.

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)
        if isinstance(k, int):
            f_xx, f_xy, f_yx, f_yy = self.func_list[k].hessian(x, y, **kwargs[k])
            return f_xx, f_xy, f_yx, f_yy

        bool_list = self._bool_list(k)
        f_xx, f_xy, f_yx, f_yy = (
            jnp.zeros_like(x),
            jnp.zeros_like(x),
            jnp.zeros_like(x),
            jnp.zeros_like(x),
        )
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_xx_i, f_xy_i, f_yx_i, f_yy_i = func.hessian(x, y, **kwargs[i])
                f_xx += f_xx_i
                f_xy += f_xy_i
                f_yx += f_yx_i
                f_yy += f_yy_i
        return f_xx, f_xy, f_yx, f_yy

    @partial(jit, static_argnums=(0, 3))
    def mass_3d(self, r, kwargs, k=None):
        """Computes the mass within a 3d sphere of radius r.

        if you want to have physical units of kg, you need to multiply by this factor:
        const.arcsec ** 2 * self._cosmo.dd * self._cosmo.ds / self._cosmo.dds *
        const.Mpc * const.c ** 2 / (4 * jnp.pi * const.G) grav_pot = -const.G * mass_dim
        / (r * const.arcsec * self._cosmo.dd * const.Mpc)

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(k)
        mass_3d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                mass_3d_i = func.mass_3d_lens(r, **kwargs_i)
                mass_3d += mass_3d_i
        return mass_3d

    @partial(jit, static_argnums=(0, 3))
    def mass_2d(self, r, kwargs, k=None):
        """Computes the mass enclosed a projected (2d) radius r.

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: projected mass (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(k)
        mass_2d = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                mass_2d_i = func.mass_2d_lens(r, **kwargs_i)
                mass_2d += mass_2d_i
        return mass_2d

    @partial(jit, static_argnums=(0, 3))
    def density(self, r, kwargs, k=None):
        """3d mass density at radius r The integral in the LOS projection of this
        quantity results in the convergence quantity.

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param k: only evaluate the k-th lens model
        :type k: None, int, or tuple of ints
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """
        bool_list = self._bool_list(k)
        density = 0
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs_i = {
                    k: v
                    for k, v in kwargs[i].items()
                    if k not in ["center_x", "center_y"]
                }
                density_i = func.density_lens(r, **kwargs_i)
                density += density_i
        return density
