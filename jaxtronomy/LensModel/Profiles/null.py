import jax
from jax import jit, numpy as jnp

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

jax.config.update("jax_enable_x64", True)

__all__ = ["Null"]

class Null(LensProfileBase):

    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}

    @staticmethod
    @jit
    def function(x, y, *args, **kwargs):
        """Returns zero potential.

        :param x: x position
        :param y: y position
        """

        return jnp.zeros_like(x)

    @staticmethod
    @jit
    def derivatives(x, y, *args, **kwargs):
        """Returns zero.

        :param x: x position
        :param y: y position
        """

        return jnp.zeros_like(x), jnp.zeros_like(y)

    @staticmethod
    @jit
    def hessian(x, y, *args, **kwargs):
        """Returns zero Hessian matrix.

        :param x: x position
        :param y: y position
        """

        return jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)