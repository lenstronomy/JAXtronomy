from jax import jit
from jaxtronomy.Util.util import shift_center
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Flexion"]


class Flexion(LensProfileBase):
    """Class for flexion."""

    param_names = ["g1", "g2", "g3", "g4", "ra_0", "dec_0"]
    lower_limit_default = {
        "g1": -0.1,
        "g2": -0.1,
        "g3": -0.1,
        "g4": -0.1,
        "ra_0": -100,
        "dec_0": -100,
    }
    upper_limit_default = {
        "g1": 0.1,
        "g2": 0.1,
        "g3": 0.1,
        "g4": 0.1,
        "ra_0": 100,
        "dec_0": 100,
    }

    @staticmethod
    @jit
    def function(x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x, y = shift_center(x, y, ra_0, dec_0)
        f_ = (
            1.0
            / 6
            * (g1 * x**3 + 3 * g2 * x**2 * y + 3 * g3 * x * y**2 + g4 * y**3)
        )
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x, y = shift_center(x, y, ra_0, dec_0)
        f_x = 1.0 / 2.0 * g1 * x**2 + g2 * x * y + 1.0 / 2.0 * g3 * y**2
        f_y = 1.0 / 2.0 * g2 * x**2 + g3 * x * y + 1.0 / 2.0 * g4 * y**2
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x, y = shift_center(x, y, ra_0, dec_0)
        f_xx = g1 * x + g2 * y
        f_yy = g3 * x + g4 * y
        f_xy = g2 * x + g3 * y
        return f_xx, f_xy, f_xy, f_yy
