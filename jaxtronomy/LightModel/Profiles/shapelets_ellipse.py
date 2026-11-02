from functools import partial
from jax import jit
from jaxtronomy.LightModel.Profiles.shapelets import ShapeletSetStatic
from jaxtronomy.Util import param_util


class ShapeletSetEllipseStatic(object):
    """Cartesian shapelets with elliptical axis ratios.
    
    The shapelet order must be declared at initialization.
    """

    param_names = ["amp", "n_max", "beta", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": -100,
        "beta": 0.01,
        "e1": -0.6,
        "e2": -0.6,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "beta": 100,
        "e1": 0.6,
        "e2": 0.6,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, n_max):
        self._shapelet_set = ShapeletSetStatic(n_max=n_max)

    @partial(jit, static_argnums=0)
    def function(self, x, y, amp, beta, e1, e2, center_x=0, center_y=0, *args, **kwargs):
        """

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param beta: shapelet scale
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: shapelet center x
        :param center_y: shapelet center y
        :return: surface brightness of combined shapelet set
        """
        x_, y_ = param_util.transform_e1e2_product_average_rotate(
            x, y, e1, e2, center_x=center_x, center_y=center_y
        )
        return self._shapelet_set.function(
            x_, y_, amp, beta, center_x=0, center_y=0
        )

    @partial(jit, static_argnums=0)
    def function_split(self, x, y, amp, beta, e1, e2, center_x=0, center_y=0, *args, **kwargs):
        """Splits shapelet set in list of individual shapelet basis function responses.

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param beta: shapelet scale
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: shapelet center x
        :param center_y: shapelet center y
        :return: list of individual shapelet basis function responses
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x=center_x, center_y=center_y
        )
        return self._shapelet_set.function_split(
            x_, y_, amp, beta, center_x=0, center_y=0
        )
