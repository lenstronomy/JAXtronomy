from jax import jit, numpy as jnp
import jaxtronomy.Util.param_util as param_util


class Gaussian(object):
    """Class for Gaussian light profile The two-dimensional Gaussian profile amplitude
    is defined such that the 2D integral leads to the 'amp' value.

    profile name in LightModel module: 'GAUSSIAN'
    """

    def __init__(self):
        self.param_names = ["amp", "sigma", "center_x", "center_y"]
        self.lower_limit_default = {
            "amp": 0,
            "sigma": 0,
            "center_x": -100,
            "center_y": -100,
        }
        self.upper_limit_default = {
            "amp": 1000,
            "sigma": 100,
            "center_x": 100,
            "center_y": 100,
        }

    @staticmethod
    @jit
    def function(x, y, amp, sigma, center_x=0, center_y=0):
        """Surface brightness per angular unit.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        c = amp / (2 * jnp.pi * sigma**2)
        r2 = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        return c * jnp.exp(-r2 / 2.0)

    @staticmethod
    def total_flux(amp, sigma, center_x=0, center_y=0):
        """Integrated flux of the profile.

        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return amp

    @staticmethod
    @jit
    def light_3d(r, amp, sigma):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :return: 3D brightness per angular volume element
        """
        amp3d = amp / jnp.sqrt(2 * sigma**2) / jnp.sqrt(jnp.pi)
        sigma3d = sigma
        return Gaussian.function(r, 0, amp3d, sigma3d)


class GaussianEllipse(object):
    """Class for Gaussian light profile with ellipticity.

    profile name in LightModel module: 'GAUSSIAN_ELLIPSE'
    """

    param_names = ["amp", "sigma", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 1000,
        "sigma": 100,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    @jit
    def function(x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x, center_y
        )
        return Gaussian.function(x_, y_, amp, sigma, center_x=0, center_y=0)

    @staticmethod
    def total_flux(amp, sigma=None, e1=None, e2=None, center_x=None, center_y=None):
        """Total integrated flux of profile.

        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return amp

    @staticmethod
    @jit
    def light_3d(r, amp, sigma, e1=0, e2=0):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: 3D brightness per angular volume element
        """
        return Gaussian.light_3d(r, amp, sigma=sigma)


class MultiGaussian(object):
    """Class for Multi Gaussian lens light (2d projected light/mass distribution.

    profile name in LightModel module: 'MULTI_GAUSSIAN'
    """

    param_names = ["amp", "sigma", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 1000,
        "sigma": 100,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    @jit
    def function(x, y, amp, sigma, center_x=0, center_y=0):
        """Surface brightness per angular unit.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        f_ = jnp.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_ += Gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
        return f_

    @staticmethod
    @jit
    def total_flux(amp, sigma, center_x=0, center_y=0):
        """Total integrated flux of profile.

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        flux = 0
        for i in range(len(amp)):
            flux += Gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    @staticmethod
    @jit
    def function_split(x, y, amp, sigma, center_x=0, center_y=0):
        """Split surface brightness in individual components.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: list of arrays of surface brightness
        """
        f_list = []
        for i in range(len(amp)):
            f_list.append(Gaussian.function(x, y, amp[i], sigma[i], center_x, center_y))
        return f_list

    @staticmethod
    @jit
    def light_3d(r, amp, sigma):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :return: 3D brightness per angular volume element
        """
        f_ = jnp.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            f_ += Gaussian.light_3d(r, amp[i], sigma[i])
        return f_


class MultiGaussianEllipse(object):
    """Class for elliptical multi Gaussian profile.

    profile name in LightModel module: 'MULTI_GAUSSIAN_ELLIPSE'
    """

    param_names = ["amp", "sigma", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 1000,
        "sigma": 100,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": 100,
        "center_y": 100,
    }

    @staticmethod
    @jit
    def function(x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """Surface brightness per angular unit.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x, center_y
        )

        f_ = jnp.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_ += Gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0)
        return f_

    @staticmethod
    @jit
    def total_flux(amp, sigma, e1, e2, center_x=0, center_y=0):
        """Total integrated flux of profile.

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        flux = 0
        for i in range(len(amp)):
            flux += Gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    @staticmethod
    @jit
    def function_split(x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """Split surface brightness in individual components.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: list of arrays of surface brightness
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x, center_y
        )
        f_list = []
        for i in range(len(amp)):
            f_list.append(
                Gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0)
            )
        return f_list

    @staticmethod
    @jit
    def light_3d(r, amp, sigma, e1=0, e2=0):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: 3D brightness per angular volume element
        """
        f_ = jnp.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            f_ += Gaussian.light_3d(r, amp[i], sigma[i])
        return f_
