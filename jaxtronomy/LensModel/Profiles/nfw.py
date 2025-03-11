__author__ = "sibirrer"

# this file contains a class to compute the Navaro-Frenk-White profile

from jax import config, jit
import jax.numpy as jnp

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

config.update("jax_enable_x64", True)

__all__ = ["NFW"]


class NFW(LensProfileBase):
    """This class contains functions concerning the NFW profile.

    relation are: R_200 = c * Rs
    The definition of 'Rs' is in angular (arc second) units and the normalization is put
    in with regard to a deflection angle at 'Rs' - 'alpha_Rs'. To convert a physical
    mass and concentration definition into those lensing quantities for a specific
    redshift configuration and cosmological model, you can find routines in
    `lenstronomy.Cosmo.lens_cosmo.py`

    Examples for converting angular to physical mass units
    ------------------------------------------------------

    >>> from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    >>> lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)

    Here we compute the angular scale of Rs on the sky (in arc seconds) and the deflection angle at Rs (in arc seconds):

    >>> Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=10**13, c=6)

    And here we perform the inverse calculation given Rs_angle and alpha_Rs to return the physical halo properties.

    >>> rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, alpha_Rs=alpha_Rs)

    The lens model calculation uses angular units as arguments! So to execute a deflection angle calculation one uses

    >>> from lenstronomy.LensModel.Profiles.nfw import NFW
    >>> nfw = NFW()
    >>> alpha_x, alpha_y = nfw.derivatives(x=1, y=1, Rs=Rs_angle, alpha_Rs=alpha_Rs, center_x=0, center_y=0)
    """

    profile_name = "NFW"
    param_names = ["Rs", "alpha_Rs", "center_x", "center_y"]
    lower_limit_default = {"Rs": 0, "alpha_Rs": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"Rs": 100, "alpha_Rs": 10, "center_x": 100, "center_y": 100}

    def __init__(self, interpol=False, **kwargs):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated
         (returning zeros outside)
        """
        if interpol:
            raise Exception(
                "This class no longer supports interpol functionality in JAXtronomy."
            )
        super(NFW, self).__init__()

    @staticmethod
    @jit
    def function(x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential
        """
        rho0_input = NFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        Rs = jnp.where(Rs < 0.0000001, 0.0000001, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        f_ = NFW.nfw_potential(R, Rs, rho0_input)
        return f_

    @staticmethod
    @jit
    def derivatives(x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function (integral of NFW), which are the
        deflection angles.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_ijnput = NFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        Rs = jnp.where(Rs < 0.0000001, 0.0000001, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        f_x, f_y = NFW.nfw_alpha(R, Rs, rho0_ijnput, x_, y_)
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        rho0_ijnput = NFW.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        Rs = jnp.where(Rs < 0.0000001, 0.0000001, Rs)
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        kappa = NFW.density_2d(R, 0, Rs, rho0_ijnput)
        gamma1, gamma2 = NFW.nfw_gamma(R, Rs, rho0_ijnput, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def density(R, Rs, rho0):
        """Three-dimensional NFW profile.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """
        return rho0 / (R / Rs * (1 + R / Rs) ** 2)

    @staticmethod
    @jit
    def density_lens(r, Rs, alpha_Rs):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: 3d radios
        :param Rs: turn-over radius of NFW profile
        :param alpha_Rs: deflection at Rs
        :return: density rho(r)
        """
        rho0 = NFW.alpha2rho0(alpha_Rs, Rs)
        return NFW.density(r, Rs, rho0)

    @staticmethod
    @jit
    def density_2d(x, y, Rs, rho0, center_x=0, center_y=0):
        """Projected two-dimensional NFW profile (kappa)

        :param x: x-coordinate
        :param y: y-coordinate
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param center_x: x-centroid position
        :param center_y: y-centroid position
        :return: Epsilon(R) projected density at radius R
        """
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        R = jnp.sqrt(x_**2 + y_**2)
        x = R / Rs
        Fx = NFW.F(x)
        return 2 * rho0 * Rs * Fx

    @staticmethod
    @jit
    def mass_3d(r, Rs, rho0):
        """Mass enclosed a 3d sphere of radius r.

        :param r: 3d radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :return: M(<r)
        """
        Rs = Rs.astype(float)
        m_3d = 4.0 * jnp.pi * rho0 * Rs**3 * (jnp.log((Rs + r) / Rs) - r / (Rs + r))
        return m_3d

    @staticmethod
    @jit
    def mass_3d_lens(r, Rs, alpha_Rs):
        """Mass enclosed a 3d sphere of radius r. This function takes as input the
        lensing parameterization.

        :param r: 3d radius
        :param Rs: scale radius
        :param alpha_Rs: deflection (angular units) at projected Rs
        :return: M(<r)
        """
        rho0 = NFW.alpha2rho0(alpha_Rs, Rs)
        m_3d = NFW.mass_3d(r, Rs, rho0)
        return m_3d

    @staticmethod
    @jit
    def mass_2d(R, Rs, rho0):
        """Mass enclosed a 2d cylinder of projected radius R.

        :param R: projected radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :return: mass in cylinder.
        """
        x = R / Rs
        gx = NFW.g(x)
        m_2d = 4 * rho0 * Rs * R**2 * gx / x**2 * jnp.pi
        return m_2d

    @staticmethod
    @jit
    def mass_2d_lens(R, Rs, alpha_Rs):
        """

        :param R: projected radius
        :param Rs: scale radius
        :param alpha_Rs: deflection (angular units) at projected Rs
        :return: mass enclosed 2d cylinder <R
        """

        rho0 = NFW.alpha2rho0(alpha_Rs, Rs)
        return NFW.mass_2d(R, Rs=Rs, rho0=rho0)

    @staticmethod
    @jit
    def nfw_potential(R, Rs, rho0):
        """Lensing potential of NFW profile (Sigma_crit D_OL**2)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: Epsilon(R) projected density at radius R
        """
        x = R / Rs
        hx = NFW.h(x)
        return 2 * rho0 * Rs**3 * hx

    @staticmethod
    @jit
    def nfw_alpha(R, Rs, rho0, ax_x, ax_y):
        """Deflection angle of NFW profile (times Sigma_crit D_OL) along the projection
        to coordinate 'axis'.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param ax_x: projection to either x- or y-axis
        :type ax_x: same as R
        :param ax_y: projection to either x- or y-axis
        :type ax_y: same as R
        :return: Epsilon(R) projected density at radius R
        """
        R = jnp.maximum(R, 0.00000001)
        x = R / Rs
        gx = NFW.g(x)
        a = 4 * rho0 * Rs * gx / x**2
        return a * ax_x, a * ax_y

    @staticmethod
    @jit
    def nfw_gamma(R, Rs, rho0, ax_x, ax_y):
        """Shear gamma of NFW profile (times Sigma_crit) along the projection to
        coordinate 'axis'.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param ax_x: projection to either x- or y-axis
        :type ax_x: same as R
        :param ax_y: projection to either x- or y-axis
        :type ax_y: same as R
        :return: Epsilon(R) projected density at radius R
        """
        c = 0.000001
        R = jnp.maximum(R, c)
        x = R / Rs
        gx = NFW.g(x)
        Fx = NFW.F(x)
        a = (
            2 * rho0 * Rs * (2 * gx / x**2 - Fx)
        )  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a * (ax_y**2 - ax_x**2) / R**2, -a * 2 * (ax_x * ax_y) / R**2

    @staticmethod
    @jit
    def F(X):
        """Analytic solution of the projection integral.

        :param X: R/Rs
        :type X: float >0
        """
        c = 0.0000001
        a = jnp.where(
            X == 0,
            (-1 + 2 * jnp.arctanh(jnp.sqrt((1 - c) / (1 + c)))),
            jnp.where(
                X < 1,
                1
                / (X**2 - 1)
                * (
                    1
                    - 2 / jnp.sqrt(1 - X**2) * jnp.arctanh(jnp.sqrt((1 - X) / (1 + X)))
                ),
                jnp.where(
                    X == 1,
                    1.0 / 3,
                    1
                    / (X**2 - 1)
                    * (
                        1
                        - 2
                        / jnp.sqrt(X**2 - 1)
                        * jnp.arctan(jnp.sqrt((X - 1) / (1 + X)))
                    ),
                ),
            ),
        )
        return a

    @staticmethod
    @jit
    def g(X):
        """Analytic solution of integral for NFW profile to compute deflection angle and
        gamma.

        :param X: R/Rs
        :type X: float >0
        """
        c = 0.000001
        X = jnp.where(X < c, c, X)
        a = jnp.where(
            X < 1,
            jnp.log(X / 2.0) + 1 / jnp.sqrt(1 - X**2) * jnp.arccosh(1.0 / X),
            jnp.where(
                X == 1,
                1 + jnp.log(1.0 / 2.0),
                jnp.log(X / 2) + 1 / jnp.sqrt(X**2 - 1) * jnp.arccos(1.0 / X),
            ),
        )
        return a

    @staticmethod
    @jit
    def h(X):
        """Analytic solution of integral for NFW profile to compute the potential.

        :param X: R/Rs
        :type X: float >0
        """
        c = 0.000001
        X = jnp.where(X < c, c, X)
        a = jnp.where(
            X < 1,
            jnp.log(X / 2.0) ** 2 - jnp.arccosh(1.0 / X) ** 2,
            jnp.log(X / 2.0) ** 2 + jnp.arccos(1.0 / X) ** 2,
        )
        return a

    @staticmethod
    @jit
    def alpha2rho0(alpha_Rs, Rs):
        """Convert angle at Rs into rho0.

        :param alpha_Rs: deflection angle at RS
        :param Rs: scale radius
        :return: density normalization (characteristic density)
        """

        rho0 = alpha_Rs / (4.0 * Rs**2 * (1.0 + jnp.log(1.0 / 2.0)))
        return rho0

    @staticmethod
    @jit
    def rho02alpha(rho0, Rs):
        """Convert rho0 to angle at Rs.

        :param rho0: density normalization (characteristic density)
        :param Rs: scale radius
        :return: deflection angle at RS
        """

        alpha_Rs = rho0 * (4 * Rs**2 * (1 + jnp.log(1.0 / 2.0)))
        return alpha_Rs
