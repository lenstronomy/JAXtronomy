__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw import NFW as NFW_ref


class TestNFW(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.nfw_ref = NFW_ref()

    def test_init(self):
        npt.assert_raises(Exception, NFW, interpol=True)

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        rho0 = 1
        alpha_Rs_ref = self.nfw_ref.rho02alpha(rho0, Rs)
        values_ref = self.nfw_ref.function(x, y, Rs, alpha_Rs_ref)
        alpha_Rs = NFW.rho02alpha(rho0, Rs)
        values = NFW.function(x, y, Rs, alpha_Rs_ref)
        npt.assert_array_almost_equal(alpha_Rs_ref, alpha_Rs, decimal=8)
        npt.assert_array_almost_equal(values_ref, values, decimal=5)

        x = np.array([0])
        y = np.array([0])
        Rs = 1.0
        rho0 = 1
        alpha_Rs_ref = self.nfw_ref.rho02alpha(rho0, Rs)
        alpha_Rs = NFW.rho02alpha(rho0, Rs)
        values_ref = self.nfw_ref.function(x, y, Rs, alpha_Rs_ref)
        values = NFW.function(x, y, Rs, alpha_Rs_ref)
        npt.assert_array_almost_equal(alpha_Rs_ref, alpha_Rs, decimal=8)
        npt.assert_array_almost_equal(values_ref, values, decimal=5)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values_ref = self.nfw_ref.function(x, y, Rs, alpha_Rs_ref)
        values = NFW.function(x, y, Rs, alpha_Rs_ref)
        npt.assert_array_almost_equal(values_ref, values, decimal=5)

    def test_derivatives(self):
        Rs = 0.1
        alpha_Rs = 0.0122741127776
        x_array = np.array(
            [
                0.0,
                0.00505050505,
                0.0101010101,
                0.0151515152,
                0.0202020202,
                0.0252525253,
                0.0303030303,
                0.0353535354,
                0.0404040404,
                0.0454545455,
                0.0505050505,
                0.0555555556,
                0.0606060606,
                0.0656565657,
                0.0707070707,
                0.0757575758,
                0.0808080808,
                0.0858585859,
                0.0909090909,
                0.095959596,
                0.101010101,
                0.106060606,
                0.111111111,
                0.116161616,
                0.121212121,
                0.126262626,
                0.131313131,
                0.136363636,
                0.141414141,
                0.146464646,
                0.151515152,
                0.156565657,
                0.161616162,
                0.166666667,
                0.171717172,
                0.176767677,
                0.181818182,
                0.186868687,
                0.191919192,
                0.196969697,
                0.202020202,
                0.207070707,
                0.212121212,
                0.217171717,
                0.222222222,
                0.227272727,
                0.232323232,
                0.237373737,
                0.242424242,
                0.247474747,
                0.252525253,
                0.257575758,
                0.262626263,
                0.267676768,
                0.272727273,
                0.277777778,
                0.282828283,
                0.287878788,
                0.292929293,
                0.297979798,
                0.303030303,
                0.308080808,
                0.313131313,
                0.318181818,
                0.323232323,
                0.328282828,
                0.333333333,
                0.338383838,
                0.343434343,
                0.348484848,
                0.353535354,
                0.358585859,
                0.363636364,
                0.368686869,
                0.373737374,
                0.378787879,
                0.383838384,
                0.388888889,
                0.393939394,
                0.398989899,
                0.404040404,
                0.409090909,
                0.414141414,
                0.419191919,
                0.424242424,
                0.429292929,
                0.434343434,
                0.439393939,
                0.444444444,
                0.449494949,
                0.454545455,
                0.45959596,
                0.464646465,
                0.46969697,
                0.474747475,
                0.47979798,
                0.484848485,
                0.48989899,
                0.494949495,
                0.5,
            ]
        )
        y_array = np.zeros_like(x_array)
        f_x_ref, f_y_ref = self.nfw_ref.derivatives(x_array, y_array, Rs, alpha_Rs)
        f_x, f_y = NFW.derivatives(x_array, y_array, Rs, alpha_Rs)

        npt.assert_array_almost_equal(f_x_ref, f_x, decimal=8)
        npt.assert_array_almost_equal(f_y_ref, f_y, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        rho0 = 1
        alpha_Rs = NFW.rho02alpha(rho0, Rs)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.nfw_ref.hessian(
            x, y, Rs, alpha_Rs
        )
        f_xx, f_xy, f_yx, f_yy = NFW.hessian(x, y, Rs, alpha_Rs)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.nfw_ref.hessian(
            x, y, Rs, alpha_Rs
        )
        f_xx, f_xy, f_yx, f_yy = NFW.hessian(x, y, Rs, alpha_Rs)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yx, f_yx_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        m_3d_ref = self.nfw_ref.mass_3d_lens(R, Rs, alpha_Rs)
        m_3d = NFW.mass_3d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_3d, m_3d_ref, decimal=8)

    def test_density_lens(self):
        R = 1.3
        Rs = 3.2
        alpha_Rs = 1.1
        density_ref = self.nfw_ref.density_lens(R, Rs, alpha_Rs)
        density = NFW.density_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(density, density_ref, decimal=8)

        R = 2.2
        Rs = 2.2
        alpha_Rs = 1.3
        density_ref = self.nfw_ref.density_lens(R, Rs, alpha_Rs)
        density = NFW.density_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(density, density_ref, decimal=8)

        R = 4.3
        Rs = 3.2
        alpha_Rs = 1.1
        density_ref = self.nfw_ref.density_lens(R, Rs, alpha_Rs)
        density = NFW.density_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_mass_2d_lens(self):
        R = 4.3
        Rs = 1.2
        alpha_Rs = 1.3
        m_2d_ref = self.nfw_ref.mass_2d_lens(R, Rs, alpha_Rs)
        m_2d = NFW.mass_2d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

        R = 2.3
        Rs = 2.3
        alpha_Rs = 1.5
        m_2d_ref = self.nfw_ref.mass_2d_lens(R, Rs, alpha_Rs)
        m_2d = NFW.mass_2d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)

        R = 1.3
        Rs = 4.2
        alpha_Rs = 1.1
        m_2d_ref = self.nfw_ref.mass_2d_lens(R, Rs, alpha_Rs)
        m_2d = NFW.mass_2d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_2d, m_2d_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
