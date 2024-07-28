__author__ = "sibirrer"

import jax
import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.Solver.lens_equation_solver import (
    LensEquationSolver as LensEquationSolver_ref,
)
from jaxtronomy.LensModel.Solver.lens_equation_solver import (
    LensEquationSolver,
    analytical_lens_model_support,
)
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from jaxtronomy.LensModel.lens_model import LensModel

jax.config.update("jax_enable_x64", True)
"""
NOTE: Trying to test the lens equation solver for profiles that have not yet been
jaxified can lead to errors, as the profiles will be automatically ported from
lenstronomy whose functions may make use of the numba compiler. The numba compiler
is not compatible with the jnp arrays used in the jaxified lens equation solver.
Thus, cross-tests between lenstronomy and jaxtronomy should only be done
using profiles that have been jaxified.
"""


class TestLensEquationSolver(object):
    def setup_method(self):
        """

        :return:
        """
        pass

    def test_epl(self):
        lens_model_list = ["EPL", "SHEAR", "CONVERGENCE"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        lensEquationSolver_ref = LensEquationSolver_ref(lensModel_ref)
        min_distance = 0.01
        search_window = 10
        gamma = 1.9
        kwargs_epl = {
            "theta_E": 1.0,
            "gamma": gamma,
            "e1": 0.2,
            "e2": -0.03,
            "center_x": 0.1,
            "center_y": -0.1,
        }

        gamma1, gamma2 = 0.1, 0.2
        kwargs_shear = {"gamma1": gamma1, "gamma2": gamma2}
        kwargs_convergence = {"kappa": 0.1}
        kwargs_lens_list = [kwargs_epl, kwargs_shear, kwargs_convergence]

        sourcePos_x = 0.22
        sourcePos_y = -0.146
        for i in range(2):
            x_pos, y_pos = lensEquationSolver.image_position_from_source(
                sourcePos_x,
                sourcePos_y,
                kwargs_lens_list,
                min_distance=min_distance,
                search_window=search_window,
                precision_limit=10 ** (-10),
                num_iter_max=100,
                initial_guess_cut=True,
                magnification_limit=0.01,
                verbose=True,
            )
            x_pos_ref, y_pos_ref = lensEquationSolver_ref.image_position_from_source(
                sourcePos_x,
                sourcePos_y,
                kwargs_lens_list,
                min_distance=min_distance,
                search_window=search_window,
                precision_limit=10 ** (-10),
                num_iter_max=100,
                initial_guess_cut=True,
                magnification_limit=0.01,
            )
            npt.assert_array_almost_equal(x_pos, x_pos_ref, decimal=8)
            npt.assert_array_almost_equal(y_pos, y_pos_ref, decimal=8)
            sourcePos_x += 3.14
            sourcePos_y -= -2.6
            search_window = 5

        x_pos, y_pos = lensEquationSolver.findBrightImage(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            numImages=4,
            min_distance=0.01,
            search_window=5,
        )
        x_pos_ref, y_pos_ref = lensEquationSolver_ref.findBrightImage(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            numImages=4,
            min_distance=0.01,
            search_window=5,
        )
        npt.assert_array_almost_equal(x_pos, x_pos_ref, decimal=8)
        npt.assert_array_almost_equal(y_pos, y_pos_ref, decimal=8)

        x_pos, y_pos = lensEquationSolver.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            min_distance=min_distance,
            search_window=search_window,
            precision_limit=10 ** (-10),
            num_iter_max=100,
            initial_guess_cut=True,
            magnification_limit=0.01,
            solver="analytical",
        )
        x_pos_ref, y_pos_ref = lensEquationSolver_ref.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            min_distance=min_distance,
            search_window=search_window,
            precision_limit=10 ** (-10),
            num_iter_max=100,
            initial_guess_cut=True,
            magnification_limit=0.01,
            solver="analytical",
        )
        npt.assert_array_almost_equal(x_pos, x_pos_ref, decimal=8)
        npt.assert_array_almost_equal(y_pos, y_pos_ref, decimal=8)

        assert analytical_lens_model_support(lens_model_list)

    def test_pjaffe(self):
        lens_model_list = ["PJAFFE"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        lensEquationSolver_ref = LensEquationSolver_ref(lensModel_ref)
        sourcePos_x = 0.17
        sourcePos_y = -0.28
        min_distance = 0.01
        search_window = 10
        kwargs_lens = [
            {"sigma0": 1.0, "Ra": 0.5, "Rs": 0.8, "center_x": -0.32, "center_y": 0.17}
        ]
        assert analytical_lens_model_support(lens_model_list) == False
        x_pos, y_pos = lensEquationSolver.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            min_distance=min_distance,
            search_window=search_window,
            precision_limit=10 ** (-10),
            num_iter_max=100,
        )
        x_pos_ref, y_pos_ref = lensEquationSolver_ref.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            min_distance=min_distance,
            search_window=search_window,
            precision_limit=10 ** (-10),
            num_iter_max=100,
        )
        npt.assert_array_almost_equal(x_pos, x_pos_ref, decimal=8)
        npt.assert_array_almost_equal(y_pos, y_pos_ref, decimal=8)


if __name__ == "__main__":
    pytest.main()
