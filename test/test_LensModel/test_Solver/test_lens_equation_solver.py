__author__ = "sibirrer"

import jax
import numpy.testing as npt
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

# NOTE: The jaxtronomy solver sometimes finds more solutions than the
#       lenstronomy solver for min_distance >= 0.05, not sure why



class TestLensEquationSolver(object):
    def setup_method(self):
        """

        :return:
        """
        pass

    def test_epl(self):
        lens_model_list = ["EPL"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        lensEquationSolver_ref = LensEquationSolver_ref(lensModel_ref)
        min_distance = 0.04
        search_window = 2.5
        gamma = 1.9
        kwargs_epl = {
            "theta_E": 1.0,
            "gamma": gamma,
            "e1": 0.2,
            "e2": -0.03,
            "center_x": 0.1,
            "center_y": -0.1,
        }

        kwargs_lens_list = [kwargs_epl]

        sourcePos_x = -0.13
        sourcePos_y = 0.15
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

        x_pos, y_pos = lensEquationSolver.findBrightImage(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            numImages=4,
            min_distance=min_distance,
            search_window=search_window,
            num_iter_max=100,
        )
        x_pos_ref, y_pos_ref = lensEquationSolver_ref.findBrightImage(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            numImages=4,
            min_distance=min_distance,
            search_window=search_window,
            num_iter_max=100,
        )
        npt.assert_array_almost_equal(x_pos, x_pos_ref, decimal=8)
        npt.assert_array_almost_equal(y_pos, y_pos_ref, decimal=8)

    def test_epl_analytical_solver(self):
        lens_model_list = ["EPL", "SHEAR"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        lensEquationSolver_ref = LensEquationSolver_ref(lensModel_ref)
        sourcePos_x = -0.11
        sourcePos_y = 0.21

        min_distance = 0.01
        search_window = 5
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
        kwargs_lens_list = [kwargs_epl, kwargs_shear]

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

        lens_model_list = ["EPL", "SHEAR", "CONVERGENCE"]
        lensModel = LensModel(lens_model_list)
        lensModel_ref = LensModel_ref(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        lensEquationSolver_ref = LensEquationSolver_ref(lensModel_ref)
        kwargs_convergence = {"kappa": 0.1}
        kwargs_lens_list = [kwargs_epl, kwargs_shear, kwargs_convergence]

        x_pos, y_pos = lensEquationSolver.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
            magnification_limit=0.01,
            solver="analytical",
        )
        x_pos_ref, y_pos_ref = lensEquationSolver_ref.image_position_from_source(
            sourcePos_x,
            sourcePos_y,
            kwargs_lens_list,
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

        npt.assert_raises(
            ValueError,
            lensEquationSolver.image_position_from_source,
            sourcePos_x,
            sourcePos_y,
            kwargs_lens,
            magnification_limit=0.01,
            solver="analytical",
        )


if __name__ == "__main__":
    pytest.main()
