import pytest
import numpy.testing as npt
import numpy as np
import copy
from jax import config

from jaxtronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from jaxtronomy.PointSource.point_source import PointSource
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from lenstronomy.Sampling.Likelihoods.position_likelihood import (
    PositionLikelihood as PositionLikelihood_ref,
)
from lenstronomy.PointSource.point_source import PointSource as PointSource_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref

config.update("jax_enable_x64", True)


class TestPositionLikelihood(object):
    def setup_method(self):
        # compute image positions
        lensModel = LensModel(lens_model_list=["EPL", "SHEAR"])
        lensModel_ref = LensModel_ref(lens_model_list=["EPL", "SHEAR"])

        solver = LensEquationSolver(lensModel=lensModel)
        self._kwargs_lens = [
            {
                "theta_E": 1,
                "gamma": 1.6,
                "e1": 0.1,
                "e2": -0.03,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.1, "gamma2": 0.2},
        ]
        self.kwargs_lens_eqn_solver = {"min_distance": 0.1, "search_window": 10}
        x_pos, y_pos = solver.image_position_from_source(
            sourcePos_x=0.01,
            sourcePos_y=-0.01,
            kwargs_lens=self._kwargs_lens,
            **self.kwargs_lens_eqn_solver
        )
        self._x_pos, self._y_pos = x_pos, y_pos

        point_source_class = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel,
        )
        point_source_class_ref = PointSource_ref(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel_ref,
        )
        self.ps_class = point_source_class

        # Has only the EPL lens model
        point_source_class2 = PointSource(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel,
            index_lens_model_list=[[0], [1]],
            point_source_frame_list=[[0] * len(x_pos)],
        )
        point_source_class_ref2 = PointSource_ref(
            point_source_type_list=["LENSED_POSITION"],
            lens_model=lensModel_ref,
            index_lens_model_list=[[0], [1]],
            point_source_frame_list=[[0] * len(x_pos)],
        )

        self.likelihood = PositionLikelihood(
            point_source_class,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=False,
            restrict_image_number=False,
            max_num_images=None,
        )
        self.likelihood_ref = PositionLikelihood_ref(
            point_source_class_ref,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=False,
            restrict_image_number=False,
            max_num_images=None,
        )

        self.likelihood2 = PositionLikelihood(
            point_source_class2,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=False,
            restrict_image_number=False,
            max_num_images=None,
        )
        self.likelihood2_ref = PositionLikelihood_ref(
            point_source_class_ref2,
            image_position_uncertainty=0.005,
            astrometric_likelihood=True,
            image_position_likelihood=True,
            ra_image_list=[x_pos],
            dec_image_list=[y_pos],
            source_position_likelihood=True,
            source_position_tolerance=0.001,
            force_no_add_image=False,
            restrict_image_number=False,
            max_num_images=None,
        )

    def test_raises(self):
        npt.assert_warns(
            UserWarning,
            PositionLikelihood,
            self.ps_class,
            source_position_tolerance=0.001,
            source_position_likelihood=False,
        )
        npt.assert_raises(
            ValueError,
            PositionLikelihood,
            self.ps_class,
            restrict_image_number=True,
        )
        npt.assert_raises(
            ValueError,
            PositionLikelihood,
            self.ps_class,
            force_no_add_image=True,
        )

    def test_image_position_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        logL_ref = self.likelihood_ref.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        kwargs_ps = [{"ra_image": self._x_pos + 0.01, "dec_image": self._y_pos - 0.01}]
        logL = self.likelihood.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        logL_ref = self.likelihood_ref.image_position_likelihood(
            kwargs_ps, self._kwargs_lens, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

    def test_astrometric_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        kwargs_special = {
            "delta_x_image": [0, 0, 0, 0.0],
            "delta_y_image": [0, 0, 0, 0.0],
        }
        logL = self.likelihood.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        logL_ref = self.likelihood_ref.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        kwargs_special = {
            "delta_x_image": [-0.02, 0, 0, 0.01],
            "delta_y_image": [-0.02, 0, 0, 0.01],
        }
        logL = self.likelihood.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        logL_ref = self.likelihood_ref.astrometric_likelihood(
            kwargs_ps, kwargs_special, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        logL = self.likelihood.astrometric_likelihood(
            [{"not_ra_image": 0.01}], kwargs_special, sigma=0.01
        )
        logL_ref = self.likelihood_ref.astrometric_likelihood(
            [{"not_ra_image": 0.1}], kwargs_special, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        logL = self.likelihood.astrometric_likelihood([], kwargs_special, sigma=0.01)
        logL_ref = self.likelihood_ref.astrometric_likelihood(
            [], kwargs_special, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        logL = self.likelihood.astrometric_likelihood(kwargs_ps, {}, sigma=0.01)
        logL_ref = self.likelihood_ref.astrometric_likelihood(kwargs_ps, {}, sigma=0.01)
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

    def test_source_position_likelihood(self):
        kwargs_ps = [{"ra_image": self._x_pos, "dec_image": self._y_pos}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.0001,
            sigma=0.001,
            verbose=False,
        )
        logL_ref = self.likelihood_ref.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.0001,
            sigma=0.001,
            verbose=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        kwargs_ps = [{"ra_image": self._x_pos + 0.01, "dec_image": self._y_pos - 0.03}]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        logL_ref = self.likelihood_ref.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        kwargs_ps = [
            {
                "ra_image": self._x_pos + np.array([0.01, -0.32, 0, 0, -0.11]),
                "dec_image": self._y_pos,
            }
        ]
        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens, kwargs_ps, sigma=0.01
        )
        logL_ref = self.likelihood_ref.source_position_likelihood(
            self._kwargs_lens, kwargs_ps, sigma=0.01
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        logL = self.likelihood.source_position_likelihood(
            self._kwargs_lens,
            [],
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        logL_ref = self.likelihood_ref.source_position_likelihood(
            self._kwargs_lens,
            [],
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

        logL = self.likelihood2.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        logL_ref = self.likelihood2_ref.source_position_likelihood(
            self._kwargs_lens,
            kwargs_ps,
            hard_bound_rms=0.001,
            sigma=0.0001,
            verbose=False,
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)

    def test_logL(self):
        kwargs_ps = [
            {
                "ra_image": self._x_pos,
                "dec_image": self._y_pos + np.array([0.01, -0.32, 0.11, 0, 0]),
            }
        ]
        kwargs_special = {
            "delta_x_image": [0, 0, 0, 0.1],
            "delta_y_image": [0, 0, 0, 0.1],
        }
        logL = self.likelihood.logL(
            self._kwargs_lens, kwargs_ps, kwargs_special, verbose=True
        )
        logL_ref = self.likelihood_ref.logL(
            self._kwargs_lens, kwargs_ps, kwargs_special, verbose=True
        )
        npt.assert_allclose(logL, logL_ref, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    pytest.main()
