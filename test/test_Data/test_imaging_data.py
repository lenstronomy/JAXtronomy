import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.imaging_data import ImageData as ImageData_ref
from jaxtronomy.Data.imaging_data import ImageData as ImageData


class Test_ImageData_noisemap(object):
    def setup_method(self):
        self.numPix = 10
        kwargs_data = {
            "image_data": np.zeros((self.numPix, self.numPix)),
            "noise_map": 1.1 * np.ones((self.numPix, self.numPix)),
        }
        self.Data = ImageData(**kwargs_data)
        self.Data_ref = ImageData_ref(**kwargs_data)

    def test_init(self):
        kwargs_data = {
            "image_data": np.zeros((self.numPix, self.numPix)),
            "noise_map": 1.1 * np.ones((self.numPix, self.numPix)),
            "likelihood_method": "incorrect",
        }
        npt.assert_raises(ValueError, ImageData, **kwargs_data)

    def test_log_likelihood(self):
        model = np.tile(np.array([0.3, -0.1, 0.4, 0.7, -0.9]), (self.numPix, 2))
        mask = np.tile(np.array([0, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data.log_likelihood(model, mask, additional_error_map)
        log_likelihood_ref = self.Data_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=6)

        model = np.tile(np.array([0.3, -0.1, 0.4, 0.7, -0.9]), (self.numPix, 2))
        mask = np.tile(np.array([1, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data.log_likelihood(model, mask, additional_error_map)
        log_likelihood_ref = self.Data_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=6)

    def test_update_data(self):
        npt.assert_raises(
            Exception,
            self.Data.update_data,
            image_data=np.ones((self.numPix, self.numPix)),
        )


class Test_ImageData_without_noisemap(object):
    def setup_method(self):
        self.numPix = 10
        kwargs_data = {
            "image_data": np.zeros((self.numPix, self.numPix)),
            "exposure_time": 2 * np.ones((self.numPix, self.numPix)),
            "background_rms": 1.103,
        }
        self.Data = ImageData(**kwargs_data)
        self.Data_ref = ImageData_ref(**kwargs_data)
        self.Data_interferometry = ImageData(
            likelihood_method="interferometry_natwt", **kwargs_data
        )
        self.Data_interferometry_ref = ImageData_ref(
            likelihood_method="interferometry_natwt", **kwargs_data
        )

    def test_log_likelihood(self):
        model = np.tile(np.array([0.3, -0.1, 0.4, 0.7, -0.9]), (self.numPix, 2))
        mask = np.tile(np.array([0, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data.log_likelihood(model, mask, additional_error_map)
        log_likelihood_ref = self.Data_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=6)

        model = np.tile(np.array([0.3, -0.1, 0.4, 0.7, -0.9]), (self.numPix, 2))
        mask = np.tile(np.array([1, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data.log_likelihood(model, mask, additional_error_map)
        log_likelihood_ref = self.Data_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=6)

    def test_log_likelihood_interferometry(self):
        x = np.tile(np.array([0.3, -0.1, 0.4, 0.7, -0.9]), (self.numPix, 2))
        model = [x, x]
        mask = np.tile(np.array([0, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data_interferometry.log_likelihood(
            model, mask, additional_error_map
        )
        log_likelihood_ref = self.Data_interferometry_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=5)

        mask = np.tile(np.array([1, 1]), (self.numPix, 5))
        additional_error_map = 0.1
        log_likelihood = self.Data_interferometry.log_likelihood(
            model, mask, additional_error_map
        )
        log_likelihood_ref = self.Data_interferometry_ref.log_likelihood(
            model, mask, additional_error_map
        )
        npt.assert_almost_equal(log_likelihood, log_likelihood_ref, decimal=5)

    def test_likelihood_method(self):
        assert self.Data.likelihood_method() == "diagonal"
        assert self.Data_interferometry.likelihood_method() == "interferometry_natwt"


if __name__ == "__main__":
    pytest.main()
