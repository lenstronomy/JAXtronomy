import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.image_noise import ImageNoise as ImageNoise_ref
from jaxtronomy.Data.image_noise import ImageNoise as ImageNoise


class Test_ImageNoise_noisemap(object):
    def setup_method(self):
        self.num_pix = 10
        kwargs_data = {
            "image_data": np.ones((self.num_pix, self.num_pix)),
            "noise_map": 1.1 * np.ones((self.num_pix, self.num_pix)),
        }
        self.Noise = ImageNoise(**kwargs_data)
        self.Noise_ref = ImageNoise_ref(**kwargs_data)

    def test_C_D_model(self):
        model = np.tile(np.linspace(-1, 1, self.num_pix), (self.num_pix, 1))
        c_d = self.Noise.C_D_model(model)
        c_d_ref = self.Noise_ref.C_D_model(model)
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=6)

        model = np.tile(np.linspace(-0.3, 1.3, self.num_pix), (self.num_pix, 1))
        c_d = self.Noise.C_D_model(model)
        c_d_ref = self.Noise_ref.C_D_model(model)
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=6)


class Test_ImageNoise_without_noisemap(object):

    def setup_method(self):
        self.num_pix = 10
        kwargs_data = {
            "image_data": np.ones((self.num_pix, self.num_pix)),
            "exposure_time": 2 * np.ones((self.num_pix, self.num_pix)),
            "background_rms": 1.103,
        }
        self.Noise = ImageNoise(**kwargs_data)
        self.Noise_ref = ImageNoise_ref(**kwargs_data)

    def test_init(self):
        image_data = np.ones((self.num_pix, self.num_pix))
        exposure_time = 2 * np.ones((self.num_pix, self.num_pix))
        noise_map = 2 * np.ones((self.num_pix, self.num_pix))
        background_rms = 1.103
        npt.assert_raises(ValueError, ImageNoise, image_data=image_data)
        npt.assert_raises(
            ValueError,
            ImageNoise,
            image_data=image_data,
            exposure_time=exposure_time,
        )
        npt.assert_raises(
            ValueError,
            ImageNoise,
            image_data=image_data,
            noise_map=noise_map,
            gradient_boost_factor=3,
        )
        test_verbose = ImageNoise(
            image_data=image_data,
            exposure_time=exposure_time,
            background_rms=background_rms / 100,
            verbose=True,
        )

    def test_C_D_model(self):
        model = np.tile(np.linspace(-1, 1, self.num_pix), (self.num_pix, 1))
        c_d = self.Noise.C_D_model(model)
        c_d_ref = self.Noise_ref.C_D_model(model)
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=6)

        model = np.tile(np.linspace(-0.3, 1.3, self.num_pix), (self.num_pix, 1))
        c_d = self.Noise.C_D_model(model)
        c_d_ref = self.Noise_ref.C_D_model(model)
        npt.assert_array_almost_equal(c_d, c_d_ref, decimal=6)


def test_big_endian_image_data():
    """FITS files store data in big-endian (>f8) byte order which JAX rejects unless it
    is converted to native byte order first."""
    num_pix = 10
    native = np.ones((num_pix, num_pix), dtype=np.float64)
    big_endian = native.astype(">f8")

    # noise-map path: image_data and noise_map both big-endian
    noise = ImageNoise(
        image_data=big_endian,
        noise_map=big_endian.copy(),
    )
    npt.assert_array_almost_equal(noise.data, native, decimal=6)

    # exposure-map path: big-endian exposure map flows through jnp.where
    noise2 = ImageNoise(
        image_data=big_endian,
        exposure_time=big_endian.copy(),
        background_rms=1.103,
    )
    npt.assert_array_almost_equal(noise2.data, native, decimal=6)


if __name__ == "__main__":
    pytest.main()
