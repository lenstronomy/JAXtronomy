from jaxtronomy.LightModel.Profiles.uniform import Uniform
from lenstronomy.LightModel.Profiles.uniform import Uniform as Uniform_ref

import numpy as np
import pytest
import numpy.testing as npt


class TestUniform(object):
    """Tests the Sersic methods."""

    def setup_method(self):
        self.uniform_ref = Uniform_ref()

    def test_function(self):
        x = np.array([1, 3, 4, 2, 7])
        y = np.array([2, 1.1, -2.4, 1.6, -3])
        values = Uniform.function(x, y, 3.3498)
        values_ref = self.uniform_ref.function(x, y, 3.3498)
        npt.assert_array_equal(values, values_ref)


if __name__ == "__main__":
    pytest.main()
