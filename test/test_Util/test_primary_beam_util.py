import lenstronomy.Util.primary_beam_util as util_ref
import jaxtronomy.Util.primary_beam_util as util

import numpy as np
import pytest
import numpy.testing as npt

def test_primary_beam_value_at_coords():

    x_pos = np.array([0, 5, 10, 11, 12.5, 13.3])
    y_pos = np.array([1, 2.2, 2.2, 2.2, 4.2, 4.2])
    np.random.seed(42)
    primary_beam = np.random.rand(20, 20)

    vals_ref = util_ref.primary_beam_value_at_coords(x_pos, y_pos, primary_beam, order=1)
    vals = util.primary_beam_value_at_coords(x_pos, y_pos, primary_beam, order=1)
    npt.assert_allclose(vals, vals_ref, atol=1e-12, rtol=1e-12)

    np.random.seed(420)
    primary_beam = np.random.rand(20, 20) + 100

    # NOTE: Default order is 3 in lenstronomy but JAX only supports order up to 1.
    # This can lead to some pretty big differences for random, highly discontinous arrays
    vals_ref = util_ref.primary_beam_value_at_coords(x_pos, y_pos, primary_beam)
    vals = util.primary_beam_value_at_coords(x_pos, y_pos, primary_beam)
    npt.assert_allclose(vals, vals_ref, atol=0.2, rtol=0.002)

    primary_beam = np.random.rand(20, 20) * 500
    vals_ref = util_ref.primary_beam_value_at_coords(x_pos, y_pos, primary_beam)
    vals = util.primary_beam_value_at_coords(x_pos, y_pos, primary_beam)
    npt.assert_allclose(vals, vals_ref, atol=40, rtol=0.6)


if __name__ == "__main__":
    pytest.main()
