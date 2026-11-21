"""Check the vacuum fields and nested meshes after the first time step."""

import sys

import numpy as np
import yt

ds = yt.load(sys.argv[1])
assert ds.index.max_level == 2
np.testing.assert_allclose(ds.current_time.to_value("s"), 1.0e-12, rtol=2e-6, atol=0)
cells = ds.all_data()
np.testing.assert_array_equal(np.unique(cells["index", "grid_level"]), [0, 1, 2])
volume = cells["index", "cell_volume"].to_value("m**3").sum()
np.testing.assert_allclose(volume, 1.0, rtol=2e-6)
np.testing.assert_equal(cells["boxlib", "Bz"].to_ndarray(), 0)
