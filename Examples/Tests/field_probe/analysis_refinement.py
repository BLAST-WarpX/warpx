"""Every probe survives collection across levels and samples a uniform E field."""

from pathlib import Path

import numpy as np

for name, expected_x in [
    ("coarse_point", [0.75]),
    ("fine_point", [0]),
    ("crossing_line", np.linspace(-0.75, 0.75, 5)),
]:
    path = Path("diags/reducedfiles") / (name + ".txt")
    lines = path.read_text().splitlines()
    assert len(lines) > 1, f"Missing samples from {name}"
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    assert len(lines[0].split()) == data.shape[1] == 12
    assert np.isfinite(data).all()
    for step in [1, 2]:
        rows = data[data[:, 0] == step]
        assert len(rows) == len(expected_x), (name, step, rows)
        np.testing.assert_allclose(rows[:, 2], expected_x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(rows[:, 3:5], 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            rows[:, 5:8], np.tile([2, 3, 5], (len(rows), 1)), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(rows[:, 8:], 0, rtol=0, atol=1e-12)
