#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native oblique transport must preserve charge/energy without populating vacuum."""

import sys
from pathlib import Path

import numpy as np
import yt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radiation_transport"))
from analysis_moving_moment_interface import nodes  # noqa: E402

yt.set_log_level(50)
energies = []
support = []
for step in (0, 2):
    plot = Path(f"diags/plt{step:06d}")
    ds = yt.load(str(plot))
    particles = ds.all_data()
    assert particles["ions", "particle_weight"].size == 1
    total_charge = float(particles["ions", "particle_weight"][0]) * 1.602176634e-19
    rho = nodes(plot / "raw_fields/Level_0/rho_fp_H", [16] * 3, periodic=True)
    temperature = nodes(
        Path(f"diags/chk{step:06d}/Level_0/hybrid_electron_temperature_fp[level=0]_H"),
        [16] * 3,
        periodic=True,
    )
    assert np.isfinite(rho).all() and np.min(rho) >= 0
    assert np.isfinite(temperature).all() and np.min(temperature[rho > 0]) > 0
    np.testing.assert_allclose(
        np.sum(rho) * (0.0076 / 16) ** 3, total_charge, rtol=1.0e-12
    )
    support.append(rho > 0)
    # Ideal-gas U is proportional to rho*T; no gather, heating or physical source.
    energies.append(np.sum(rho * temperature))
assert np.any(support[0] & ~support[1]) and np.any(support[1] & ~support[0])
np.testing.assert_allclose(energies[1], energies[0], rtol=1.0e-10)
print(
    "Oblique vacuum crossing preserves charge, supported temperature and thermal energy"
)
