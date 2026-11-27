#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Thin/thick RZ interface handoff, zero-residence exclusion and reemission."""

import argparse
import json
from pathlib import Path

import numpy as np
from analysis_precision import add_precision_arguments, precision_dtypes
from read_raw_data import _read_buffer

parser = argparse.ArgumentParser()
add_precision_arguments(parser)
parser.add_argument(
    "--case", choices=("entry", "departure", "reemission", "tiling"), required=True
)
args = parser.parse_args()
_, _, dtype = precision_dtypes(args)
plotfile = Path("diags/diag000001")
with (plotfile / "Header").open() as header:
    header.readline()
    names = [header.readline().strip() for _ in range(int(header.readline()))]
fields = _read_buffer(str(plotfile), str(plotfile / "Level_0/Cell_H"), names)
material = np.asarray(fields["radiation_material_energy"]).reshape(4, 4)
diffusion = np.asarray(fields["radiation_diffusion_energy"]).reshape(4, 4)
particles = np.atleast_2d(np.loadtxt("diags/particle_energy.txt"))
radiation = np.atleast_2d(np.loadtxt("diags/radiation_energy.txt"))
initial = radiation[0, 2]  # total radiation, already streaming + diffusion
np.testing.assert_allclose(initial, radiation[0, 3] + radiation[0, 4], rtol=1e-12)
expected_material = np.zeros((4, 4))
expected_diffusion = np.zeros((4, 4))
expected_streaming = 0.0
if args.case == "entry":
    # Circle radius .5, y=.3 => x=.4. Unit ray nx=.6 starts at x=.35.
    entry_distance = (0.4 - 0.35) / 0.6
    transmission = np.exp(-2 * entry_distance)
    expected_material[1, 0] = 1 - transmission
    expected_diffusion[2, 0] = transmission
elif args.case == "tiling":
    # Half the packets traverse .05 m in the thin cell; half start in the
    # receiving thick cell. Distinct CPU particle tiles share that recipient.
    expected_material[1, 0] = 0.5 * -np.expm1(-0.1)
    expected_diffusion[2, 0] = 1 - expected_material[1, 0]
elif args.case == "departure":
    # Starts on r=.5 directed toward the thin side: no residence in the thick cell.
    expected_streaming = np.exp(-2 * 0.1)
    expected_material[1, 0] = 1 - expected_streaming
else:
    np.testing.assert_allclose(initial, 100 * np.pi, rtol=3e-6)
    expected_streaming = 1.0

report = {
    "case": args.case,
    "initial_energy_J": float(initial),
    "material_fraction": (material / initial).tolist(),
    "diffusion_fraction": (diffusion / initial).tolist(),
    "streaming_fraction": float(particles[-1, 2] / initial),
}
Path("exact_rz_conversion.json").write_text(json.dumps(report, indent=2) + "\n")
rtol = 3.0e-5 if dtype == np.float32 else 8.0e-13
atol = 3.0e-6 if dtype == np.float32 else 2.0e-12
np.testing.assert_allclose(material / initial, expected_material, rtol=rtol, atol=atol)
np.testing.assert_allclose(
    diffusion / initial, expected_diffusion, rtol=rtol, atol=atol
)
np.testing.assert_allclose(
    particles[-1, 2] / initial, expected_streaming, rtol=rtol, atol=atol
)
np.testing.assert_allclose(
    (np.sum(material) + np.sum(diffusion) + particles[-1, 2]) / initial,
    1.0,
    rtol=rtol,
    atol=atol,
)
print(f"Face-exact RZ packet/diffusion {args.case}: energy and cell ownership PASS")
