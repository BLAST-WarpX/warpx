#!/usr/bin/env python3
"""Analysis for test_2d_dsmc_rotational_flux.

Regression test for the internal-DOF runtime components under flux injection + resampling.
D2 carrying rotational energy (E_rot) is injected as a flux into a multi-box domain with
leveling-thinning resampling enabled. Previously the E_rot runtime component was dropped during
the flux-injection / migration Redistribute (yielding NaN), and resampling was blocked entirely
for internal-DOF species.

Checks:
  1. particles were actually injected,
  2. every E_rot value is finite (no NaN from a lost runtime component),
  3. the rotational temperature matches the 300 K injection temperature (energy carried intact).
"""

import numpy as np
import openpmd_api as io

kB_eV = 8.617333262e-5  # [eV/K]

series = io.Series("diags/diag1/openpmd_%T.h5", io.Access.read_only)
it = sorted(series.iterations)[-1]
p = series.iterations[it].particles["D2"]
w = p["weighting"][io.Mesh_Record_Component.SCALAR].load_chunk()
er_name = next((n for n in ("E_rot", "eRot", "erot") if n in p), None)
assert er_name is not None, "E_rot component was not dumped"
er_rc = p[er_name][io.Mesh_Record_Component.SCALAR]
er = er_rc.load_chunk()
er_si = er_rc.unit_SI if er_rc.unit_SI else 1.0
series.flush()

# 1. injection happened
print(f"injected macroparticles at final step: {len(w)}")
assert len(w) > 100

# 2. no NaN / inf in the runtime component (the core regression)
n_bad = np.count_nonzero(~np.isfinite(er))
print(f"non-finite E_rot values: {n_bad}")
assert n_bad == 0, (
    "E_rot has non-finite values -> runtime component lost in Redistribute"
)

# 3. rotational temperature matches the injection temperature (<E_rot> = kB T_rot for zeta_rot = 2)
W = w.sum()
T_rot = (w * er).sum() / W * er_si / kB_eV
print(f"injected rotational temperature: {T_rot:.1f} K  (expected ~300 K)")
assert 220.0 < T_rot < 380.0, (
    "rotational energy not carried through injection/resampling intact"
)

print(
    "PASS: E_rot survives flux injection, box migration, and leveling-thinning resampling."
)
