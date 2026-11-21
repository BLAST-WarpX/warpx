#!/usr/bin/env python3
"""RCYLINDER hybrid magnetic-diffusion HARSH-vacuum oracle (T1, AMReX GMRES).

Strong oracle (not "ran N steps, no NaN"):
  - Bt min/max finite at t=0 and t=end
  - initial |Bt| amplitude nonzero (was initialized)
  - final |Bt| amplitude < 1e-3 * initial  (>99.9% damped in 5 steps at
    eta=129 Ohm*m, dt=1 ns: the slowest cylindrical Bessel mode damps by
    ~1/(1 + dt*chi*(j'_{1,1}/R)^2) per step ~ 1/871, so it is annihilated)
  - paired ANALYTIC explicit-resistive CFL-violation analysis: at this eta and
    const_dt the explicit Faraday resistive CFL dt_explicit ~ mu0*dr^2/(2*eta)
    is ~1e-6 of const_dt, so an explicit Faraday step (implicit_mag_diffusion=0
    or eta_explicit_max uncapped) is wildly unstable. This is what the
    operator-split implicit mag-diff (eta_explicit_max=0) enables.
"""

import math
import os
import re
import sys

# Test parameters (fixed by inputs_test_rcylinder_hybrid_mag_diffusion_harsh); used for
# the analytic explicit-CFL-violation ratio. Edit here if the inputs change.
MU0 = 4.0 * math.pi * 1.0e-7
DR = 2.0e-2 / 64.0  # prob_hi / n_cell
ETA = 129.0  # plasma_resistivity (Ohm*m), uniform vacuum value
DT = 1.0e-9  # warpx.const_dt (s)
CFL_VIOLATION_MIN = 1.0e3  # explicit must be >1000x unstable to count as "harsh"


def component_minmax(diag_dir, component):
    """Return the selected Cell_H component's (min, max)."""
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as infile:
        text = infile.read()
    float_lines = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            try:
                values = [
                    float(value)
                    for value in line.strip().rstrip(",").split(",")
                    if value
                ]
            except ValueError:
                continue
            if len(values) >= 3:
                float_lines.append(values)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    minimum, maximum = float_lines[-2], float_lines[-1]
    return minimum[component], maximum[component]


def main():
    plotfiles = sorted(
        p for p in (__import__("glob").glob("diags/diag1*")) if os.path.isdir(p)
    )
    if len(plotfiles) < 2:
        print("FAILED: expected initial and final RCYLINDER plotfiles")
        return 1

    # RCYLINDER component order is Br, Bt, Bz (same as RZ); Bt is component 1.
    bt_min0, bt_max0 = component_minmax(plotfiles[0], 1)
    bt_min1, bt_max1 = component_minmax(plotfiles[-1], 1)
    amp0 = max(abs(bt_min0), abs(bt_max0))
    amp1 = max(abs(bt_min1), abs(bt_max1))

    # Analytic explicit-resistive CFL: a 1D explicit B-diffusion step is stable
    # only for dt < mu0*dr^2/(2*eta). The ratio const_dt / dt_explicit must be
    # huge for this to be a "harsh vacuum explicit cannot resolve" deck.
    dt_explicit = MU0 * DR * DR / (2.0 * ETA)
    cfl_ratio = DT / dt_explicit

    print(f"first={plotfiles[0]} last={plotfiles[-1]}")
    print(f"Bt range t=0   = [{bt_min0:.8e}, {bt_max0:.8e}] T")
    print(f"Bt range t=end = [{bt_min1:.8e}, {bt_max1:.8e}] T")
    print(
        f"Bt amplitude: init={amp0:.4e}  final={amp1:.4e}  damp_frac={amp1 / amp0:.4e}"
    )
    print(
        f"explicit resistive CFL: dt_explicit=mu0*dr^2/(2*eta)="
        f"{dt_explicit:.4e} s  const_dt/dt_explicit={cfl_ratio:.4e}"
    )

    if not all(math.isfinite(v) for v in (bt_min0, bt_max0, bt_min1, bt_max1)):
        print("FAILED: non-finite Bt value")
        return 1
    if amp0 < 1.0e-8:
        print("FAILED: initial Bt was not initialized")
        return 1
    if amp1 >= 1.0e-3 * amp0:
        print("FAILED: Bt did not damp by >99.9% in the harsh-vacuum deck")
        return 1
    if cfl_ratio < CFL_VIOLATION_MIN:
        print("FAILED: explicit resistive CFL not violated (not a harsh deck)")
        return 1

    print(
        "PASSED: RCYLINDER harsh-vacuum implicit mag-diff (finite, >99.9% "
        "damped, explicit CFL violated)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
