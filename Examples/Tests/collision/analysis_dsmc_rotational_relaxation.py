#!/usr/bin/env python3
"""Analysis for test_3d_dsmc_rotational_relaxation.

Validates the Borgnakke-Larsen rotational internal-DOF exchange in the DSMC collision
operator. The gas starts at T_trans = 1000 K, T_rot = 0. Over collisions, energy moves
from translation into rotation (classical mode, zeta_rot = 2); energy conservation fixes
the equilibrium at T_eq = (3/5) * 1000 = 600 K.

Checks (device-portable, physics-based; no checksum):
  1. total energy (translational + rotational) is conserved,
  2. rotational energy has activated (T_rot rises well above zero),
  3. the translational temperature has dropped from its initial value.
"""

import openpmd_api as io

kB = 1.380649e-23
q_e = 1.602176634e-19
m = 6.689e-27  # D2 mass [kg]
kB_eV = 8.617333262e-5  # [eV/K]

series = io.Series("diags/diag1/openpmd_%T.h5", io.Access.read_only)
iterations = sorted(series.iterations)
assert len(iterations) >= 2, "need at least the first and last iteration"


def state(it):
    p = series.iterations[it].particles["D2"]
    w = p["weighting"][io.Mesh_Record_Component.SCALAR].load_chunk()
    mom = p["momentum"]
    ux = mom["x"].load_chunk()
    uy = mom["y"].load_chunk()
    uz = mom["z"].load_chunk()
    ux_si, uy_si, uz_si = mom["x"].unit_SI, mom["y"].unit_SI, mom["z"].unit_SI
    er_name = next((n for n in ("E_rot", "eRot", "erot") if n in p), None)
    assert er_name is not None, "E_rot component was not dumped"
    er_rc = p[er_name][io.Mesh_Record_Component.SCALAR]
    er = er_rc.load_chunk()
    er_si = er_rc.unit_SI if er_rc.unit_SI else 1.0
    series.flush()

    # velocities [m/s] (neutrals: SI momentum = m v)
    vx, vy, vz = ux * ux_si / m, uy * uy_si / m, uz * uz_si / m
    W = w.sum()
    mvx, mvy, mvz = (w * vx).sum() / W, (w * vy).sum() / W, (w * vz).sum() / W
    var = (w * ((vx - mvx) ** 2 + (vy - mvy) ** 2 + (vz - mvz) ** 2)).sum() / W
    T_trans = m * var / (3.0 * kB)

    # rotational energy stored in eV; <E_rot> = (zeta_rot/2) kB T_rot = kB T_rot (zeta=2)
    Erot_eV = (w * er).sum() / W * er_si
    T_rot = Erot_eV / kB_eV

    # total energy [J]: translational (0.5 m <v^2> per real particle) + rotational
    E_trans_J = 0.5 * m * (w * (vx * vx + vy * vy + vz * vz)).sum()
    E_rot_J = (w * er).sum() * er_si * q_e
    E_tot_J = E_trans_J + E_rot_J
    return T_trans, T_rot, E_tot_J


T_trans0, T_rot0, E0 = state(iterations[0])
T_trans1, T_rot1, E1 = state(iterations[-1])

print(
    f"initial:  T_trans = {T_trans0:8.1f} K   T_rot = {T_rot0:8.1f} K   E_tot = {E0:.6e} J"
)
print(
    f"final:    T_trans = {T_trans1:8.1f} K   T_rot = {T_rot1:8.1f} K   E_tot = {E1:.6e} J"
)

# 1. energy conservation
E_err = abs(E1 - E0) / E0
print(f"energy conservation error = {E_err:.3e}  (tol 2e-2)")
assert E_err < 2e-2

# 2. rotational activation (started at 0; equilibrium is 600 K, so a partial run clears 300 K)
print(f"final T_rot = {T_rot1:.1f} K  (must exceed 300 K)")
assert T_rot1 > 300.0

# 3. translational cooling toward equilibrium
print(f"T_trans dropped {T_trans0:.1f} -> {T_trans1:.1f} K  (must fall below 800 K)")
assert T_trans1 < 800.0 and T_trans1 < T_trans0

print(
    "PASS: rotational Borgnakke-Larsen relaxation conserves energy and relaxes correctly."
)
