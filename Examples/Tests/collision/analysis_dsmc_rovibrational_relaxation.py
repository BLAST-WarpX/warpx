#!/usr/bin/env python3
"""Analysis for test_3d_dsmc_rovibrational_relaxation.

Validates the Borgnakke-Larsen rotational (classical) + vibrational (quantum-harmonic) exchange.
The gas starts at T_trans = 6000 K with rotation and vibration cold. At near-equilibrium:
  - translation and rotation equipartition to a common T_eq (~3050 K),
  - vibration activates but remains sub-equipartition (<E_vib> < kB T_eq), the signature of the
    quantum-harmonic treatment (a classical vibrational mode would reach <E_vib> = kB T_eq).

Checks (device-portable, physics-based; no checksum):
  1. total energy (translational + rotational + vibrational) is conserved,
  2. translation and rotation have equipartitioned (T_trans ~ T_rot),
  3. vibration has activated (<E_vib> rises from 0),
  4. vibration is quantum-suppressed (<E_vib> < kB T_trans_final).
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
    sx, sy, sz = mom["x"].unit_SI, mom["y"].unit_SI, mom["z"].unit_SI

    def rt(name_options):
        nm = next((n for n in name_options if n in p), None)
        assert nm is not None, f"{name_options} component was not dumped"
        rc = p[nm][io.Mesh_Record_Component.SCALAR]
        return rc.load_chunk(), (rc.unit_SI if rc.unit_SI else 1.0)

    er, er_si = rt(("E_rot", "eRot", "erot"))
    ev, ev_si = rt(("E_vib", "eVib", "evib"))
    series.flush()

    vx, vy, vz = ux * sx / m, uy * sy / m, uz * sz / m
    W = w.sum()
    mvx, mvy, mvz = (w * vx).sum() / W, (w * vy).sum() / W, (w * vz).sum() / W
    var = (w * ((vx - mvx) ** 2 + (vy - mvy) ** 2 + (vz - mvz) ** 2)).sum() / W
    T_trans = m * var / (3.0 * kB)
    Erot_eV = (w * er).sum() / W * er_si
    T_rot = Erot_eV / kB_eV  # <E_rot> = kB T_rot for zeta_rot = 2
    Evib_eV = (w * ev).sum() / W * ev_si

    E_trans_J = 0.5 * m * (w * (vx * vx + vy * vy + vz * vz)).sum()
    E_rot_J = (w * er).sum() * er_si * q_e
    E_vib_J = (w * ev).sum() * ev_si * q_e
    E_tot_J = E_trans_J + E_rot_J + E_vib_J
    return T_trans, T_rot, Evib_eV, E_tot_J


T_trans0, T_rot0, Evib0, E0 = state(iterations[0])
T_trans1, T_rot1, Evib1, E1 = state(iterations[-1])

print(
    f"initial:  T_trans={T_trans0:8.1f} K  T_rot={T_rot0:8.1f} K  <E_vib>={Evib0:.4f} eV  E={E0:.6e} J"
)
print(
    f"final:    T_trans={T_trans1:8.1f} K  T_rot={T_rot1:8.1f} K  <E_vib>={Evib1:.4f} eV  E={E1:.6e} J"
)

# 1. total energy conservation
E_err = abs(E1 - E0) / E0
print(f"energy conservation error = {E_err:.3e}  (tol 2e-2)")
assert E_err < 2e-2

# 2. translation/rotation equipartition
equi = abs(T_trans1 - T_rot1) / T_trans1
print(f"|T_trans - T_rot|/T_trans = {equi:.3f}  (tol 0.12)")
assert equi < 0.12

# 3. vibrational activation
print(f"<E_vib>: {Evib0:.4f} -> {Evib1:.4f} eV  (must rise above 0.02 eV)")
assert Evib1 > 0.02 and Evib1 > Evib0

# 4. quantum suppression: classical equipartition would give <E_vib> = kB T; require well below it
Evib_classical = kB_eV * T_trans1  # eV
print(
    f"<E_vib>={Evib1:.4f} eV  vs classical kB*T={Evib_classical:.4f} eV  (must be sub-equipartition)"
)
assert Evib1 < 0.8 * Evib_classical

print(
    "PASS: rovibrational Borgnakke-Larsen conserves energy, equipartitions rot, and keeps vib quantum."
)
