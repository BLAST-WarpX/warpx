#!/usr/bin/env python3
"""Validate the eta*J^2 Joule source of the electron energy equation with two
independent measurements of the resistivity from the force-free run:

1. FIELD DECAY (primary): the force-free mode decays resistively,
       E_B(t) = E_B(0) exp(-2 t / tau_R),   tau_R = mu0 / (eta k^2),
   so eta = mu0 * rate / (2 k^2) from the FieldEnergy reduced diagnostic.
   This measures the Ohm's-law friction directly and is immune to PIC-noise
   heating of T_e.

2. Te RAMP (secondary): the Joule source gives
       dTe(t) = (gamma-1) eta J0^2/(n0 kB) * (tau_R/2)(1 - e^{-2t/tau_R}),
   fitted for eta by a 1-parameter least-squares scan.  This checks that the
   heating deposited into T_e uses the same eta.  It sits on a small
   ion-current shot-noise heating floor (~1/N_ppc), hence the looser
   tolerance.

The figure additionally shows the cumulative energy budget: the electron
thermal gain Delta E_e tracks the magnetic-field loss Delta E_B plus the
(small, shot-noise driven) ion kinetic drain Delta E_ion, with the total
conserved.

PASS if the field-decay fit is within --tol-field and the Te fit within
--tol-te of the input resistivity.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

Q_E = 1.602176634e-19
K_B = 1.380649e-23
MU0 = 4.0e-7 * np.pi
K_PER_EV = Q_E / K_B

# Must match the input deck.
N0, B0, LX = 2.0e20, 0.1, 0.5
GAMMA = 5.0 / 3.0
KWAVE = 2.0 * np.pi / LX
J0 = KWAVE * B0 / MU0


def read_te_series(diag_dir):
    """Read the openPMD field diags.

    Returns (t[s], density-weighted <Te>[eV], electron thermal energy E_e[J])
    with E_e = kB/(gamma-1) * sum(n_e Te dV) integrated over the domain
    (per meter in the ignored y direction, consistent with the reduced
    diagnostics in 2D).
    """
    ts = OpenPMDTimeSeries(str(diag_dir))
    t = np.asarray(ts.t, dtype=float)
    Te_m, E_e = [], []
    for it in ts.iterations:
        Te, info = ts.get_field("Te", iteration=it)
        rho, _ = ts.get_field("rho", iteration=it)
        Te = np.asarray(Te, dtype=float)  # K
        ne = np.asarray(rho, dtype=float) / Q_E  # m^-3
        Te_m.append(float(np.sum(Te * ne) / np.sum(ne)) / K_PER_EV)
        dV = info.dx * info.dz  # (x 1 m in y)
        E_e.append(K_B / (GAMMA - 1.0) * float(np.sum(ne * Te)) * dV)
    return t, np.asarray(Te_m), np.asarray(E_e)


def eta_from_field_decay(t, E_B):
    """eta from the exponential decay of the magnetic field energy."""
    # E_B ~ exp(-2t/tau_R): linear fit of log E_B.
    rate = -np.polyfit(t, np.log(E_B), 1)[0]  # = 2/tau_R
    return MU0 * rate / (2.0 * KWAVE**2)


def model_dTe(t, eta):
    """Joule Te ramp [eV] with the resistive J decay folded in."""
    tau = MU0 / (eta * KWAVE**2)
    pref = (GAMMA - 1.0) * eta * J0**2 / (N0 * K_B)  # K/s at t=0
    return pref * (tau / 2.0) * (1.0 - np.exp(-2.0 * t / tau)) / K_PER_EV


def eta_from_te_ramp(t, dTe, eta_input):
    """1-parameter least-squares fit of eta (coarse scan)."""
    grid = np.linspace(0.1 * eta_input, 3.0 * eta_input, 4001)
    ssr = [np.sum((model_dTe(t, e) - dTe) ** 2) for e in grid]
    return grid[int(np.argmin(ssr))]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--diag-dir", default="diags/field_diags")
    ap.add_argument("--reduced-dir", default="diags")
    ap.add_argument(
        "--eta-scale",
        type=float,
        default=1.0,
        help="multiplier on the base resistivity eta=1e-5; must match the run",
    )
    ap.add_argument(
        "--tol-field",
        type=float,
        default=0.05,
        help="allowed relative error on the field-decay eta fit",
    )
    ap.add_argument(
        "--tol-te",
        type=float,
        default=0.20,
        help="allowed relative error on the Te-ramp eta fit",
    )
    ap.add_argument("--out", default="joule_check.png")
    args = ap.parse_args(argv)

    eta_input = 1.0e-5 * args.eta_scale

    # Reduced diagnostics: magnetic field and ion kinetic energies.
    fdata = np.loadtxt(Path(args.reduced_dir) / "field_energy.txt", skiprows=1)
    t_r, E_B = fdata[:, 1], fdata[:, 4]  # column 4 = B_lev0 (J)
    pdata = np.loadtxt(Path(args.reduced_dir) / "part_energy.txt", skiprows=1)
    E_ion = pdata[:, 2]  # column 2 = total (J)

    eta_B = eta_from_field_decay(t_r, E_B)
    err_B = abs(eta_B - eta_input) / eta_input

    t, Te, E_e = read_te_series(args.diag_dir)
    # Skip the iteration-0 dump in the Te-ramp fit (fields are written
    # before the first current deposition, so J = 0 there).
    t_fit, dTe = t[1:] - t[1], Te[1:] - Te[1]
    eta_T = eta_from_te_ramp(t_fit, dTe, eta_input)
    err_T = abs(eta_T - eta_input) / eta_input

    # Cumulative energy budget (each series relative to its first sample).
    dE_e = E_e - E_e[0]
    dE_B = E_B - E_B[0]
    dE_i = E_ion - E_ion[0]
    n_b = min(t_r.size, t.size)
    dE_tot = dE_e[:n_b] + dE_B[:n_b] + dE_i[:n_b]
    noncons = dE_tot[-1] / dE_e[n_b - 1] if dE_e[n_b - 1] != 0.0 else 0.0

    print("=" * 66)
    print("Force-free Joule-heating check, dTe/dt = (gamma-1) eta J^2/(n kB)")
    print(f"  eta (input)      = {eta_input:.4e} Ohm*m")
    print(
        f"  eta (field decay)= {eta_B:.4e} Ohm*m  "
        f"({100 * err_B:+.2f}%, tol {100 * args.tol_field:.1f}%)"
    )
    print(
        f"  eta (Te ramp)    = {eta_T:.4e} Ohm*m  "
        f"({100 * err_T:+.2f}%, tol {100 * args.tol_te:.1f}%)"
    )
    print(
        f"  energy budget: dE_e = {dE_e[n_b - 1]:+.3f} J, dE_B = {dE_B[n_b - 1]:+.3f} J, "
        f"dE_ion = {dE_i[n_b - 1]:+.3f} J"
    )
    print(f"  final non-conservation    = {100 * noncons:+.2f}% of dE_e")
    print("=" * 66)

    fig, (axE, axT) = plt.subplots(1, 2, figsize=(12, 4.6))

    tus_r = t_r * 1e6
    axE.plot(t * 1e6, dE_e, "o-", ms=4, label=r"$\Delta E_e$ (electron thermal)")
    axE.plot(tus_r, dE_B, "s-", ms=4, label=r"$\Delta E_B$ (magnetic)")
    axE.plot(tus_r, dE_i, "^-", ms=4, label=r"$\Delta E_{ion}$")
    axE.plot(tus_r[:n_b], dE_tot, "k-", lw=2.5, label=r"$\Delta E_{tot}$ (should be 0)")
    axE.axhline(0.0, color="gray", lw=0.8, ls=":")
    axE.set_xlabel(r"time ($\mu$s)")
    axE.set_ylabel("cumulative energy change (J)")
    axE.set_title(
        f"energy budget (non-conservation {100 * noncons:+.2f}% of "
        r"$\Delta E_e$)"
    )
    axE.legend(fontsize=9)
    axE.grid(alpha=0.3)

    tm = np.linspace(0.0, t_fit[-1], 200)
    axT.plot(t_fit * 1e6, dTe, "o", ms=5, label="measured")
    axT.plot(
        tm * 1e6, model_dTe(tm, eta_input), "-", lw=1.5, label=r"analytic, input $\eta$"
    )
    axT.plot(
        tm * 1e6,
        model_dTe(tm, eta_T),
        "--",
        lw=1.2,
        label=rf"fit, $\eta$ = {eta_T:.3e}",
    )
    axT.set_xlabel(r"time ($\mu$s)")
    axT.set_ylabel(r"$\Delta\langle T_e\rangle_n$ (eV)")
    axT.set_title("electron temperature ramp")
    axT.legend(fontsize=9)
    axT.grid(alpha=0.3)

    fig.suptitle("Joule heating of the force-free equilibrium")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

    ok = err_B <= args.tol_field and err_T <= args.tol_te
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
