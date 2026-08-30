#!/usr/bin/env python3
"""Validate the eta*J^2 Joule source of the electron energy equation with two
independent measurements of the resistivity from the force-free run:

1. FIELD DECAY (primary): the force-free mode decays resistively,
       E_B(t) = E_B(0) exp(-2 t / tau_R),   tau_R = mu0 / (eta k^2),
   so eta = mu0 * rate / (2 k^2) from the FieldEnergy reduced diagnostic.
   This measures the Ohm's-law friction directly and is immune to PIC-noise
   heating of T_e.

2. SOURCE RESPONSE (secondary): for the filtered ideal-gas case, the raw
   temperature ramp remains a clean independent signature.  For a nonlinear
   caloric EOS the filter must be disabled, and pressure work can exchange
   energy between the kinetic ions and the electrons.  Removing that internal
   exchange gives
       Delta E_J = Delta E_e + Delta E_ion
                 = integral eta J(t)^2 dV dt.
   Evaluating Delta E_e with the configured caloric EOS and fitting Delta E_J
   for eta checks that the nonlinear source update uses the same resistivity
   without mistaking reversible ion-electron work for Joule heating.

The figure additionally shows the cumulative energy budget: the electron
thermal gain Delta E_e tracks the magnetic-field loss Delta E_B plus the
(small, shot-noise driven) ion kinetic drain Delta E_ion, with the total
conserved.

PASS if the field-decay fit is within --tol-field and either the ideal-gas
temperature fit is within --tol-te or the nonlinear caloric-energy fit is
within --tol-energy of the input resistivity.
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
N0, B0, LX, TE0_EV = 2.0e20, 0.1, 0.5, 500.0
GAMMA = 5.0 / 3.0
KWAVE = 2.0 * np.pi / LX
J0 = KWAVE * B0 / MU0


def latent_fraction(temperature_eV, transition_temperature_eV, sharpness):
    ratio = np.asarray(temperature_eV, dtype=float) / transition_temperature_eV
    return ratio**sharpness / (1.0 + ratio**sharpness)


def caloric_energy_eV(
    temperature_eV, latent_energy_eV, transition_temperature_eV, sharpness
):
    """Electron internal energy in eV per represented electron."""
    return np.asarray(temperature_eV, dtype=float) / (GAMMA - 1.0) + (
        latent_energy_eV
        * latent_fraction(temperature_eV, transition_temperature_eV, sharpness)
    )


def temperature_from_caloric_energy_eV(
    energy_eV, latent_energy_eV, transition_temperature_eV, sharpness
):
    """Independent monotone bisection of the manufactured caloric EOS."""
    energy = np.asarray(energy_eV, dtype=float)
    lower = np.zeros_like(energy)
    upper = np.maximum((GAMMA - 1.0) * energy, TE0_EV)
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        middle_energy = caloric_energy_eV(
            middle, latent_energy_eV, transition_temperature_eV, sharpness
        )
        lower = np.where(middle_energy < energy, middle, lower)
        upper = np.where(middle_energy < energy, upper, middle)
    return 0.5 * (lower + upper)


def read_te_series(diag_dir, latent_energy_eV, transition_temperature_eV, sharpness):
    """Read the openPMD field diags.

    Returns (t[s], density-weighted <Te>[eV], electron thermal energy E_e[J])
    with E_e = kB/(gamma-1) * sum(n_e Te dV) integrated over the domain
    (per meter in the ignored y direction, consistent with the reduced
    diagnostics in 2D).
    """
    ts = OpenPMDTimeSeries(str(diag_dir))
    t = np.asarray(ts.t, dtype=float)
    Te_m, E_e = [], []
    volume = None
    for it in ts.iterations:
        Te, info = ts.get_field("Te", iteration=it)
        rho, _ = ts.get_field("rho", iteration=it)
        Te = np.asarray(Te, dtype=float)  # K
        ne = np.asarray(rho, dtype=float) / Q_E  # m^-3
        Te_m.append(float(np.sum(Te * ne) / np.sum(ne)) / K_PER_EV)
        dV = info.dx * info.dz  # (x 1 m in y)
        if volume is None:
            volume = dV * Te.size
        epsilon_eV = caloric_energy_eV(
            Te / K_PER_EV,
            latent_energy_eV,
            transition_temperature_eV,
            sharpness,
        )
        E_e.append(Q_E * float(np.sum(ne * epsilon_eV)) * dV)
    return t, np.asarray(Te_m), np.asarray(E_e), volume


def eta_from_field_decay(t, E_B):
    """eta from the exponential decay of the magnetic field energy."""
    # E_B ~ exp(-2t/tau_R): linear fit of log E_B.
    rate = -np.polyfit(t, np.log(E_B), 1)[0]  # = 2/tau_R
    return MU0 * rate / (2.0 * KWAVE**2)


def model_dTe(t, eta, latent_energy_eV, transition_temperature_eV, sharpness):
    """Joule Te ramp [eV] after the nonlinear caloric inversion."""
    tau = MU0 / (eta * KWAVE**2)
    energy_gain_eV = (
        eta * J0**2 / (N0 * Q_E) * (tau / 2.0) * (1.0 - np.exp(-2.0 * t / tau))
    )
    target_energy_eV = (
        caloric_energy_eV(
            TE0_EV, latent_energy_eV, transition_temperature_eV, sharpness
        )
        + energy_gain_eV
    )
    return (
        temperature_from_caloric_energy_eV(
            target_energy_eV,
            latent_energy_eV,
            transition_temperature_eV,
            sharpness,
        )
        - TE0_EV
    )


def model_joule_energy(t, eta, volume):
    """Domain-integrated resistive energy transfer [J]."""
    tau = MU0 / (eta * KWAVE**2)
    energy_gain_eV = (
        eta * J0**2 / (N0 * Q_E) * (tau / 2.0) * (1.0 - np.exp(-2.0 * t / tau))
    )
    return N0 * volume * Q_E * energy_gain_eV


def eta_from_te_ramp(
    t,
    dTe,
    eta_input,
    latent_energy_eV,
    transition_temperature_eV,
    sharpness,
):
    """1-parameter least-squares fit of eta (coarse scan)."""
    grid = np.linspace(0.1 * eta_input, 3.0 * eta_input, 4001)
    ssr = [
        np.sum(
            (
                model_dTe(
                    t,
                    e,
                    latent_energy_eV,
                    transition_temperature_eV,
                    sharpness,
                )
                - dTe
            )
            ** 2
        )
        for e in grid
    ]
    return grid[int(np.argmin(ssr))]


def eta_from_caloric_budget(t, energy, eta_input, volume):
    """Fit eta to the conservative electron-plus-ion matter-energy gain."""
    grid = np.linspace(0.1 * eta_input, 3.0 * eta_input, 4001)
    ssr = [np.sum((model_joule_energy(t, e, volume) - energy) ** 2) for e in grid]
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
        help="allowed relative error on the ideal-gas Te-ramp eta fit",
    )
    ap.add_argument(
        "--tol-energy",
        type=float,
        default=0.05,
        help="allowed relative error on the caloric-energy eta fit",
    )
    ap.add_argument("--out", default="joule_check.png")
    ap.add_argument("--latent-energy-eV", type=float, default=0.0)
    ap.add_argument("--transition-temperature-eV", type=float, default=500.0)
    ap.add_argument("--latent-sharpness", type=float, default=6.0)
    args = ap.parse_args(argv)

    eta_input = 1.0e-5 * args.eta_scale

    # Reduced diagnostics: magnetic field and ion kinetic energies.
    fdata = np.loadtxt(Path(args.reduced_dir) / "field_energy.txt", skiprows=1)
    t_r, E_B = fdata[:, 1], fdata[:, 4]  # column 4 = B_lev0 (J)
    pdata = np.loadtxt(Path(args.reduced_dir) / "part_energy.txt", skiprows=1)
    E_ion = pdata[:, 2]  # column 2 = total (J)

    eta_B = eta_from_field_decay(t_r, E_B)
    err_B = abs(eta_B - eta_input) / eta_input

    t, Te, E_e, volume = read_te_series(
        args.diag_dir,
        args.latent_energy_eV,
        args.transition_temperature_eV,
        args.latent_sharpness,
    )
    # Skip the iteration-0 dump in the Te-ramp fit (fields are written
    # before the first current deposition, so J = 0 there).
    t_fit, dTe = t[1:] - t[1], Te[1:] - Te[1]
    eta_T = eta_from_te_ramp(
        t_fit,
        dTe,
        eta_input,
        args.latent_energy_eV,
        args.transition_temperature_eV,
        args.latent_sharpness,
    )
    err_T = abs(eta_T - eta_input) / eta_input

    # Cumulative energy budget (each series relative to its first sample).
    dE_e = E_e - E_e[0]
    dE_B = E_B - E_B[0]
    dE_i = E_ion - E_ion[0]
    dE_i_on_t = np.interp(t, t_r, dE_i)
    dE_joule = dE_e + dE_i_on_t
    eta_U = eta_from_caloric_budget(t, dE_joule, eta_input, volume)
    err_U = abs(eta_U - eta_input) / eta_input
    n_b = min(t_r.size, t.size)
    dE_tot = dE_e[:n_b] + dE_B[:n_b] + dE_i[:n_b]
    noncons = dE_tot[-1] / dE_e[n_b - 1] if dE_e[n_b - 1] != 0.0 else 0.0

    print("=" * 66)
    print("Force-free Joule-heating check, dUe/dt = eta J^2")
    print(
        "  caloric latent model: "
        f"L={args.latent_energy_eV:g} eV/e, "
        f"Ta={args.transition_temperature_eV:g} eV, "
        f"s={args.latent_sharpness:g}"
    )
    print(f"  eta (input)      = {eta_input:.4e} Ohm*m")
    print(
        f"  eta (field decay)= {eta_B:.4e} Ohm*m  "
        f"({100 * err_B:+.2f}%, tol {100 * args.tol_field:.1f}%)"
    )
    nonlinear_caloric = args.latent_energy_eV > 0.0
    if nonlinear_caloric:
        print(
            f"  eta (caloric E)  = {eta_U:.4e} Ohm*m  "
            f"({100 * err_U:+.2f}%, tol {100 * args.tol_energy:.1f}%)"
        )
        print(
            f"  eta (raw Te; diagnostic only) = {eta_T:.4e} Ohm*m ({100 * err_T:+.2f}%)"
        )
    else:
        print(
            f"  eta (Te ramp)    = {eta_T:.4e} Ohm*m  "
            f"({100 * err_T:+.2f}%, tol {100 * args.tol_te:.1f}%)"
        )
        print(
            f"  eta (caloric E; diagnostic only) = {eta_U:.4e} Ohm*m "
            f"({100 * err_U:+.2f}%)"
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
        tm * 1e6,
        model_dTe(
            tm,
            eta_input,
            args.latent_energy_eV,
            args.transition_temperature_eV,
            args.latent_sharpness,
        ),
        "-",
        lw=1.5,
        label=r"analytic, input $\eta$",
    )
    axT.plot(
        tm * 1e6,
        model_dTe(
            tm,
            eta_T,
            args.latent_energy_eV,
            args.transition_temperature_eV,
            args.latent_sharpness,
        ),
        "--",
        lw=1.2,
        label=rf"raw-$T_e$ fit, $\eta$ = {eta_T:.3e}",
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

    secondary_ok = (
        err_U <= args.tol_energy if nonlinear_caloric else err_T <= args.tol_te
    )
    ok = err_B <= args.tol_field and secondary_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
