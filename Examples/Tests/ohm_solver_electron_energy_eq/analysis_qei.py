#!/usr/bin/env python3
"""Validate the electron-ion temperature relaxation (Q_ei), both the
electron-side sink AND the conjugate ion heating -- i.e. that the exchange is
energy-conserving.

The companion deck evolves a uniform, unmagnetized, zero-resistivity plasma
with the ions at rest and hot electrons (Te0 >> Ti0), with ONLY the Q_ei
exchange active:

    dU_e/dt = -Q_ei,   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),     (electron sink)
    ions GAIN exactly Q_ei via a thermal-velocity rescale.     (ion source)

With U_e = n_e k_B T_e/(gamma_e-1), the (3/2) n_i k_B T_i ion thermal energy,
a single proton species (Z=1, n_e=n_i) and constant nu_ei, the two
temperatures relax toward a common value:

    dT_e/dt = -3(gamma_e-1) nu_ei (T_e - T_i)        [electron side]
    dT_i/dt = +2 nu_ei (T_e - T_i)                   [ion side, gamma-indep.]

so the difference decays exponentially,

    (T_e - T_i)(t) = (T_e0 - T_i0) exp(-rate t),  rate = [3(gamma_e-1) + 2] nu_ei,

(= 4 nu_ei for gamma_e = 5/3), and the total thermal energy is conserved:

    C_e T_e + C_i T_i = const,   C_e = n_e k_B/(gamma_e-1),  C_i = (3/2) n_i k_B.

For gamma_e=5/3, C_e=C_i so T_e and T_i meet at (T_e0+T_i0)/2.

This script reads domain-mean T_e(t) (Kelvin->eV) and T_i(t) (eV) and checks
  (1) the difference-decay rate vs [3(gamma_e-1)+2] nu_ei, and
  (2) energy conservation: C_e T_e + C_i T_i constant over the run.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

Q_E = 1.602176634e-19
K_B = 1.380649e-23
K_PER_EV = Q_E / K_B  # T[eV] * this = T[K];  T[K] / this = T[eV]


def domain_means(diag_dir):
    """Return (t[s], <Te>[eV], <Ti>[eV]) density-weighted domain means."""
    ts = OpenPMDTimeSeries(str(diag_dir))
    t = np.asarray(ts.t, dtype=float)

    Te_m, Ti_m = [], []
    for it in ts.iterations:
        Te, _ = ts.get_field("Te", iteration=it)
        Ti, _ = ts.get_field("T_ions", iteration=it)
        rho, _ = ts.get_field("rho", iteration=it)
        w = np.asarray(rho, dtype=float) / Q_E
        wsum = float(np.sum(w))
        Te_m.append(float(np.sum(np.asarray(Te, float) * w) / wsum) / K_PER_EV)
        Ti_m.append(float(np.sum(np.asarray(Ti, float) * w) / wsum))  # already eV
    return t, np.array(Te_m), np.array(Ti_m)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--diag-dir",
        default="diags/field_diags",
        help="openPMD field-diagnostics directory",
    )
    ap.add_argument(
        "--nu-ei",
        type=float,
        default=1.0e6,
        help="constant relaxation rate used in the run (1/s); must match the deck",
    )
    ap.add_argument(
        "--gamma", type=float, default=5.0 / 3.0, help="electron adiabatic index"
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="allowed relative error on the fitted difference-rate",
    )
    ap.add_argument(
        "--etol",
        type=float,
        default=0.02,
        help="allowed relative drift of total thermal energy",
    )
    ap.add_argument("--out", default="qei_check.png")
    args = ap.parse_args(argv)

    t, Te, Ti = domain_means(args.diag_dir)
    if t.size < 3:
        print(f"ERROR: need >=3 dumps, found {t.size} in {args.diag_dir}")
        return 1

    g = args.gamma
    # rate at which (Te - Ti) decays = [3(g-1) + 2] nu_ei.
    rate_pred = (3.0 * (g - 1.0) + 2.0) * args.nu_ei
    Te0, Ti0 = Te[0], Ti[0]

    # (1) fit ln((Te-Ti)/(Te0-Ti0)) = -rate t.
    d = (Te - Ti) / (Te0 - Ti0)
    good = d > 1e-3
    rate_fit = -np.polyfit(t[good], np.log(d[good]), 1)[0]
    rel_err = abs(rate_fit - rate_pred) / rate_pred

    # (2) energy conservation: heat capacities (per n k_B; n_e=n_i cancels).
    ce = 1.0 / (g - 1.0)  # C_e / (n k_B)
    ci = 1.5  # C_i / (n k_B)
    E = ce * Te + ci * Ti  # total thermal "energy" per n k_B (eV units)
    E_drift = (E - E[0]) / E[0]
    e_max = float(np.max(np.abs(E_drift)))
    T_eq_pred = (ce * Te0 + ci * Ti0) / (ce + ci)

    print("=" * 66)
    print("Electron-ion relaxation (Q_ei), energy-conserving exchange")
    print(f"  Te0 = {Te0:.2f} eV,  Ti0 = {Ti0:.2f} eV,  gamma = {g:.4f}")
    print(f"  nu_ei (input)             = {args.nu_ei:.4e} 1/s")
    print(f"  diff-rate predicted [3(g-1)+2]nu_ei = {rate_pred:.4e} 1/s")
    print(f"  diff-rate fitted          = {rate_fit:.4e} 1/s")
    print(
        f"  relative error            = {rel_err * 100:.2f}%   (tol {args.rtol * 100:.1f}%)"
    )
    print(f"  equilibrium T predicted   = {T_eq_pred:.2f} eV")
    print(f"  Te_end / Ti_end (meet?)   = {Te[-1]:.2f} / {Ti[-1]:.2f} eV")
    print(
        f"  total-energy max drift    = {e_max * 100:.3f}%   (tol {args.etol * 100:.2f}%)"
    )
    print("=" * 66)

    tus = t * 1e6
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(tus, Te, "o-", ms=4, label=r"$\langle T_e\rangle$")
    ax[0].plot(tus, Ti, "s-", ms=4, color="C3", label=r"$\langle T_i\rangle$")
    ax[0].axhline(T_eq_pred, color="gray", lw=0.9, ls=":", label=r"$T_{eq}$ pred")
    ax[0].set_xlabel(r"time ($\mu$s)")
    ax[0].set_ylabel("temperature (eV)")
    ax[0].set_title("e-i relaxation to common T")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(tus[good], d[good], "o", ms=5, label="measured")
    ax[1].semilogy(
        tus, np.exp(-rate_fit * t), "-", lw=2, label=f"fit  rate={rate_fit:.2e}"
    )
    ax[1].semilogy(
        tus, np.exp(-rate_pred * t), "--", lw=2, label=f"pred rate={rate_pred:.2e}"
    )
    ax[1].set_xlabel(r"time ($\mu$s)")
    ax[1].set_ylabel(r"$(T_e-T_i)/(T_{e0}-T_{i0})$")
    ax[1].set_title(f"difference decay (err {rel_err * 100:.1f}%)")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")

    ax[2].plot(tus, E_drift * 100, "o-", ms=4, color="C2")
    ax[2].axhline(0.0, color="gray", lw=0.8, ls=":")
    ax[2].set_xlabel(r"time ($\mu$s)")
    ax[2].set_ylabel(r"$(E-E_0)/E_0$ (%)")
    ax[2].set_title(f"total thermal energy (max {e_max * 100:.2f}%)")
    ax[2].grid(alpha=0.3)

    fig.suptitle("$Q_{ei}$ energy-conserving electron-ion relaxation")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

    ok = (rel_err <= args.rtol) and (e_max <= args.etol)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
