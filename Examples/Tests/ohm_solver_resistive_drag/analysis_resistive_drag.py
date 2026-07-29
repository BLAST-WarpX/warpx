#!/usr/bin/env python3
#
# --- Analysis for the resistive-drag momentum-consistency test.
# ---
# --- Primary check: the deposited ion current projected onto the force-free
# --- pattern [0, sin(kx), cos(kx)] stays at the shot-noise level. The drag
# --- (-R_s) and the resistive push-field force (+(rho_s/rho) Sum_t R_t)
# --- cancel exactly for a global eta, so the ions must not pick up the
# --- electron current. A missing push-field resistive term would drive the
# --- projection to -J(t) (1 - exp(-nu t)), an order-unity fraction of J0
# --- for the CI parameters (nu * t_end ~ 1).
# ---
# --- Secondary check: the magnetic-field energy still decays resistively at
# --- the analytic rate 2/tau_R, tau_R = mu0/(eta k^2) (the drag must not
# --- disturb the field evolution).

import argparse

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e, m_p, mu_0

# Parameters must match inputs_test_2d_ohm_solver_resistive_drag_picmi.py
n0 = 2.0e20
B0 = 0.1
Lx = 0.5
k = 2.0 * np.pi / Lx
J0 = k * B0 / mu_0

parser = argparse.ArgumentParser()
parser.add_argument("--eta-scale", type=float, default=1.0)
parser.add_argument(
    "--tol-current",
    type=float,
    default=0.15,
    help="max allowed |projected ion current| / J0 (broken pairing gives ~0.6)",
)
parser.add_argument(
    "--tol-decay",
    type=float,
    default=0.5,
    help="max relative error of the fitted B-energy decay rate vs 2/tau_R",
)
args = parser.parse_args()

eta = 1.0e-5 * args.eta_scale
tau_R = mu_0 / (eta * k**2)
nu_drag = e**2 * eta * n0 / m_p

ts = OpenPMDTimeSeries("diags/field_diags")
t = np.asarray(ts.t)
t_end = t[-1]

# --- primary: projection of the deposited ion current on the pattern -------
proj = []
for it in ts.iterations:
    Jy, info_y = ts.get_field(field="j", coord="y", iteration=it)
    Jz, info_z = ts.get_field(field="j", coord="z", iteration=it)
    # openPMD arrays are (z, x); project each component on its own mesh
    ay = np.mean(Jy * np.sin(k * info_y.x)[np.newaxis, :])
    az = np.mean(Jz * np.cos(k * info_z.x)[np.newaxis, :])
    # <sin^2> = <cos^2> = 1/2 on the periodic mesh -> a = <Jy sin> + <Jz cos>
    proj.append(2.0 * (ay + az) / 2.0)
proj = np.asarray(proj)

frac_end = abs(proj[-1]) / J0
frac_max = np.max(np.abs(proj)) / J0
broken_pred = 1.0 - np.exp(-nu_drag * t_end)

print("Resistive-drag momentum-consistency analysis")
print(f"  J0                     = {J0:.4e} A/m^2")
print(f"  nu_drag * t_end        = {nu_drag * t_end:.3f}")
print(f"  |proj(J_i)|/J0 (end)   = {frac_end:.4f}")
print(f"  |proj(J_i)|/J0 (max)   = {frac_max:.4f}")
print(f"  broken pairing would give ~ {broken_pred:.2f}")
print(f"  tolerance              = {args.tol_current:.2f}")

assert frac_max < args.tol_current, (
    f"Deposited ion current picked up {frac_max:.2f} of the force-free "
    f"current (tolerance {args.tol_current}): the drag and the resistive "
    "push-field force are not cancelling."
)

# --- secondary: resistive B-energy decay ------------------------------------
data = np.loadtxt("diags/field_energy.txt", skiprows=1)
t_red = data[:, 1]
E_B = data[:, 4]  # B-field energy column of the FieldEnergy diagnostic
good = E_B > 0
# ln E_B decays at 2/tau_R while the force-free mode dominates
fit = np.polyfit(t_red[good], np.log(E_B[good]), 1)
rate_fit = -fit[0]
rate_pred = 2.0 / tau_R
rel_err = abs(rate_fit - rate_pred) / rate_pred

print(f"  B-energy decay rate    = {rate_fit:.4e} 1/s (fit)")
print(f"  analytic 2/tau_R       = {rate_pred:.4e} 1/s")
print(f"  relative error         = {rel_err:.3f} (tolerance {args.tol_decay})")

assert rel_err < args.tol_decay, (
    f"B-energy decay rate off by {rel_err:.2f} relative (tolerance "
    f"{args.tol_decay}): the drag is disturbing the resistive field decay."
)

print("PASS")
