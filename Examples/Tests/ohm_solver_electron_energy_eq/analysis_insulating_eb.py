#!/usr/bin/env python3
#
# --- Analysis for the insulating embedded-boundary wall test
# --- (inputs_test_2d_ohm_solver_insulating_eb_picmi.py). The deck writes
# --- its measured gates to insulating_eb_metrics.npz; this script asserts
# --- them:
# ---
# ---  * exact collection bookkeeping: the probe species (loaded entirely
# ---    inside the standoff band) is collected in full during the first
# ---    step, and the per-species tallies reproduce its total charge and
# ---    kinetic energy; the bulk ions (loaded clear of the band) are never
# ---    collected;
# ---  * the open-set electron entropy Sigma K_e N_e holds: exactly (to
# ---    machine accumulation) in the at-rest case where it is an invariant
# ---    and any drift is wall leakage, and to the marker/ion transport-
# ---    mismatch level in the compress case.

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tol-entropy", type=float, required=True)
args = parser.parse_args()

m = np.load("insulating_eb_metrics.npz")
case = str(m["case"])

print(f"case: {case}")
print(f"open-set entropy rel drift: {float(m['d_entropy']):+.3e}")

# --- collection tallies -------------------------------------------------
probe_q = float(m["probe_q"])
probe_q_expected = float(m["probe_q_expected"])
probe_e = float(m["probe_e"])
probe_e_expected = float(m["probe_e_expected"])

# charge: an exact bookkeeping identity through the scrape path
q_err = abs(probe_q - probe_q_expected) / abs(probe_q_expected)
print(f"probe charge tally rel err: {q_err:.3e}")
assert q_err < 1.0e-9, f"probe collected-charge tally off: {q_err:.3e}"

# energy: the tally uses the (exact) relativistic kinetic energy, the
# expectation the non-relativistic m v^2 / 2 -- they differ at (v/c)^2
e_err = abs(probe_e - probe_e_expected) / abs(probe_e_expected)
print(f"probe energy tally rel err: {e_err:.3e}")
assert e_err < 1.0e-6, f"probe collected-energy tally off: {e_err:.3e}"

# the probe must be collected in full and the bulk ions not at all
assert float(m["probe_w_left"]) == 0.0, "probe survived the standoff band"
assert float(m["ions_q"]) == 0.0, "bulk ions were collected (standoff breached)"
assert float(m["ions_e"]) == 0.0, "bulk ions were collected (standoff breached)"

# --- open-set entropy hold ----------------------------------------------
d_entropy = abs(float(m["d_entropy"]))
assert d_entropy < args.tol_entropy, (
    f"open-set entropy drift {d_entropy:.3e} exceeds {args.tol_entropy:.1e}"
)

print("insulating EB analysis: all gates passed")
