#!/usr/bin/env python3
"""Analysis for inputs_test_3d_darwin_solver_oblique_modes_picmi.py.

This CI test is a heavily reduced (16^3, 130 step, <20s on 2 cores) version
of a full verification run (32^3, ~4700 steps, ES off) that showed every
low-k (k l_i < 0.9) transverse-theory branch has a spectral peak within 1-2
frequency bins of the cold two-fluid dispersion relation, and that wave
vectors related by a permutation of (x,y,z) axes (equivalent because
B0 || (1,1,1)/sqrt(3)) give identical spectra. At the reduced scale used
here there are only ~5 cyclotron periods (vs. 30 in the full run), far too
few for an FFT to resolve a spectral peak, so this script does not attempt
to reproduce that check via a periodogram. It checks the same physics with
methods that work on short, noisy records:

1. Energy conservation: |dE/E0| should stay small over the run.
2. Permutation symmetry: (1,0,0)/(0,1,0)/(0,0,1) are physically equivalent
   (same |k|, same angle to B0), and so are (1,1,0)/(0,1,1)/(1,0,1). The
   input deck loads particles so that the FULL ensemble (all species) is
   exactly invariant under the cyclic axis permutation (x,y,z)->(y,z,x) that
   relates each group (see its docstring): every macroparticle is loaded
   once, then 2 more copies are added with position and velocity cyclically
   permuted and weight rescaled by 1/3. This turns "symmetric to within
   realization noise" into "symmetric to within floating-point roundoff":
   measured mean |dB(k)|^2 across a group's 3 members agreed to 1e-10ish
   relative on every seed tried (see measurements below) -- a mass matrix
   that mixed up field/axis indices would have to conspire to preserve this
   to be missed.
3. Coarse frequency check: each recorded k(t) is fit to a constrained sum of
   two complex exponentials at +w and -w (a "matched two-tone" fit; see
   below for why), fit jointly across all 3 permutation members and 3 field
   components (9 channels sharing one w). This is compared to the cold-
   plasma transverse-theory root, with a tolerance in the ~30-70% range
   (still much tighter than an earlier single-lag estimator's 10x-plus
   scatter -- see "what changed" below).

Why a two-tone fit, not a single complex-exponential fit: the recorded
dB(k, t) is the k-th spatial Fourier coefficient of a REAL field, so at any
given wave vector both a "+w" and a "-w" time dependence are physically
present simultaneously (the two propagation directions along +/-k of the one
existing transverse-theory branch at this obliqueness), generically with
comparable amplitude since nothing breaks that symmetry for thermal-noise
seeding. A single complex-exponential (single-lag / Prony order-1) fit to
such a two-tone signal is not just noisy but systematically BIASED TOWARD
ZERO (the +w and -w contributions partially cancel in the estimator's phase
average), which is what an earlier version of this test used and is why its
measured ratios were stuck around 0.2-0.8x theory instead of the ~1.0-1.1x
(4-5% high from finite beta) the full-resolution run found. Fitting the +w
and -w amplitudes simultaneously (by a joint least-squares grid scan over
|w|) removes that bias.

Measured on 4 independent noise seeds (repository CI uses one fixed seed for
a reproducible checksum; the tolerances below come from all 4). "ratio" is
the fitted |w|/w_theory; "resid_frac" is the two-tone fit's leftover
variance fraction (0 = perfect fit, 1 = no better than the mean):

                                seed1   seed2   seed3   seed4
  max|dE/E0|                   3.3e-5  1.1e-5  3.5e-5  1.0e-4
  pow_ratio-1, (1,0,0) group    ~1e-13  ~1e-13  ~1e-13  ~1e-13   (see note)
  pow_ratio-1, (1,1,0) group    ~1e-13  ~1e-13  ~1e-13  ~1e-13
  ratio, (1,0,0) group           0.88    0.99    0.71    0.98
  ratio, (1,1,0) group           1.39    1.05    0.96    0.80

(pow_ratio is (max/min mean |dB(k)|^2 among the 3 group members); it was
1.0 +/- a few parts in 1e13 on every seed and every group tried, i.e. at
the floating-point-roundoff floor of an FFT computed from a MultiFab whose
own fill order differs from rank to rank -- this is as tight as this
diagnostic can be without bit-identical parallel reductions.)

What changed from an earlier version of this test (single-lag estimator,
statistical-only symmetry check, dx=1 l_i, N=8^3, 200 steps): the grid was
refined to N=16^3 (dx=0.5 l_i, matching the full-resolution deck; the
earlier dx=1 l_i put the tested k dx above ~0.6, where the full-resolution
run showed dispersion measurements degrade), particle loading was made
exactly permutation-symmetric (see above), and the frequency estimator was
replaced. The step count was reduced from 200 to 130 to keep wall time
comfortably under budget (see the input deck's timing measurements): the
finer grid costs ~1.6-1.9x more per step than the old 8^3 grid (the solve
is GMRES/MLMG-dominated, not particle-work-dominated, so increasing NPPC is
not an effective way to buy back accuracy -- more steps or a finer grid are).
The k=(1,1,1) wave vector (parallel to B0, no permutation partner) was
dropped: it is the one wave vector left invariant by the symmetrizing
rotation, and a transverse (div(B)=0) field can have no component along its
only rotation-invariant polarization (parallel to k), so it is a genuine
selection-rule zero under this loading (measured at the ~1e-36 floor) and
carries no signal to check.
"""

import argparse
import sys

import dill
import numpy as np
from scipy.optimize import brentq

p = argparse.ArgumentParser()
p.add_argument(
    "--energy-tol", type=float, default=5e-4, help="max allowed |dE/E0| over the run"
)
p.add_argument(
    "--symmetry-tol",
    type=float,
    default=1e-6,
    help="max allowed ratio-minus-one of the largest to smallest mean "
    "|dB(k)|^2 among wave vectors related by a permutation of axes "
    "(near machine precision with the symmetrized particle loading)",
)
p.add_argument(
    "--w-lo",
    type=float,
    default=0.6,
    help="lower bound on the two-tone-fit |w|/w_theory for the (1,0,0) group",
)
p.add_argument(
    "--w-hi",
    type=float,
    default=1.15,
    help="upper bound on the two-tone-fit |w|/w_theory for the (1,0,0) group",
)
p.add_argument(
    "--w-lo-110",
    type=float,
    default=0.6,
    help="lower bound on the two-tone-fit |w|/w_theory for the (1,1,0) group",
)
p.add_argument(
    "--w-hi-110",
    type=float,
    default=1.6,
    help="upper bound on the two-tone-fit |w|/w_theory for the (1,1,0) group",
)
p.add_argument(
    "--resid-max",
    type=float,
    default=0.8,
    help="max allowed two-tone fit residual fraction (sanity bound: a "
    "value near 1 means the fit found no better-than-random-phase signal)",
)
a = p.parse_args()

with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)
d = np.load("diags/k_modes.npz")
t = d["time"]
amp = d["amp"]  # (nt, nk, 3) complex
k_list = [tuple(int(x) for x in m) for m in d["k_list"]]
dt = np.mean(np.diff(t))
nt = len(t)

fail = False

# --------------------------------------------------------------- theory
# cold two-fluid (ions + electrons with inertia), E restricted to the plane
# perpendicular to k (what an ES-off Darwin run solves); same construction
# as the full verification deck's analysis script.
c = 2.99792458e8
species = [
    ((sim.w_pi / sim.w_ci) ** 2, +1.0),
    ((sim.w_pe / sim.w_ci) ** 2, -sim.w_ce / sim.w_ci),
]
kfac = c / (sim.w_ci * sim.l_i)


def stix(w):
    R = 1.0 - sum(wp2 / (w * (w + Wc)) for wp2, Wc in species)
    L = 1.0 - sum(wp2 / (w * (w - Wc)) for wp2, Wc in species)
    P = 1.0 - sum(wp2 / w**2 for wp2, _ in species)
    return R, L, 0.5 * (R + L), P


def G(w, kl, th):
    R, L, S, P = stix(w)
    D = 0.5 * (R - L)
    n2 = (kl * kfac / w) ** 2
    s2, c2 = np.sin(th) ** 2, np.cos(th) ** 2
    a_ = S * c2 + P * s2
    A, B, C = 1.0, a_ + S, a_ * S - D**2 * c2
    return (A * n2**2 - B * n2 + C) / (abs(A) * n2**2 + abs(B) * n2 + abs(C))


def lowest_root(kl, th, wmax=3.0):
    th = float(np.clip(th, 1e-6, np.pi / 2 - 1e-6))
    ws = np.geomspace(1e-3, wmax, 20000)
    g = G(ws, kl, th)
    for i in np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        try:
            r = brentq(G, ws[i], ws[i + 1], args=(kl, th), xtol=1e-10)
        except ValueError:
            continue
        if abs(G(r, kl, th)) < 1e-6 and abs(r - 1.0) > 1e-6:
            return r
    return None


def fit_two_tone(channels, t_wci, w_grid):
    """Joint least-squares fit of channel(t) ~ c1*exp(-i*w*t) + c2*exp(i*w*t)
    (per-channel c1, c2; w shared across all channels), by a grid scan over
    w. Returns (residual(w) array, total power) so the caller can find the
    best-fit w and judge the fit quality."""
    resid = np.empty(len(w_grid))
    total_power = sum(np.sum(np.abs(z) ** 2) for z in channels)
    for iw, w in enumerate(w_grid):
        b1 = np.exp(-1j * w * t_wci)
        b2 = np.exp(1j * w * t_wci)
        gmat = np.array(
            [
                [np.sum(np.conj(b1) * b1), np.sum(np.conj(b1) * b2)],
                [np.sum(np.conj(b2) * b1), np.sum(np.conj(b2) * b2)],
            ]
        )
        ginv = np.linalg.inv(gmat)
        r = 0.0
        for z in channels:
            proj = np.array([np.sum(np.conj(b1) * z), np.sum(np.conj(b2) * z)])
            c1, c2 = ginv @ proj
            model = c1 * b1 + c2 * b2
            r += np.sum(np.abs(z - model) ** 2)
        resid[iw] = r
    return resid, total_power


bhat = np.asarray(sim.bhat)
k0 = 2 * np.pi / sim.L_over_li

groups = {}
for j, m in enumerate(k_list):
    key = tuple(sorted(abs(x) for x in m))
    groups.setdefault(key, []).append(j)

print(
    f"nt={nt}  dt*W_ci={dt * sim.w_ci:.4f}  duration={nt * dt * sim.w_ci:.2f} W_ci^-1"
)
print(f"k0 l_i={k0:.3f}  b={np.round(bhat, 3)}")

# --------------------------------------------------------- symmetry + freq
# (0, 0, 1) = the (1,0,0)-family group (theta=54.7deg); (0, 1, 1) = the
# (1,1,0)-family group (theta=35.3deg) -- see K_LIST in the input deck
w_bounds = {(0, 0, 1): (a.w_lo, a.w_hi), (0, 1, 1): (a.w_lo_110, a.w_hi_110)}
for key, members in sorted(groups.items()):
    m0 = np.array(k_list[members[0]], float)
    kl = k0 * np.linalg.norm(m0)
    th = np.arccos(np.clip(abs(m0 @ bhat) / np.linalg.norm(m0), 0.0, 1.0))
    w_theory = lowest_root(kl, th)

    pows = [np.mean(np.sum(np.abs(amp[:, j, :]) ** 2, axis=1)) for j in members]
    pow_ratio_m1 = max(pows) / min(pows) - 1.0
    sym_ok = pow_ratio_m1 <= a.symmetry_tol
    fail |= not sym_ok

    channels = [amp[:, j, c] for j in members for c in range(3)]
    w_lo, w_hi = w_bounds[key]
    w_grid = np.linspace(0.3 * w_theory, 1.8 * w_theory, 600)
    resid, totpow = fit_two_tone(channels, t * sim.w_ci, w_grid)
    i_best = np.argmin(resid)
    w_best = w_grid[i_best]
    resid_frac = resid[i_best] / totpow

    ratio = w_best / w_theory
    freq_ok = w_lo <= ratio <= w_hi
    resid_ok = resid_frac <= a.resid_max
    fail |= not (freq_ok and resid_ok)

    print(
        f"group {key}: members={[k_list[j] for j in members]}  "
        f"k l_i={kl:.3f} theta={np.degrees(th):.1f}deg  w_theory={w_theory:.3f}\n"
        f"    power ratio - 1 (max/min) = {pow_ratio_m1:.2e}"
        f" (tol {a.symmetry_tol:g})  -> {'ok' if sym_ok else 'FAIL'}\n"
        f"    two-tone fit: |w|/w_theory = {ratio:.3f}"
        f" (bounds [{w_lo:g}, {w_hi:g}]), resid_frac={resid_frac:.3f}"
        f" (max {a.resid_max:g})  -> {'ok' if (freq_ok and resid_ok) else 'FAIL'}"
    )


# --------------------------------------------------------------- energy
def tot(path):
    header = open(path).readline().split()
    data = np.loadtxt(path)
    idx = [i for i, x in enumerate(header) if "total" in x.split("]")[-1]]
    return data[:, idx].sum(axis=1)


fE = tot("diags/field_energy.txt")
pE = tot("diags/part_energy.txt")
E = fE + pE
max_dE = np.max(np.abs(E - E[0]) / E[0])
energy_ok = max_dE <= a.energy_tol
fail |= not energy_ok
print(
    f"\nenergy: max|dE/E0|={max_dE:.3e} (tol {a.energy_tol:g}), "
    f"final={(E[-1] - E[0]) / E[0]:+.3e}  -> {'ok' if energy_ok else 'FAIL'}"
)

print("\nRESULT:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
