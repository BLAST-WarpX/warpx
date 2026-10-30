#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp for embedded-boundary conductors (RZ, Yee, Esirkepov, periodic z).

A rod (cathode) and a sleeve (anode) inside a grounded PEC wall. With --sleeve joined
the sleeve reaches the wall, so it is part of the ground and only the rod is listed.

--case vacuum:  capacitance against the discrete coaxial formula, and the clamped
                vacuum field stays static.
--case diode:   electrons emitted from the rod at the Langmuir-Blodgett current.
                The clamp observer must equal the line integral of E_r at every step,
                the free-node Gauss law must hold, and B stays zero. With --arm on
                the gap stays at 1 kV and the grounded charge -psi^T q must equal a
                grounded Poisson solve; with --arm off the gap discharges.
--case restart: --mode base writes a checkpoint at step N_RESTART; --mode restart
                resumes from it and must reproduce the base run.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["vacuum", "diode", "restart"], required=True)
parser.add_argument("--arm", choices=["on", "off"], default="on")
parser.add_argument("--mode", choices=["base", "restart"], default="base")
parser.add_argument("--sleeve", choices=["separate", "joined"], default="separate")
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()

EPS0, Q_E, M_E = picmi.constants.ep0, picmi.constants.q_e, picmi.constants.m_e

# --- Geometry: rod (cathode) and sleeve (anode) inside the grounded outer wall
NR, NZ, R_WALL, LENGTH = 48, 8, 6.0e-2, 4.0e-2
R_ROD, R_SLEEVE_IN = 6.2e-3, 5.03e-2
# A joined sleeve extends beyond the wall; the gap to the rod is the same
R_SLEEVE_OUT = 5.53e-2 if args.sleeve == "separate" else 2.0 * R_WALL
V_CATHODE = -1000.0
r2 = "(x*x+y*y)"
rod = f"({R_ROD}*{R_ROD}-{r2})"
sleeve = f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
implicit = f"-({rod})*({sleeve})"
# The selector only has to separate the two conductors; x is the radius in RZ
electrodes = [{"name": "cathode", "region": "(x<0.03)", "potential": V_CATHODE}]
if args.sleeve == "separate":
    electrodes.append({"name": "anode", "region": "(x>0.03)", "potential": 0.0})
dr, dz = R_WALL / NR, LENGTH / NZ

# --- Run control
NO_STEP_LIMIT = 10**8  # max_steps is only an upper bound: the script calls sim.step(n)
N_VACUUM_STEPS = 20
N_RESTART = 30  # checkpoint step; the restart case then runs to 2 * N_RESTART
# CTest runs each test in its own directory; the restart test depends on the base test
BASE_DIR = Path("../test_rz_voltage_clamp_checkpoint_picmi")
CORRECTION_INTERVAL_VACUUM = 5  # vacuum: also tests that skipped steps change nothing
GROUNDED_SOLVE_RTOL, GROUNDED_SOLVE_MAX_ITER = 1.0e-13, 200
DIAGNOSTIC_INTERVAL = 100  # steps between the (costlier) Gauss and B checks
BIRTH_FRACTION = 0.1  # birth radius between the geometric and represented rod surface
BIRTH_U = 1.0e6  # initial radial momentum per mass of emitted electrons [m/s]
RESTART_WEIGHT = 2.0e6  # macroparticle weight in the restart case
N_TRANSITS = 1.2  # diode run length in space-charge-limited transit times

# --- Tolerances (absolute in V or relative as named)
# The discrete coaxial formula is exact for the staircase; the limit is the solver
CAPACITANCE_RTOL = 1.0e-9
VOLTAGE_ATOL = 1.0e-8  # vacuum readback and line integral, 1 kV scale
FIELD_RTOL = 1.0e-12  # E unchanged by repeated corrections, relative to max |E|
B_RTOL = 1.0e-12  # c |B| relative to max |E|
RESTART_FIELD_RTOL = (
    1.0e-12  # particle order may change on restart: roundoff, not bitwise
)
RESTART_VOLTAGE_ATOL = 1.0e-9
LINE_OBSERVER_ATOL = 1.0e-6  # observer vs independent line integral, over ~2500 steps
TARGET_ATOL = 1.0e-9  # observer reading after each correction
GAP_ATOL = 1.0e-6  # line integral at the end of the diode run
# Roundoff accumulates over ~2500 steps; real violations are 1e-3 or larger
GAUSS_RTOL = 1.0e-10
# B_theta stays zero: a z-uniform radial current is balanced by the displacement current
B_ATOL = 1.0e-12
GROUNDED_RTOL = 1.0e-10  # grounded solve vs reciprocity, relative to the induced charge

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)
restart = args.case == "restart" and args.mode == "restart"
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9),
    max_steps=NO_STEP_LIMIT,
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=implicit, potential=0.0
    ),
    warpx_use_filter=False,
    # The clamp requires charge-conserving deposition
    warpx_current_deposition_algo="esirkepov",
    warpx_amr_restart=str(BASE_DIR / "diags" / f"chk{N_RESTART:06d}")
    if restart
    else None,
    verbose=0,
)
sim.add_species(
    picmi.Species(
        name="electrons", particle_type="electron", initial_distribution=None
    ),
    layout=None,
)
if args.case == "restart" and args.mode == "base":
    sim.add_diagnostic(picmi.Checkpoint(period=N_RESTART, name="chk"))
sim.initialize_inputs()
sim.initialize_warpx()

warpx = sim.extension.warpx
amr = sim.extension.amr
particles = sim.particles.get("electrons")
is_io_rank = amr.ParallelDescriptor.MyProc() == 0

clamp = StaircaseBiasCorrector(
    sim,
    correction_interval=CORRECTION_INTERVAL_VACUUM if args.case == "vacuum" else 1,
    electrodes=electrodes,
)
clamp.setup_after_init()
n = clamp.n


def field(name, direction=None):
    """Valid data of a registered field on all ranks (radius, z), without the
    duplicate periodic z node. pyAMReX global indexing gathers across boxes and ranks."""
    if direction is None:
        mf = sim.fields.get(name, level=0)
    else:
        mf = sim.fields.get(name, dir=direction, level=0)
    return mf[...][:, :NZ]


RZ_DIRECTIONS = ("r", "theta", "z")


def efield(d):
    return field("Efield_fp", RZ_DIRECTIONS[d])


# Represented conductor surfaces at the midplane: the last cathode node (from the
# clamp's weight field of conductor 0) and the first fixed node beyond it (anode).
z_mid = NZ // 2
cathode = np.nonzero(field("voltage_clamp_weight_0")[:, z_mid])[0]
fixed = field("voltage_clamp_fixed")[:, z_mid]
k_cathode = int(cathode.max())
k_anode = int(np.nonzero(fixed[k_cathode + 1 :])[0][0]) + k_cathode + 1
r_node = np.arange(NR + 1) * dr
r_half = (np.arange(NR) + 0.5) * dr


def gap_voltage():
    """phi(cathode) - phi(anode) as the line integral of E_r at the midplane."""
    return float(np.sum(efield(0)[k_cathode:k_anode, z_mid]) * dr)


def gauss_residual():
    """Largest |eps0 div E - rho| times node volume on the free nodes in the gap.

    div E is computed here with the cylindrical Yee divergence because WarpX's nodal
    ComputeDivE has no Python binding and the divE diagnostic is cell-centered."""
    er, ez = efield(0), efield(2)
    mpc = warpx.multi_particle_container()
    rho = mpc.get_charge_density(lev=0, local=False, finalize=True)[...][:, :NZ]
    div = np.zeros((NR + 1, NZ))
    for i in range(1, NR):
        div[i] = (r_half[i] * er[i] - r_half[i - 1] * er[i - 1]) / (r_node[i] * dr)
    div += (ez - np.roll(ez, 1, axis=1)) / dz
    volume = 2.0 * np.pi * r_node * dr * dz
    residual = ((EPS0 * div - rho) * volume[:, None]).sum(axis=1)
    return float(np.max(np.abs(residual[k_cathode + 1 : k_anode])))


def grounded_check():
    """Plasma-induced charge: reciprocity -psi^T q against a grounded solve."""
    warpx.voltage_clamp_solve_grounded(GROUNDED_SOLVE_RTOL, GROUNDED_SOLVE_MAX_ITER)
    field_charge, grounded = (
        np.asarray(q) for q in warpx.voltage_clamp_charge_state(n)
    )
    # Live charge on the fixed nodes = Gauss charge of live E - field charge
    on_fixed = np.asarray(warpx.voltage_clamp_gauss_charge(n)) - field_charge
    solved = np.asarray(warpx.voltage_clamp_gauss_charge(n, "voltage_clamp_E_grounded"))
    return solved - on_fixed - grounded


installed = []


def install(name, func):
    callbacks.installcallback(name, func)
    installed.append((name, func))


def finish(checks, summary):
    if is_io_rank:
        print(f"{summary}; {checks}")
    # Callbacks and Python-held AMReX objects must be released before finalize
    for name, func in installed:
        callbacks.uninstallcallback(name, func)
    gc.collect()
    sim.finalize()
    failures = [k for k, ok in checks.items() if not ok]
    assert not failures, failures
    sys.exit(0)


if restart:
    assert warpx.getistep(0) == N_RESTART
    clamp.resume_from_checkpoint()
else:
    clamp.initialize_vacuum_bias()

if args.case == "vacuum":
    # Discrete coaxial capacitance between the represented rod and sleeve: the
    # series sum of the radial cell "resistances" dr / r_half across the gap
    c_exact = 2.0 * np.pi * EPS0 * LENGTH / np.sum(dr / r_half[k_cathode:k_anode])
    cap = clamp.capacitance
    e_initial = [efield(d) for d in range(3)]
    voltage_initial = clamp.measure_voltage_state()["voltage"]
    install("afterstep", clamp.correct_after_step)
    sim.step(N_VACUUM_STEPS)
    e_change = max(np.max(np.abs(efield(d) - e_initial[d])) for d in range(3))
    e_scale = max(np.max(np.abs(e)) for e in e_initial)
    b_max = max(np.max(np.abs(field("Bfield_fp", d))) for d in RZ_DIRECTIONS)
    checks = {
        "C = discrete coax": abs(cap[0, 0] / c_exact - 1.0) < CAPACITANCE_RTOL,
        "C symmetric": np.max(np.abs(cap - cap.T)) < CAPACITANCE_RTOL * abs(cap[0, 0]),
        "initial voltage": np.max(np.abs(voltage_initial - clamp.v_target))
        < VOLTAGE_ATOL,
        "line integral": abs(gap_voltage() - V_CATHODE) < VOLTAGE_ATOL,
        "E static": e_change < FIELD_RTOL * e_scale,
        "B zero": picmi.constants.c * b_max < B_RTOL * e_scale,
    }
    finish(checks, f"vacuum: C = {cap[0, 0]:.10e} F, discrete coax {c_exact:.10e} F")

# Electrons are born between the geometric rod surface (R_ROD) and the represented
# cathode surface (the last fixed node, a_eff > R_ROD). There, a new particle's charge
# lands on fixed nodes, i.e. on the conductor, so creating it does not break the
# discrete Gauss law in the gap (charge created in free vacuum would have no current).
a_eff, b_eff = k_cathode * dr, k_anode * dr
r_birth = R_ROD + BIRTH_FRACTION * (a_eff - R_ROD)
z_birth = (np.arange(NZ) + 0.5) * dz

if args.case == "restart":

    def inject():
        particles.add_particles(
            x=np.full(NZ, r_birth), y=np.zeros(NZ), z=z_birth,
            ux=np.full(NZ, BIRTH_U), uy=np.zeros(NZ), uz=np.zeros(NZ),
            w=np.full(NZ, RESTART_WEIGHT), unique_particles=False,
        )  # fmt: skip

    trace = {"observer": [], "er": []}

    def record():
        if warpx.getistep(0) > N_RESTART:
            trace["observer"].append(clamp.measure_voltage_state()["voltage"])
            trace["er"].append(efield(0))

    install("beforestep", inject)
    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    sim.step(2 * N_RESTART - warpx.getistep(0))
    result = {key: np.asarray(value) for key, value in trace.items()}
    checks = {}
    if args.mode == "base":
        if is_io_rank:
            np.savez("restart_trace.npz", **result)
    else:
        base = np.load(BASE_DIR / "restart_trace.npz")
        scale = np.max(np.abs(base["er"]))
        observer_change = np.max(np.abs(result["observer"] - base["observer"]))
        checks = {
            "E_r trace": np.max(np.abs(result["er"] - base["er"]))
            <= RESTART_FIELD_RTOL * scale,
            "observer trace": observer_change <= RESTART_VOLTAGE_ATOL,
            "target held": np.max(np.abs(result["observer"][:, 0] - V_CATHODE))
            <= RESTART_VOLTAGE_ATOL,
        }
    finish(checks, f"restart {args.mode}")


# --case diode: Langmuir-Blodgett current of the represented gap (a_eff, b_eff).
# beta^2 solves 3 b b'' + b'^2 + 4 b b' + b^2 = 1 in gamma = ln(r/a), with the series
# start b = g - 0.4 g^2 near g = 0 (Langmuir & Blodgett 1923, as reviewed in Zhang et al.,
# Appl. Phys. Rev. 4, 011304 (2017), https://doi.org/10.1063/1.4978231).
def lb_rhs(_, y):
    b, db = y
    return [db, (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)]


GAMMA_START = 1.0e-6  # the ODE is singular at gamma = 0; start on the series
lb = solve_ivp(
    lb_rhs,
    (GAMMA_START, np.log(b_eff / a_eff)),
    [GAMMA_START - 0.4 * GAMMA_START**2, 1.0 - 0.8 * GAMMA_START],
    method="DOP853",
    rtol=1.0e-12,
    atol=1.0e-14,
)
beta_end = lb.y[0, -1]
current = (
    8.0 * np.pi * EPS0 / 9.0 * np.sqrt(2.0 * Q_E / M_E) * abs(V_CATHODE) ** 1.5
    * LENGTH / (b_eff * beta_end ** 2)
)  # fmt: skip
dt = warpx.getdt(0)
# Space-charge-limited transit time: 3 d / v_final (planar Child-Langmuir)
transit = 3.0 * (b_eff - a_eff) / np.sqrt(2.0 * Q_E * abs(V_CATHODE) / M_E)
n_steps = int(N_TRANSITS * transit / dt)
weight = current * dt / (Q_E * NZ)  # NZ macroparticles per step carry I dt
injected = 0.0
worst = {"line - observer": 0.0, "observer - target": 0.0, "gauss": 0.0, "B": 0.0}


def inject():
    global injected
    particles.add_particles(
        x=np.full(NZ, r_birth), y=np.zeros(NZ), z=z_birth,
        ux=np.full(NZ, BIRTH_U), uy=np.zeros(NZ), uz=np.zeros(NZ),
        w=np.full(NZ, weight), unique_particles=False,
    )  # fmt: skip
    injected += Q_E * weight * NZ


def record():
    voltage = clamp.measure_voltage_state()["voltage"]
    gap = float(voltage[0] - (voltage[1] if n > 1 else 0.0))
    worst["line - observer"] = max(worst["line - observer"], abs(gap_voltage() - gap))
    worst["observer - target"] = max(worst["observer - target"], abs(gap - V_CATHODE))
    step = warpx.getistep(0)
    if step % DIAGNOSTIC_INTERVAL == 0 or step == n_steps:
        worst["gauss"] = max(worst["gauss"], gauss_residual())
        b_theta = field("Bfield_fp", "theta")
        worst["B"] = max(worst["B"], float(np.max(np.abs(b_theta))))


install("beforestep", inject)
if args.arm == "on":
    install("afterstep", clamp.correct_after_step)
install("afterstep", record)
sim.step(n_steps)

final_gap = gap_voltage()
collected = particles.number_of_particles(False) < round(injected / (Q_E * weight))
checks = {
    "electrons collected": collected,
    "observer = line integral": worst["line - observer"] < LINE_OBSERVER_ATOL,
    "Gauss law": worst["gauss"] / injected < GAUSS_RTOL,
    "B zero": worst["B"] < B_ATOL,
}
summary = f"diode arm={args.arm} steps={n_steps}: gap {final_gap:.4f} V; " + "; ".join(
    f"max|{k}| = {v:.2e}" for k, v in worst.items()
)
if args.arm == "on":
    grounded_error = np.max(np.abs(grounded_check())) / np.sum(
        np.abs(clamp.measure_voltage_state()["grounded_charge"])
    )
    checks["observer held target"] = worst["observer - target"] < TARGET_ATOL
    checks["gap held"] = abs(final_gap - V_CATHODE) < GAP_ATOL
    checks["grounded solve = reciprocity"] = grounded_error < GROUNDED_RTOL
    summary += f"; grounded check {grounded_error:.2e}"
else:
    checks["gap discharged"] = abs(final_gap) < 0.5 * abs(V_CATHODE)
finish(checks, summary)
