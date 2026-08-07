#!/usr/bin/env python3
#
# --- Test of the insulating embedded-boundary wall type
# --- (boundary.eb_type = insulating) with the hybrid-PIC electron energy
# --- equation, in a 2D (x,z) box with a cylindrical containing wall.
# ---
# --- The insulating wall is a collecting wall held off the plasma by a
# --- maintained density standoff band: ions are collected where the signed
# --- distance to the EB falls below eb_standoff_cells cells (tallied per
# --- species), the electron temperature is filled with zero normal gradient
# --- into the band and the covered region each step, and the fraction of
# --- the QDSMC marker deposits that lands on below-floor/covered nodes is
# --- folded back onto the neighboring live-plasma nodes instead of being
# --- dropped.
# ---
# --- Two cases:
# ---
# ---   at-rest  : static heavy ions loaded clear of the standoff band, all
# ---              sources off. V_e = 0, so the transported electron entropy
# ---              Sigma K_e N_e over the open set (n_e > n_floor) is an
# ---              exact invariant; without the deposit fold the hat deposit
# ---              of the plasma-edge markers splits fractionally across the
# ---              floor boundary and is dropped there (a measured
# ---              ~1e-6/step one-way drain on this class of setups). The
# ---              case gates the open-set entropy drift at the
# ---              machine-accumulation level.
# ---
# ---   compress : the same plasma with a slow radial inflow drive
# ---              (u_r = -v0 r/R on the heavy ions), so V_e = J_i/(q n_e)
# ---              advects the markers against the standoff band while the
# ---              bulk compresses and heats adiabatically. Exercises the
# ---              fill + fold under motion; the entropy budget gate is
# ---              correspondingly looser (marker/ion transport mismatch,
# ---              not a wall property).
# ---
# --- Both cases load a small "probe" species entirely inside the standoff
# --- band: it is collected in the first step, and the per-species tallies
# --- must reproduce its total charge and kinetic energy exactly while the
# --- bulk-ion tallies stay identically zero.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, particle_containers, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["at-rest", "compress"], default="at-rest")
parser.add_argument("--test", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--N", type=int, default=32, help="cells per side")
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--standoff", type=float, default=2.0, help="standoff band [cells]")
parser.add_argument(
    "--n-floor-frac",
    type=float,
    default=0.05,
    help="n_floor/n0; the production-class floor keeps E = -grad Pe/(q n) "
    "bounded at the plasma edge",
)
parser.add_argument(
    "--v0", type=float, default=2.0e4, help="compress: radial inflow speed at r=R [m/s]"
)
parser.add_argument(
    "--eb-type",
    choices=["insulating", "absorbing"],
    default="insulating",
    help="A/B control: 'absorbing' turns the insulating wall machinery off "
    "(the collection tallies then stay empty and the probe asserts do not "
    "apply)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

N = args.N
L = 1.0  # domain side [m]
dx = L / N
R = 0.4 * L  # wall radius (EB body at r > R)
s_cells = args.standoff
shell = s_cells * dx  # collection shell: signed distance < shell

n0 = 2.0e20  # bulk density [m^-3]
Te0_eV = 10.0
gamma_e = 5.0 / 3.0
mass_factor = 1.0e6  # heavy ions: static/slow background (see the
# ohm_solver_electron_energy_eq README; light ions scatter off the
# grad-Pe edge field and pollute the budget metric)
v0 = args.v0 if args.case == "compress" else 0.0
dt = 5.0e-8
steps = args.steps

# bulk ions loaded clear of the standoff band (the production standoff
# pattern: orbits never graze the collection shell, so the bulk tally
# stays identically zero)
r_load = R - (s_cells + 1.5) * dx

# probe: a thin ring entirely inside the collection shell -> collected in
# full during the first step (exact tally targets)
probe_r1 = R - 0.75 * shell
probe_r2 = R - 0.25 * shell
probe_n = 1.0e-3 * n0
probe_v = 1.0e5  # uniform +x drift [m/s]; KE per real particle = m v^2/2

# ----------------------------------------------------------------------
# Simulation setup
# ----------------------------------------------------------------------
grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[-L / 2.0, -L / 2.0],
    upper_bound=[L / 2.0, L / 2.0],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic"],
    warpx_max_grid_size=N,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=gamma_e,
    Te=Te0_eV,
    n0=n0,
    n_floor=args.n_floor_frac * n0,
    plasma_resistivity=0.0,
    substeps=4,
    solve_electron_energy_equation=True,
)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"((x*x+z*z) - {R * R})",  # body outside r = R
    eb_type=args.eb_type,
    eb_standoff_cells=s_cells,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=steps,
    verbose=args.verbose or args.test,
    particle_shape=1,
    warpx_embedded_boundary=embedded_boundary,
    warpx_serialize_initial_conditions=True,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=True,
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=mass_factor * constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}*((x*x+z*z)<{r_load * r_load})",
        momentum_expressions=[
            f"-{v0}*x/{R}",
            "0",
            f"-{v0}*z/{R}",
        ],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=16),
)

probe = picmi.Species(
    name="probe",
    charge="q_e",
    mass=constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=(
            f"{probe_n}*((x*x+z*z)>{probe_r1 * probe_r1})"
            f"*((x*x+z*z)<{probe_r2 * probe_r2})"
        ),
        momentum_expressions=[f"{probe_v}", "0", "0"],
    ),
)
simulation.add_species(
    probe,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=steps,
    data_list=["rho", "Te"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

simulation.initialize_inputs()
simulation.initialize_warpx()

# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------
Te_wrap = fields.MultiFabWrapper(mf_name="hybrid_electron_temperature_fp", level=0)
rho_wrap = fields.MultiFabWrapper(mf_name="rho_fp", level=0)

qe = constants.q_e
n_floor = args.n_floor_frac * n0


def open_set_entropy():
    """Sigma K_e N_e over the open set (n_e > n_floor), up to constant
    factors: Sigma Te * n_e^(2-gamma). The open set is the transport's own
    conserved set; a geometry-based mask would misread energy legitimately
    shared with above-floor edge nodes as a drain."""
    te = Te_wrap[:, :]
    ne = rho_wrap[:, :] / qe
    open_mask = ne > n_floor
    return float(np.sum(te[open_mask] * ne[open_mask] ** (2.0 - gamma_e)))


def global_weight_sum(name):
    pw = particle_containers.ParticleContainerWrapper(name)
    local = 0.0
    for w in pw.get_particle_real_arrays("w", level=0, copy_to_host=True):
        local += float(np.sum(w))
    return comm.allreduce(local, op=mpi.SUM)


state = {}


def capture0():
    if "S0" in state:
        return
    state["S0"] = open_set_entropy()
    state["probe_w0"] = global_weight_sum("probe")
    state["ions_w0"] = global_weight_sum("ions")


callbacks.installparticleinjection(capture0)

# ----------------------------------------------------------------------
# Run + measure
# ----------------------------------------------------------------------
simulation.step(steps)

wx = simulation.extension.warpx

S0 = state["S0"]
S1 = open_set_entropy()
d_entropy = (S1 - S0) / S0

probe_q_expected = qe * state["probe_w0"]
probe_e_expected = 0.5 * constants.m_p * probe_v**2 * state["probe_w0"]
probe_q = wx.get_eb_collected_charge("probe")
probe_e = wx.get_eb_collected_energy("probe")
ions_q = wx.get_eb_collected_charge("ions")
ions_e = wx.get_eb_collected_energy("ions")
probe_w_left = global_weight_sum("probe")

if comm.rank == 0:
    print(f"case                 : {args.case}")
    print(f"open-set entropy     : S0 {S0:.9e} -> S1 {S1:.9e}")
    print(f"  rel drift          : {d_entropy:+.3e}")
    print(f"probe charge         : tally {probe_q:.9e} expect {probe_q_expected:.9e}")
    print(f"probe energy         : tally {probe_e:.9e} expect {probe_e_expected:.9e}")
    print(f"probe weight left    : {probe_w_left:.3e} (of {state['probe_w0']:.3e})")
    print(f"ions tallies         : q {ions_q:.3e}  E {ions_e:.3e}")

    np.savez_compressed(
        "insulating_eb_metrics.npz",
        case=args.case,
        S0=S0,
        S1=S1,
        d_entropy=d_entropy,
        probe_q=probe_q,
        probe_q_expected=probe_q_expected,
        probe_e=probe_e,
        probe_e_expected=probe_e_expected,
        probe_w_left=probe_w_left,
        ions_q=ions_q,
        ions_e=ions_e,
    )
