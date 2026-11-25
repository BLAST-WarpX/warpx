#!/usr/bin/env python3
#
# --- Continuation test for the hybrid-PIC solver with a time-dependent
# --- external vector potential: stepping a simulation in several
# --- Evolve() segments (repeated sim.step() calls) must produce the
# --- same result as stepping it continuously. Guards the external-field
# --- staging at the Evolve entry point, which historically re-added the
# --- external contribution on every re-entry.

import argparse
import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

parser = argparse.ArgumentParser()
parser.add_argument(
    "--segments",
    type=str,
    default="10",
    help="comma-separated sim.step() segment lengths (must sum to 10)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

segments = [int(s) for s in args.segments.split(",")]
assert sum(segments) == 10, "segments must sum to the 10-step horizon"

# uniform proton plasma in a guide field, driven by a ramped external
# vector potential (uniform dB_z/dt through curl A)
N = 32
L = 1.0
N0 = 1.0e19
B0 = 5.0e-2
DB = 5.0e-3
T_PLASMA = 20.0  # eV

grid = picmi.Cartesian2DGrid(
    number_of_cells=[N, N],
    lower_bound=[0.0, 0.0],
    upper_bound=[L, L],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
)

w_ci = constants.q_e * B0 / constants.m_p
DT = 0.02 / w_ci
T_RAMP = 10.0 * DT

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_PLASMA,
    n0=N0,
    n_floor=0.05 * N0,
    plasma_resistivity=1.0e-6,
    substeps=32,
    A_external={
        "ramp": {
            "Ax_external_function": "-0.5*y",
            "Ay_external_function": "0.5*x",
            "Az_external_function": "0",
            "A_time_external_function": f"{2.0 * DB}*t/{T_RAMP}",
        }
    },
)

ions = picmi.Species(
    particle_type="proton",
    name="ions",
    initial_distribution=picmi.UniformDistribution(
        density=N0,
        rms_velocity=[np.sqrt(constants.q_e * T_PLASMA / constants.m_p)] * 3,
    ),
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=sum(segments),
    verbose=0,
)
sim.add_species(
    ions, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=16)
)
sim.add_applied_field(
    picmi.AnalyticInitialField(
        Bx_expression="0.0",
        By_expression="0.0",
        Bz_expression=f"{B0}",
    )
)
sim.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=10,
        data_list=["B", "E", "rho"],
    )
)

sim.initialize_inputs()
sim.initialize_warpx()

for seg in segments:
    sim.step(seg)
