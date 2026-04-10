#!/usr/bin/env python3
import argparse
import sys

import numpy as np

from pywarpx import callbacks, libwarpx, picmi
from pywarpx.LoadThirdParty import load_cupy

# Create the parser and add the argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-u",
    "--unique",
    action="store_true",
    help="Whether injected particles should be treated as unique",
)

# Parse the input
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

##########################
# numerics parameters
##########################

dt = 7.5e-10

# --- Nb time steps

max_steps = 10

# --- grid

nx = 64
ny = 64

xmin = 0
xmax = 0.03
ymin = 0
ymax = 0.03

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    lower_boundary_conditions=["dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    moving_window_velocity=None,
    warpx_max_grid_size=32,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-6,
    warpx_self_fields_verbosity=0,
)

##########################
# physics components
##########################

electrons = picmi.Species(particle_type="electron", name="electrons")

##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=["phi"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(solver=solver, time_step_size=dt, max_steps=max_steps, verbose=1)

sim.add_species(
    electrons, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0], grid=grid)
)
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

sim.initialize_inputs()
sim.initialize_warpx()

##########################
# python particle data access
##########################

# set numpy random seed so that the particle properties generated
# below will be reproducible from run to run
xp, _ = load_cupy()
xp.random.seed(30025025)

electrons = sim.particles.get("electrons")
electrons.add_real_comp("newPid")

my_id = libwarpx.amr.ParallelDescriptor.MyProc()
num_ranks = libwarpx.amr.ParallelDescriptor.NProcs()


def to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    else:
        return np.asarray(arr)


def nps_per_rank(rank):
    return 10 * (rank + 1)


def add_particles():
    global electrons
    nps = nps_per_rank(my_id)
    x = np.linspace(0.005, 0.025, nps)
    y = np.zeros(nps)
    z = np.linspace(0.005, 0.025, nps)
    ux = np.random.normal(loc=0, scale=1e3, size=nps)
    uy = np.random.normal(loc=0, scale=1e3, size=nps)
    uz = np.random.normal(loc=0, scale=1e3, size=nps)
    w = np.ones(nps) * 2.0
    newPid = np.ones(nps) * 5.0

    electrons.add_particles(
        x=x,
        y=y,
        z=z,
        ux=ux,
        uy=uy,
        uz=uz,
        w=w,
        newPid=newPid,
        unique_particles=True,
    )


callbacks.installbeforestep(add_particles)

##########################
# simulation run
##########################

sim.step(max_steps - 1)

##########################
# check that the new PIDs
# are properly set
##########################
total_expected_electrons_size = sum(
    nps_per_rank(rank) * (max_steps - 1) for rank in range(num_ranks)
)

assert electrons.size == total_expected_electrons_size
assert electrons.get_real_comp_index("w") == 2
assert electrons.get_real_comp_index("newPid") == 6

for pti in electrons.iterator(level=0):
    vals = pti["newPid"]
    assert np.allclose(vals, 5)

#####################################
# check that current deposition works
#####################################

electrons.deposit_current(
    "current_fp", 0, sim.extension.warpx.getdt(0), -0.5 * sim.extension.warpx.getdt(0)
)
assert np.isclose(
    np.max(np.abs(sim.fields.get("current_fp", "x", 0)[...])), 6.889716673402935e-09
)
assert np.isclose(
    np.max(np.abs(sim.fields.get("current_fp", "z", 0)[...])), 4.888491863315444e-09
)

##########################
# take the final sim step
##########################

sim.step(1)

del electrons
