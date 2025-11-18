#!/usr/bin/env python3
#
# This is a script that adds particle components at runtime,
# then performs checkpoint / restart and compares the result
# to the original simulation.

import sys

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi
from pywarpx.LoadThirdParty import load_cupy

##########################
# physics parameters
##########################

dt = 7.5e-10

##########################
# numerics parameters
##########################

max_steps = 10

nx = 64
nz = 64

xmin = 0
xmax = 0.03
zmin = 0
zmax = 0.03

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
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

checkpoint = picmi.Checkpoint(name="chk", period=5)

##########################
# simulation setup
##########################

sim = picmi.Simulation(solver=solver, time_step_size=dt, max_steps=max_steps, verbose=1)

sim.add_species(
    electrons, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0], grid=grid)
)

for arg in sys.argv:
    if arg.startswith("amr.restart"):
        restart_file_name = arg.split("=")[1]
        sim.amr_restart = restart_file_name
        sys.argv.remove(arg)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(checkpoint)
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
if not sim.amr_restart:
    electrons.add_real_comp("newPid")


def to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    else:
        return np.asarray(arr)


def add_particles():
    rank = mpi.COMM_WORLD.Get_rank()
    nps = 10 if rank == 0 else 0
    x = xp.linspace(0.005, 0.025, nps)
    y = xp.zeros(nps)
    z = xp.linspace(0.005, 0.025, nps)
    ux = xp.random.normal(loc=0, scale=1e3, size=nps)
    uy = xp.random.normal(loc=0, scale=1e3, size=nps)
    uz = xp.random.normal(loc=0, scale=1e3, size=nps)
    w = xp.ones(nps) * 2.0
    newPid = xp.ones(nps) * 5.0

    electrons.add_particles(
        x=to_numpy(x),
        y=to_numpy(y),
        z=to_numpy(z),
        ux=to_numpy(ux),
        uy=to_numpy(uy),
        uz=to_numpy(uz),
        w=to_numpy(w),
        newPid=to_numpy(newPid),
    )


callbacks.installbeforestep(add_particles)

##########################
# simulation run
##########################

step_number = sim.extension.warpx.getistep(lev=0)
sim.step(max_steps - 1 - step_number)

#######################################
# check that the new PIDs are properly set
#######################################

assert electrons.size == 90
assert electrons.get_real_comp_index("w") == 2
assert electrons.get_real_comp_index("newPid") == 6

for pti in electrons.iterator(level=0):
    vals = pti["newPid"]
    assert np.allclose(vals, 5)

##########################
# take the final sim step
##########################

sim.step(1)

if not sim.amr_restart:
    del electrons
