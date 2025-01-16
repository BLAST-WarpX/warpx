#!/usr/bin/env python3
#
# This is a script that adds particle components at runtime,
# then performs checkpoint / restart and compares the result
# to the original simulation.

import sys

import numpy as np

from pywarpx import callbacks, particle_containers, picmi

##########################
# physics parameters
##########################

dt = 7.5e-10

##########################
# numerics parameters
##########################

max_steps = 10

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
np.random.seed(30025025)

electron_wrapper = particle_containers.ParticleContainerWrapper("electrons")
electron_wrapper.add_real_comp("newPid")


def add_particles():
    nps = 10
    x = np.linspace(0.005, 0.025, nps)
    y = np.zeros(nps)
    z = np.linspace(0.005, 0.025, nps)
    ux = np.random.normal(loc=0, scale=1e3, size=nps)
    uy = np.random.normal(loc=0, scale=1e3, size=nps)
    uz = np.random.normal(loc=0, scale=1e3, size=nps)
    w = np.ones(nps) * 2.0
    newPid = 5.0

    electron_wrapper.add_particles(
        x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, newPid=newPid
    )


callbacks.installbeforestep(add_particles)

##########################
# simulation run
##########################

step_number = sim.extension.warpx.getistep(lev=0)
sim.step(max_steps - 1 - step_number)

##########################
# check that the new PIDs are properly set
##########################

assert electron_wrapper.nps == 90
assert electron_wrapper.particle_container.get_comp_index("w") == 2
assert electron_wrapper.particle_container.get_comp_index("newPid") == 6

new_pid_vals = electron_wrapper.get_particle_real_arrays("newPid", 0)
for vals in new_pid_vals:
    assert np.allclose(vals, 5)

##########################
# take the final sim step
##########################

sim.step(1)
