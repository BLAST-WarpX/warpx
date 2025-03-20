#!/usr/bin/env python3
#
# Mirroring Tests/ParticleBoundaryScrape setup, this tests that restarting
# with an EB works properly.

import sys

from pywarpx import picmi

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 60
diagnostic_intervals = 30

# --- Grid
nx = 64
ny = 64
nz = 128

cfl = 0.99

xmin = -125e-6
ymin = -125e-6
zmin = -149e-6
xmax = 125e-6
ymax = 125e-6
zmax = 1e-6

##########################
# physics components
##########################

uniform_plasma_elec = picmi.UniformDistribution(
    density=1e23,  # number of electrons per m^3
    lower_bound=[-1e-5, -1e-5, -149e-6],
    upper_bound=[1e-5, 1e-5, -129e-6],
    directed_velocity=[
        0.0,
        0.0,
        2000.0 * picmi.constants.c,
    ],  # uth the std of the (unitless) momentum
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=uniform_plasma_elec,
    warpx_save_particles_at_xhi=1,
    warpx_save_particles_at_eb=1,
)

##########################
# numerics components
##########################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["none", "none", "none"],
    upper_boundary_conditions=["none", "none", "none"],
    lower_boundary_conditions_particles=["open", "open", "open"],
    upper_boundary_conditions_particles=["open", "open", "open"],
    warpx_max_grid_size=32,
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=cfl)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="-max(max(max(x-12.5e-6,-12.5e-6-x),max(y-12.5e-6,-12.5e-6-y)),max(z-(-6.15e-5),-8.65e-5-z))"
)

##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
)

checkpoint = picmi.Checkpoint(
    name="chk",
    period=diagnostic_intervals,
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    warpx_embedded_boundary=embedded_boundary,
    verbose=True,
    warpx_load_balance_intervals=40,
    warpx_load_balance_efficiency_ratio_threshold=0.9,
)

sim.add_species(
    electrons, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[1, 1, 1], grid=grid)
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
# simulation run
##########################

step_number = sim.extension.warpx.getistep(lev=0)
sim.step(max_steps - step_number)
