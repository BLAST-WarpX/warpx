#!/usr/bin/env python3
#
# --- Simple example of Langmuir oscillations in a uniform plasma
# --- in two dimensions

from pywarpx import picmi

constants = picmi.constants

##########################
# physics parameters
##########################

plasma_density = 1.0e25
plasma_xmin = 0.0
plasma_x_velocity = 0.1 * constants.c

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 40
diagnostic_intervals = "::10"

# --- Grid
nx = 64
ny = 64

xmin = -20.0e-6
ymin = -20.0e-6
xmax = +20.0e-6
ymax = +20.0e-6

number_per_cell_each_dim = [2, 2]

##########################
# physics components
##########################

uniform_plasma = picmi.UniformDistribution(
    density=1.0e25,
    upper_bound=[0.0, None, None],
    directed_velocity=[0.1 * constants.c, 0.0, 0.0],
)

electrons = picmi.Species(
    particle_type="electron", name="electrons", initial_distribution=uniform_plasma
)

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    moving_window_velocity=[0.0, 0.0, 0.0],
    warpx_max_grid_size=32,
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=1.0)

##########################
# diagnostics
##########################

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["Ex", "Jx"],
)

part_diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
    species=[electrons],
    data_list=["weighting", "ux"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=0,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)

sim.add_diagnostic(field_diag1)
sim.add_diagnostic(part_diag1)

##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
sim.write_input_file(file_name="inputs2d_from_PICMI")

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
