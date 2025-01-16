#!/usr/bin/env python3

from pywarpx import picmi

##########################
# physics parameters
##########################

V_xmin = "150.0*sin(2*pi*6.78e6*t)"
V_xmax = "450.0*sin(2*pi*13.56e6*t)"


##########################
# numerics parameters
##########################

dt = 7.5e-10

# --- Nb time steps

max_steps = 100

# --- grid

nx = 64
ny = 8

xmin = 0
xmax = 0.032
ymin = 0
ymax = 0.004


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
    warpx_potential_lo_x=V_xmin,
    warpx_potential_hi_x=V_xmax,
    moving_window_velocity=None,
    warpx_max_grid_size=32,
)

solver = picmi.ElectrostaticSolver(
    grid=grid, method="Multigrid", required_precision=1e-6
)


##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(name="diag1", period=4)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=4,
    data_list=["phi"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    particle_shape=None,
    verbose=0,
)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
# sim.write_input_file(file_name = 'inputs_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step(max_steps)
