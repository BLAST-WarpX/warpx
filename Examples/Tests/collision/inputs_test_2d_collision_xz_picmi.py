#!/usr/bin/env python3
#
# --- Input file for binary Coulomb collision testing. This input script
# --- runs the same test as inputs_test_2d_collision_xz but via Python, therefore the input
# --- values where directly copied from inputs_2d.

from pywarpx import picmi

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 150

max_grid_size = 8
max_level = 0
nx = max_grid_size
ny = max_grid_size

xmin = 0
xmax = 4.154046151855669e2
ymin = xmin
ymax = xmax

plasma_density = 1e21
elec_rms_velocity = 0.044237441120300 * constants.c
ion_rms_velocity = 0.006256118919701 * constants.c
number_per_cell = 200

#################################
############ NUMERICS ###########
#################################
serialize_initial_conditions = 1
verbose = 1
cfl = 1.0

# Order of particle shape factors
particle_shape = 1

#################################
############ PLASMA #############
#################################

elec_dist = picmi.UniformDistribution(
    density=plasma_density,
    rms_velocity=[elec_rms_velocity] * 3,
    directed_velocity=[elec_rms_velocity, 0.0, 0.0],
)

ion_dist = picmi.UniformDistribution(
    density=plasma_density,
    rms_velocity=[ion_rms_velocity] * 3,
)

electrons = picmi.Species(
    particle_type="electron",
    name="electron",
    warpx_do_not_deposit=1,
    initial_distribution=elec_dist,
)
ions = picmi.Species(
    particle_type="H",
    name="ion",
    charge="q_e",
    mass="5*m_e",
    warpx_do_not_deposit=1,
    initial_distribution=ion_dist,
)

#################################
########## COLLISIONS ###########
#################################

collision1 = picmi.CoulombCollisions(
    name="collisions1", species=[electrons, ions], CoulombLog=15.9
)
collision2 = picmi.CoulombCollisions(
    name="collisions2", species=[electrons, electrons], CoulombLog=15.9
)
collision3 = picmi.CoulombCollisions(
    name="collisions3", species=[ions, ions], CoulombLog=15.9
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=max_grid_size,
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(grid=grid, cfl=cfl)

#################################
######### DIAGNOSTICS ###########
#################################

particle_diag = picmi.ParticleDiagnostic(name="diag1", period=10)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=[],
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=verbose,
    warpx_serialize_initial_conditions=serialize_initial_conditions,
    warpx_collisions=[collision1, collision2, collision3],
)

sim.add_species(
    electrons,
    layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=number_per_cell, grid=grid
    ),
)
sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=number_per_cell, grid=grid
    ),
)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

#################################
##### SIMULATION EXECUTION ######
#################################

# sim.write_input_file('inputs_test_2d_collision_xz')
sim.step(max_steps)
