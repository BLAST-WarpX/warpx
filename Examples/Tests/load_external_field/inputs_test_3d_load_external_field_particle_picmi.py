#!/usr/bin/env python3
#
# --- Input file for loading initial field from openPMD file.

from pywarpx import picmi

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 300

max_grid_size = 40
nx = max_grid_size
ny = max_grid_size
nz = max_grid_size

xmin = -1
xmax = 1
ymin = xmin
ymax = xmax
zmin = 0
zmax = 5

number_per_cell = 200

#################################
############ NUMERICS ###########
#################################

verbose = 1
dt = 4.4e-7
use_filter = 0

# Order of particle shape factors
particle_shape = 1

#################################
############ PLASMA #############
#################################

ion_dist = picmi.ParticleListDistribution(
    x=0.0,
    y=0.2,
    z=2.5,
    ux=9.5e-05 * constants.c,
    uy=0.0 * constants.c,
    uz=1.34e-4 * constants.c,
    weight=1.0,
)

ions = picmi.Species(
    particle_type="H",
    name="proton",
    charge="q_e",
    mass="m_p",
    warpx_do_not_deposit=1,
    initial_distribution=ion_dist,
)

#################################
######## INITIAL FIELD ##########
#################################

applied_field = picmi.LoadAppliedField(
    read_fields_from_path="../../../../openPMD-example-datasets/example-femm-3d.h5",
    load_E=False,
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)
solver = picmi.ElectrostaticSolver(grid=grid)

#################################
######### DIAGNOSTICS ###########
#################################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=300,
    species=[ions],
    data_list=["ux", "uy", "uz", "x", "y", "z", "weighting"],
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=300,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=verbose,
    warpx_serialize_initial_conditions=False,
    warpx_grid_type="collocated",
    warpx_do_dynamic_scheduling=False,
    warpx_use_filter=use_filter,
    time_step_size=dt,
    particle_shape=particle_shape,
)

sim.add_applied_field(applied_field)

sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=number_per_cell, grid=grid
    ),
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_diag)

#################################
##### SIMULATION EXECUTION ######
#################################

# sim.write_input_file('inputs_test_3d_load_external_field_particle')
sim.step(max_steps)
