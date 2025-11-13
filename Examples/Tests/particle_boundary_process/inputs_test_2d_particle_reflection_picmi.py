#!/usr/bin/env python3
#
# --- Input file to test particle reflection off an absorbing boundary

from pywarpx import particle_containers, picmi

constants = picmi.constants

##########################
# numerics parameters
##########################

dt = 7.5e-12

# --- Nb time steps

max_steps = 10

# --- grid

nx = 64
nz = 64

xmin = -125e-6
zmin = -149e-6
xmax = 125e-6
zmax = 1e-6


##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["open", "absorbing"],
    upper_boundary_conditions_particles=["open", "absorbing"],
    warpx_max_grid_size=32,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-6,
    warpx_self_fields_verbosity=0,
)

# embedded_boundary = picmi.EmbeddedBoundary(
#    implicit_function="-max(max(x-12.5e-6,-12.5e-6-x),max(z+6.15e-5,-8.65e-5-z))"
# )

##########################
# physics components
##########################

uniform_plasma_elec = picmi.UniformDistribution(
    density=1e15,  # number of electrons per m^3
    lower_bound=[-1e-5, -1e-5, -125e-6],
    upper_bound=[1e-5, 1e-5, -120e-6],
    directed_velocity=[0.0, 0.0, 5e6],  # uth the std of the (unitless) momentum
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=uniform_plasma_elec,
    warpx_save_particles_at_zhi=1,
    warpx_save_particles_at_zlo=1,
    warpx_reflection_model_zhi="0.5",
)

##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
)
field_diag = picmi.FieldDiagnostic(
    grid=grid,
    name="diag1",
    data_list=["E"],
    period=10,
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    # warpx_embedded_boundary=embedded_boundary,
    verbose=1,
)

sim.add_species(
    electrons, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[5, 2], grid=grid)
)
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################

sim.step(max_steps)

################################################
# check that the wrappers to access the particle
# buffer functions as intended
################################################

buffer = particle_containers.ParticleBoundaryBufferWrapper()

n = buffer.get_particle_boundary_buffer_size("electrons", "z_hi")
print("Number of electrons in upper buffer:", n)
assert n == 63

n = buffer.get_particle_boundary_buffer_size("electrons", "z_lo")
print("Number of electrons in lower buffer:", n)
assert n == 67

scraped_steps = buffer.get_particle_boundary_buffer(
    "electrons", "z_hi", "stepScraped", 0
)
for arr in scraped_steps:
    # print(arr)
    assert all(arr == 4)

scraped_steps = buffer.get_particle_boundary_buffer(
    "electrons", "z_lo", "stepScraped", 0
)
for arr in scraped_steps:
    # print(arr)
    assert all(arr == 8)
