#!/usr/bin/env python3

from pywarpx import picmi

constants = picmi.constants

# Constants from the focusing beam example
nano = 1.0e-9
micro = 1.0e-6

sigmax = 516.0 * nano
sigmay = 7.7 * nano
sigmaz = 300.0 * micro

# Box dimensions
Lx = 20 * sigmax
Ly = 20 * sigmay
Lz = 20 * sigmaz

nx = 256
ny = 256
nz = 256

xmin = -0.5 * Lx
xmax = 0.5 * Lx
ymin = -0.5 * Ly
ymax = 0.5 * Ly
zmin = -0.5 * Lz
zmax = 0.5 * Lz

em_order = 3

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_max_grid_size=256,
)

solver = picmi.ElectromagneticSolver(grid=grid)

# Create species with external file injection

beam1_distribution = picmi.FromFileDistribution(
    file_path="../test_3d_focusing_gaussian_beam_from_openpmd_prepare/openpmd_generated_particles.h5",
)

beam1 = picmi.Species(
    particle_type="electron",
    name="beam1",
    initial_distribution=beam1_distribution,
)

diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    period=1,
    warpx_dump_last_timestep=1,
)

openpmd = picmi.ParticleDiagnostic(
    name="openpmd",
    period=1,
    species=[beam1],
    data_list=["weighting", "x", "y", "z"],
    warpx_format="openpmd",
    warpx_dump_last_timestep=1,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=0,
    verbose=1,
)

sim.add_species(beam1, layout=None)

sim.add_diagnostic(diag1)
sim.add_diagnostic(openpmd)

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
# sim.write_input_file(file_name = 'inputs_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
