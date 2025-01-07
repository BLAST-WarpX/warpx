#!/usr/bin/env python3

from pywarpx import picmi

# from warp import picmi

constants = picmi.constants

nz = 64

zmin = -200.0e-6
zmax = +200.0e-6

moving_window_velocity = [constants.c]

number_per_cell_each_dim = [10]

max_steps = 1000

grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["dirichlet"],
    upper_boundary_conditions=["dirichlet"],
    lower_boundary_conditions_particles=["absorbing"],
    upper_boundary_conditions_particles=["absorbing"],
    moving_window_velocity=moving_window_velocity,
    warpx_max_grid_size=32,
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=0.999)

beam_distribution = picmi.UniformDistribution(
    density=1.0e23,
    lower_bound=[None, None, -150.0e-6],
    upper_bound=[None, None, -100.0e-6],
    directed_velocity=[0.0, 0.0, 1.0e9],
)

plasma_distribution = picmi.UniformDistribution(
    density=1.0e22,
    lower_bound=[None, None, 0.0],
    upper_bound=[None, None, None],
    fill_in=True,
)

beam = picmi.Species(
    particle_type="electron", name="beam", initial_distribution=beam_distribution
)
plasma = picmi.Species(
    particle_type="electron", name="plasma", initial_distribution=plasma_distribution
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=0,
)

sim.add_species(
    beam,
    layout=picmi.GriddedLayout(
        grid=grid, n_macroparticle_per_cell=number_per_cell_each_dim
    ),
)
sim.add_species(
    plasma,
    layout=picmi.GriddedLayout(
        grid=grid, n_macroparticle_per_cell=number_per_cell_each_dim
    ),
)

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Ex", "Ey", "Ez", "Jx", "Jy", "Jz", "part_per_cell"],
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=max_steps,
    species=[beam, plasma],
    data_list=["ux", "uy", "uz", "weighting"],
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
# sim.write_input_file(file_name = 'inputs_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
