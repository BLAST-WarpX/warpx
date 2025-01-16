#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

# Number of time steps
max_steps = 100

# Number of cells
nz = 256

# Physical domain
zmin = -56e-06
zmax = 12e-06

# Domain decomposition
max_grid_size = 64
blocking_factor = 32

# Create grid
grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["dirichlet"],
    upper_boundary_conditions=["dirichlet"],
    lower_boundary_conditions_particles=["absorbing"],
    upper_boundary_conditions_particles=["absorbing"],
    moving_window_velocity=[c],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
)

# Particles: plasma electrons
plasma_density = 2e23
plasma_xmin = None
plasma_ymin = None
plasma_zmin = 10e-06
plasma_xmax = None
plasma_ymax = None
plasma_zmax = None
uniform_distribution = picmi.UniformDistribution(
    density=plasma_density,
    lower_bound=[plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound=[plasma_xmax, plasma_ymax, plasma_zmax],
    fill_in=True,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=uniform_distribution,
)

# Laser
e_max = 16e12
position_z = 9e-06
profile_t_peak = 30.0e-15
profile_focal_distance = 100e-06
laser = picmi.GaussianLaser(
    wavelength=0.8e-06,
    waist=5e-06,
    duration=15e-15,
    focal_position=[0, 0, profile_focal_distance + position_z],
    centroid_position=[0, 0, position_z - c * profile_t_peak],
    propagation_direction=[0, 0, 1],
    polarization_direction=[0, 1, 0],
    E0=e_max,
    fill_in=False,
)
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 0.0, position_z], normal_vector=[0, 0, 1]
)

# Electromagnetic solver
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9, divE_cleaning=0)

# Diagnostics
diag_field_list = ["B", "E", "J", "rho"]
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=100,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=100,
    data_list=diag_field_list,
)

# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape="cubic",
    warpx_use_filter=1,
    warpx_serialize_initial_conditions=1,
    warpx_do_dynamic_scheduling=0,
)

# Add plasma electrons
sim.add_species(
    electrons, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[10])
)

# Add laser
sim.add_laser(laser, injection_method=laser_antenna)

# Add diagnostics
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_1d_picmi")

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_steps)
