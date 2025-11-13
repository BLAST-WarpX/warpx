#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

# Number of time steps
max_steps = 10

# Number of cells
nr = 64
nz = 512

# Physical domain
rmin = 0
rmax = 30e-06
zmin = -56e-06
zmax = 12e-06

# Domain decomposition
max_grid_size = 64
blocking_factor = 32

# Create grid
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=2,
    lower_bound=[rmin, zmin],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    moving_window_velocity=[0.0, c],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
)

# Particles: plasma electrons
plasma_density = 2e23
plasma_xmin = -20e-06
plasma_ymin = None
plasma_zmin = 10e-06
plasma_xmax = 20e-06
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

# Particles: beam electrons
q_tot = 1e-12
x_m = 0.0
y_m = 0.0
z_m = -28e-06
x_rms = 0.5e-06
y_rms = 0.5e-06
z_rms = 0.5e-06
ux_m = 0.0
uy_m = 0.0
uz_m = 500.0
ux_th = 2.0
uy_th = 2.0
uz_th = 50.0
gaussian_bunch_distribution = picmi.GaussianBunchDistribution(
    n_physical_particles=q_tot / q_e,
    rms_bunch_size=[x_rms, y_rms, z_rms],
    rms_velocity=[c * ux_th, c * uy_th, c * uz_th],
    centroid_position=[x_m, y_m, z_m],
    centroid_velocity=[c * ux_m, c * uy_m, c * uz_m],
)
beam = picmi.Species(
    particle_type="electron",
    name="beam",
    initial_distribution=gaussian_bunch_distribution,
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
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=1.0, divE_cleaning=0)

# Diagnostics
diag_field_list = ["B", "E", "J", "rho"]
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=diag_field_list,
    warpx_dump_rz_modes=1,
)
diag_particle_list = ["weighting", "momentum"]
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
    species=[electrons, beam],
    data_list=diag_particle_list,
)

# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape="cubic",
    warpx_use_filter=0,
)

# Add plasma electrons
sim.add_species(
    electrons, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1, 4, 1])
)

# Add beam electrons
sim.add_species(beam, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles=100))

# Add laser
sim.add_laser(laser, injection_method=laser_antenna)

# Add diagnostics
sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_diag)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_rz_picmi")

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_steps)
