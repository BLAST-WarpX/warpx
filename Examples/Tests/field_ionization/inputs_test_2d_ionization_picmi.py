#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c

# Number of time steps
max_steps = 1600

# Number of cells
nx = 16
nz = 800

# Physical domain
xmin = -5e-06
xmax = 5e-06
zmin = 0e-06
zmax = 20e-06

# Domain decomposition
max_grid_size = 64
blocking_factor = 16

# Create grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=["periodic", "open"],
    upper_boundary_conditions=["periodic", "open"],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
)

# Particles: electrons and ions
ions_density = 1
ions_xmin = None
ions_ymin = None
ions_zmin = 5e-06
ions_xmax = None
ions_ymax = None
ions_zmax = 15e-06
uniform_distribution = picmi.UniformDistribution(
    density=ions_density,
    lower_bound=[ions_xmin, ions_ymin, ions_zmin],
    upper_bound=[ions_xmax, ions_ymax, ions_zmax],
    fill_in=True,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    warpx_add_real_attributes={"orig_z": "z"},
)
ions = picmi.Species(
    particle_type="N",
    name="ions",
    charge_state=2,
    initial_distribution=uniform_distribution,
    warpx_add_real_attributes={"orig_z": "z"},
)

# Field ionization
nitrogen_ionization = picmi.FieldIonization(
    model="ADK",  # Ammosov-Delone-Krainov model
    ionized_species=ions,
    product_species=electrons,
)

# Laser
position_z = 3e-06
profile_t_peak = 60.0e-15
laser = picmi.GaussianLaser(
    wavelength=0.8e-06,
    waist=1e10,
    duration=26.685e-15,
    focal_position=[0, 0, position_z],
    centroid_position=[0, 0, position_z - c * profile_t_peak],
    propagation_direction=[0, 0, 1],
    polarization_direction=[1, 0, 0],
    a0=1.8,
    fill_in=False,
)
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 0.0, position_z], normal_vector=[0, 0, 1]
)

# Electromagnetic solver
solver = picmi.ElectromagneticSolver(grid=grid, method="CKC", cfl=0.999)

# Diagnostics
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10000,
    species=[electrons, ions],
    data_list=["ux", "uy", "uz", "x", "z", "weighting", "orig_z"],
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10000,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
)

# Set up simulation
sim = picmi.Simulation(
    solver=solver, max_steps=max_steps, particle_shape="linear", warpx_use_filter=0
)

# Add electrons and ions
sim.add_species(
    electrons, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[0, 0, 0])
)
sim.add_species(
    ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 1, 1])
)

# Add field ionization
sim.add_interaction(nitrogen_ionization)

# Add laser
sim.add_laser(laser, injection_method=laser_antenna)

# Add diagnostics
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_2d_picmi")

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_steps)
