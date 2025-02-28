#!/usr/bin/env python3
import math
from pywarpx import picmi

# -----------------------------
# General Simulation Parameters
# -----------------------------
max_steps = 20

# -----------------------------
# User-defined Constants (from my_constants)
# -----------------------------
zc    = 20.e-6
zp    = 20.05545177444479562e-6
lgrad = 0.08e-6
nc    = 1.74e27
zp2   = 24.e-6
zc2   = 24.05545177444479562e-6

# -----------------------------
# Simulation Domain Setup
# -----------------------------
nx, nz = 256, 128
xmin, zmin = -100e-6, 0.0
xmax, zmax = 100e-6, 100e-6

# Use 'open' as the absorbing boundary (since "pml" is not supported by PICMI)
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=['open', 'open'],
    upper_boundary_conditions=['open', 'open'],
    warpx_max_grid_size=128,
    warpx_blocking_factor=32,
)

# -----------------------------
# Numerical Solver Setup
# -----------------------------
solver = picmi.ElectromagneticSolver(
    grid=grid, method="Yee", cfl=1.0
)

# -----------------------------
# Plasma Species Initialization
# -----------------------------
# For electrons:
zmin_e = zc - lgrad * math.log(400)
zmax_e = 25.47931e-6
electrons_distribution = picmi.UniformDistribution(
    density=nc,  # constant density placeholder
    lower_bound=[xmin, 0.0, zmin_e],
    upper_bound=[xmax, 0.0, zmax_e],
    fill_in=True,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    charge=-picmi.constants.q_e,
    mass=picmi.constants.m_e,
    initial_distribution=electrons_distribution,
)

# For ions:
zmin_i = 19.520e-6
zmax_i = 25.47931e-6
ions_distribution = picmi.UniformDistribution(
    density=nc,  # constant density placeholder
    lower_bound=[xmin, 0.0, zmin_i],
    upper_bound=[xmax, 0.0, zmax_i],
    fill_in=True,
)
ions = picmi.Species(
    particle_type="proton",
    name="ions",
    charge=picmi.constants.q_e,
    mass=picmi.constants.m_p,
    initial_distribution=ions_distribution,
)

# -----------------------------
# Laser Initialization
# -----------------------------
# Use 3-element vectors for laser parameters.
# Original input:
#   position     = 0. 0. 5.e-6
#   direction    = 0. 0. 1.
#   polarization = 1. 0. 0.
#   e_max        = 4.e12, wavelength = 0.8e-6,
#   profile      = Gaussian, profile_waist = 5.e-6,
#   profile_duration = 15.e-15, profile_t_peak = 25.e-15,
#   profile_focal_distance = 15.e-6
laser1 = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5e-6,
    duration=15e-15,
    # Provide a 3-element focal_position: [x, y, z]
    focal_position=[0.0, 0.0, 15e-6 + 5e-6],
    # Provide a 3-element centroid_position: [x, y, z]
    centroid_position=[0.0, 0.0, 5e-6 - picmi.constants.c * 25e-15],
    propagation_direction=[0, 0, 1],
    polarization_direction=[1, 0, 0],
    E0=4.e12,
)
# Similarly, provide 3-element vectors for the laser antenna.
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 0.0, 5e-6],
    normal_vector=[0, 0, 1],
)

# -----------------------------
# Diagnostics Setup
# -----------------------------
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=["E", "B", "J", "rho"],
)

# -----------------------------
# Simulation Setup and Execution
# -----------------------------
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape="cubic",
    warpx_use_filter=1,
    warpx_serialize_initial_conditions=1,
)

# Use GriddedLayout with 2 macroparticles per cell (2D)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2])
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2])
)

sim.add_laser(laser1, injection_method=laser_antenna)
sim.add_diagnostic(field_diag)

sim.write_input_file(file_name="inputs_2d_picmi.txt")
sim.initialize_inputs()
sim.initialize_warpx()
sim.step(max_steps)

