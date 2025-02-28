#!/usr/bin/env python3
import math
from pywarpx import picmi

# -----------------------------
# General Simulation Parameters
# -----------------------------
max_steps = 20

# -----------------------------
# User-defined Constants (my_constants from input file)
# -----------------------------
zc    = 20.e-6
zp    = 20.05545177444479562e-6
lgrad = 0.08e-6
nc    = 1.74e27
zp2   = 24.e-6
zc2   = 24.05545177444479562e-6

# -----------------------------
# Simulation Domain Setup (from geometry.prob_lo/prob_hi and amr.n_cell)
# -----------------------------
# geometry.prob_lo = -100.e-6   0.
# geometry.prob_hi =  100.e-6   100.e-6
# amr.n_cell = 256 128
nx, nz = 256, 128
xmin, zmin = -100e-6, 0.0
xmax, zmax = 100e-6, 100e-6

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    # boundary.field_lo and boundary.field_hi are both set to 'pml'
    lower_boundary_conditions=['pml', 'pml'],
    upper_boundary_conditions=['pml', 'pml'],
    warpx_max_grid_size=128,          # from amr.max_grid_size
    warpx_blocking_factor=32,         # from amr.blocking_factor
)

# -----------------------------
# Numerical Solver Setup
# -----------------------------
# warpx.cfl = 1.0 and warpx.use_filter = 1 are used here.
solver = picmi.ElectromagneticSolver(
    grid=grid, method="Yee", cfl=1.0
)

# -----------------------------
# Plasma Density Function
# -----------------------------
# electrons.density_function and ions.density_function are identical:
# "if(z<zp, nc*exp((z-zc)/lgrad), if(z<=zp2, 2.*nc, nc*exp(-(z-zc2)/lgrad)))"
def plasma_density(x, y, z):
    if z < zp:
        return nc * math.exp((z - zc) / lgrad)
    elif z <= zp2:
        return 2.0 * nc
    else:
        return nc * math.exp(-(z - zc2) / lgrad)

# -----------------------------
# Plasma Species Initialization
# -----------------------------
# For electrons:
# electrons.zmin = "zc-lgrad*log(400)"
# electrons.zmax = 25.47931e-6
zmin_e = zc - lgrad * math.log(400)
zmax_e = 25.47931e-6

electrons_distribution = picmi.UniformDistribution(
    # The density value here is a placeholder;
    # the actual spatial dependence is provided by plasma_density.
    density=1.0,
    lower_bound=[xmin, zmin_e],
    upper_bound=[xmax, zmax_e],
    fill_in=True,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    charge=-picmi.constants.q_e,
    mass=picmi.constants.m_e,
    initial_distribution=electrons_distribution,
    # The density function implements the parsed density profile.
    density_function=plasma_density,
    # electrons.momentum_distribution_type = "gaussian" with ux_th = 0.01 and uz_th = 0.01.
    # PICMI does not directly support specifying a momentum spread;
    # custom initialization would be required to exactly reproduce this.
)

# For ions:
# ions.zmin = 19.520e-6 and ions.zmax = 25.47931e-6
zmin_i = 19.520e-6
zmax_i = 25.47931e-6

ions_distribution = picmi.UniformDistribution(
    density=1.0,
    lower_bound=[xmin, zmin_i],
    upper_bound=[xmax, zmax_i],
    fill_in=True,
)
ions = picmi.Species(
    particle_type="proton",  # assuming ions are protons (m_p)
    name="ions",
    charge=picmi.constants.q_e,
    mass=picmi.constants.m_p,
    initial_distribution=ions_distribution,
    density_function=plasma_density,
    # ions.momentum_distribution_type = "at_rest" is default (particles start with zero momentum)
)

# -----------------------------
# Laser Initialization
# -----------------------------
# From the input:
# laser1.position     = 0. 0. 5.e-6   --> in 2D, use [0, 5e-6] (x,z)
# laser1.direction    = 0. 0. 1.       --> propagation_direction: [0, 1]
# laser1.polarization = 1. 0. 0.       --> polarization_direction: [1, 0]
# laser1.e_max        = 4.e12         (V/m)
# laser1.wavelength   = 0.8e-6        (m)
# laser1.profile      = Gaussian
# laser1.profile_waist = 5.e-6       (m)
# laser1.profile_duration = 15.e-15   (s)
# laser1.profile_t_peak = 25.e-15     (s)
# laser1.profile_focal_distance = 15.e-6 (m)
laser1 = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.e-6,
    duration=15e-15,
    # In 2D, positions are [x, z]. Following the 3D convention:
    focal_position=[0.0, 15e-6 + 5e-6],  # focal distance added to laser plane z-position
    centroid_position=[0.0, 5e-6 - picmi.constants.c * 25e-15],
    propagation_direction=[0, 1],
    polarization_direction=[1, 0],
    E0=4.e12,
)
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 5e-6],
    normal_vector=[0, 1],
)

# -----------------------------
# Diagnostics Setup
# -----------------------------
# diagnostics.diags_names = diag1, with diag1.intervals = 10 and diag1.diag_type = Full
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
    particle_shape="cubic",  # corresponds to order 3 particle shape
    warpx_use_filter=1,
    warpx_serialize_initial_conditions=1,
    # Additional parameters (e.g. algo.load_balance_intervals) are not directly exposed via PICMI.
)

# Add species using a GriddedLayout with 2 particles per cell in each direction (as specified)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticles_per_cell=[2, 2])
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticles_per_cell=[2, 2])
)

# Add the laser with its antenna injection method
sim.add_laser(laser1, injection_method=laser_antenna)

# Add the field diagnostic
sim.add_diagnostic(field_diag)

# (Optional) Write a text file with the input parameters for inspection
sim.write_input_file(file_name="inputs_2d_picmi.txt")

# Initialize inputs and the WarpX instance, then advance the simulation
sim.initialize_inputs()
sim.initialize_warpx()
sim.step(max_steps)

