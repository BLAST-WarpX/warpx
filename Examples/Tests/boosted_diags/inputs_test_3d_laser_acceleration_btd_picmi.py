#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

# Physical domain
xmin = -128.0e-6
xmax = 128.0e-6
ymin = -128.0e-6
ymax = 128.0e-6
zmin = -40.0e-6
zmax = 0.0

# Grid
nx = 32
ny = 32
nz = 64

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["periodic", "periodic", "dirichlet"],
    upper_boundary_conditions=["periodic", "periodic", "dirichlet"],
    lower_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    upper_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    moving_window_velocity=[0.0, 0.0, c],
    warpx_max_grid_size=64,
    warpx_blocking_factor=32,
)

# Solver
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="CKC",
    cfl=1.0,
)

# --- Plasma species

plasma_density = 3.5e24

electron_dist = picmi.UniformDistribution(
    density=plasma_density,
    lower_bound=[-120.0e-6, -120.0e-6, 0.0],
    upper_bound=[120.0e-6, 120.0e-6, 0.003],
    fill_in=True,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=electron_dist,
)

ion_dist = picmi.UniformDistribution(
    density=plasma_density,
    lower_bound=[-120.0e-6, -120.0e-6, 0.0],
    upper_bound=[120.0e-6, 120.0e-6, 0.003],
    fill_in=True,
)
ions = picmi.Species(
    particle_type="proton",
    name="ions",
    initial_distribution=ion_dist,
)

# --- Beam

beam_dist = picmi.GaussianBunchDistribution(
    n_physical_particles=1.0e-14 / q_e,
    rms_bunch_size=[1.0e-6, 1.0e-6, 0.2e-6],
    centroid_position=[0.0, 0.0, -20.0e-6],
    centroid_velocity=[0.0, 0.0, 200000.0 * c],
    rms_velocity=[0.2 * c, 0.2 * c, 20.0 * c],
)
beam = picmi.Species(
    particle_type="electron",
    name="beam",
    initial_distribution=beam_dist,
)

# --- Laser

position_z = -0.1e-6
profile_t_peak = 40.0e-15
profile_focal_distance = 0.5e-3

laser = picmi.GaussianLaser(
    wavelength=0.81e-6,
    waist=45.0e-6,
    duration=20.0e-15,
    focal_position=[0.0, 0.0, profile_focal_distance + position_z],
    centroid_position=[0.0, 0.0, position_z - c * profile_t_peak],
    propagation_direction=[0, 0, 1],
    polarization_direction=[0, 1, 0],
    E0=2.0e12,
    fill_in=False,
)
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 0.0, position_z],
    normal_vector=[0, 0, 1],
)

# --- Back-transformed diagnostics

# diag1: BTD fields (plotfile)
btd_field_diag = picmi.LabFrameFieldDiagnostic(
    name="diag1",
    grid=grid,
    num_snapshots=4,
    dt_snapshots=0.001 / c,
    data_list=["E", "B", "J", "rho"],
    warpx_format="plotfile",
    warpx_buffer_size=32,
)

# diag2: BTD particles with beam sub-sampling (openPMD)
btd_particle_diag = picmi.LabFrameParticleDiagnostic(
    name="diag2",
    grid=grid,
    num_snapshots=4,
    dt_snapshots=0.001 / c,
    species=[beam],
    warpx_random_fraction={beam: 0.5},
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
    warpx_intervals="0:3:2, 1:3:2",
    warpx_buffer_size=32,
)

# --- Simulation

sim = picmi.Simulation(
    solver=solver,
    verbose=1,
    gamma_boost=10.0,
    particle_shape="cubic",
    warpx_use_filter=1,
    warpx_use_fdtd_nci_corr=1,
    warpx_serialize_initial_conditions=1,
    warpx_current_deposition_algo="esirkepov",
    warpx_particle_pusher_algo="vay",
    warpx_zmax_plasma_to_compute_max_step=0.0055,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1, 1, 1]),
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1, 1, 1]),
)
sim.add_species(
    beam,
    layout=picmi.PseudoRandomLayout(n_macroparticles=1000, grid=grid),
)

sim.add_laser(laser, injection_method=laser_antenna)

sim.add_diagnostic(btd_field_diag)
sim.add_diagnostic(btd_particle_diag)

sim.step()
