#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

# We only run 100 steps for tests
# Disable `max_step` below to run until the physical `stop_time`.
max_step = 100
# time-scale with highly kinetic dynamics
stop_time = 0.2e-12

# proper resolution for 30 n_c (dx<=3.33nm) incl. acc. length
# (>=6x V100)
# --> choose larger `max_grid_size` and `blocking_factor` for 1 to 8 grids per GPU accordingly
# nx = 7488
# nz = 14720

# Number of cells
nx = 384
nz = 512

# Domain decomposition (deactivate `warpx_numprocs` in `picmi.Simulation` for this to take effect)
max_grid_size = 64
blocking_factor = 32

# Physical domain
xmin = -7.5e-06
xmax = 7.5e-06
zmin = -5.0e-06
zmax = 25.0e-06

# Create grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=["open", "open"],
    upper_boundary_conditions=["open", "open"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
)

# Particles: plasma parameters
# critical plasma density
nc = 1.742e27  # [m^-3]  1.11485e21 * 1.e6 / 0.8**2
# number density: "fully ionized" electron density as reference
#   [material 1] cryogenic H2
n0 = 30.0  # [n_c]
#   [material 2] liquid crystal
# n0 = 192
#   [material 3] PMMA
# n0 = 230
#   [material 4] Copper (ion density: 8.49e28/m^3; times ionization level)
# n0 = 1400
plasma_density = n0 * nc
preplasma_L = 0.05e-6  # [m] scale length (>0)
preplasma_Lcut = 2.0e-6  # [m] hard cutoff from surface
plasma_r0 = 2.5e-6  # [m] radius or half-thickness
plasma_eps_z = 0.05e-6  # [m] small offset in z to make zmin, zmax interval larger than 2*(r0 + Lcut)
plasma_creation_limit_z = (
    plasma_r0 + preplasma_Lcut + plasma_eps_z
)  # [m] upper limit in z for particle creation

plasma_xmin = None
plasma_ymin = None
plasma_zmin = -plasma_creation_limit_z
plasma_xmax = None
plasma_ymax = None
plasma_zmax = plasma_creation_limit_z

density_expression_str = f"{plasma_density}*((abs(z)<={plasma_r0}) + (abs(z)<{plasma_r0}+{preplasma_Lcut}) * (abs(z)>{plasma_r0}) * exp(-(abs(z)-{plasma_r0})/{preplasma_L}))"

slab_with_ramp_dist_hydrogen = picmi.AnalyticDistribution(
    density_expression=density_expression_str,
    lower_bound=[plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound=[plasma_xmax, plasma_ymax, plasma_zmax],
)

# thermal velocity spread for electrons in gamma*beta
ux_th = 0.01
uz_th = 0.01

slab_with_ramp_dist_electrons = picmi.AnalyticDistribution(
    density_expression=density_expression_str,
    lower_bound=[plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound=[plasma_xmax, plasma_ymax, plasma_zmax],
    # if `momentum_expressions` and `momentum_spread_expressions` are unset,
    # a Gaussian momentum distribution is assumed given that `rms_velocity` has any non-zero elements
    rms_velocity=[c * ux_th, 0.0, c * uz_th],  # thermal velocity spread in m/s
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=slab_with_ramp_dist_electrons,
)

hydrogen = picmi.Species(
    particle_type="proton",
    name="hydrogen",
    initial_distribution=slab_with_ramp_dist_hydrogen,
    warpx_add_real_attributes={"orig_x": "x", "orig_z": "z"},
)

# Laser
# e_max = a0 * 3.211e12 / lambda_0[mu]
#   a0 = 16, lambda_0 = 0.8mu -> e_max = 64.22 TV/m
e_max = 64.22e12
position_z = -4.0e-06
profile_t_peak = 50.0e-15
profile_focal_distance = 4.0e-06
laser = picmi.GaussianLaser(
    wavelength=0.8e-06,
    waist=4.0e-06,
    duration=30.0e-15,
    focal_position=[0, 0, profile_focal_distance + position_z],
    centroid_position=[0, 0, position_z - c * profile_t_peak],
    propagation_direction=[0, 0, 1],
    polarization_direction=[1, 0, 0],
    E0=e_max,
    fill_in=False,
)
laser_antenna = picmi.LaserAntenna(
    position=[0.0, 0.0, position_z], normal_vector=[0, 0, 1]
)

# Electromagnetic solver
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
    cfl=0.999,
    divE_cleaning=0,
    # warpx_pml_ncell=10
)

# Diagnostics
particle_diag = picmi.ParticleDiagnostic(
    name="diagInst",
    period=100,
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
    # demonstration of a spatial and momentum filter
    warpx_plot_filter_function="(uz>=0) * (x<1.0e-6) * (x>-1.0e-6)",
)
# reduce resolution of output fields
coarsening_ratio = [4, 4]
ncell_field = []
for ncell_comp, cr in zip([nx, nz], coarsening_ratio):
    ncell_field.append(int(ncell_comp / cr))
field_diag = picmi.FieldDiagnostic(
    name="diagInst",
    grid=grid,
    period=100,
    number_of_cells=ncell_field,
    data_list=["B", "E", "J", "rho", "rho_electrons", "rho_hydrogen"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

field_time_avg_diag = picmi.TimeAveragedFieldDiagnostic(
    name="diagTimeAvg",
    grid=grid,
    period=100,
    number_of_cells=ncell_field,
    data_list=["B", "E", "J", "rho", "rho_electrons", "rho_hydrogen"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
    warpx_time_average_mode="dynamic_start",
    warpx_average_period_time=2.67e-15,
)

particle_fw_diag = picmi.ParticleDiagnostic(
    name="openPMDfw",
    period=100,
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
    warpx_plot_filter_function="(uz>=0) * (x<1.0e-6) * (x>-1.0e-6)",
)

particle_bw_diag = picmi.ParticleDiagnostic(
    name="openPMDbw",
    period=100,
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
    warpx_plot_filter_function="(uz<0)",
)

# histograms with 2.0 degree acceptance angle in fw direction
# 2 deg * pi / 180 : 0.03490658503 rad
# half-angle +/-   : 0.017453292515 rad
histuH_rdiag = picmi.ReducedDiagnostic(
    diag_type="ParticleHistogram",
    name="histuH",
    species=hydrogen,
    bin_number=1000,
    bin_min=0.0,
    bin_max=0.474,  # 100 MeV protons
    histogram_function="u2=ux*ux+uy*uy+uz*uz; if(u2>0, sqrt(u2), 0.0)",
    filter_function="u2=ux*ux+uy*uy+uz*uz; if(u2>0, abs(acos(uz / sqrt(u2))) <= 0.017453, 0)",
)

histue_rdiag = picmi.ReducedDiagnostic(
    diag_type="ParticleHistogram",
    name="histue",
    species=electrons,
    bin_number=1000,
    bin_min=0.0,
    bin_max=197.0,  # 100 MeV electrons
    histogram_function="u2=ux*ux+uy*uy+uz*uz; if(u2>0, sqrt(u2), 0.0)",
    filter_function="u2=ux*ux+uy*uy+uz*uz; if(u2>0, abs(acos(uz / sqrt(u2))) <= 0.017453, 0)",
)

# just a test entry to make sure that the histogram filter is purely optional:
# this one just records uz of all hydrogen ions, independent of their pointing
histuzAll_rdiag = picmi.ReducedDiagnostic(
    diag_type="ParticleHistogram",
    name="histuzAll",
    species=hydrogen,
    bin_number=1000,
    bin_min=-0.474,
    bin_max=0.474,
    histogram_function="uz",
)

field_probe_z_rdiag = picmi.ReducedDiagnostic(
    diag_type="FieldProbe",
    name="FieldProbe_Z",
    integrate=0,
    probe_geometry="Line",
    x_probe=0.0,
    z_probe=-5.0e-6,
    x1_probe=0.0,
    z1_probe=25.0e-6,
    resolution=3712,
)

field_probe_scat_point_rdiag = picmi.ReducedDiagnostic(
    diag_type="FieldProbe",
    name="FieldProbe_ScatPoint",
    integrate=0,
    probe_geometry="Point",
    x_probe=0.0,
    z_probe=15.0e-6,
)

field_probe_scat_line_rdiag = picmi.ReducedDiagnostic(
    diag_type="FieldProbe",
    name="FieldProbe_ScatLine",
    integrate=1,
    probe_geometry="Line",
    x_probe=-2.5e-6,
    z_probe=15.0e-6,
    x1_probe=2.5e-6,
    z1_probe=15e-6,
    resolution=201,
)

load_balance_costs_rdiag = picmi.ReducedDiagnostic(
    diag_type="LoadBalanceCosts",
    name="LBC",
)

# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_time=stop_time,  # need to remove `max_step` to run this far
    verbose=1,
    particle_shape="cubic",
    warpx_numprocs=[1, 2],  # deactivate `numprocs` for dynamic load balancing
    warpx_use_filter=1,
    warpx_reduced_diags_intervals=100,
    warpx_load_balance_intervals=100,
    warpx_load_balance_costs_update="heuristic",
)

# Add plasma electrons
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2]),
    # for more realistic simulations, try to avoid that macro-particles represent more than 1 n_c
    # layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[4,8])
)

# Add hydrogen ions
sim.add_species(
    hydrogen,
    layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2]),
    # for more realistic simulations, try to avoid that macro-particles represent more than 1 n_c
    # layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[4,8])
)

# Add laser
sim.add_laser(laser, injection_method=laser_antenna)

# Add full diagnostics
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(field_time_avg_diag)
sim.add_diagnostic(particle_fw_diag)
sim.add_diagnostic(particle_bw_diag)
# Add reduced diagnostics
sim.add_diagnostic(histuH_rdiag)
sim.add_diagnostic(histue_rdiag)
sim.add_diagnostic(histuzAll_rdiag)
sim.add_diagnostic(field_probe_z_rdiag)
sim.add_diagnostic(field_probe_scat_point_rdiag)
sim.add_diagnostic(field_probe_scat_line_rdiag)
sim.add_diagnostic(load_balance_costs_rdiag)
# TODO: make ParticleHistogram2D available

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name="inputs_2d_picmi")

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_step)
