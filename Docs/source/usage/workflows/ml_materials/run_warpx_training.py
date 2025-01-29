#!/usr/bin/env python3
import math

import numpy as np

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e
m_e = picmi.constants.m_e
m_p = picmi.constants.m_p
ep0 = picmi.constants.ep0

# Number of cells
dim = "3"
nx = ny = 128
nz = 35328  # 17664 #8832
if dim == "rz":
    nr = nx // 2

# Computational domain
rmin = 0.0
rmax = 128e-6
zmin = -180e-6
zmax = 0.0

# Number of processes for static load balancing
# Check with your submit script
num_procs = [1, 1, 64 * 4]
if dim == "rz":
    num_procs = [1, 64]

# Number of time steps
gamma_boost = 60.0
beta_boost = np.sqrt(1.0 - gamma_boost**-2)

# Create grid
if dim == "rz":
    grid = picmi.CylindricalGrid(
        number_of_cells=[nr, nz],
        guard_cells=[32, 32],
        n_azimuthal_modes=2,
        lower_bound=[rmin, zmin],
        upper_bound=[rmax, zmax],
        lower_boundary_conditions=["none", "damped"],
        upper_boundary_conditions=["none", "damped"],
        lower_boundary_conditions_particles=["absorbing", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        moving_window_velocity=[0.0, c],
        warpx_max_grid_size=256,
        warpx_blocking_factor=64,
    )
else:
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[nx, ny, nz],
        guard_cells=[11, 11, 12],
        lower_bound=[-rmax, -rmax, zmin],
        upper_bound=[rmax, rmax, zmax],
        lower_boundary_conditions=["periodic", "periodic", "damped"],
        upper_boundary_conditions=["periodic", "periodic", "damped"],
        lower_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
        upper_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
        moving_window_velocity=[0.0, 0.0, c],
        warpx_max_grid_size=256,
        warpx_blocking_factor=32,
    )


# plasma region
plasma_rlim = 100.0e-6
N_stage = 15
L_plasma_bulk = 0.28
L_ramp = 1.0e-9
L_ramp_up = L_ramp
L_ramp_down = L_ramp
L_stage = L_plasma_bulk + 2 * L_ramp

# focusing
# lens external fields
beam_gamma1 = 15095
lens_focal_length = 0.015
lens_width = 0.003

stage_spacing = L_plasma_bulk + 2 * lens_focal_length


def get_species_of_accelerator_stage(
    stage_idx,
    stage_zmin,
    stage_zmax,
    stage_xmin=-plasma_rlim,
    stage_xmax=plasma_rlim,
    stage_ymin=-plasma_rlim,
    stage_ymax=plasma_rlim,
    Lplus=L_ramp_up,
    Lp=L_plasma_bulk,
    Lminus=L_ramp_down,
):
    # Parabolic density profile
    n0 = 1.7e23
    Rc = 40.0e-6
    Lstage = Lplus + Lp + Lminus
    if not np.isclose(stage_zmax - stage_zmin, Lstage):
        print("Warning: zmax disagrees with stage length")
    parabolic_distribution = picmi.AnalyticDistribution(
        density_expression=f"n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))*(0.5*(1.-cos(pi*(z-{stage_zmin})/Lplus)))*((z-{stage_zmin})<Lplus)"
        + f"+n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))*((z-{stage_zmin})>=Lplus)*((z-{stage_zmin})<(Lplus+Lp))"
        + f"+n0*(1.+4.*(x**2+y**2)/(kp**2*Rc**4))*(0.5*(1.+cos(pi*((z-{stage_zmin})-Lplus-Lp)/Lminus)))*((z-{stage_zmin})>=(Lplus+Lp))*((z-{stage_zmin})<(Lplus+Lp+Lminus))",
        pi=3.141592653589793,
        n0=n0,
        kp=q_e / c * math.sqrt(n0 / (m_e * ep0)),
        Rc=Rc,
        Lplus=Lplus,
        Lp=Lp,
        Lminus=Lminus,
        lower_bound=[stage_xmin, stage_ymin, stage_zmin],
        upper_bound=[stage_xmax, stage_ymax, stage_zmax],
        fill_in=True,
    )

    electrons = picmi.Species(
        particle_type="electron",
        name=f"electrons{stage_idx}",
        initial_distribution=parabolic_distribution,
    )

    ions = picmi.Species(
        particle_type="proton",
        name=f"ions{stage_idx}",
        initial_distribution=parabolic_distribution,
    )

    return electrons, ions


species_list = []
for i_stage in range(1):
    # Add plasma
    zmin_stage = i_stage * stage_spacing
    zmax_stage = zmin_stage + L_stage
    electrons, ions = get_species_of_accelerator_stage(
        i_stage + 1, zmin_stage, zmax_stage
    )
    species_list.append(electrons)
    species_list.append(ions)

# add beam to species_list
beam_charge = -10.0e-15  # in Coulombs
N_beam_particles = int(1e6)
beam_centroid_z = -107.0e-6
beam_rms_z = 2.0e-6
beam_gammas = [1960 + 13246 * i_stage for i_stage in range(N_stage)]
# beam_gammas = [1957, 15188, 28432, 41678, 54926, 68174, 81423,94672, 107922,121171] # From 3D run
beams = []
for i_stage in range(N_stage):
    beam_gamma = beam_gammas[i_stage]
    sigma_gamma = 0.06 * beam_gamma
    gaussian_distribution = picmi.GaussianBunchDistribution(
        n_physical_particles=abs(beam_charge) / q_e,
        rms_bunch_size=[2.0e-6, 2.0e-6, beam_rms_z],
        rms_velocity=[8 * c, 8 * c, sigma_gamma * c],
        centroid_position=[0.0, 0.0, beam_centroid_z],
        centroid_velocity=[0.0, 0.0, beam_gamma * c],
    )
    beam = picmi.Species(
        particle_type="electron",
        name=f"beam_stage_{i_stage}",
        initial_distribution=gaussian_distribution,
    )
    beams.append(beam)

# Laser
antenna_z = -1e-9
profile_t_peak = 1.46764864e-13


def get_laser(antenna_z, profile_t_peak, fill_in=True):
    profile_focal_distance = 0.0
    laser = picmi.GaussianLaser(
        wavelength=0.8e-06,
        waist=36e-06,
        duration=7.33841e-14,
        focal_position=[0.0, 0.0, profile_focal_distance + antenna_z],
        centroid_position=[0.0, 0.0, antenna_z - c * profile_t_peak],
        propagation_direction=[0.0, 0.0, 1.0],
        polarization_direction=[0.0, 1.0, 0.0],
        a0=2.36,
        fill_in=fill_in,
    )
    laser_antenna = picmi.LaserAntenna(
        position=[0.0, 0.0, antenna_z], normal_vector=[0.0, 0.0, 1.0]
    )
    return (laser, laser_antenna)


lasers = []
for i_stage in range(1):
    fill_in = True
    if i_stage == 0:
        fill_in = False
    lasers.append(
        get_laser(
            antenna_z + i_stage * stage_spacing,
            profile_t_peak + i_stage * stage_spacing / c,
            fill_in,
        )
    )

# Electromagnetic solver

psatd_algo = "multij"
if psatd_algo == "galilean":
    galilean_velocity = [0.0, 0.0] if dim == "3" else [0.0]
    galilean_velocity += [-c * beta_boost]
    n_pass_z = 1
    do_multiJ = None
    do_multi_J_n_depositions = None
    J_in_time = None
    current_correction = True
    divE_cleaning = False
elif psatd_algo == "multij":
    n_pass_z = 4
    galilean_velocity = None
    do_multiJ = True
    do_multi_J_n_depositions = 2
    J_in_time = "linear"
    current_correction = False
    divE_cleaning = True
else:
    raise Exception(
        f"PSATD algorithm '{psatd_algo}' is not recognized!\n"
        "Valid options are 'multiJ' or 'galilean'."
    )
if dim == "rz":
    stencil_order = [8, 16]
    smoother = picmi.BinomialSmoother(n_pass=[1, n_pass_z])
    grid_type = "collocated"
else:
    stencil_order = [8, 8, 16]
    smoother = picmi.BinomialSmoother(n_pass=[1, 1, n_pass_z])
    grid_type = "hybrid"


solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="PSATD",
    cfl=0.9999,
    source_smoother=smoother,
    stencil_order=stencil_order,
    galilean_velocity=galilean_velocity,
    warpx_psatd_update_with_rho=True,
    warpx_current_correction=current_correction,
    divE_cleaning=divE_cleaning,
    warpx_psatd_J_in_time=J_in_time,
)

# Diagnostics
diag_field_list = ["B", "E", "J", "rho"]
diag_particle_list = ["weighting", "position", "momentum"]
coarse_btd_end = int((L_plasma_bulk + 0.001 + stage_spacing * (N_stage - 1)) * 100000)
stage_end_snapshots = [
    f"{int((L_plasma_bulk + stage_spacing * ii) * 100000)}:{int((L_plasma_bulk + stage_spacing * ii) * 100000 + 50)}:5"
    for ii in range(1)
]
btd_particle_diag = picmi.LabFrameParticleDiagnostic(
    name="lab_particle_diags",
    species=beams,
    grid=grid,
    num_snapshots=25 * N_stage,
    # warpx_intervals=', '.join([f':{coarse_btd_end}:1000']+stage_end_snapshots),
    warpx_intervals=", ".join(["0:0"] + stage_end_snapshots),
    dt_snapshots=0.00001 / c,
    data_list=diag_particle_list,
    write_dir="lab_particle_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="bp",
)

btd_field_diag = picmi.LabFrameFieldDiagnostic(
    name="lab_field_diags",
    grid=grid,
    num_snapshots=25 * N_stage,
    dt_snapshots=stage_spacing / 25 / c,
    data_list=diag_field_list,
    warpx_lower_bound=[-128.0e-6, 0.0e-6, -180.0e-6],
    warpx_upper_bound=[128.0e-6, 0.0e-6, 0.0],
    write_dir="lab_field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="bp",
)

field_diag = picmi.FieldDiagnostic(
    name="field_diags",
    data_list=diag_field_list,
    grid=grid,
    period=100,
    write_dir="field_diags",
    lower_bound=[-128.0e-6, 0.0e-6, -180.0e-6],
    upper_bound=[128.0e-6, 0.0e-6, 0.0],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

particle_diag = picmi.ParticleDiagnostic(
    name="particle_diags",
    species=beams,
    period=100,
    write_dir="particle_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

beamrel_red_diag = picmi.ReducedDiagnostic(
    diag_type="BeamRelevant", name="beamrel", species=beam, period=1
)

# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    warpx_numprocs=num_procs,
    warpx_compute_max_step_from_btd=True,
    verbose=2,
    particle_shape="cubic",
    gamma_boost=gamma_boost,
    warpx_charge_deposition_algo="standard",
    warpx_current_deposition_algo="direct",
    warpx_field_gathering_algo="momentum-conserving",
    warpx_particle_pusher_algo="vay",
    warpx_amrex_the_arena_is_managed=False,
    warpx_amrex_use_gpu_aware_mpi=True,
    warpx_do_multi_J=do_multiJ,
    warpx_do_multi_J_n_depositions=do_multi_J_n_depositions,
    warpx_grid_type=grid_type,
    # default: 2 for staggered grids, 8 for hybrid grids
    warpx_field_centering_order=[16, 16, 16],
    # only for hybrid grids, default: 8
    warpx_current_centering_order=[16, 16, 16],
)

for species in species_list:
    if dim == "rz":
        n_macroparticle_per_cell = [2, 4, 2]
    else:
        n_macroparticle_per_cell = [2, 2, 2]
    sim.add_species(
        species,
        layout=picmi.GriddedLayout(
            grid=grid, n_macroparticle_per_cell=n_macroparticle_per_cell
        ),
    )

for i_stage in range(N_stage):
    sim.add_species_through_plane(
        species=beams[i_stage],
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles=N_beam_particles),
        injection_plane_position=0.0,
        injection_plane_normal_vector=[0.0, 0.0, 1.0],
    )

for i_stage in range(1):
    # Add laser
    (laser, laser_antenna) = lasers[i_stage]
    sim.add_laser(laser, injection_method=laser_antenna)

# Add diagnostics
sim.add_diagnostic(btd_particle_diag)
# sim.add_diagnostic(btd_field_diag)
# sim.add_diagnostic(field_diag)
# sim.add_diagnostic(particle_diag)

# Add reduced diagnostic
sim.add_diagnostic(beamrel_red_diag)

sim.write_input_file(f"inputs_training_{N_stage}_stages")

# Advance simulation until last time step
sim.step()
