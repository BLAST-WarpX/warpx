#!/usr/bin/env python3
#
# --- Simple example of Langmuir oscillations in a uniform plasma
# --- in two dimensions
import numpy as np

from pywarpx import picmi

constants = picmi.constants

##########################
# physics parameters
##########################

e_mass = 100.0 * constants.m_e
i_mass = constants.m_p
N = 1.0e25
T = 1.0  # 1 eV
mean_velocity = 10.0e3

##########################
# numerics parameters
##########################

# --- Grid
nz = 200

zmin = 0.0
zmax = 2.0e-7

dz = (zmax - zmin) / nz

dt = 1.0e-14
max_steps = 100
diagnostic_intervals = 100

number_per_cell = 1000

##########################
# physics components
##########################

vthe = np.sqrt(T * constants.q_e / e_mass)
vthi = np.sqrt(T * constants.q_e / i_mass)

electrons_fill = picmi.UniformDistribution(
    density=N,
    rms_velocity=[vthe, vthe, vthe],
    directed_velocity=[mean_velocity, 0.0, 0.0],
)
electrons_flux_left = picmi.UniformFluxDistribution(
    warpx_fixed_num_particles_per_cell=True,
    warpx_density=N,
    flux_normal_axis="z",
    surface_flux_position=0.0,
    flux_direction=+1,
    rms_velocity=[vthe, vthe, vthe],
    directed_velocity=[mean_velocity, 0.0, 0.0],
    gaussian_flux_momentum_distribution=True,
    flux=None,
)
electrons_flux_right = picmi.UniformFluxDistribution(
    warpx_fixed_num_particles_per_cell=True,
    warpx_density=N,
    flux_normal_axis="z",
    surface_flux_position=zmax,
    flux_direction=-1,
    rms_velocity=[vthe, vthe, vthe],
    directed_velocity=[mean_velocity, 0.0, 0.0],
    gaussian_flux_momentum_distribution=True,
    flux=None,
)


ions_fill = picmi.UniformDistribution(
    density=N,
    rms_velocity=[vthi, vthi, vthi],
    directed_velocity=[mean_velocity, 0.0, 0.0],
)
ions_flux_left = picmi.UniformFluxDistribution(
    warpx_fixed_num_particles_per_cell=True,
    warpx_density=N,
    flux_normal_axis="z",
    surface_flux_position=0.0,
    flux_direction=+1,
    rms_velocity=[vthi, vthi, vthi],
    directed_velocity=[mean_velocity, 0.0, 0.0],
    gaussian_flux_momentum_distribution=True,
    flux=None,
)
ions_flux_right = picmi.UniformFluxDistribution(
    warpx_fixed_num_particles_per_cell=True,
    warpx_density=N,
    flux_normal_axis="z",
    surface_flux_position=zmax,
    flux_direction=-1,
    rms_velocity=[vthi, vthi, vthi],
    directed_velocity=[mean_velocity, 0.0, 0.0],
    gaussian_flux_momentum_distribution=True,
    flux=None,
)

electrons = picmi.Species(
    particle_type="electron",
    mass=e_mass,
    name="electrons",
    initial_distribution=[electrons_fill, electrons_flux_left, electrons_flux_right],
)
ions = picmi.Species(
    particle_type="proton",
    name="ions",
    initial_distribution=[ions_fill, ions_flux_left, ions_flux_right],
)

##########################
# numerics components
##########################

grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["neumann"],
    upper_boundary_conditions=["neumann"],
    lower_boundary_conditions_particles=["absorbing"],
    upper_boundary_conditions_particles=["absorbing"],
    moving_window_velocity=[0.0, 0.0, 0.0],
)

solver = picmi.ElectrostaticSolver(grid=grid)

##########################
# diagnostics
##########################

particle_diags = [
    picmi.ParticleFieldDiagnostic("nn", "1.", do_average=0),
    picmi.ParticleFieldDiagnostic("vx", "ux", do_average=0),
    picmi.ParticleFieldDiagnostic("vy", "uy", do_average=0),
    picmi.ParticleFieldDiagnostic("vz", "uz", do_average=0),
    picmi.ParticleFieldDiagnostic("vxvx", "ux*ux", do_average=0),
    picmi.ParticleFieldDiagnostic("vyvy", "uy*uy", do_average=0),
    picmi.ParticleFieldDiagnostic("vzvz", "uz*uz", do_average=0),
]

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    warpx_format="openpmd",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
    warpx_particle_fields_to_plot=particle_diags,
    write_dir=".",
    warpx_file_prefix="diags/diag1",
)

part_diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    warpx_format="openpmd",
    period=diagnostic_intervals,
    species=[electrons, ions],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_field_gathering_algo="energy-conserving",
    particle_shape=2,
    verbose=1,
    warpx_use_filter=0,
)

sim.add_species(
    electrons,
    layout=[
        picmi.GriddedLayout(n_macroparticle_per_cell=[number_per_cell], grid=grid),
        picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=[number_per_cell], grid=grid
        ),
        picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=[number_per_cell], grid=grid
        ),
    ],
)

sim.add_species(
    ions,
    layout=[
        picmi.GriddedLayout(n_macroparticle_per_cell=[number_per_cell], grid=grid),
        picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=[number_per_cell], grid=grid
        ),
        picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=[number_per_cell], grid=grid
        ),
    ],
)

sim.add_diagnostic(field_diag1)
sim.add_diagnostic(part_diag1)


##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
# sim.write_input_file(file_name = 'inputs1d_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
