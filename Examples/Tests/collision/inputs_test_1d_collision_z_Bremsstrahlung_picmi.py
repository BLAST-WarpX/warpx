#!/usr/bin/env python3
#
# --- Input file for binary Bremsstrahlung collision testing. This input script
# --- runs the same test as inputs_test_1d_collision_z_Bremsstrahlung but via Python

import math

from pywarpx import picmi

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 100

max_level = 0
nz = 100

zmin = 0.0
zmax = 1.0e-6

n_e = 5.47e30  # electron density, m^-3
Np_e = 1024  # electrons per cell
T1 = 1.0e6  # 1 MeV, electron energy
m_e_eV = constants.m_e * constants.c**2 / constants.q_e  # electron rest mass in eV
gamma = T1 / m_e_eV + 1.0  # electron gamma factor
gamma_beta = math.sqrt(gamma**2 - 1.0)

n_i = 5.47e31  # ion density, m^-3, for rho = 1000 g/cm^3
Np_i = 1024  # ions per cell
T_i = 2.0  # ion temperature, eV

m_B11 = 20065.0 * constants.m_e  # 11.01*amu/me - 5, Boron
q_B11 = 5.0 * constants.q_e

#################################
############ NUMERICS ###########
#################################
serialize_initial_conditions = 1
verbose = 1

# Order of particle shape factors
particle_shape = 1

#################################
############ PLASMA #############
#################################

elec_dist = picmi.UniformDistribution(
    density=n_e,
    directed_velocity=[0.0, 0.0, gamma_beta * constants.c],
)

ion_dist = picmi.UniformDistribution(
    density=n_i,
    rms_velocity=[math.sqrt(T_i * constants.q_e / m_B11)] * 3,
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    warpx_do_not_deposit=1,
    initial_distribution=elec_dist,
)
ions = picmi.Species(
    name="ions",
    charge=q_B11,
    mass=m_B11,
    warpx_do_not_deposit=1,
    initial_distribution=ion_dist,
)
photons = picmi.Species(
    particle_type="photon",
    name="photons",
)

#################################
########## COLLISIONS ###########
#################################

collision1 = picmi.BremsstrahlungCollisions(
    name="collisions1",
    species=[electrons, ions],
    Z=5,
    product_species=photons,
    multiplier=100.0,
)

#################################
###### GRID #####################
#################################

grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    warpx_blocking_factor=4,
    warpx_max_grid_size=128,
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["periodic"],
    upper_boundary_conditions=["periodic"],
)

#################################
######### DIAGNOSTICS ###########
#################################

particle_diag = picmi.ParticleDiagnostic(name="diag1", period=100)
field_diag = picmi.FieldDiagnostic(name="diag1", grid=grid, period=100, data_list=[])

particle_energy_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy", name="particle_energy", period=10
)

particle_momentum_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleMomentum", name="particle_momentum", period=10
)

particle_number_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleNumber", name="particle_number", period=10
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    warpx_grid=grid,
    max_steps=max_steps,
    verbose=verbose,
    time_step_size=1.0e-2 * 1.0e-15,
    warpx_serialize_initial_conditions=serialize_initial_conditions,
    warpx_collisions=[collision1],
    warpx_reduced_diags_precision=18,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[Np_e], grid=grid),
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[Np_i], grid=grid),
)
sim.add_species(photons, layout=None)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_energy_diag)
sim.add_diagnostic(particle_momentum_diag)
sim.add_diagnostic(particle_number_diag)

#################################
##### SIMULATION EXECUTION ######
#################################

# sim.write_input_file('inputs_test_2d_collision_xz')
sim.step(max_steps)
