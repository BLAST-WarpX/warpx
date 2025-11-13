#!/usr/bin/env python3
#
# --- Simple example of Langmuir oscillations in a uniform plasma
# --- in two dimensions
import numpy as np
import openpmd_viewer

from pywarpx import picmi

constants = picmi.constants

##########################
# physics parameters
##########################

e_mass = 100.*constants.m_e
i_mass = constants.m_p
N = 1.e25
T = 1.  # 1 eV
mean_velocity = 10.e3

##########################
# numerics parameters
##########################

# --- Grid
nz = 200

zmin = 0.
zmax = 2.e-7

dz = (zmax - zmin)/nz

dt = 1.e-14
max_steps = 100
diagnostic_intervals = 100

number_per_cell = 10000

##########################
# physics components
##########################

vthe = np.sqrt(T*constants.q_e/e_mass)
vthi = np.sqrt(T*constants.q_e/i_mass)

electrons_fill = picmi.UniformDistribution(density = N,
                                           rms_velocity = [vthe, vthe, vthe],
                                           directed_velocity = [mean_velocity, 0., 0.])
electrons_flux_left = picmi.UniformThermalPLasmaFluxDistribution(density = N,
                                                                 flux_normal_axis = 'z',
                                                                 surface_flux_position = 0.,
                                                                 flux_direction = +1,
                                                                 rms_velocity = [vthe, vthe, vthe],
                                                                 directed_velocity = [mean_velocity, 0., 0.],
                                                                 gaussian_flux_momentum_distribution = True)
electrons_flux_right = picmi.UniformThermalPLasmaFluxDistribution(density = N,
                                                                  flux_normal_axis = 'z',
                                                                  surface_flux_position = zmax,
                                                                  flux_direction = -1,
                                                                  rms_velocity = [vthe, vthe, vthe],
                                                                  directed_velocity = [mean_velocity, 0., 0.],
                                                                  gaussian_flux_momentum_distribution = True)


ions_fill = picmi.UniformDistribution(density = N,
                                      rms_velocity = [vthi, vthi, vthi],
                                      directed_velocity = [mean_velocity, 0., 0.])
ions_flux_left = picmi.UniformThermalPLasmaFluxDistribution(density = N,
                                                            flux_normal_axis = 'z',
                                                            surface_flux_position = 0.,
                                                            flux_direction = +1,
                                                            rms_velocity = [vthi, vthi, vthi],
                                                            directed_velocity = [mean_velocity, 0., 0.],
                                                            gaussian_flux_momentum_distribution = True)
ions_flux_right = picmi.UniformThermalPLasmaFluxDistribution(density = N,
                                                             flux_normal_axis = 'z',
                                                             surface_flux_position = zmax,
                                                             flux_direction = -1,
                                                             rms_velocity = [vthi, vthi, vthi],
                                                             directed_velocity = [mean_velocity, 0., 0.],
                                                             gaussian_flux_momentum_distribution = True)


electrons = picmi.Species(particle_type='electron', mass=e_mass, name='electrons',
                          initial_distribution=[electrons_fill, electrons_flux_left, electrons_flux_right])
ions = picmi.Species(particle_type='proton', name='ions',
                     initial_distribution=[ions_fill, ions_flux_left, ions_flux_right])

##########################
# numerics components
##########################

grid = picmi.Cartesian1DGrid(number_of_cells = [nz],
                             lower_bound = [zmin],
                             upper_bound = [zmax],
                             lower_boundary_conditions = ['neumann'],
                             upper_boundary_conditions = ['neumann'],
                             lower_boundary_conditions_particles = ['absorbing'],
                             upper_boundary_conditions_particles = ['absorbing'],
                             moving_window_velocity = [0., 0., 0.])

solver = picmi.ElectrostaticSolver(grid=grid)

##########################
# diagnostics
##########################

particle_diags = [picmi.ParticleFieldDiagnostic('nn', '1.', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vx', 'ux', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vy', 'uy', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vz', 'uz', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vxvx', 'ux*ux', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vyvy', 'uy*uy', do_average = 0),
                  picmi.ParticleFieldDiagnostic('vzvz', 'uz*uz', do_average = 0)]

field_diag1 = picmi.FieldDiagnostic(name = 'diag1',
                                    warpx_format = 'openpmd',
                                    grid = grid,
                                    period = diagnostic_intervals,
                                    data_list = ['Ez', 'Jz'],
                                    warpx_particle_fields_to_plot = particle_diags,
                                    write_dir = '.',
                                    warpx_file_prefix = 'Python_FluxInjection1D_plt')

part_diag1 = picmi.ParticleDiagnostic(name = 'diag1',
                                      warpx_format = 'openpmd',
                                      period = diagnostic_intervals,
                                      species = [electrons, ions])

##########################
# simulation setup
##########################

sim = picmi.Simulation(solver = solver,
                       time_step_size = dt,
                       max_steps = max_steps,
                       warpx_field_gathering_algo = 'energy-conserving',
                       particle_shape = 2,
                       verbose = 1,
                       warpx_use_filter = 0)

sim.add_species(electrons,
                layout = [picmi.GriddedLayout(n_macroparticle_per_cell=[number_per_cell], grid=grid),
                          picmi.PseudoRandomLayout(n_macroparticles_per_cell=[number_per_cell], grid=grid),
                          picmi.PseudoRandomLayout(n_macroparticles_per_cell=[number_per_cell], grid=grid)])

sim.add_species(ions,
                layout = [picmi.GriddedLayout(n_macroparticle_per_cell=[number_per_cell], grid=grid),
                          picmi.PseudoRandomLayout(n_macroparticles_per_cell=[number_per_cell], grid=grid),
                          picmi.PseudoRandomLayout(n_macroparticles_per_cell=[number_per_cell], grid=grid)])

sim.add_diagnostic(field_diag1)
sim.add_diagnostic(part_diag1)


##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
#sim.write_input_file(file_name = 'inputs1d_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()


# Check results

def calcdensity(species, it):
    nn, info = ts.get_field(f'nn_{species}', iteration=it)
    return nn/dz, info

def calctemperature(species, mass, it):
    "Returns temperature in eV"
    nn, info = ts.get_field(f'nn_{species}', iteration=it)
    vx, info = ts.get_field(f'vx_{species}', iteration=it)
    vy, info = ts.get_field(f'vy_{species}', iteration=it)
    vz, info = ts.get_field(f'vz_{species}', iteration=it)
    vxvx, info = ts.get_field(f'vxvx_{species}', iteration=it)
    vyvy, info = ts.get_field(f'vyvy_{species}', iteration=it)
    vzvz, info = ts.get_field(f'vzvz_{species}', iteration=it)
    nn1 = nn.clip(1)
    T = mass/3.*(vxvx - vx*vx/nn1 + vyvy - vy*vy/nn1 + vzvz - vz*vz/nn1)*constants.c**2/nn1
    return T/constants.q_e, info

ts = openpmd_viewer.OpenPMDTimeSeries('Python_FluxInjection1D_plt')

it = 100

nn_electrons, info = calcdensity('electrons', it)
nn_ions, info = calcdensity('ions', it)

T_electrons, info = calctemperature('electrons', e_mass, it)
T_ions, info = calctemperature('ions', i_mass, it)

nn_electrons_error = np.abs((nn_electrons/N - 1.))
nn_ions_error = np.abs((nn_ions/N - 1.))
T_electrons_error = np.abs((T_electrons/T - 1.))
T_ions_error = np.abs((T_ions/T - 1.))

print(f'nn_electrons_error.max() = {nn_electrons_error.max()}, tolerance = 0.03')
print(f'nn_ions_error.max() = {nn_ions_error.max()}, tolerance = 0.03')
print(f'T_electrons_error.max() = {T_electrons_error.max()}, tolerance = 0.03')
print(f'T_ions_error.max() = {T_ions_error.max()}, tolerance = 0.03')

assert nn_electrons_error.max() < 0.03
assert nn_ions_error.max() < 0.03
assert T_electrons_error.max() < 0.03
assert T_ions_error.max() < 0.03
