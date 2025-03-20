#!/usr/bin/env python3
#
# --- Tests the python interface to the Implicit solver

import numpy as np

from pywarpx import picmi

constants = picmi.constants

##########################
# physics parameters
##########################

n0 = 1.0e30  # m^-3
Ti = 100.0  # eV
Te = 100.0  # eV
wpe = constants.q_e * np.sqrt(n0 / (constants.m_e * constants.ep0))
de0 = constants.c / wpe
nppcz = 10  # number of particles/cell in z
dt = 0.1 / wpe  # s

vthe = np.sqrt(Te * constants.q_e / constants.m_e)
vthi = np.sqrt(Ti * constants.q_e / constants.m_p)

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 20
diagnostic_intervals = "::20"

# --- Grid
nx = 40
ny = 40

xmin = 0.0
ymin = 0.0
xmax = 10.0 * de0
ymax = 10.0 * de0

number_per_cell_each_dim = [nppcz, nppcz]

##########################
# physics components
##########################

electrons_uniform_plasma = picmi.UniformDistribution(
    density=n0, rms_velocity=[vthe, vthe, vthe]
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=electrons_uniform_plasma,
)

protons_uniform_plasma = picmi.UniformDistribution(
    density=n0, rms_velocity=[vthi, vthi, vthi]
)

protons = picmi.Species(
    particle_type="proton", name="protons", initial_distribution=protons_uniform_plasma
)

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    warpx_max_grid_size=8,
    warpx_blocking_factor=8,
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")

GMRES_solver = picmi.GMRESLinearSolver(
    verbose_int=2,
    max_iterations=1000,
    relative_tolerance=1.0e-8,
    absolute_tolerance=0.0,
)

newton_solver = picmi.NewtonNonlinearSolver(
    verbose=True,
    max_iterations=20,
    relative_tolerance=1.0e-12,
    absolute_tolerance=0.0,
    require_convergence=False,
    linear_solver=GMRES_solver,
    max_particle_iterations=21,
    particle_tolerance=1.0e-12,
)

evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
    theta=0.5, nonlinear_solver=newton_solver
)

##########################
# diagnostics
##########################

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "Jx", "Jy", "Jz", "rho", "divE"],
)

part_diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
    species=[electrons, protons],
    data_list=["weighting", "position", "momentum"],
)

particle_energy_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy", name="particle_energy", period=1
)

field_energy_diag = picmi.ReducedDiagnostic(
    diag_type="FieldEnergy", name="field_energy", period=1
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    particle_shape=2,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=1,
    warpx_evolve_scheme=evolve_scheme,
    warpx_current_deposition_algo="villasenor",
    warpx_particle_pusher_algo="boris",
    warpx_serialize_initial_conditions=1,
    warpx_use_filter=0,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)
sim.add_species(
    protons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)

sim.add_diagnostic(field_diag1)
sim.add_diagnostic(part_diag1)
sim.add_diagnostic(particle_energy_diag)
sim.add_diagnostic(field_energy_diag)

##########################
# simulation run
##########################

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
sim.write_input_file(file_name="inputs2d_from_PICMI")

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
