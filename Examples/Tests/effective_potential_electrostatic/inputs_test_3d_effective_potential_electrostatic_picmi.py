#!/usr/bin/env python3
#
# --- Test script for the effective potential Poisson solver. This test is based
# --- on the adiabatic plasma expansion benchmark from Connor et al. (2021)
# --- doi.org/10.1109/TPS.2021.3072353.
# --- In the benchmark an expanding plasma ball with Gaussian density distribution
# --- in the radial direction is simulated and the time evolution of the
# --- density of the electron species is compared to an approximate analytic solution.
# --- The example is modified slightly in the following ways:
# --- 1) The original example used an electromagnetic solver with absorbing
# ---    boundaries while the present case encloses the plasma in a conducting
# ---    sphere.
# --- 2) The domain and plasma parameters for this case has been modified to
# ---    set the cell-size and time step such that the explicit electrostatic
# ---    solver is unstable.

import dill
import numpy as np
from mpi4py import MPI as mpi
from scipy.special import erf

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)

m_ion = 25  # Ion mass (electron masses)

# Plasma parameters
n_plasma = 5e12  # Plasma density (m^-3)
sigma_0 = 22  # Initial Gaussian distribution standard deviation (Debye lengths)
T_e = 100.0  # Electron temperature (K)
T_i = 10.0  # Ion temperature (K)

# Spatial domain
R = 86  # Radius of metallic sphere (Debye lengths)
NZ = 72  # Number of cells in each direction

# Temporal domain (if not run as a CI test)
LT = 0.6e-6  # Simulation temporal length (s)

# Numerical parameters
NPARTS = 500000  # Seed number of particles
DT = 0.8  # Time step (electron streaming)

# Solver parameter
C_EP = 1.0  # Effective potential factor

#######################################################################
# Calculate various plasma parameters based on the simulation input   #
#######################################################################

# Ion mass (kg)
M = m_ion * constants.m_e

# Electron plasma frequency (Hz)
f_pe = np.sqrt(constants.q_e**2 * n_plasma / (constants.m_e * constants.ep0)) / (
    2.0 * np.pi
)

# Debye length (m)
lambda_e = np.sqrt(constants.ep0 * constants.kb * T_e / (n_plasma * constants.q_e**2))

# Thermal velocities (m/s) from v_th = np.sqrt(kT / m)
v_ti = np.sqrt(T_i * constants.kb / M)
v_te = np.sqrt(T_e * constants.kb / constants.m_e)

R *= lambda_e
sigma_0 *= lambda_e

dz = 2.0 * R / (NZ - 4)
Lz = dz * NZ
dt = DT * dz / v_te

total_steps = int(LT / dt)
diag_steps = total_steps // 3
total_steps = diag_steps * 3

# dump attributes needed for analysis to a dill pickle file
if comm.rank == 0:
    parameter_dict = {
        "sigma_0": sigma_0,
        "M": M,
        "T_i": T_i,
        "T_e": T_e,
        "n_plasma": n_plasma,
    }
    with open("sim_parameters.dpkl", "wb") as f:
        dill.dump(parameter_dict, f)

# print out plasma parameters
if comm.rank == 0:
    print(
        f"Initializing simulation with input parameters:\n"
        f"\tT_e = {T_e:.1f} K\n"
        f"\tT_i = {T_i:.1f} K\n"
        f"\tn = {n_plasma:.1e} m^-3\n"
    )
    print(
        f"Plasma parameters:\n"
        f"\tlambda_e = {lambda_e:.1e} m\n"
        f"\tt_pe = {1.0 / f_pe:.1e} s\n"
        f"\tv_ti = {v_ti:.1e} m/s\n"
    )
    print(
        f"Numerical parameters:\n"
        f"\tdz/lambda_e = {dz / lambda_e:.2f}\n"
        f"\tdt*w_pe = {dt * f_pe * 2.0 * np.pi:.2f}\n"
        f"\tdiag steps = {diag_steps:d}\n"
        f"\ttotal steps = {total_steps:d}\n"
    )


#######################################################################
# Set geometry and boundary conditions                                #
#######################################################################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[NZ] * 3,
    lower_bound=[-Lz / 2.0] * 3,
    upper_bound=[Lz / 2.0] * 3,
    lower_boundary_conditions=["neumann"] * 3,
    upper_boundary_conditions=["neumann"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_max_grid_size=NZ // 2,
)
simulation.time_step_size = dt
simulation.max_steps = total_steps
simulation.current_deposition_algo = "direct"
simulation.particle_shape = 1
simulation.verbose = 1

#######################################################################
# Insert spherical boundary as EB                                     #
#######################################################################

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"(x**2+y**2+z**2-{R**2})",
    potential=0.0,
)
simulation.embedded_boundary = embedded_boundary

#######################################################################
# Field solver                                                        #
#######################################################################

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    warpx_effective_potential=True,
    warpx_effective_potential_factor=C_EP,
    warpx_self_fields_verbosity=0,
)
simulation.solver = solver

#######################################################################
# Particle types setup                                                #
#######################################################################

total_parts = (
    n_plasma
    * sigma_0**2
    * (
        (2.0 * np.pi) ** 1.5 * sigma_0 * erf(R / (np.sqrt(2) * sigma_0))
        + 4.0 * np.pi * R * np.exp(-(R**2) / (2.0 * sigma_0**2))
    )
)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=picmi.GaussianBunchDistribution(
        n_physical_particles=total_parts,
        rms_bunch_size=[sigma_0] * 3,
        rms_velocity=[v_te] * 3,
    ),
)
simulation.add_species(
    electrons,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles=NPARTS),
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=M,
    initial_distribution=picmi.GaussianBunchDistribution(
        n_physical_particles=total_parts,
        rms_bunch_size=[sigma_0] * 3,
        rms_velocity=[v_ti] * 3,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles=NPARTS),
)

#######################################################################
# Add diagnostics                                                     #
#######################################################################

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=diag_steps,
    data_list=[
        "E",
        "J",
        "T_electrons",
        "T_ions",
        "phi",
        "rho_electrons",
        "rho_ions",
    ],
    write_dir="diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

#######################################################################
# Initialize simulation                                               #
#######################################################################

simulation.initialize_inputs()
simulation.initialize_warpx()

#######################################################################
# Execute simulation                                                  #
#######################################################################

simulation.step()
