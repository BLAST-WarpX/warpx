#!/usr/bin/env python3
#
# --- Test script for the effective potential Poisson solver. This test is based on the
# --- adiabatic plasma expansion benchmark from Connor et al. (2021)
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

import argparse
import sys

import dill
import numpy as np
from mpi4py import MPI as mpi
from scipy.special import erf

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class PlasmaExpansionSimulation(object):
    """Simulation setup for an expanding plasma ball in 3d."""

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

    def __init__(self, verbose):
        """Get input parameters for the specific case desired."""
        self.verbose = verbose

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.R *= self.lambda_e
        self.sigma_0 *= self.lambda_e

        # self.dz = self.DZ * self.lambda_e
        self.dz = 2.0 * self.R / (self.NZ - 4)
        self.Lz = self.dz * self.NZ
        self.dt = self.DT * self.dz / self.v_te

        self.total_steps = int(self.LT / self.dt)
        self.diag_steps = self.total_steps // 3
        self.total_steps = self.diag_steps * 3

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tT_e = {self.T_e:.1f} K\n"
                f"\tT_i = {self.T_i:.1f} K\n"
                f"\tn = {self.n_plasma:.1e} m^-3\n"
            )
            print(
                f"Plasma parameters:\n"
                f"\tlambda_e = {self.lambda_e:.1e} m\n"
                f"\tt_pe = {1.0/self.f_pe:.1e} s\n"
                f"\tv_ti = {self.v_ti:.1e} m/s\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdz/lambda_e = {self.dz/self.lambda_e:.2f}\n"
                f"\tdt*w_pe = {self.dt*self.f_pe*2.0*np.pi:.2f}\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )
        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        # Ion mass (kg)
        self.M = self.m_ion * constants.m_e

        # Electron plasma frequency (Hz)
        self.f_pe = np.sqrt(
            constants.q_e**2 * self.n_plasma / (constants.m_e * constants.ep0)
        ) / (2.0 * np.pi)

        # Debye length (m)
        self.lambda_e = np.sqrt(
            constants.ep0 * constants.kb * self.T_e / (self.n_plasma * constants.q_e**2)
        )

        # Thermal velocities (m/s) from v_th = np.sqrt(kT / m)
        self.v_ti = np.sqrt(self.T_i * constants.kb / self.M)
        self.v_te = np.sqrt(self.T_e * constants.kb / constants.m_e)

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.NZ] * 3,
            lower_bound=[-self.Lz / 2.0] * 3,
            upper_bound=[self.Lz / 2.0] * 3,
            lower_boundary_conditions=["neumann"] * 3,
            upper_boundary_conditions=["neumann"] * 3,
            lower_boundary_conditions_particles=["absorbing"] * 3,
            upper_boundary_conditions_particles=["absorbing"] * 3,
            warpx_max_grid_size=self.NZ // 2,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.verbose = self.verbose

        #######################################################################
        # Insert spherical boundary as EB                                     #
        #######################################################################

        embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function=f"(x**2+y**2+z**2-{self.R**2})",
            potential=0.0,
        )
        simulation.embedded_boundary = embedded_boundary

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        solver = picmi.ElectrostaticSolver(
            grid=self.grid,
            method="Multigrid",
            warpx_effective_potential=True,
            warpx_effective_potential_factor=self.C_EP,
            warpx_self_fields_verbosity=self.verbose,
        )
        simulation.solver = solver

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        total_parts = (
            self.n_plasma
            * self.sigma_0**2
            * (
                (2.0 * np.pi) ** 1.5
                * self.sigma_0
                * erf(self.R / (np.sqrt(2) * self.sigma_0))
                + 4.0 * np.pi * self.R * np.exp(-(self.R**2) / (2.0 * self.sigma_0**2))
            )
        )

        self.electrons = picmi.Species(
            name="electrons",
            particle_type="electron",
            initial_distribution=picmi.GaussianBunchDistribution(
                n_physical_particles=total_parts,
                rms_bunch_size=[self.sigma_0] * 3,
                rms_velocity=[self.v_te] * 3,
            ),
        )
        simulation.add_species(
            self.electrons,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles=self.NPARTS
            ),
        )

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.GaussianBunchDistribution(
                n_physical_particles=total_parts,
                rms_bunch_size=[self.sigma_0] * 3,
                rms_velocity=[self.v_ti] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles=self.NPARTS
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
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


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = PlasmaExpansionSimulation(verbose=args.verbose)
simulation.step()
