#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. The script is set up to produce parallel normal EM modes
# --- in a metallic cylinder and is run in RZ geometry.
# --- As a CI test only a small number of steps are taken.

import argparse
import sys

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=0)


class CylindricalNormalModes(object):
    """The following runs a simulation of an uniform plasma at a set ion
    temperature (and Te = 0) with an external magnetic field applied in the
    z-direction (parallel to domain).
    The analysis script (in this same directory) analyzes the output field
    data for EM modes.
    """

    # Applied field parameters
    B0 = 0.5  # Initial magnetic field strength (T)
    beta = 0.01  # Plasma beta, used to calculate temperature

    # Plasma species parameters
    m_ion = 400.0  # Ion mass (electron masses)
    vA_over_c = 5e-3  # ratio of Alfven speed and the speed of light

    # Spatial domain
    Nz = 512  # number of cells in z direction
    Nr = 128  # number of cells in r direction

    # Temporal domain (if not run as a CI test)
    LT = 800.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = 8000  # Seed number of particles per cell
    DZ = 0.4  # Cell size (ion skin depths)
    DR = 0.4  # Cell size (ion skin depths)
    DT = 0.02  # Time step (ion cyclotron periods)

    # Plasma resistivity - used to dampen the mode excitation
    eta = 5e-4
    # Number of substeps used to update B
    substeps = 20

    def __init__(self, test, verbose):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.verbose = verbose or self.test

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        if not self.test:
            self.total_steps = int(self.LT / self.DT)
        else:
            # if this is a test case run for only a small number of steps
            self.total_steps = 100
            # and make the grid and particle count smaller
            self.Nz = 128
            self.Nr = 64
            self.NPPC = 200
        # output diagnostics 5 times per cyclotron period
        self.diag_steps = max(10, int(1.0 / 5 / self.DT))

        self.Lz = self.Nz * self.DZ * self.l_i
        self.Lr = self.Nr * self.DR * self.l_i

        self.dt = self.DT * self.t_ci

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tT = {self.T_plasma:.3f} eV\n"
                f"\tn = {self.n_plasma:.1e} m^-3\n"
                f"\tB0 = {self.B0:.2f} T\n"
                f"\tM/m = {self.m_ion:.0f}\n"
            )
            print(
                f"Plasma parameters:\n"
                f"\tl_i = {self.l_i:.1e} m\n"
                f"\tt_ci = {self.t_ci:.1e} s\n"
                f"\tv_ti = {self.v_ti:.1e} m/s\n"
                f"\tvA = {self.vA:.1e} m/s\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n",
                flush=True,
            )
        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        # Ion mass (kg)
        self.M = self.m_ion * constants.m_e

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Alfven speed (m/s): vA = B / sqrt(mu0 * n * (M + m)) = c * omega_ci / w_pi
        self.vA = self.vA_over_c * constants.c
        self.n_plasma = (self.B0 / self.vA) ** 2 / (
            constants.mu0 * (self.M + constants.m_e)
        )

        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n_plasma / (self.M * constants.ep0))

        # Skin depth (m)
        self.l_i = constants.c / self.w_pi

        # Ion thermal velocity (m/s) from beta = 2 * (v_ti / vA)**2
        self.v_ti = np.sqrt(self.beta / 2.0) * self.vA

        # Temperature (eV) from thermal speed: v_ti = sqrt(kT / M)
        self.T_plasma = self.v_ti**2 * self.M / constants.q_e  # eV

        # Larmor radius (m)
        self.rho_i = self.v_ti / self.w_ci

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.CylindricalGrid(
            number_of_cells=[self.Nr, self.Nz],
            warpx_max_grid_size=self.Nz,
            lower_bound=[0, -self.Lz / 2.0],
            upper_bound=[self.Lr, self.Lz / 2.0],
            lower_boundary_conditions=["none", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic"],
            lower_boundary_conditions_particles=["none", "periodic"],
            upper_boundary_conditions_particles=["reflecting", "periodic"],
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.verbose = self.verbose

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            Te=0.0,
            n0=self.n_plasma,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
            n_floor=self.n_plasma * 0.05,
        )
        simulation.solver = self.solver

        B_ext = picmi.AnalyticInitialField(Bz_expression=self.B0)
        simulation.add_applied_field(B_ext)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.UniformDistribution(
                density=self.n_plasma,
                rms_velocity=[self.v_ti] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "E"],
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

        # add particle diagnostic for checksum
        if self.test:
            part_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.total_steps,
                species=[self.ions],
                data_list=["ux", "uy", "uz", "weighting"],
            )
            simulation.add_diagnostic(part_diag)


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short CI test",
    action="store_true",
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = CylindricalNormalModes(test=args.test, verbose=args.verbose)
simulation.step()
