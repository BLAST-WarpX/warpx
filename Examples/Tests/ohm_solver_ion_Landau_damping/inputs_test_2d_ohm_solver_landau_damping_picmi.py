#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. The script simulates ion Landau damping as described
# --- in section 4.5 of Munoz et al. (2018).

import argparse
import os
import sys
import time

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, libwarpx, particle_containers, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class IonLandauDamping(object):
    """This input is based on the ion Landau damping test as described by
    Munoz et al. (2018).
    """

    # Applied field parameters
    B0 = 0.1  # Initial magnetic field strength (T)
    beta = 2.0  # Plasma beta, used to calculate temperature

    # Plasma species parameters
    m_ion = 100.0  # Ion mass (electron masses)
    vA_over_c = 1e-3  # ratio of Alfven speed and the speed of light

    # Spatial domain
    Nz = 256  # number of cells in z direction
    Nx = 4  # number of cells in x (and y) direction for >1 dimensions

    # Temporal domain (if not run as a CI test)
    LT = 40.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = [8192, 4096, 1024]  # Seed number of particles per cell
    DZ = 1.0 / 6.0  # Cell size (ion skin depths)
    DT = 1e-3  # Time step (ion cyclotron periods)

    # density perturbation strength
    epsilon = 0.03

    # Plasma resistivity - used to dampen the mode excitation
    eta = 1e-7
    # Number of substeps used to update B
    substeps = 10

    def __init__(self, test, dim, m, T_ratio, verbose):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.dim = int(dim)
        self.m = m
        self.T_ratio = T_ratio
        self.verbose = verbose or self.test

        # sanity check
        assert dim > 0 and dim < 4, f"{dim}-dimensions not a valid input"

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.dz = self.DZ * self.l_i
        self.Lz = self.Nz * self.dz
        self.Lx = self.Nx * self.dz

        diag_period = 1 / 16.0  # Output interval (ion cyclotron periods)
        self.diag_steps = int(diag_period / self.DT)

        self.total_steps = int(np.ceil(self.LT / self.DT))
        # if this is a test case run for only 100 steps
        if self.test:
            self.total_steps = 100

        self.dt = self.DT / self.w_ci  # self.DT * self.t_ci

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tT = {self.T_plasma * 1e-3:.1f} keV\n"
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
                f"\tdz = {self.dz:.1e} m\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n"
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

        if self.dim == 1:
            grid_object = picmi.Cartesian1DGrid
        elif self.dim == 2:
            grid_object = picmi.Cartesian2DGrid
        else:
            grid_object = picmi.Cartesian3DGrid

        self.grid = grid_object(
            number_of_cells=[self.Nx, self.Nx, self.Nz][-self.dim :],
            warpx_max_grid_size=self.Nz,
            lower_bound=[-self.Lx / 2.0, -self.Lx / 2.0, 0][-self.dim :],
            upper_bound=[self.Lx / 2.0, self.Lx / 2.0, self.Lz][-self.dim :],
            lower_boundary_conditions=["periodic"] * self.dim,
            upper_boundary_conditions=["periodic"] * self.dim,
            warpx_blocking_factor=4,
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
            gamma=1.0,
            Te=self.T_plasma / self.T_ratio,
            n0=self.n_plasma,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
        )
        simulation.solver = self.solver

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        k_m = 2.0 * np.pi * self.m / self.Lz
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression=f"{self.n_plasma}*(1+{self.epsilon}*cos({k_m}*z))",
                rms_velocity=[self.v_ti] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC[self.dim - 1]
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterstep(self.text_diag)

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=100,
                species=[self.ions],
                data_list=["ux", "uy", "uz", "x", "z", "weighting"],
            )
            simulation.add_diagnostic(particle_diag)
            field_diag = picmi.FieldDiagnostic(
                name="diag1",
                grid=self.grid,
                period=100,
                data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            )
            simulation.add_diagnostic(field_diag)

        self.output_file_name = "field_data.txt"
        # install a custom "reduced diagnostic" to save the average field
        callbacks.installafterEsolve(self._record_average_fields)
        try:
            os.mkdir("diags")
        except OSError:
            # diags directory already exists
            pass
        with open(f"diags/{self.output_file_name}", "w") as f:
            f.write("[0]step() [1]time(s) [2]z_coord(m) [3]Ez_lev0-(V/m)\n")

        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_step = 0

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()

        # get ion particle container wrapper
        self.ion_part_container = particle_containers.ParticleContainerWrapper("ions")

    def text_diag(self):
        """Diagnostic function to print out timing data and particle numbers."""
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % (self.total_steps // 10) != 0:
            return

        wall_time = time.time() - self.prev_time
        steps = step - self.prev_step
        step_rate = steps / wall_time

        status_dict = {
            "step": step,
            "nplive ions": self.ion_part_container.nps,
            "wall_time": wall_time,
            "step_rate": step_rate,
            "diag_steps": self.diag_steps,
            "iproc": None,
        }

        diag_string = (
            "Step #{step:6d}; "
            "{nplive ions} core ions; "
            "{wall_time:6.1f} s wall time; "
            "{step_rate:4.2f} steps/s"
        )

        if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
            print(diag_string.format(**status_dict))

        self.prev_time = time.time()
        self.prev_step = step

    def _record_average_fields(self):
        """A custom reduced diagnostic to store the average E&M fields in a
        similar format as the reduced diagnostic so that the same analysis
        script can be used regardless of the simulation dimension.
        """
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % self.diag_steps != 0:
            return

        Ez_warpx = fields.EzWrapper()[...]

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        t = step * self.dt
        z_vals = np.linspace(0, self.Lz, self.Nz, endpoint=False)

        if self.dim == 1:
            Ez = Ez_warpx
        elif self.dim == 2:
            Ez = np.mean(Ez_warpx, axis=0)
        else:
            Ez = np.mean(Ez_warpx, axis=(0, 1))

        with open(f"diags/{self.output_file_name}", "a") as f:
            for ii in range(self.Nz):
                f.write(f"{step:05d} {t:.10e} {z_vals[ii]:.10e} {Ez[ii]:+.10e}\n")


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
    "-d", "--dim", help="Simulation dimension", required=False, type=int, default=1
)
parser.add_argument(
    "-m", help="Mode number to excite", required=False, type=int, default=4
)
parser.add_argument(
    "--temp_ratio",
    help="Ratio of ion to electron temperature",
    required=False,
    type=float,
    default=1.0 / 3,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = IonLandauDamping(
    test=args.test,
    dim=args.dim,
    m=args.m,
    T_ratio=args.temp_ratio,
    verbose=args.verbose,
)
simulation.step()
