#!/usr/bin/env python3
#
# --- Test script for the Darwin model in WarpX.
# --- The script is set up to produce either parallel or perpendicular
# --- (Bernstein) EM modes and can be run in 1d, 2d or 3d
# --- Cartesian geometries. See Section 4.2 and 4.3 of Munoz et al. (2018).
# --- As a CI test only a small number of steps are taken using the 1d version.

import argparse
import os
import sys

# import prototype solvers
import darwin_pic_model
import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, libwarpx, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class EMModes(object):
    """The following runs a simulation of an uniform plasma at a set
    temperature (Te = Ti) with an external magnetic field applied in either the
    z-direction (parallel to domain) or x-direction (perpendicular to domain).
    The analysis script (in this same directory) analyzes the output field data
    for EM modes. This input is based on the EM modes tests as described by
    Munoz et al. (2018) and tests done by Scott Nicks at TAE Technologies.
    """

    # Applied field parameters
    B0 = 0.15  # Initial magnetic field strength (T)

    # Plasma parameters
    vA_over_c = 0.015  # ratio of Alfven speed to c, sets density
    beta = [0.025, 0.1]  # Plasma beta, sets temperature

    # Plasma species parameters
    m_ion = 10.0  # Ion mass (electron masses)

    # Spatial domain
    Nz = [256, 128]  # [1024, 1920]  # number of cells in z direction
    Nx = 8  # number of cells in x (and y) direction for >1 dimensions

    # Temporal domain (if not run as a CI test)
    LT = 150.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = 256  # Seed number of particles per cell
    DZ = [300, 60]  # Cell size (Debye lengths)
    DT = 10.0  # Time step (electron plasma periods)
    C_SI = 4.0

    def __init__(self, test, dim, B_dir, verbose):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.dim = int(dim)
        self.B_dir = B_dir
        self.verbose = verbose or self.test

        # sanity check
        assert dim > 0 and dim < 4, f"{dim}-dimensions not a valid input"

        # get simulation parameters from the defaults given the direction of
        # the initial B-field and the dimensionality
        self.get_simulation_parameters()

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.dz = self.DZ * self.lambda_e
        self.Lz = self.Nz * self.dz
        self.Lx = self.Nx * self.dz

        self.dt = self.DT / self.w_pe

        if not self.test:
            self.total_steps = int(self.LT * self.t_ci / self.dt)
        else:
            # if this is a test case run for only a small number of steps
            self.total_steps = 1000  # 50
        # output diagnostics every 0.01 ion cyclotron periods
        self.diag_steps = 1  # max(int(0.01 * self.t_ci / self.dt), 1)

        # calculate SIPIC modified plasma quantities
        sipic_factor = np.sqrt(1.0 + self.C_SI * (self.w_pe * self.dt) ** 2 / 4.0)
        self.lambda_e_SI = self.lambda_e * sipic_factor
        self.w_pe_SI = self.w_pe / sipic_factor

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
                f"\tdz = {self.dz:.1e} m\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_simulation_parameters(self):
        """Pick appropriate parameters from the defaults given the direction
        of the B-field and the simulation dimensionality."""
        if self.B_dir == "z":
            idx = 0
            self.Bx = 0.0
            self.By = 0.0
            self.Bz = self.B0
        elif self.B_dir == "y":
            idx = 1
            self.Bx = 0.0
            self.By = self.B0
            self.Bz = 0.0
        else:
            idx = 1
            self.Bx = self.B0
            self.By = 0.0
            self.Bz = 0.0

        self.beta = self.beta[idx]
        self.Nz = self.Nz[idx]
        self.DZ = self.DZ[idx]

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""
        # Ion mass (kg)
        self.M = self.m_ion * constants.m_e

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ce = constants.q_e * abs(self.B0) / constants.m_e
        self.t_ce = 2.0 * np.pi / self.w_ce

        # Alfven speed (m/s): vA = B / sqrt(mu0 * n * (M + m)) = c * omega_ci / w_pi
        self.vA = self.vA_over_c * constants.c
        self.n_plasma = (self.B0 / self.vA) ** 2 / (
            constants.mu0 * (self.M + constants.m_e)
        )

        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n_plasma / (self.M * constants.ep0))
        self.w_pe = np.sqrt(
            constants.q_e**2 * self.n_plasma / (constants.m_e * constants.ep0)
        )

        # Skin depth (m)
        self.l_i = constants.c / self.w_pi
        self.l_e = constants.c / self.w_pe

        # Ion thermal velocity (m/s) from beta = 2 * (v_ti / vA)**2
        self.v_ti = np.sqrt(self.beta / 2.0) * self.vA

        # Temperature (eV) from thermal speed: v_ti = sqrt(kT / M)
        self.T_plasma = self.v_ti**2 * self.M / constants.q_e  # eV

        # Electron thermal velocity (m/s) from v_th = sqrt(kB*T/m)
        self.v_te = np.sqrt(self.T_plasma * constants.q_e / constants.m_e)

        # Larmor radius (m)
        self.rho_i = self.v_ti / self.w_ci
        self.rho_e = self.v_te / self.w_ce

        # Debye length
        self.lambda_e = np.sqrt(
            constants.ep0 * self.T_plasma / (self.n_plasma * constants.q_e)
        )

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
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.particle_shape = 1
        simulation.verbose = self.verbose
        simulation.current_deposition_algo = "direct"
        simulation.evolve_scheme = picmi.SemiImplicitDarwinEvolveScheme(
            linear_solver=picmi.GMRESLinearSolver(max_iterations=0),
            projection_div_cleaner_rtol=1e-6,
        )

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge=constants.q_e,
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
        self.electrons = picmi.Species(
            name="electron",
            charge=-constants.q_e,
            mass=constants.m_e,
            initial_distribution=picmi.UniformDistribution(
                density=self.n_plasma,
                rms_velocity=[self.v_te] * 3,
            ),
        )
        simulation.add_species(
            self.electrons,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = darwin_pic_model.OneD_DarwinSolver(
            simulation=simulation,
            grid=self.grid,
            dt=self.dt,
            Csi=self.C_SI,
            skip_es=True,
            python_ms_solve=(
                simulation.evolve_scheme.linear_solver.max_iterations == 0
            ),
        )
        simulation.solver = self.solver

        B_ext = picmi.AnalyticInitialField(
            Bx_expression=self.Bx,
            By_expression=self.By,
            Bz_expression=self.Bz,
            warpx_do_initial_div_cleaning=False,
        )
        simulation.add_applied_field(B_ext)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        if self.B_dir == "z":
            self.output_file_name = "par_field_data.txt"
        else:
            self.output_file_name = "perp_field_data.txt"

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="field_diag",
                period=self.total_steps,
                # warpx_format = 'openpmd',
                # warpx_openpmd_backend = 'h5'
            )
            simulation.add_diagnostic(particle_diag)
            field_diag = picmi.FieldDiagnostic(
                name="field_diag",
                grid=self.grid,
                period=self.total_steps,
                data_list=["B", "E"],
                # warpx_verbose=0,
                # warpx_format = 'openpmd',
                # warpx_openpmd_backend = 'h5'
            )
            simulation.add_diagnostic(field_diag)

        if self.B_dir == "z" or self.dim == 1:
            line_diag = picmi.ReducedDiagnostic(
                diag_type="FieldProbe",
                probe_geometry="Line",
                z_probe=0,
                z1_probe=self.Lz,
                resolution=self.Nz - 1,
                name=self.output_file_name[:-4],
                period=self.diag_steps,
                path="diags/",
            )
            simulation.add_diagnostic(line_diag)
        else:
            # install a custom "reduced diagnostic" to save the average field
            callbacks.installafterEsolve(self._record_average_fields)
            try:
                os.mkdir("diags")
            except OSError:
                # diags directory already exists
                pass
            with open(f"diags/{self.output_file_name}", "w") as f:
                f.write(
                    "[0]step() [1]time(s) [2]z_coord(m) "
                    "[3]Ez_lev0-(V/m) [4]Bx_lev0-(T) [5]By_lev0-(T)\n"
                )

        write_dir = "diags/"
        field_energy = picmi.ReducedDiagnostic(
            diag_type="FieldEnergy",
            name="field_energy",
            period=self.diag_steps,
            path=write_dir,
        )
        simulation.add_diagnostic(field_energy)

        part_energy = picmi.ReducedDiagnostic(
            diag_type="ParticleEnergy",
            name="part_energy",
            period=self.diag_steps,
            path=write_dir,
        )
        simulation.add_diagnostic(part_energy)

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()

    def _record_average_fields(self):
        """A custom reduced diagnostic to store the average E&M fields in a
        similar format as the reduced diagnostic so that the same analysis
        script can be used regardless of the simulation dimension.
        """
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % self.diag_steps != 0:
            return

        Bx_warpx = fields.BxWrapper()[...]
        By_warpx = fields.ByWrapper()[...]
        Ez_warpx = fields.EzWrapper()[...]

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        t = step * self.dt
        z_vals = np.linspace(0, self.Lz, self.Nz, endpoint=False)

        if self.dim == 2:
            Ez = np.mean(Ez_warpx[:-1], axis=0)
            Bx = np.mean(Bx_warpx[:-1], axis=0)
            By = np.mean(By_warpx[:-1], axis=0)
        else:
            Ez = np.mean(Ez_warpx[:-1, :-1], axis=(0, 1))
            Bx = np.mean(Bx_warpx[:-1], axis=(0, 1))
            By = np.mean(By_warpx[:-1], axis=(0, 1))

        with open(f"diags/{self.output_file_name}", "a") as f:
            for ii in range(self.Nz):
                f.write(
                    f"{step:05d} {t:.10e} {z_vals[ii]:.10e} {Ez[ii]:+.10e} "
                    f"{Bx[ii]:+.10e} {By[ii]:+.10e}\n"
                )


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
    "--bdir",
    help="Direction of the B-field",
    required=False,
    choices=["x", "y", "z"],
    default="z",
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = EMModes(test=args.test, dim=args.dim, B_dir=args.bdir, verbose=args.verbose)
simulation.step()
