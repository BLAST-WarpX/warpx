#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. The script simulates an ion beam instability wherein a
# --- low density ion beam interacts with background plasma. See Section 6.5 of
# --- Matthews (1994) and Section 4.4 of Munoz et al. (2018).

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


class HybridPICBeamInstability(object):
    """This input is based on the ion beam R instability test as described by
    Munoz et al. (2018).
    """

    # Applied field parameters
    B0 = 0.25  # Initial magnetic field strength (T)
    beta = 1.0  # Plasma beta, used to calculate temperature

    # Plasma species parameters
    m_ion = 100.0  # Ion mass (electron masses)
    vA_over_c = 1e-4  # ratio of Alfven speed and the speed of light

    # Spatial domain
    Nz = 1024  # number of cells in z direction
    Nx = 8  # number of cells in x (and y) direction for >1 dimensions

    # Temporal domain (if not run as a CI test)
    LT = 120.0  # Simulation temporal length (ion cyclotron periods)

    # Numerical parameters
    NPPC = [1024, 256, 64]  # Seed number of particles per cell
    DZ = 1.0 / 4.0  # Cell size (ion skin depths)
    DT = 0.01  # Time step (ion cyclotron periods)

    # Plasma resistivity - used to dampen the mode excitation
    eta = 1e-7
    # Number of substeps used to update B
    substeps = 10

    # Beam parameters
    n_beam = [0.02, 0.1]
    U_bc = 10.0  # relative drifts between beam and core in Alfven speeds

    def __init__(self, test, dim, resonant, verbose):
        """Get input parameters for the specific case desired."""
        self.test = test
        self.dim = int(dim)
        self.resonant = resonant
        self.verbose = verbose or self.test

        # sanity check
        assert dim > 0 and dim < 4, f"{dim}-dimensions not a valid input"

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.n_beam = self.n_beam[1 - int(resonant)]
        self.u_beam = 1.0 / (1.0 + self.n_beam) * self.U_bc * self.vA
        self.u_c = -1.0 * self.n_beam / (1.0 + self.n_beam) * self.U_bc * self.vA
        self.n_beam = self.n_beam * self.n_plasma

        self.dz = self.DZ * self.l_i
        self.Lz = self.Nz * self.dz
        self.Lx = self.Nx * self.dz

        if self.dim == 3:
            self.volume = self.Lx * self.Lx * self.Lz
            self.N_cells = self.Nx * self.Nx * self.Nz
        elif self.dim == 2:
            self.volume = self.Lx * self.Lz
            self.N_cells = self.Nx * self.Nz
        else:
            self.volume = self.Lz
            self.N_cells = self.Nz

        diag_period = 1 / 4.0  # Output interval (ion cyclotron periods)
        self.diag_steps = int(diag_period / self.DT)

        # if this is a test case run for only 25 cyclotron periods
        if self.test:
            self.LT = 25.0

        self.total_steps = int(np.ceil(self.LT / self.DT))

        self.dt = self.DT / self.w_ci

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
            Te=self.T_plasma / 10.0,
            n0=self.n_plasma + self.n_beam,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
        )
        simulation.solver = self.solver

        B_ext = picmi.AnalyticInitialField(
            Bx_expression=0.0, By_expression=0.0, Bz_expression=self.B0
        )
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
                directed_velocity=[0, 0, self.u_c],
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC[self.dim - 1]
            ),
        )
        self.beam_ions = picmi.Species(
            name="beam_ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.UniformDistribution(
                density=self.n_beam,
                rms_velocity=[self.v_ti] * 3,
                directed_velocity=[0, 0, self.u_beam],
            ),
        )
        simulation.add_species(
            self.beam_ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC[self.dim - 1] / 2
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterstep(self.energy_diagnostic)
        callbacks.installafterstep(self.text_diag)

        if self.test:
            part_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=1250,
                species=[self.ions, self.beam_ions],
                data_list=["ux", "uy", "uz", "z", "weighting"],
            )
            simulation.add_diagnostic(part_diag)
            field_diag = picmi.FieldDiagnostic(
                name="diag1",
                grid=self.grid,
                period=1250,
                data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
            )
            simulation.add_diagnostic(field_diag)

        # output the full particle data at t*w_ci = 40
        step = int(40.0 / self.DT)
        parts_diag = picmi.ParticleDiagnostic(
            name="parts_diag",
            period=f"{step}:{step}",
            species=[self.ions, self.beam_ions],
            write_dir="diags",
            warpx_file_prefix="Python_hybrid_PIC_plt",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(parts_diag)

        self.output_file_name = "field_data.txt"
        if self.dim == 1:
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
                f.write("[0]step() [1]time(s) [2]z_coord(m) [3]By_lev0-(T)\n")

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()

        # create particle container wrapper for the ion species to access
        # particle data
        self.ion_container_wrapper = particle_containers.ParticleContainerWrapper(
            self.ions.name
        )
        self.beam_ion_container_wrapper = particle_containers.ParticleContainerWrapper(
            self.beam_ions.name
        )

    def _create_data_arrays(self):
        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_step = 0

        if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
            # allocate arrays for storing energy values
            self.energy_vals = np.zeros((self.total_steps // self.diag_steps, 4))

    def text_diag(self):
        """Diagnostic function to print out timing data and particle numbers."""
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if not hasattr(self, "prev_time"):
            self._create_data_arrays()

        if step % (self.total_steps // 10) != 0:
            return

        wall_time = time.time() - self.prev_time
        steps = step - self.prev_step
        step_rate = steps / wall_time

        status_dict = {
            "step": step,
            "nplive beam ions": self.ion_container_wrapper.nps,
            "nplive ions": self.beam_ion_container_wrapper.nps,
            "wall_time": wall_time,
            "step_rate": step_rate,
            "diag_steps": self.diag_steps,
            "iproc": None,
        }

        diag_string = (
            "Step #{step:6d}; "
            "{nplive beam ions} beam ions; "
            "{nplive ions} core ions; "
            "{wall_time:6.1f} s wall time; "
            "{step_rate:4.2f} steps/s"
        )

        if libwarpx.amr.ParallelDescriptor.MyProc() == 0:
            print(diag_string.format(**status_dict))

        self.prev_time = time.time()
        self.prev_step = step

    def energy_diagnostic(self):
        """Diagnostic to get the total, magnetic and kinetic energies in the
        simulation."""
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % self.diag_steps != 1:
            return

        idx = (step - 1) // self.diag_steps

        if not hasattr(self, "prev_time"):
            self._create_data_arrays()

        # get the simulation energies
        Ec_par, Ec_perp = self._get_kinetic_energy(self.ion_container_wrapper)
        Eb_par, Eb_perp = self._get_kinetic_energy(self.beam_ion_container_wrapper)

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        self.energy_vals[idx, 0] = Ec_par
        self.energy_vals[idx, 1] = Ec_perp
        self.energy_vals[idx, 2] = Eb_par
        self.energy_vals[idx, 3] = Eb_perp

        if step == self.total_steps:
            np.save("diags/energies.npy", run.energy_vals)

    def _get_kinetic_energy(self, container_wrapper):
        """Utility function to retrieve the total kinetic energy in the
        simulation."""
        try:
            ux = np.concatenate(container_wrapper.get_particle_ux())
            uy = np.concatenate(container_wrapper.get_particle_uy())
            uz = np.concatenate(container_wrapper.get_particle_uz())
            w = np.concatenate(container_wrapper.get_particle_weight())
        except ValueError:
            return 0.0, 0.0

        my_E_perp = 0.5 * self.M * np.sum(w * (ux**2 + uy**2))
        E_perp = comm.allreduce(my_E_perp, op=mpi.SUM)

        my_E_par = 0.5 * self.M * np.sum(w * uz**2)
        E_par = comm.allreduce(my_E_par, op=mpi.SUM)

        return E_par, E_perp

    def _record_average_fields(self):
        """A custom reduced diagnostic to store the average E&M fields in a
        similar format as the reduced diagnostic so that the same analysis
        script can be used regardless of the simulation dimension.
        """
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % self.diag_steps != 0:
            return

        By_warpx = fields.BxWrapper()[...]

        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return

        t = step * self.dt
        z_vals = np.linspace(0, self.Lz, self.Nz, endpoint=False)

        if self.dim == 2:
            By = np.mean(By_warpx[:-1], axis=0)
        else:
            By = np.mean(By_warpx[:-1], axis=(0, 1))

        with open(f"diags/{self.output_file_name}", "a") as f:
            for ii in range(self.Nz):
                f.write(f"{step:05d} {t:.10e} {z_vals[ii]:.10e} {By[ii]:+.10e}\n")


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
    "-r",
    "--resonant",
    help="Run the resonant case",
    required=False,
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

run = HybridPICBeamInstability(
    test=args.test, dim=args.dim, resonant=args.resonant, verbose=args.verbose
)
simulation.step()
