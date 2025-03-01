#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein ions are
# --- treated as kinetic particles and electrons as an isothermal, inertialess
# --- background fluid. The script demonstrates the use of this model to
# --- simulate adiabatic compression of a plasma cylinder initialized from an
# --- analytical Grad-Shafranov solution.

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import openpmd_api as io
from mpi4py import MPI as mpi

from pywarpx import fields, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)


class PlasmaCylinderCompression(object):
    # B0 is chosen with all other quantities scaled by it
    n0 = 1e20
    T_i = 10  # eV
    T_e = 0
    p0 = n0 * constants.q_e * T_i

    B0 = np.sqrt(2 * constants.mu0 * p0)  # Initial magnetic field strength (T)

    # Do a 2x uniform B-field compression
    dB = B0

    # Flux Conserver radius
    R_c = 0.5

    # Plasma Radius (These values control the analytical GS solution)
    R_p = 0.25
    delta_p = 0.025

    # Domain parameters
    LX = 2.0 * R_c * 1.05  # m
    LY = 2.0 * R_c * 1.05
    LZ = 0.5  # m

    LT = 10  # ion cyclotron periods
    DT = 1e-3  # ion cyclotron periods

    # Resolution parameters
    NX = 256
    NY = 256
    NZ = 128

    # Starting number of particles per cell
    NPPC = 100

    # Number of substeps used to update B
    substeps = 20

    def Bz(self, r):
        return np.sqrt(
            self.B0**2
            - 2.0
            * constants.mu0
            * self.n0
            * constants.q_e
            * self.T_i
            / (1.0 + np.exp((r - self.R_p) / self.delta_p))
        )

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or self.test

        self.Lx = self.LX
        self.Ly = self.LY
        self.Lz = self.LZ

        self.DX = self.LX / self.NX
        self.DY = self.LY / self.NY
        self.DZ = self.LZ / self.NZ

        if comm.rank == 0:
            # Write uniform compression dataset to OpenPMD to exercise reading openPMD data
            # for the time varying external fields
            xvec = np.linspace(-self.LX, self.LX, num=2 * self.NX)
            yvec = np.linspace(-self.LY, self.LY, num=2 * self.NY)
            zvec = np.linspace(-self.LZ, self.LZ, num=2 * self.NZ)
            XM, YM, ZM = np.meshgrid(xvec, yvec, zvec, indexing="ij")

            RM = np.sqrt(XM**2 + YM**2)

            Ax_data = -0.25 * YM * self.dB
            Ay_data = 0.25 * XM * self.dB
            Az_data = np.zeros_like(RM)

            # Write vector potential to file to exercise field loading via OpenPMD
            series = io.Series("Afield.h5", io.Access.create)

            it = series.iterations[0]

            A = it.meshes["A"]
            A.grid_spacing = [self.DX, self.DY, self.DZ]
            A.grid_global_offset = [-self.LX, -self.LY, -self.LZ]
            A.grid_unit_SI = 1.0
            A.axis_labels = ["x", "y", "z"]
            A.data_order = "C"
            A.unit_dimension = {
                io.Unit_Dimension.M: 1.0,
                io.Unit_Dimension.T: -2.0,
                io.Unit_Dimension.I: -1.0,
                io.Unit_Dimension.L: -1.0,
            }

            Ax = A["x"]
            Ay = A["y"]
            Az = A["z"]

            Ax.position = [0.0, 0.0]
            Ay.position = [0.0, 0.0]
            Az.position = [0.0, 0.0]

            Ax_dataset = io.Dataset(Ax_data.dtype, Ax_data.shape)

            Ay_dataset = io.Dataset(Ay_data.dtype, Ay_data.shape)

            Az_dataset = io.Dataset(Az_data.dtype, Az_data.shape)

            Ax.reset_dataset(Ax_dataset)
            Ay.reset_dataset(Ay_dataset)
            Az.reset_dataset(Az_dataset)

            Ax.store_chunk(Ax_data)
            Ay.store_chunk(Ay_data)
            Az.store_chunk(Az_data)

            series.flush()
            series.close()

        comm.Barrier()

        # calculate various plasma parameters based on the simulation input
        self.get_plasma_quantities()

        self.dt = self.DT * self.t_ci

        # run very low resolution as a CI test
        if self.test:
            self.total_steps = 20
            self.diag_steps = self.total_steps // 5
            self.NX = 64
            self.NY = 64
            self.NZ = 32
        else:
            self.total_steps = int(self.LT / self.DT)
            self.diag_steps = 100

        # print out plasma parameters
        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tTi = {self.T_i:.1f} eV\n"
                f"\tn0 = {self.n0:.1e} m^-3\n"
                f"\tB0 = {self.B0:.2f} T\n",
                f"\tDX/DY = {self.DX / self.l_i:.3f} c/w_pi\n"
                f"\tDZ = {self.DZ / self.l_i:.3f} c/w_pi\n",
            )
            print(
                f"Plasma parameters:\n"
                f"\tl_i = {self.l_i:.1e} m\n"
                f"\tt_ci = {self.t_ci:.1e} s\n"
                f"\tv_ti = {self.vi_th:.1e} m/s\n"
                f"\tvA = {self.vA:.1e} m/s\n"
            )
            print(
                f"Numerical parameters:\n"
                f"\tdz = {self.Lz / self.NZ:.1e} m\n"
                f"\tdt = {self.dt:.1e} s\n"
                f"\tdiag steps = {self.diag_steps:d}\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate various plasma parameters based on the simulation input."""

        # Ion mass (kg)
        self.M = constants.m_p

        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci = constants.q_e * abs(self.B0) / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci

        # Ion plasma frequency (Hz)
        self.w_pi = np.sqrt(constants.q_e**2 * self.n0 / (self.M * constants.ep0))

        # Ion skin depth (m)
        self.l_i = constants.c / self.w_pi

        # # Alfven speed (m/s): vA = B / sqrt(mu0 * n * (M + m)) = c * omega_ci / w_pi
        self.vA = abs(self.B0) / np.sqrt(
            constants.mu0 * self.n0 * (constants.m_e + self.M)
        )

        # calculate thermal speeds
        self.vi_th = np.sqrt(self.T_i * constants.q_e / self.M)

        # Ion Larmor radius (m)
        self.rho_i = self.vi_th / self.w_ci

    def load_fields(self):
        Bx = fields.BxFPExternalWrapper(include_ghosts=False)
        By = fields.ByFPExternalWrapper(include_ghosts=False)
        Bz = fields.BzFPExternalWrapper(include_ghosts=False)

        Bx[:, :] = 0.0
        By[:, :] = 0.0

        XM, YM, ZM = np.meshgrid(
            Bz.mesh("x"), Bz.mesh("y"), Bz.mesh("z"), indexing="ij"
        )

        RM = np.sqrt(XM**2 + YM**2)

        Bz[:, :] = self.Bz(RM)
        comm.Barrier()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        # Create grid
        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.NX, self.NY, self.NZ],
            lower_bound=[-0.5 * self.Lx, -0.5 * self.Ly, -0.5 * self.Lz],
            upper_bound=[0.5 * self.Lx, 0.5 * self.Ly, 0.5 * self.Lz],
            lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
            upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
            lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
            upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
            warpx_max_grid_size=self.NZ,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1
        simulation.use_filter = True
        simulation.verbose = self.verbose

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################
        # External Field definition. Sigmoid starting around 2.5 us
        A_ext = {
            "uniform_file": {
                "read_from_file": True,
                "path": "Afield.h5",
                "A_time_external_function": "1/(1+exp(5*(1-(t-t0_ramp)*sqrt(2)/tau_ramp)))",
            },
            "uniform_analytical": {
                "Ax_external_function": f"-0.25*y*{self.dB}",
                "Ay_external_function": f"0.25*x*{self.dB}",
                "Az_external_function": "0",
                "A_time_external_function": "1/(1+exp(5*(1-(t-t0_ramp)*sqrt(2)/tau_ramp)))",
            },
        }

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=1.0,
            Te=self.T_e,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity="if(rho<=rho_floor,eta_v,eta_p)",
            plasma_hyper_resistivity=1e-8,
            substeps=self.substeps,
            A_external=A_ext,
            tau_ramp=20e-6,
            t0_ramp=5e-6,
            rho_floor=0.05 * self.n0 * constants.q_e,
            eta_p=1e-8,
            eta_v=1e-3,
        )
        simulation.solver = self.solver

        simulation.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="(x**2+y**2-R_w**2)", R_w=self.R_c
        )

        # Add field loader callback
        B_ext = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_fields,
            warpx_do_divb_cleaning_external=True,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_ext)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################
        r_omega = "(sqrt(x*x+y*y)*q_e*B0/m_p)"
        dlnndr = "((-1/delta_p)/(1+exp(-(sqrt(x*x+y*y)-R_p)/delta_p)))"
        vth = f"0.5*(-{r_omega}+sqrt({r_omega}*{r_omega}+4*q_e*T_i*{dlnndr}/m_p))"

        momentum_expr = [f"y*{vth}", f"-x*{vth}", "0"]

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.M,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0_p/(1+exp((sqrt(x*x+y*y)-R_p)/delta_p))",
                momentum_expressions=momentum_expr,
                warpx_momentum_spread_expressions=[f"{str(self.vi_th)}"] * 3,
                warpx_density_min=0.01 * self.n0,
                R_p=self.R_p,
                delta_p=self.delta_p,
                n0_p=self.n0,
                B0=self.B0,
                T_i=self.T_i,
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

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="diag1",
                period=self.diag_steps,
                species=[self.ions],
                data_list=["ux", "uy", "uz", "x", "z", "weighting"],
                write_dir="diags",
                warpx_format="plotfile",
            )
            simulation.add_diagnostic(particle_diag)
        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "E", "rho"],
            write_dir="diags",
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(field_diag)

        #######################################################################
        # Initialize                                                          #
        #######################################################################

        if comm.rank == 0:
            if Path.exists(Path("diags")):
                shutil.rmtree("diags")
            Path("diags").mkdir(parents=True, exist_ok=True)

        # Initialize inputs and WarpX instance
        simulation.initialize_inputs()
        simulation.initialize_warpx()


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

run = PlasmaCylinderCompression(test=args.test, verbose=args.verbose)
simulation.step()
