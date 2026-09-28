#!/usr/bin/env python3
#
# --- Theta-implicit hybrid-PIC solve of a magnetized plasma with a shear-Alfven perturbation,
# --- with the ion current response of the Jacobian actions from re-pushing the particles
# --- (default) or from the grid moments (--fluid: implicit_evolve.use_fluid_ion_response).
# ---
# --- A uniform plasma in a doubly periodic (x, z) domain carries a uniform B0 along z and an
# --- ion velocity perturbation delta v_x = eps v_A sin(k z). The time step makes the implicit
# --- gyro-rotation b = Omega_ci dt / 2 = 0.1, so the magnetized ion response matters in the
# --- Jacobian. The Jacobian only changes how Newton converges, not the solution it converges
# --- to: analysis_fluid_ion_response.py checks that the --fluid run reproduces the fields of
# --- the particle-push run to within the Newton tolerance and converges in as many iterations.

import argparse
import math
import shutil
import sys
from pathlib import Path

from mpi4py import MPI as mpi

import pywarpx
from pywarpx import picmi

constants = picmi.constants
comm = mpi.COMM_WORLD


class MagnetizedShearAlfven(object):
    # plasma
    B0 = 0.1  # uniform field along z (T)
    n0 = 1.0e20  # density (m^-3)
    Te = 10.0  # isothermal electrons (eV)
    Ti = 1.0  # ion temperature (eV)
    eps = 0.05  # ion velocity perturbation / v_A

    # numerics
    NX = 32
    NZ = 32
    NPPC = 64
    L_over_di = 16.0  # domain length in ion inertial lengths
    b = 0.1  # Omega_ci dt / 2
    steps = 10

    def __init__(self, fluid, verbose):
        self.fluid = fluid
        self.verbose = verbose
        self.m_i = constants.m_p
        self.w_ci = constants.q_e * self.B0 / self.m_i
        self.vA = self.B0 / (constants.mu0 * self.n0 * self.m_i) ** 0.5
        w_pi = (constants.q_e**2 * self.n0 / (constants.ep0 * self.m_i)) ** 0.5
        self.d_i = constants.c / w_pi
        self.L = self.L_over_di * self.d_i
        self.k = 2.0 * math.pi / self.L
        self.dt = 2.0 * self.b / self.w_ci
        self.vi_th = (constants.q_e * self.Ti / self.m_i) ** 0.5
        self.setup_run()

    def setup_run(self):
        grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[0.0, 0.0],
            upper_bound=[self.L, self.L],
            lower_boundary_conditions=["periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic"],
            warpx_max_grid_size=16,
        )
        solver = picmi.HybridPICSolver(
            grid=grid,
            gamma=1.0,
            Te=self.Te,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=1.0e-6,
        )

        linear_solver = picmi.GMRESLinearSolver(
            max_iterations=200,
            relative_tolerance=1.0e-6,
            absolute_tolerance=0.0,
            restart_length=100,
        )
        nonlinear_solver = picmi.NewtonNonlinearSolver(
            diagnostic_file="newton.txt",
            diagnostic_interval=1,
            verbose=True,
            max_iterations=20,
            relative_tolerance=1.0e-9,
            absolute_tolerance=0.0,
            require_convergence=True,
            linear_solver=linear_solver,
            max_particle_iterations=21,
            particle_tolerance=1.0e-12,
        )
        evolve_scheme = picmi.ThetaImplicitHybridEvolveScheme(
            theta=0.5,
            nonlinear_solver=nonlinear_solver,
            use_fluid_ion_response=self.fluid,
        )
        # physics-based preconditioner in its linear (fixed) V-cycle form, which the plain
        # GMRES accepts; with --fluid its ion block comes from the same grid moments
        pywarpx.warpx.get_bucket("jacobian").pc_type = "pc_hybrid_pic"
        pywarpx.warpx.get_bucket("pc_hybrid_pic").inner_max = 0
        # guard range of the implicit deposit (the default admits no cell crossing)
        pywarpx.particles.max_grid_crossings = 2

        simulation = picmi.Simulation(
            solver=solver,
            warpx_evolve_scheme=evolve_scheme,
            time_step_size=self.dt,
            max_steps=self.steps,
            verbose=self.verbose,
            particle_shape=1,
            warpx_grid_type="collocated",
            warpx_current_deposition_algo="direct",
            warpx_serialize_initial_conditions=True,
            warpx_random_seed=1,
        )
        simulation.add_applied_field(
            picmi.AnalyticInitialField(
                Bx_expression="0.0", By_expression="0.0", Bz_expression=f"{self.B0}"
            )
        )

        ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=self.m_i,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression=f"{self.n0}",
                momentum_expressions=[
                    f"{self.eps * self.vA}*sin({self.k}*z)",
                    "0.0",
                    "0.0",
                ],
                warpx_momentum_spread_expressions=[f"{self.vi_th}"] * 3,
            ),
        )
        simulation.add_species(
            ions,
            layout=picmi.PseudoRandomLayout(
                grid=grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        if comm.rank == 0 and Path("diags").exists():
            shutil.rmtree("diags")
        comm.Barrier()
        simulation.add_diagnostic(
            picmi.FieldDiagnostic(
                name="field_diags",
                grid=grid,
                period=self.steps,
                data_list=["B", "E", "J", "rho"],
                write_dir="diags",
                warpx_format="openpmd",
                warpx_openpmd_backend="h5",
            )
        )
        simulation.initialize_inputs()
        simulation.initialize_warpx()
        self.simulation = simulation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fluid", action="store_true", help="implicit_evolve.use_fluid_ion_response"
)
parser.add_argument("-v", "--verbose", action="store_true")
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = MagnetizedShearAlfven(fluid=args.fluid, verbose=args.verbose)
run.simulation.step()
