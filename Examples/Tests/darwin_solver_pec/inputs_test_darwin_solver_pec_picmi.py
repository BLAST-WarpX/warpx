#!/usr/bin/env python3
#
# --- Test script for conducting (PEC) wall support in the semi-implicit
# --- Darwin solver. A quasi-neutral electron-ion plasma sits between two PEC
# --- walls, normal to z in 1D and to x in 2D and 3D (the other directions are
# --- periodic). The electrons carry a transverse (y) drift velocity
# --- perturbation proportional to sin(pi*s/L), with s the wall-normal
# --- coordinate, which vanishes at both walls and is symmetric about the
# --- domain midplane. This sources a current sheet that drives a magnetic
# --- field through the Darwin solve.
# --- The analysis script (analysis_pec.py) checks that: (a) the tangential
# --- E-field (Ey) vanishes at the walls, as required by the PEC condition;
# --- (b) the driven tangential B-field has the profile, sign and symmetry
# --- about the domain midplane dictated by the drive and the identical
# --- treatment of both walls; and (c) the field+particle energy is not still
# --- growing unboundedly by the end of the run.

import argparse

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class DummyES_Solver(picmi.ElectrostaticSolver):
    """A no-op electrostatic solver. The plasma is quasi-neutral (only a
    velocity, not a density, perturbation is applied), so it generates no
    appreciable electrostatic field and the Poisson solve is skipped."""

    def __init__(self, grid):
        super(DummyES_Solver, self).__init__(
            grid=grid, method="Multigrid", required_precision=1
        )

    def solver_initialize_inputs(self):
        super(DummyES_Solver, self).solver_initialize_inputs()
        callbacks.installpoissonsolver(self.skip_poisson_solve)

    def skip_poisson_solve(self):
        pass


class DarwinPECWalls(object):
    """Two PEC walls bound a quasi-neutral electron-ion plasma. Particles are
    reflected at the walls, keeping the plasma confined."""

    # Plasma parameters
    n0 = 1.0e18  # density (m^-3)
    Te = 10.0  # electron temperature (eV)
    Ti = 10.0  # ion temperature (eV)
    m_ion = 10.0  # ion mass (electron masses)
    v0_over_vte = 1.0  # amplitude of the electron drift perturbation

    # Spatial domain: number of cells across the walls, and along each
    # periodic direction (the drive is uniform along those, so only a few
    # cells are needed)
    N_wall = {1: 128, 2: 32, 3: 32}
    N_periodic = 8
    DX = 1.5  # cell size (electron skin depths)

    # Numerical parameters
    NPPC = {1: 1000, 2: 100, 3: 100}
    DT = 2.0  # time step (electron plasma periods)

    def __init__(self, dim, test, verbose):
        self.dim = dim
        self.test = test
        self.verbose = verbose or self.test

        # Wall-normal axis, and the FieldProbe columns (see
        # Source/Diagnostics/ReducedDiags/FieldProbe.cpp:
        # [0]step [1]time [2]x [3]y [4]z [5]Ex [6]Ey [7]Ez [8]Bx [9]By [10]Bz)
        # used by the analysis script: the coordinate across the walls, the
        # tangential E-field component parallel to the current sheet (Ey) and
        # the tangential B-field component driven by the current sheet. The
        # latter is proportional to cos(pi*s/L), with the sign given below
        # (from curl(B) = mu0*J, with Jy proportional to -sin(pi*s/L)).
        self.wall_axis = "z" if self.dim == 1 else "x"
        self.coord_col = 4 if self.dim == 1 else 2
        self.tangential_E_col = 6  # Ey
        self.driven_B_col = 8 if self.dim == 1 else 10  # Bx in 1D, Bz otherwise
        self.driven_B_sign = 1.0 if self.dim == 1 else -1.0

        self.get_plasma_quantities()

        self.d_e = constants.c / self.w_pe
        self.dx = self.DX * self.d_e
        self.L_wall = self.N_wall[self.dim] * self.dx
        self.L_periodic = self.N_periodic * self.dx
        self.dt = self.DT / self.w_pe
        self.v0 = self.v0_over_vte * self.v_te

        if not self.test:
            self.total_steps = 4000
            self.diag_steps = 40
        else:
            # if this is a test case run for only a small number of steps.
            # The probe samples every step: the scheme carries an undamped
            # parasitic mode that alternates sign every step, which the
            # analysis script removes by averaging consecutive samples.
            self.total_steps = 30
            self.diag_steps = 1

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        if comm.rank == 0:
            print(
                f"Initializing simulation with input parameters:\n"
                f"\tn0 = {self.n0:.1e} m^-3\n"
                f"\tTe = {self.Te:.1f} eV\n"
                f"\tw_pe = {self.w_pe:.2e} rad/s\n"
                f"\tL = {self.L_wall:.2e} m ({self.N_wall[self.dim]} cells) "
                f"between the walls\n"
                f"\tdt = {self.dt:.2e} s\n"
                f"\ttotal steps = {self.total_steps:d}\n"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        """Calculate the plasma quantities used to set the spatial and
        temporal resolution of the simulation."""
        self.M = self.m_ion * constants.m_e

        self.w_pe = np.sqrt(
            constants.q_e**2 * self.n0 / (constants.m_e * constants.ep0)
        )
        self.v_te = np.sqrt(self.Te * constants.q_e / constants.m_e)
        self.v_ti = np.sqrt(self.Ti * constants.q_e / self.M)

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        n_wall = self.N_wall[self.dim]
        if self.dim == 1:
            grid_object = picmi.Cartesian1DGrid
            number_of_cells = [n_wall]
            upper_bound = [self.L_wall]
            field_bc = ["dirichlet"]
            particle_bc = ["reflecting"]
        elif self.dim == 2:
            grid_object = picmi.Cartesian2DGrid
            number_of_cells = [n_wall, self.N_periodic]
            upper_bound = [self.L_wall, self.L_periodic]
            field_bc = ["dirichlet", "periodic"]
            particle_bc = ["reflecting", "periodic"]
        else:
            grid_object = picmi.Cartesian3DGrid
            number_of_cells = [n_wall, self.N_periodic, self.N_periodic]
            upper_bound = [self.L_wall, self.L_periodic, self.L_periodic]
            field_bc = ["dirichlet", "periodic", "periodic"]
            particle_bc = ["reflecting", "periodic", "periodic"]

        self.grid = grid_object(
            number_of_cells=number_of_cells,
            warpx_max_grid_size=n_wall,
            lower_bound=[0.0] * self.dim,
            upper_bound=upper_bound,
            lower_boundary_conditions=field_bc,
            upper_boundary_conditions=field_bc,
            lower_boundary_conditions_particles=particle_bc,
            upper_boundary_conditions_particles=particle_bc,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.particle_shape = 1
        simulation.verbose = self.verbose
        simulation.evolve_scheme = picmi.SemiImplicitDarwinEvolveScheme(
            graddiv_factor=1.0,
            linear_solver=picmi.GMRESLinearSolver(
                relative_tolerance=5e-5,
                max_iterations=200,
                verbose_int=(2 if self.test else 0),
                pc_type=picmi.DarwinMLMGPreconditioner(),
            ),
        )

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge=constants.q_e,
            mass=self.M,
            initial_distribution=picmi.UniformDistribution(
                density=self.n0,
                rms_velocity=[self.v_ti] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC[self.dim]
            ),
        )

        # Electrons carry a transverse (y) drift velocity that vanishes at
        # both walls and is symmetric about the domain midplane, sourcing a
        # current sheet with the same symmetry.
        self.electrons = picmi.Species(
            name="electrons",
            charge=-constants.q_e,
            mass=constants.m_e,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression=f"{self.n0}",
                momentum_expressions=[
                    None,
                    f"{self.v0}*sin({np.pi}*{self.wall_axis}/{self.L_wall})",
                    None,
                ],
                warpx_momentum_spread_expressions=[f"{self.v_te}"] * 3,
            ),
        )
        simulation.add_species(
            self.electrons,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC[self.dim]
            ),
        )

        #######################################################################
        # Field solver                                                        #
        #######################################################################

        self.solver = DummyES_Solver(self.grid)
        simulation.solver = self.solver

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        if self.test:
            particle_diag = picmi.ParticleDiagnostic(
                name="field_diag",
                period=self.total_steps,
            )
            simulation.add_diagnostic(particle_diag)
            field_diag = picmi.FieldDiagnostic(
                name="field_diag",
                grid=self.grid,
                period=self.total_steps,
                data_list=["B", "E"],
            )
            simulation.add_diagnostic(field_diag)

        write_dir = "diags/"

        # Line probe of E and B from one wall to the other (at a fixed
        # position along the periodic directions, which is immaterial since
        # the drive is uniform along those), used by the analysis script.
        # The end point is moved just inside the domain, since a probe point
        # on the upper domain boundary is discarded.
        probe_kwargs = {
            f"{self.wall_axis}_probe": 0.0,
            f"{self.wall_axis}1_probe": self.L_wall * (1.0 - 1e-10),
        }
        periodic_axes = {1: [], 2: ["z"], 3: ["y", "z"]}[self.dim]
        for axis in periodic_axes:
            probe_kwargs[f"{axis}_probe"] = 0.5 * self.L_periodic
            probe_kwargs[f"{axis}1_probe"] = 0.5 * self.L_periodic
        line_diag = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            probe_geometry="Line",
            resolution=n_wall + 1,
            name="line_probe",
            period=self.diag_steps,
            path=write_dir,
            **probe_kwargs,
        )
        simulation.add_diagnostic(line_diag)

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
parser.add_argument("-d", "--dim", help="Simulation dimension", required=True, type=int)
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()

run = DarwinPECWalls(dim=args.dim, test=args.test, verbose=args.verbose)
simulation.step()
