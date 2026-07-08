#!/usr/bin/env python3
#
# --- Test script for conducting (PEC) boundary support in the semi-implicit
# --- Darwin solver. A quasi-neutral electron-ion plasma sits between two PEC
# --- walls at z=0 and z=Lz. The electrons carry a transverse drift velocity
# --- perturbation proportional to sin(pi*z/Lz), which vanishes at both walls
# --- and is symmetric about the domain midplane. This sources a current
# --- sheet that drives inductive E/B fields through the Darwin solve.
# --- The analysis script (analysis_pec.py, shared with the 2D PEC test)
# --- checks that: (a) the wall-normal B-field stays regular (zero) at the
# --- walls, as required by the PEC condition; (b) the driven field's energy
# --- is balanced between the two halves of the domain, as dictated by the
# --- symmetric drive and the identical treatment of both walls; and (c) the
# --- field+particle energy is not still growing unboundedly by the end of
# --- the run.

import argparse

import dill
import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, particles, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class DummyES_Solver(picmi.ElectrostaticSolver):
    """A no-op electrostatic solver. The plasma here is quasi-neutral (only
    a velocity, not a density, perturbation is applied) so it generates no
    appreciable electrostatic field; skip the Poisson solve, as the periodic
    Darwin EM-modes test does for its parallel-mode case."""

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
    """Two PEC walls bound a quasi-neutral electron-ion plasma at z=0 and
    z=Lz. Particle boundaries are reflecting, keeping the plasma confined.
    """

    # Dimension and wall-normal axis, used by the shared analysis script.
    # Efield_fp only ever holds the electrostatic component between steps
    # (see analysis_pec.py), so the checks are built from B instead: the
    # wall-normal component (zero at the wall under PEC) and the driven
    # tangential component(s) (the ones excited by the drive below) for a
    # wall whose normal is z.
    dim = 1
    wall_axis = "z"
    normal_B_col = 10  # Bz
    driven_B_cols = [8]  # Bx (driven by the Jy current sheet)

    # Plasma parameters
    n0 = 1.0e18  # density (m^-3)
    Te = 10.0  # electron temperature (eV)
    Ti = 10.0  # ion temperature (eV)
    m_ion = 10.0  # ion mass (electron masses)
    v0_over_vte = 0.1  # amplitude of the electron drift perturbation

    # Spatial domain. The grid spacing is set relative to the electron skin
    # depth c/w_pe, not the Debye length: the Darwin GMRES solve has no
    # preconditioner, and its mass-matrix (susceptibility) term - which is
    # what regularizes the otherwise poorly conditioned biharmonic operator -
    # only dominates when the grid is comparable to or coarser than a skin
    # depth. A Debye-length-based grid (fine compared to the skin depth)
    # leaves the biharmonic term dominant and GMRES stalls.
    Nz = 128
    DZ = 1.5  # cell size (electron skin depths)

    # Numerical parameters. DT is deliberately smaller than the periodic
    # Darwin test's DT=10: reflecting particles near the walls exercise
    # WarpXParticleContainer's local-redistribute path, which is only sized
    # for a bounded number of cell crossings per step
    # (particles.max_grid_crossings below) - the Maxwellian velocity tail
    # combined with a large dt occasionally produces a particle that exceeds
    # that bound and segfaults in MultiParticleContainer::RedistributeLocal
    # (or aborts in the charge-deposition shape-fit assert). A smaller dt
    # shrinks the per-step crossing distance for every particle, including
    # rare tail draws, without changing GMRES conditioning (verified
    # dt-independent - the susceptibility term's dt-scaling cancels against
    # the mass-matrix deposit's own dt-linear scaling).
    NPPC = 200
    DT = 2.0  # time step (electron plasma periods)
    MAX_GRID_CROSSINGS = 20

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or self.test

        self.get_plasma_quantities()

        self.d_e = constants.c / self.w_pe
        self.dz = self.DZ * self.d_e
        self.Lz = self.Nz * self.dz
        self.dt = self.DT / self.w_pe
        self.v0 = self.v0_over_vte * self.v_te

        if not self.test:
            self.total_steps = 4000
            self.diag_steps = 40
        else:
            # if this is a test case run for only a small number of steps
            self.total_steps = 30
            self.diag_steps = 2

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
                f"\tLz = {self.Lz:.2e} m ({self.Nz} cells)\n"
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

        # Debye length (m)
        self.lambda_e = self.v_te / self.w_pe

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian1DGrid(
            number_of_cells=[self.Nz],
            warpx_max_grid_size=self.Nz,
            lower_bound=[0.0],
            upper_bound=[self.Lz],
            lower_boundary_conditions=["dirichlet"],
            upper_boundary_conditions=["dirichlet"],
            lower_boundary_conditions_particles=["reflecting"],
            upper_boundary_conditions_particles=["reflecting"],
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.particle_shape = 1
        simulation.verbose = self.verbose
        simulation.current_deposition_algo = "direct"
        # The PEC wall treatment removes a few near-wall modes from the
        # well-conditioned (mass-matrix-dominated) part of the operator's
        # spectrum; resolving them needs a single Krylov cycle of a few
        # hundred vectors rather than the ~30 that suffice for the periodic
        # Darwin solve, or GMRES stalls on a restart before it can build a
        # large enough subspace.
        simulation.evolve_scheme = picmi.SemiImplicitDarwinEvolveScheme(
            linear_solver=picmi.GMRESLinearSolver(
                relative_tolerance=5e-5,
                restart_length=350,
                max_iterations=700,
                verbose_int=(2 if self.test else 0),
            ),
        )
        particles.max_grid_crossings = self.MAX_GRID_CROSSINGS

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
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
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
                    f"{self.v0}*sin({np.pi}*z/{self.Lz})",
                    None,
                ],
                warpx_momentum_spread_expressions=[f"{self.v_te}"] * 3,
            ),
        )
        simulation.add_species(
            self.electrons,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
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

        # Line probe of E and B along z, from one wall to the other, used by
        # the analysis script to check wall regularity and mirror symmetry.
        line_diag = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            probe_geometry="Line",
            z_probe=0.0,
            z1_probe=self.Lz,
            resolution=self.Nz - 1,
            name="line_probe",
            period=self.diag_steps,
            path=write_dir,
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
parser.add_argument(
    "-v",
    "--verbose",
    help="Verbose output",
    action="store_true",
)
args, left = parser.parse_known_args()

run = DarwinPECWalls(test=args.test, verbose=args.verbose)
simulation.step()
