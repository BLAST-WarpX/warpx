#!/usr/bin/env python3
#
# --- 3D Darwin solver test: noise-seeded low-frequency EM waves in a periodic
# --- box with B0 along (1,1,1)/sqrt(3), oblique to every grid axis, so all nine
# --- 3D mass-matrix blocks carry a full stencil. Runs with ES off, using global
# --- particle shape 3 but particle shape 1 for the MS part of the solve, and
# --- with the grad-div penalty term in the Darwin field operator.
# ---
# --- Each step, the dB = B - B0 Fourier amplitudes are recorded at two groups of
# --- wave vectors that are physically equivalent under axis permutation:
# ---   (1,0,0), (0,1,0), (0,0,1)   theta_kB = 54.7 deg
# ---   (1,1,0), (0,1,1), (1,0,1)   theta_kB = 35.3 deg
# --- Particles are loaded symmetric under the cyclic permutation
# --- (x,y,z)->(y,z,x) (each macroparticle plus 2 permuted copies at 1/3 weight),
# --- so the members of each group should agree to roundoff.
# ---
# --- This is a reduced version of a longer verification run; see
# --- analysis_oblique_modes.py for what is checked and why.

import os
import sys

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, libwarpx, picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)

# integer wave vectors recorded (units of 2 pi / L); with b = (1,1,1)/sqrt(3)
# each row is a group of permutations sharing the same |k| and theta_kB, used
# for the permutation-symmetry check
K_LIST = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),  # theta_kB = 54.7 deg
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),  # theta_kB = 35.3 deg
]


class DummyES_Solver(picmi.ElectrostaticSolver):
    """No-op electrostatic solver: the Darwin solve is exercised with ES off,
    matching the physics regime the full-resolution verification deck found
    usable (the effective-potential ES solver overheats electrons at the
    dx/lambda_De of this problem; that is a separate, known issue)."""

    def __init__(self, grid):
        super(DummyES_Solver, self).__init__(
            grid=grid, method="Multigrid", required_precision=1
        )

    def solver_initialize_inputs(self):
        super(DummyES_Solver, self).solver_initialize_inputs()
        callbacks.installpoissonsolver(self.skip_poisson_solve)

    def skip_poisson_solve(self):
        pass


class ObliqueModes(object):
    B0 = 0.15  # T
    m_ion = 25.0  # ion/electron mass ratio
    beta = 0.1  # plasma beta (Te = Ti)
    vA_over_c = 0.02  # sets the density
    N = 16  # cells per side (dx = 0.5 l_i, matching the full-resolution deck)
    L_over_li = 8.0  # box side, in ion skin depths
    NPPC = 4  # seed macroparticles/cell/species before symmetrization (x3 after)
    SHAPE = 3  # particle shape for the electrostatic part of the solve
    MS_SHAPE = 1  # particle shape for the magnetostatic part of the solve
    GRADDIV_FACTOR = 1.0  # grad-div penalty factor of the Darwin field operator
    GMRES_TOL = 1e-5
    DT_WCI = 0.04  # dt * Omega_ci
    TOTAL_STEPS = 130

    def __init__(self, test, seed, graddiv_factor):
        self.test = test
        self.seed = seed
        self.graddiv_factor = graddiv_factor
        self.nprocs = comm.size

        b = np.array([1.0, 1.0, 1.0])
        self.bhat = b / np.linalg.norm(b)
        self.Bvec = self.B0 * self.bhat

        self.get_plasma_quantities()

        self.L = self.L_over_li * self.l_i
        self.dx = self.L / self.N
        self.dt = self.DT_WCI / self.w_ci
        self.total_steps = self.TOTAL_STEPS
        self.diag_steps = 1
        self.k_list = K_LIST

        if comm.rank == 0:
            print(
                f"ObliqueModes: M/m={self.m_ion:.0f} beta={self.beta} "
                f"vA/c={self.vA_over_c}\n"
                f"\tb = {self.bhat}\n"
                f"\tN={self.N}^3, L={self.L_over_li} l_i, dx={self.dx / self.l_i:.3f} l_i, "
                f"k0 l_i = {2 * np.pi / self.L_over_li:.3f}\n"
                f"\tdt W_ci = {self.dt * self.w_ci:.4f}, steps={self.total_steps}, "
                f"NPPC={self.NPPC}, ES/MS shape={self.SHAPE}/{self.MS_SHAPE}"
            )

        self.setup_run()

    def get_plasma_quantities(self):
        self.M = self.m_ion * constants.m_e
        self.w_ci = constants.q_e * self.B0 / self.M
        self.t_ci = 2.0 * np.pi / self.w_ci
        self.w_ce = constants.q_e * self.B0 / constants.m_e
        self.vA = self.vA_over_c * constants.c
        self.n_plasma = (self.B0 / self.vA) ** 2 / (
            constants.mu0 * (self.M + constants.m_e)
        )
        self.w_pi = np.sqrt(constants.q_e**2 * self.n_plasma / (self.M * constants.ep0))
        self.w_pe = np.sqrt(
            constants.q_e**2 * self.n_plasma / (constants.m_e * constants.ep0)
        )
        self.l_i = constants.c / self.w_pi
        self.v_ti = np.sqrt(self.beta / 2.0) * self.vA
        self.T_plasma = self.v_ti**2 * self.M / constants.q_e  # eV, Te = Ti
        self.v_te = np.sqrt(self.T_plasma * constants.q_e / constants.m_e)

    def setup_run(self):
        if self.seed:
            # picked up by Simulation.initialize_inputs() below; a fixed,
            # explicit seed keeps the checksum comparison reproducible
            simulation.random_seed = self.seed

        # the cost is dominated by the GMRES/MLMG field solve, so the box
        # decomposition barely matters
        mgs = self.N // 2 if self.nprocs > 1 else self.N
        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.N] * 3,
            warpx_max_grid_size=mgs,
            lower_bound=[0.0] * 3,
            upper_bound=[self.L] * 3,
            lower_boundary_conditions=["periodic"] * 3,
            upper_boundary_conditions=["periodic"] * 3,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        simulation.particle_shape = self.SHAPE
        simulation.use_filter = False
        simulation.verbose = self.test
        simulation.current_deposition_algo = "direct"
        simulation.evolve_scheme = picmi.SemiImplicitDarwinEvolveScheme(
            magnetostatic_particle_shape=self.MS_SHAPE,
            graddiv_factor=self.graddiv_factor,
            linear_solver=picmi.GMRESLinearSolver(
                relative_tolerance=self.GMRES_TOL,
                max_iterations=2048,
                restart_length=512,
                verbose_int=(2 if self.test else 0),
                pc_type=picmi.DarwinMLMGPreconditioner(),
            ),
        )
        self.solver = DummyES_Solver(self.grid)
        simulation.solver = self.solver

        simulation.add_applied_field(
            picmi.AnalyticInitialField(
                Bx_expression=f"{self.Bvec[0]}",
                By_expression=f"{self.Bvec[1]}",
                Bz_expression=f"{self.Bvec[2]}",
                warpx_do_initial_div_cleaning=False,
            )
        )

        for name, q, m, vth in (
            ("ions", constants.q_e, self.M, self.v_ti),
            ("electrons", -constants.q_e, constants.m_e, self.v_te),
        ):
            sp = picmi.Species(
                name=name,
                charge=q,
                mass=m,
                initial_distribution=picmi.UniformDistribution(
                    density=self.n_plasma, rms_velocity=[vth] * 3
                ),
            )
            simulation.add_species(
                sp,
                layout=picmi.PseudoRandomLayout(
                    grid=self.grid, n_macroparticles_per_cell=self.NPPC
                ),
            )

        for name, kind in (
            ("field_energy", "FieldEnergy"),
            ("part_energy", "ParticleEnergy"),
        ):
            simulation.add_diagnostic(
                picmi.ReducedDiagnostic(
                    diag_type=kind, name=name, period=self.diag_steps, path="diags/"
                )
            )

        # checksum regression diagnostic, at the final step only
        particle_diag = picmi.ParticleDiagnostic(
            name="field_diag", period=self.total_steps
        )
        simulation.add_diagnostic(particle_diag)
        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.total_steps,
            data_list=["B", "E"],
        )
        simulation.add_diagnostic(field_diag)

        # Fourier-mode recorder: FFT indices of the selected wave vectors
        # (numpy's exp(-2 pi i m.n/N) convention, m taken modulo N)
        m_arr = np.array(self.k_list)  # (nk, 3)
        self._fidx = tuple(m_arr[:, a] % self.N for a in range(3))
        self._records = []
        callbacks.installafterstep(self._record_modes)

        simulation.initialize_inputs()
        simulation.initialize_warpx()
        self._symmetrize_particles()

    def _symmetrize_particles(self):
        """Add 2 cyclic-permuted (x,y,z)->(y,z,x)->(z,x,y) copies of every
        macroparticle (position and velocity both permuted, weight rescaled
        by 1/3) so the full particle ensemble is exactly invariant under the
        axis permutation that relates our two K_LIST groups. Runs once,
        locally on each rank, right after the normal random loading; added
        particles that land outside the calling rank's box are moved to
        their owner by the normal per-step Redistribute()."""
        comps = ("x", "y", "z", "ux", "uy", "uz", "w")
        for name in ("ions", "electrons"):
            pc = simulation.particles.get(name)
            tiles = []
            for pti in pc.iterator(level=0):
                soa = pti.soa()
                tiles.append([soa.get_real_data(c).to_numpy(copy=True) for c in comps])
                # scale the ORIGINAL (live) particles' weight down by 1/3 in
                # place, so tripling the macroparticle count leaves the physical
                # density unchanged
                pti["w"][:] /= 3.0
            x, y, z, ux, uy, uz, w = (
                np.concatenate([t[i] for t in tiles]) for i in range(len(comps))
            )
            w = w / 3.0
            pc.add_particles(x=y, y=z, z=x, ux=uy, uy=uz, uz=ux, w=w)
            pc.add_particles(x=z, y=x, z=y, ux=uz, uy=ux, uz=uy, w=w)
        if comm.rank == 0 and self.test:
            print("symmetrize_particles: added 2 cyclic-permuted copies per species")

    def _record_modes(self):
        step = simulation.extension.warpx.getistep(lev=0)
        comps = []
        for d in "xyz":
            a = simulation.fields.get("Bfield_aux", dir=d, level=0)[...]
            comps.append(a)
        if libwarpx.amr.ParallelDescriptor.MyProc() != 0:
            return
        row = [step, step * self.dt]
        for c, a in enumerate(comps):
            # drop the duplicated periodic node along each nodal axis
            a = a[tuple(slice(0, self.N) for _ in range(3))] - self.Bvec[c]
            # A(m) = sum_ijk a_ijk exp(-2 pi i m.(i,j,k)/N), for each m in k_list
            amp = np.fft.fftn(a)[self._fidx]
            row.append(amp / self.N**3)
        self._records.append((row[0], row[1], np.stack(row[2:], axis=1)))
        if step == self.total_steps:
            self._flush()

    def _flush(self):
        os.makedirs("diags", exist_ok=True)
        steps = np.array([r[0] for r in self._records])
        times = np.array([r[1] for r in self._records])
        amps = np.array([r[2] for r in self._records])  # (nt, nk, 3)
        np.savez(
            "diags/k_modes.npz",
            step=steps,
            time=times,
            amp=amps,
            k_list=np.array(self.k_list),
        )


test = "--test" in sys.argv or "-t" in sys.argv
seed = 1
graddiv_factor = ObliqueModes.GRADDIV_FACTOR
for i, a in enumerate(sys.argv):
    if a == "--seed" and i + 1 < len(sys.argv):
        seed = int(sys.argv[i + 1])
    if a == "--graddiv_factor" and i + 1 < len(sys.argv):
        graddiv_factor = float(sys.argv[i + 1])
sys.argv = [sys.argv[0]]

run = ObliqueModes(test=test, seed=seed, graddiv_factor=graddiv_factor)
simulation.step()
