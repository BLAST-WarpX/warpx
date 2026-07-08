#!/usr/bin/env python3
#
# --- Force-free Joule-heating test for the hybrid-PIC (Ohm's law) electron
# --- energy equation.  A 2D Cartesian (x,z) periodic box holds a linear
# --- (constant-alpha) force-free magnetic field that varies only in x:
# ---
# ---     B(x) = B0 [ 0, sin(k x), cos(k x) ],    k = 2 pi / Lx
# ---
# --- This field satisfies curl(B) = k B, hence the plasma current
# ---     J = curl(B)/mu0 = (k B0/mu0) [0, sin(kx), cos(kx)]
# --- is parallel to B (J x B = 0, force-free) and has uniform magnitude
# ---     |J| = k B0 / mu0.
# ---
# --- Why this is the ideal Joule-heating unit test:
# ---   * No J x B force and no pressure gradient -> no bulk motion and no
# ---     free energy for instability.  The plasma just sits there.
# ---   * |J| is spatially uniform -> ohmic heating eta*J^2 is uniform -> T_e
# ---     stays spatially uniform -> the transport terms div(U_e V_e) and
# ---     P_e div(V_e) are identically zero.
# ---   * The electrons carry the current (V_e = -curl(B)/(mu0 e n_e)); ions
# ---     start uniform and stationary.  The relative drift is
# ---     |V_i - V_e| = |J|/(e n_e), so the source reduces to eta J^2.
# ---
# --- The electron temperature therefore obeys
# ---     dT_e/dt = (gamma_e - 1) * eta * J^2 / (n_e k_B),
# --- a linear ramp that turns sub-linear only as the current resistively
# --- decays on tau_R = mu0/(eta k^2).  Both signatures are checked by
# --- analysis_joule.py: the resistive decay of the magnetic field energy
# --- (primary, immune to PIC noise) and the T_e ramp (secondary).

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import picmi

constants = picmi.constants
comm = mpi.COMM_WORLD

simulation = None


class ForceFreeJoule(object):
    # ---- Plasma parameters ------------------------------------------------
    ti_eV = 500.0  # ion temperature (eV)
    te_eV = 500.0  # initial electron temperature (eV)
    gamma_e = 5.0 / 3.0  # electron adiabatic index -> (gamma_e - 1) = 2/3
    n0 = 2.0e20  # uniform density (m^-3)

    # ---- Force-free field -------------------------------------------------
    B0 = 0.1  # field magnitude (T); |B| is uniform
    n_wave = 1  # number of full wavelengths of B across Lx

    # ---- Geometry ---------------------------------------------------------
    Lx = 0.5  # domain length in x (m); the field varies in x
    NX = 128  # cells in x (full run)
    NZ = 16  # cells in z (field is z-independent; periodic)

    # ---- Numerics ---------------------------------------------------------
    NPPC = 800  # particles per cell; the T_e ramp sits on an ion shot-noise
    #   heating floor that converges as 1/NPPC
    DT = 0.0025  # timestep as a fraction of the ion cyclotron period; small
    #   enough for the forward-Euler Joule deposit to be converged
    TOTAL_STEPS = 3000  # full run
    DIAG_EVERY = 150  # diagnostic cadence (steps)
    substeps = 20

    def __init__(self, test, verbose, eta_scale):
        self.test = test
        self.verbose = verbose or test
        self.eta_scale = eta_scale

        if self.test:
            self.NX = 32
            self.NZ = 8
            self.NPPC = 64
            self.DT = 0.01
            self.total_steps = 50
            self.diag_steps = 10
        else:
            self.total_steps = self.TOTAL_STEPS
            self.diag_steps = self.DIAG_EVERY

        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

    def get_plasma_quantities(self):
        mi = constants.m_p

        self.dx = self.Lx / self.NX
        self.Lz = self.dx * self.NZ  # square cells
        self.k = 2.0 * np.pi * self.n_wave / self.Lx

        # Uniform plasma current magnitude from curl(B) = k B.
        self.J0 = self.k * self.B0 / constants.mu0  # A/m^2
        # Electron drift carrying it (ions start at rest): V_e = J/(e n0).
        self.v_drift = self.J0 / (constants.q_e * self.n0)

        # Ion cyclotron period at B0 sets the timestep scale.
        self.w_ci = constants.q_e * self.B0 / mi
        self.t_ci = 2.0 * np.pi / self.w_ci
        self.dt = self.DT * self.t_ci

        self.vi_th = np.sqrt(constants.q_e * self.ti_eV / mi)

        # Constant resistivity (Ohm*m), scaled for the heating-signal amplitude.
        self.eta = 1.0e-5 * self.eta_scale
        # Resistive decay time of the force-free current: tau_R = mu0/(eta k^2).
        self.tau_R = constants.mu0 / (self.eta * self.k**2)

        # Hyper-resistivity off: grid-scale damping is not needed for a smooth,
        # single-wavelength field and would complicate the eta*J^2 budget.
        self.eta_h = 0.0

        # Analytic prediction (for the printout / cross-check).
        self.dTe_dt_pred = (
            (self.gamma_e - 1.0) * self.eta * self.J0**2 / (self.n0 * constants.kb)
        )  # K/s

    def _print_params(self):
        print(
            f"\n[setup] Force-free Joule-heating test\n"
            f"  Te0 = {self.te_eV:.1f} eV,  Ti = {self.ti_eV:.1f} eV,  gamma_e = {self.gamma_e:.4f}\n"
            f"  n0        = {self.n0:.3e} m^-3\n"
            f"  B0        = {self.B0:.3e} T  (|B| uniform)\n"
            f"  k         = {self.k:.4e} 1/m  ({self.n_wave} wavelength(s) across Lx)\n"
            f"  |J|       = {self.J0:.3e} A/m^2  (uniform, force-free)\n"
            f"  V_e drift = {self.v_drift:.3e} m/s\n"
            f"  eta       = {self.eta:.3e} Ohm*m  (1e-5 x scale {self.eta_scale:g})\n"
            f"  tau_R     = {self.tau_R:.3e} s  (current resistive-decay time)\n"
            f"  Grid      = {self.NX} x {self.NZ}  (x x z),  dx = {self.dx:.3e} m\n"
            f"  t_ci      = {self.t_ci:.3e} s,  dt = {self.dt:.3e} s ({self.DT:.4f} t_ci)\n"
            f"  steps     = {self.total_steps},  diag every {self.diag_steps}\n"
            f"  ----\n"
            f"  PREDICTED dTe/dt = (gamma_e-1) eta J^2 / (n0 kB) = {self.dTe_dt_pred:.4e} K/s\n"
            f"                   = {self.dTe_dt_pred * constants.kb / constants.q_e:.4e} eV/s\n"
        )

    def load_initial_B(self):
        """Set the linear force-free field B(x) = B0[0, sin(kx), cos(kx)].

        WarpX folds Bfield_fp_external into Bfield_fp at initialization, after
        which it evolves self-consistently. div(B) = 0 analytically (B has no
        x-component and the others depend only on x), so no cleaning needed.
        """
        Bx = simulation.fields.get("Bfield_fp_external", dir="x", level=0)
        By = simulation.fields.get("Bfield_fp_external", dir="y", level=0)
        Bz = simulation.fields.get("Bfield_fp_external", dir="z", level=0)

        Bx[:, :] = 0.0
        # Each component on its own (possibly staggered) mesh.
        XBy, _ = np.meshgrid(By.mesh("x"), By.mesh("z"), indexing="ij")
        XBz, _ = np.meshgrid(Bz.mesh("x"), Bz.mesh("z"), indexing="ij")
        By[:, :] = self.B0 * np.sin(self.k * XBy)
        Bz[:, :] = self.B0 * np.cos(self.k * XBz)
        comm.Barrier()

    def setup_run(self):
        global simulation

        self.grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.NX, self.NZ],
            lower_bound=[0.0, -self.Lz / 2.0],
            upper_bound=[self.Lx, self.Lz / 2.0],
            lower_boundary_conditions=["periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic"],
            lower_boundary_conditions_particles=["periodic", "periodic"],
            upper_boundary_conditions_particles=["periodic", "periodic"],
            warpx_max_grid_size=self.NZ,
        )

        # Electron energy equation ON with only the eta*J^2 Joule source.
        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.gamma_e,
            Te=self.te_eV,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=self.eta,
            plasma_hyper_resistivity=self.eta_h,
            substeps=self.substeps,
            solve_electron_energy_equation=True,
            include_joule_heating=True,
        )

        simulation = picmi.Simulation(
            solver=self.solver,
            time_step_size=self.dt,
            max_steps=self.total_steps,
            verbose=self.verbose,
            particle_shape=1,
            warpx_serialize_initial_conditions=True,
            warpx_current_deposition_algo="direct",
            warpx_use_filter=True,
        )

        B_init = picmi.LoadInitialFieldFromPython(
            load_from_python=self.load_initial_B,
            load_B=True,
            load_E=False,
        )
        simulation.add_applied_field(B_init)

        # Stationary ions -> the electron fluid carries the current.
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            warpx_do_temperature_deposition=True,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0",
                momentum_expressions=["0", "0", "0"],
                warpx_momentum_spread_expressions=[str(self.vi_th)] * 3,
                n0=self.n0,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        # Remove any diags from a previous run in the same directory, so
        # stale openPMD dumps (one file per iteration) cannot mix into the
        # analysis of this run.
        if comm.rank == 0 and Path("diags").exists():
            shutil.rmtree("diags")
        comm.Barrier()

        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B", "E", "rho", "J", "Te", "T_ions"],
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

        simulation.add_diagnostic(
            picmi.ReducedDiagnostic(
                diag_type="FieldEnergy",
                name="field_energy",
                period=self.diag_steps,
                path="diags/",
            )
        )
        simulation.add_diagnostic(
            picmi.ReducedDiagnostic(
                diag_type="ParticleEnergy",
                name="part_energy",
                period=self.diag_steps,
                path="diags/",
            )
        )

        simulation.initialize_inputs()
        simulation.initialize_warpx()


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
parser.add_argument(
    "--eta-scale",
    type=float,
    default=1.0,
    help="multiplier on the base resistivity eta=1e-5 (amplifies the eta*J^2 "
    "heating signal; the CI test uses 100)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = ForceFreeJoule(test=args.test, verbose=args.verbose, eta_scale=args.eta_scale)
simulation.step()
