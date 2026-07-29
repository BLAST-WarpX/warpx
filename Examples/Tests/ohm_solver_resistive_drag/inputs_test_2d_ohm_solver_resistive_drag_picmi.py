#!/usr/bin/env python3
#
# --- Momentum-consistency test for the hybrid-PIC resistive-drag collision.
# ---
# --- A 2D Cartesian (x,z) periodic box holds a linear force-free field
# ---
# ---     B(x) = B0 [ 0, sin(k x), cos(k x) ],    k = 2 pi / Lx,
# ---
# --- so the plasma current J = curl(B)/mu0 = (k B0/mu0)[0, sin(kx), cos(kx)]
# --- is parallel to B (no J x B force) with uniform magnitude, and is carried
# --- entirely by the electron fluid (the ions start at rest).
# ---
# --- The hybrid_resistive_drag collision applies the species-resolved half of
# --- the electron-ion friction, -R_s, relaxing the ion bulk velocity toward
# --- V_e at nu = e^2 eta n_e / m_i.  With the drag registered, the resistive
# --- terms of Ohm's law are included in the particle-push E-field, whose
# --- Lorentz force applies the other half, +(rho_s/rho) Sum_t R_t.  For a
# --- global eta the two halves cancel exactly -- per species and pointwise --
# --- so the ions must STAY AT REST (deposited ion current ~ shot noise) while
# --- the magnetic field still decays resistively on tau_R = mu0/(eta k^2).
# ---
# --- This cancellation is the sharp observable: if the push field dropped the
# --- resistive terms (the bug this test guards against), the unbalanced drag
# --- would drive the deposited ion current toward the full force-free current
# --- as J_i -> -J (1 - exp(-nu t)), with nu*t_end ~ 1 for the CI parameters
# --- -- an order-unity signal against a few-percent noise floor.
# ---
# --- Analyse with analysis_resistive_drag.py: the projection of the deposited
# --- ion current onto the force-free pattern (primary) and the resistive
# --- B-energy decay rate (secondary).

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


class HybridResistiveDragCollision(object):
    """Minimal PICMI-style wrapper for the hybrid_resistive_drag collision
    (not yet in picmistandard): one species, no further parameters -- the
    drag rate comes from the Ohm's-law resistivity."""

    def __init__(self, name, species):
        self.name = name
        self.species = species

    def collision_initialize_inputs(self):
        import pywarpx

        coll = pywarpx.Collisions.newcollision(self.name)
        coll.type = "hybrid_resistive_drag"
        coll.species = [s.name for s in self.species]


class ResistiveDragMomentum(object):
    # ---- Plasma parameters --------------------------------------------------
    ti_eV = 500.0  # ion temperature (eV)
    te_eV = 500.0  # initial electron temperature (eV)
    gamma_e = 5.0 / 3.0  # electron adiabatic index
    n0 = 2.0e20  # uniform density (m^-3)

    # ---- Force-free field ---------------------------------------------------
    B0 = 0.1  # field magnitude (T); |B| is uniform
    n_wave = 1  # number of full wavelengths of B across Lx

    # ---- Geometry -----------------------------------------------------------
    Lx = 0.5  # domain length in x (m); the field varies in x
    NX = 128
    NZ = 16

    # ---- Numerics -----------------------------------------------------------
    NPPC = 800
    DT = 0.0025  # timestep as a fraction of the ion cyclotron period
    TOTAL_STEPS = 3000
    DIAG_EVERY = 150
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

        # Uniform plasma current magnitude from curl(B) = k B, carried by the
        # electron fluid: V_e = -J/(e n0), ions at rest.
        self.J0 = self.k * self.B0 / constants.mu0  # A/m^2
        self.v_drift = self.J0 / (constants.q_e * self.n0)

        # Ion cyclotron period at B0 sets the timestep scale.
        self.w_ci = constants.q_e * self.B0 / mi
        self.t_ci = 2.0 * np.pi / self.w_ci
        self.dt = self.DT * self.t_ci
        self.t_end = self.total_steps * self.dt

        self.vi_th = np.sqrt(constants.q_e * self.ti_eV / mi)

        # Constant resistivity (Ohm*m), scaled so nu_drag * t_end ~ 1.
        self.eta = 1.0e-5 * self.eta_scale
        # Resistive decay time of the force-free current: tau_R = mu0/(eta k^2).
        self.tau_R = constants.mu0 / (self.eta * self.k**2)
        # Hyper-resistivity off (stays out of the drag/push-field pairing).
        self.eta_h = 0.0

        # Drag rate implied by the Ohm's-law resistivity (Z = 1). Without the
        # resistive push-field term the deposited ion current would grow as
        # -J (1 - exp(-nu t)); with it, the drag and the E-force cancel.
        self.nu_drag = constants.q_e**2 * self.eta * self.n0 / mi  # 1/s

    def _print_params(self):
        print(
            f"\n[setup] Resistive-drag momentum-consistency test\n"
            f"  Te0 = {self.te_eV:.1f} eV,  Ti = {self.ti_eV:.1f} eV,  gamma_e = {self.gamma_e:.4f}\n"
            f"  n0        = {self.n0:.3e} m^-3\n"
            f"  B0        = {self.B0:.3e} T  (|B| uniform, force-free)\n"
            f"  k         = {self.k:.4e} 1/m  ({self.n_wave} wavelength(s) across Lx)\n"
            f"  |J|       = {self.J0:.3e} A/m^2  (uniform; carried by electrons)\n"
            f"  V_e drift = {self.v_drift:.3e} m/s\n"
            f"  eta       = {self.eta:.3e} Ohm*m  (1e-5 x scale {self.eta_scale:g})\n"
            f"  tau_R     = {self.tau_R:.3e} s  (current resistive-decay time)\n"
            f"  nu_drag   = {self.nu_drag:.3e} 1/s  (= e^2 eta n0 / m_i)\n"
            f"  Grid      = {self.NX} x {self.NZ}  (x x z),  dx = {self.dx:.3e} m\n"
            f"  t_ci      = {self.t_ci:.3e} s,  dt = {self.dt:.3e} s ({self.DT:.4f} t_ci)\n"
            f"  steps     = {self.total_steps},  diag every {self.diag_steps}\n"
            f"  ----\n"
            f"  nu_drag * t_end = {self.nu_drag * self.t_end:.3f}\n"
            f"  CHECK: deposited ion current stays at shot noise (<< J0); a\n"
            f"  missing push-field resistive term would give |J_i|/J0 ->\n"
            f"  {1.0 - np.exp(-self.nu_drag * self.t_end):.2f}. B-energy decays on tau_R.\n"
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

        # Electron energy equation ON with the eta*J^2 Joule source, so the
        # resistive dissipation the drag/push-field pairing tracks in momentum
        # is also returned to the electrons in energy (and the drag
        # consistency warning stays quiet).
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

        # The feature under test: ion side of the electron-ion friction.
        simulation.collisions = [
            HybridResistiveDragCollision(name="ion_drag", species=[self.ions])
        ]

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
    help="multiplier on the base resistivity eta=1e-5 (the CI test uses 100, "
    "putting nu_drag * t_end ~ 1 so a broken drag/push-field pairing gives "
    "an order-unity spurious ion current)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = ResistiveDragMomentum(
    test=args.test, verbose=args.verbose, eta_scale=args.eta_scale
)
simulation.step()
