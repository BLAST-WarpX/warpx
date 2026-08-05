#!/usr/bin/env python3
#
# --- Test suite for the hybrid-PIC (Ohm's law) electron energy equation.
# --- One script, three cases (selected with --case), each isolating one
# --- term of the equation in a 2D Cartesian (x,z) periodic box:
# ---
# ---   adiabat : transport terms only (B=0, eta=0, all sources off),
# ---                 dU_e/dt + div(U_e V_e) + P_e div(V_e) = 0,
# ---             solved by the QDSMC scheme advecting the electron entropy
# ---             K_e = T_e n_e^(1-gamma) with Lagrangian markers moving at
# ---             V_e.  A sinusoidal ion velocity perturbation v_x = V0
# ---             sin(kx) drives an ion-acoustic compression, and entropy
# ---             conservation gives the pointwise check
# ---                 T_e(x,t) = T_e0 (n(x,t)/n0)^(gamma_e-1).
# ---             Analyse with analysis_adiabat.py.
# ---
# ---   joule   : eta*J^2 source only.  A linear force-free field
# ---                 B(x) = B0 [0, sin(kx), cos(kx)],  k = 2 pi/Lx,
# ---             carries the uniform, parallel current J = curl(B)/mu0
# ---             (J x B = 0): no bulk motion, uniform heating, transport
# ---             terms identically zero, so
# ---                 dT_e/dt = (gamma_e - 1) eta J^2 / (n_e k_B),
# ---             a linear ramp turning sub-linear on the resistive-decay
# ---             time tau_R = mu0/(eta k^2).  Analyse with
# ---             analysis_joule.py (B-energy decay + T_e ramp).
# ---
# ---   qei     : electron-ion thermal-equilibration sink only (B=0, eta=0),
# ---                 dU_e/dt = -Q_ei,  Q_ei = 3 n_e k_B nu_ei (T_e - T_i),
# ---             enabled by the rate parser
# ---             hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t).
# ---             The single parameter enables BOTH the electron-side sink
# ---             AND the conjugate ion heating, so the exchange conserves
# ---             energy.  For constant nu_ei (single proton species, Z=1),
# ---                 (T_e - T_i)(t) = (T_e0 - T_i0) exp(-rate t),
# ---                 rate = [3(gamma_e-1) + 2] nu_ei,
# ---             and C_e T_e + C_i T_i is conserved.  Analyse with
# ---             analysis_qei.py (difference rate + budget).

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


class ElectronEnergyCase(object):
    """Shared 2D periodic-box setup; the case subclasses supply the physics
    parameters, the solver sources and the diagnostic field list."""

    # ---- Common plasma parameters ------------------------------------------
    gamma_e = 5.0 / 3.0  # electron adiabatic index
    n0 = 2.0e20  # uniform density (m^-3)
    Lx = 0.5  # domain length in x (m)

    # ---- Case hooks (overridden where a case differs) -----------------------
    include_joule_heating = False
    relaxation_rate = None  # qei only: nu_ei expression (str)
    eta_h = None  # joule only: hyper-resistivity
    # qei leaves do_temperature_deposition unset on purpose -- it is enabled
    # automatically on charged species when the Q_ei relaxation is configured,
    # which that case also exercises.
    set_temperature_deposition = True
    load_B = False  # joule only: force-free initial field
    reduced_diags = False  # joule only: field/particle energy

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or test

        self.configure()
        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

    def momentum_expressions(self):
        return ["0", "0", "0"]

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

        # Electron energy equation ON; each case turns on exactly one source
        # (or none, for the pure-transport adiabat case).
        solver_kwargs = {}
        if self.eta_h is not None:
            solver_kwargs["plasma_hyper_resistivity"] = self.eta_h
        if self.relaxation_rate is not None:
            solver_kwargs["electron_ion_relaxation_rate"] = self.relaxation_rate
        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.gamma_e,
            Te=self.te_eV,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
            solve_electron_energy_equation=True,
            include_joule_heating=self.include_joule_heating,
            **solver_kwargs,
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

        if self.load_B:
            B_init = picmi.LoadInitialFieldFromPython(
                load_from_python=self.load_initial_B,
                load_B=True,
                load_E=False,
            )
            simulation.add_applied_field(B_init)

        species_kwargs = {}
        if self.set_temperature_deposition:
            species_kwargs["warpx_do_temperature_deposition"] = True
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0",
                momentum_expressions=self.momentum_expressions(),
                warpx_momentum_spread_expressions=[str(self.vi_th)] * 3,
                n0=self.n0,
            ),
            **species_kwargs,
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
            data_list=self.diag_data_list,
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

        if self.reduced_diags:
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


class AdiabaticCompression(ElectronEnergyCase):
    """Transport-terms (LHS) test: entropy-conserving compression."""

    te_eV = 100.0  # initial (uniform) electron temperature (eV)
    ti_eV = 10.0  # ion temperature (eV); cold vs Te for a clean,
    #   electron-pressure-driven acoustic wave

    # ---- Perturbation -------------------------------------------------------
    pert_frac = 0.30  # ion velocity amplitude V0 = pert_frac * c_s
    n_wave = 1  # wavelengths across Lx

    # ---- Geometry / numerics ------------------------------------------------
    NX = 128
    NZ = 16
    NPPC = 800
    periods = 2.0  # acoustic periods to simulate
    steps_per_period = 400
    substeps = 10

    diag_data_list = ["rho", "Te", "J", "B"]

    def configure(self):
        if self.test:
            self.NX = 32
            self.NZ = 8
            self.NPPC = 64
            self._steps_override = 60
            self.ndiag = 10
        else:
            self._steps_override = None
            self.ndiag = 40

    def get_plasma_quantities(self):
        mi = constants.m_p
        self.dx = self.Lx / self.NX
        self.Lz = self.dx * self.NZ
        self.k = 2.0 * np.pi * self.n_wave / self.Lx

        # Electron-pressure sound speed (cold-ion limit) sets the wave period.
        self.c_s = np.sqrt(self.gamma_e * constants.q_e * self.te_eV / mi)
        self.omega = self.k * self.c_s  # acoustic angular frequency
        self.T_period = 2.0 * np.pi / self.omega  # = Lx / c_s for n_wave=1
        self.V0 = self.pert_frac * self.c_s  # velocity perturbation amplitude

        self.dt = self.T_period / self.steps_per_period
        if self._steps_override is not None:
            self.total_steps = self._steps_override
        else:
            self.total_steps = int(self.periods * self.steps_per_period)
        self.diag_steps = max(1, self.total_steps // self.ndiag)

        self.vi_th = np.sqrt(constants.q_e * self.ti_eV / mi)
        # No applied B (B=0). No resistivity (eta=0 -> no Joule, pure LHS).
        self.eta = 0.0

    def momentum_expressions(self):
        # Sinusoidal x-velocity perturbation v_x = V0 sin(kx); uniform n0 and
        # uniform Te0 -> uniform initial entropy.
        return [f"({self.V0})*sin(({self.k})*x)", "0", "0"]

    def _print_params(self):
        print(
            f"\n[setup] Adiabatic-compression (electron-energy-equation LHS) test\n"
            f"  Te0 = {self.te_eV:.1f} eV,  Ti = {self.ti_eV:.1f} eV,  gamma_e = {self.gamma_e:.4f}\n"
            f"  n0        = {self.n0:.3e} m^-3\n"
            f"  c_s       = {self.c_s:.3e} m/s   (electron-pressure sound speed)\n"
            f"  V0        = {self.V0:.3e} m/s   (= {self.pert_frac:.2f} c_s)\n"
            f"  k         = {self.k:.4e} 1/m   ({self.n_wave} wavelength(s))\n"
            f"  T_period  = {self.T_period:.3e} s   (acoustic)\n"
            f"  Grid      = {self.NX} x {self.NZ},  Lx x Lz = {self.Lx:.3f} x {self.Lz:.4f} m\n"
            f"  dt        = {self.dt:.3e} s   ({self.steps_per_period}/period)\n"
            f"  steps     = {self.total_steps},  diag every {self.diag_steps}\n"
            f"  B = 0,  eta = 0  ->  Joule OFF, pure advection+compression\n"
            f"  CHECK:  T_e(x,t) = Te0 (n/n0)^(gamma_e-1)  pointwise\n"
        )


class ForceFreeJoule(ElectronEnergyCase):
    """eta*J^2 source test: force-free field, uniform Joule ramp."""

    ti_eV = 500.0  # ion temperature (eV)
    te_eV = 500.0  # initial electron temperature (eV)

    # ---- Force-free field ---------------------------------------------------
    B0 = 0.1  # field magnitude (T); |B| is uniform
    n_wave = 1  # number of full wavelengths of B across Lx

    # ---- Geometry / numerics ------------------------------------------------
    NX = 128  # cells in x (full run)
    NZ = 16  # cells in z (field is z-independent; periodic)
    NPPC = 800  # particles per cell; the T_e ramp sits on an ion shot-noise
    #   heating floor that converges as 1/NPPC
    DT = 0.0025  # timestep as a fraction of the ion cyclotron period; small
    #   enough for the forward-Euler Joule deposit to be converged
    TOTAL_STEPS = 3000  # full run
    DIAG_EVERY = 150  # diagnostic cadence (steps)
    substeps = 20

    include_joule_heating = True
    load_B = True
    reduced_diags = True
    diag_data_list = ["B", "E", "rho", "J", "Te", "T_ions"]

    def configure(self):
        self.eta_scale = self.args.eta_scale
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


class QeiRelaxation(ElectronEnergyCase):
    """Q_ei electron-ion thermal-equilibration test: pure exponential."""

    te_eV = 300.0  # initial (uniform) electron temperature (eV), hot
    ti_eV = 50.0  # ion temperature (eV); the relaxation target

    # ---- Relaxation ---------------------------------------------------------
    nu_ei = 1.0e6  # electron-ion relaxation rate (1/s), constant;
    #   Te sink rate = 3(gamma_e-1)*nu_ei = 2e6 1/s -> tau = 0.5 us

    # ---- Geometry (small; the physics is 0-D / uniform) / numerics ----------
    NX = 32
    NZ = 8
    NPPC = 400
    n_tau = 3.0  # number of relaxation times to simulate
    steps_per_tau = 100  # rate*dt = 0.01 -> forward-Euler ~ exponential
    substeps = 10

    # do_temperature_deposition is NOT set on purpose -- it is enabled
    # automatically on charged species when the Q_ei relaxation is
    # configured, which this test also exercises (T_ions is dumped below).
    set_temperature_deposition = False
    diag_data_list = ["rho", "Te", "T_ions"]

    def configure(self):
        if self.test:
            self.NX = 16
            self.NZ = 8
            self.NPPC = 200
            self._steps_override = 80
            self.ndiag = 10
        else:
            self._steps_override = None
            self.ndiag = 20

    def get_plasma_quantities(self):
        mi = constants.m_p
        self.dx = self.Lx / self.NX
        self.Lz = self.dx * self.NZ

        # Analytic electron-sink rate and e-folding time.
        self.rate = 3.0 * (self.gamma_e - 1.0) * self.nu_ei
        self.tau = 1.0 / self.rate

        self.dt = self.tau / self.steps_per_tau
        if self._steps_override is not None:
            self.total_steps = self._steps_override
        else:
            self.total_steps = int(self.n_tau * self.steps_per_tau)
        self.diag_steps = max(1, self.total_steps // self.ndiag)

        self.vi_th = np.sqrt(constants.q_e * self.ti_eV / mi)
        # No applied B (B=0). No resistivity (eta=0 -> no Joule, pure Q_ei).
        self.eta = 0.0
        # Constant relaxation rate so the relaxation is a pure exponential.
        self.relaxation_rate = f"{self.nu_ei}"

    def _print_params(self):
        print(
            f"\n[setup] Electron-ion relaxation (Q_ei) test\n"
            f"  Te0 = {self.te_eV:.1f} eV,  Ti0 = {self.ti_eV:.1f} eV,  gamma_e = {self.gamma_e:.4f}\n"
            f"  n0        = {self.n0:.3e} m^-3\n"
            f"  nu_ei     = {self.nu_ei:.3e} 1/s   (constant)\n"
            f"  rate      = 3(gamma-1)nu_ei = {self.rate:.3e} 1/s\n"
            f"  tau       = 1/rate = {self.tau:.3e} s\n"
            f"  Grid      = {self.NX} x {self.NZ},  Lx x Lz = {self.Lx:.3f} x {self.Lz:.4f} m\n"
            f"  dt        = {self.dt:.3e} s   (rate*dt = {self.rate * self.dt:.3f})\n"
            f"  steps     = {self.total_steps},  diag every {self.diag_steps}\n"
            f"  B = 0,  eta = 0  ->  Joule OFF, Q_ei ON (e-sink + conjugate ion heating)\n"
            f"  CHECK:  (Te-Ti)(t) = (Te0-Ti0) exp(-[3(g-1)+2]nu t),  energy conserved\n"
        )


CASES = {
    "adiabat": AdiabaticCompression,
    "joule": ForceFreeJoule,
    "qei": QeiRelaxation,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    required=True,
    choices=sorted(CASES.keys()),
    help="which electron-energy-equation term to test",
)
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
    help="joule case only: multiplier on the base resistivity eta=1e-5 "
    "(amplifies the eta*J^2 heating signal; the CI test uses 100)",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

case_class = CASES[args.case]
case_class.args = args
run = case_class(test=args.test, verbose=args.verbose)
simulation.step()
