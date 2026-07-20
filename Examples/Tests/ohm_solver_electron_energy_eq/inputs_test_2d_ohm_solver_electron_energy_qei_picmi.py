#!/usr/bin/env python3
#
# --- Electron-ion temperature-relaxation (Q_ei) test for the hybrid-PIC
# --- (Ohm's law) electron energy equation.  Where the force-free test
# --- exercises the eta*J^2 source and the adiabatic-compression test the
# --- transport terms, this test isolates the electron-ion thermal-
# --- equilibration sink,
# ---
# ---     dU_e/dt = -Q_ei,   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),
# ---
# --- enabled by specifying the rate parser
# --- hybrid_pic_model.electron_ion_relaxation_rate(rho,Te,Ti,t).  The single
# --- parameter enables BOTH the electron-side sink AND the conjugate ion
# --- heating, so the exchange is energy-conserving (ion gain == electron loss).
# ---
# --- Set-up: a uniform, unmagnetized (B=0), zero-resistivity (eta=0, so no
# --- Joule heating) plasma with the ions at rest (only a thermal spread at
# --- T_i0) and a hot electron fluid (T_e0 >> T_i0).  With no B, no flow and
# --- uniform fields, grad(P_e)=0 -> E=0, nothing moves macroscopically, and
# --- T_e and T_i relax toward a common temperature purely through Q_ei.
# ---
# --- Clean analytic check (single proton species, Z=1 -> n_e=n_i):
# ---     dT_e/dt = -3(gamma_e-1) nu_ei (T_e - T_i)      [electron side]
# ---     dT_i/dt = +2 nu_ei (T_e - T_i)                 [ion side, gamma-indep.]
# --- so for a constant nu_ei the difference decays exponentially,
# ---
# ---     (T_e - T_i)(t) = (T_e0 - T_i0) exp(-rate t),
# ---     rate = [3(gamma_e-1) + 2] nu_ei  (= 4 nu_ei for gamma_e = 5/3),
# ---
# --- the total thermal energy C_e T_e + C_i T_i is conserved (C_e = n_e k_B/
# --- (gamma_e-1), C_i = 3/2 n_i k_B), and for gamma_e=5/3 they meet at
# --- (T_e0+T_i0)/2.  Analyse with analysis_qei.py (difference rate + budget).

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


class QeiRelaxation(object):
    # ---- Plasma parameters ------------------------------------------------
    te_eV = 300.0  # initial (uniform) electron temperature (eV), hot
    ti_eV = 50.0  # ion temperature (eV); the relaxation target
    gamma_e = 5.0 / 3.0  # electron adiabatic index
    n0 = 2.0e20  # uniform density (m^-3)

    # ---- Relaxation -------------------------------------------------------
    nu_ei = 1.0e6  # electron-ion relaxation rate (1/s), constant;
    #   Te sink rate = 3(gamma_e-1)*nu_ei = 2e6 1/s -> tau = 0.5 us

    # ---- Geometry (small; the physics is 0-D / uniform) -------------------
    Lx = 0.5  # domain length in x (m)
    NX = 32
    NZ = 8

    # ---- Numerics ---------------------------------------------------------
    NPPC = 400
    n_tau = 3.0  # number of relaxation times to simulate
    steps_per_tau = 100  # rate*dt = 0.01 -> forward-Euler ~ exponential
    substeps = 10

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or test

        if self.test:
            self.NX = 16
            self.NZ = 8
            self.NPPC = 200
            self._steps_override = 80
            self.ndiag = 10
        else:
            self._steps_override = None
            self.ndiag = 20

        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

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

        # Electron energy equation ON; Joule source OFF; Q_ei exchange ON with
        # a constant relaxation rate so the relaxation is a pure exponential.
        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.gamma_e,
            Te=self.te_eV,
            n0=self.n0,
            n_floor=0.05 * self.n0,
            plasma_resistivity=self.eta,
            substeps=self.substeps,
            solve_electron_energy_equation=True,
            include_joule_heating=False,
            electron_ion_relaxation_rate=f"{self.nu_ei}",
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

        # Uniform density, ions at rest (thermal spread at Ti0 only). Uniform
        # n0 and Te0 -> uniform fields -> no flow; T_e relaxes purely via Q_ei.
        # Note: do_temperature_deposition is NOT set here on purpose -- it is
        # enabled automatically on charged species when the Q_ei relaxation is
        # configured, which this test also exercises (T_ions is dumped below).
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
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
            data_list=["rho", "Te", "T_ions"],
            write_dir="diags",
            warpx_file_prefix="field_diags",
            warpx_format="openpmd",
            warpx_openpmd_backend="h5",
        )
        simulation.add_diagnostic(field_diag)

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
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = QeiRelaxation(test=args.test, verbose=args.verbose)
simulation.step()
