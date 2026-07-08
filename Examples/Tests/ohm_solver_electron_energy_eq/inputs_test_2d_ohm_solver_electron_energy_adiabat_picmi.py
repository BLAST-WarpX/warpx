#!/usr/bin/env python3
#
# --- Adiabatic-compression test for the hybrid-PIC (Ohm's law) electron
# --- energy equation.  With all sources off the equation reduces to the
# --- transport terms,
# ---
# ---     dU_e/dt + div(U_e V_e) + P_e div(V_e) = 0,
# ---
# --- which the QDSMC scheme solves by advecting the electron entropy
# --- K_e = T_e n_e^(1-gamma) with Lagrangian markers moving at V_e and
# --- recovering T_e = K_e n_e^(gamma-1).
# ---
# --- Set-up: a uniform, unmagnetized (B=0), zero-resistivity (eta=0, so no
# --- Joule heating) plasma is given a sinusoidal ion velocity perturbation
# --- v_x = V0 sin(kx).  The electron-pressure gradient in Ohm's law
# --- (E = -grad Pe/(e n)) drives an ion-acoustic compression/rarefaction, so
# --- n(x,t) and T_e(x,t) oscillate.
# ---
# --- Clean analytic check (entropy conservation): the initial entropy is
# --- uniform (uniform n0, T_e0), and entropy-conserving advection keeps it
# --- uniform, so at every cell and time
# ---
# ---     T_e(x,t) = T_e0 * ( n(x,t) / n0 )^(gamma_e - 1)    [pointwise adiabat]
# ---
# --- regardless of the flow.  Deviations measure the error in the QDSMC
# --- compression term and the K_e <-> T_e polytropic reconstruction.
# --- Analyse with analysis_adiabat.py.

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


class AdiabaticCompression(object):
    # ---- Plasma parameters ------------------------------------------------
    te_eV = 100.0  # initial (uniform) electron temperature (eV)
    ti_eV = 10.0  # ion temperature (eV); cold vs Te for a clean,
    #   electron-pressure-driven acoustic wave
    gamma_e = 5.0 / 3.0  # electron adiabatic index -> adiabat exponent 2/3
    n0 = 2.0e20  # uniform density (m^-3)

    # ---- Perturbation -----------------------------------------------------
    pert_frac = 0.30  # ion velocity amplitude V0 = pert_frac * c_s
    n_wave = 1  # wavelengths across Lx

    # ---- Geometry ---------------------------------------------------------
    Lx = 0.5  # domain length in x (m); the wave runs along x
    NX = 128
    NZ = 16

    # ---- Numerics ---------------------------------------------------------
    NPPC = 800
    periods = 2.0  # acoustic periods to simulate
    steps_per_period = 400
    substeps = 10

    def __init__(self, test, verbose):
        self.test = test
        self.verbose = verbose or test

        if self.test:
            self.NX = 32
            self.NZ = 8
            self.NPPC = 64
            self._steps_override = 60
            self.ndiag = 10
        else:
            self._steps_override = None
            self.ndiag = 40

        self.get_plasma_quantities()
        if comm.rank == 0:
            self._print_params()
        self.setup_run()

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

        # Electron energy equation ON, Joule source OFF -> only the LHS
        # (advection + compression) evolves T_e.
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

        # Uniform density; sinusoidal x-velocity perturbation v_x = V0 sin(kx).
        # Uniform n0 and uniform Te0 -> uniform initial entropy.
        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            warpx_do_temperature_deposition=True,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression="n0",
                momentum_expressions=[f"({self.V0})*sin(({self.k})*x)", "0", "0"],
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
            data_list=["rho", "Te", "J", "B"],
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

run = AdiabaticCompression(test=args.test, verbose=args.verbose)
simulation.step()
