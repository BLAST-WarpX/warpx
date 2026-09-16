#!/usr/bin/env python3
#
# --- Input file for MCC testing. There is already a test of the MCC
# --- functionality. This tests the PICMI interface for the MCC and
# --- provides an example of how an external Poisson solver can be
# --- used for the field solve step.

from typing import Literal

import numpy as np
from pydantic import PrivateAttr
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sla

from pywarpx import callbacks, picmi

constants = picmi.constants

##########################
# physics parameters
##########################

D_CA = 0.067  # m

N_INERT = 9.64e20  # m^-3
T_INERT = 300.0  # K

FREQ = 13.56e6  # Hz

VOLTAGE = 450.0

M_ION = 6.67e-27  # kg

PLASMA_DENSITY = 2.56e14  # m^-3
T_ELEC = 30000.0  # K

DT = 1.0 / (400 * FREQ)

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 50
diagnostic_intervals = "::50"

# --- Grid
nx = 128
ny = 8

xmin = 0.0
ymin = 0.0
xmax = D_CA
ymax = D_CA / nx * ny

number_per_cell_each_dim = [32, 16]

#############################
# specialized Poisson solver
# using superLU decomposition
#############################


class PoissonSolverPseudo1D(picmi.ElectrostaticSolver):
    """Direct solver for the Poisson equation using superLU. This solver is
    useful for pseudo 1D cases i.e. diode simulations with small x extent.

    The grid (picmi.Cartesian2DGrid) is the instance of the grid on which the
    solver will be installed.
    """

    # Runtime state of the solver
    _time_sum: float = PrivateAttr(default=0.0)
    _right_voltage: float | str | None = PrivateAttr(default=None)
    _nx: int | None = PrivateAttr(default=None)
    _nz: int | None = PrivateAttr(default=None)
    _dx: float | None = PrivateAttr(default=None)
    _dz: float | None = PrivateAttr(default=None)
    _nxguardphi: int | None = PrivateAttr(default=None)
    _nzguardphi: int | None = PrivateAttr(default=None)
    _phi: np.ndarray | None = PrivateAttr(default=None)
    _nxsolve: int | None = PrivateAttr(default=None)
    _nzsolve: int | None = PrivateAttr(default=None)
    _lu: sla.SuperLU | None = PrivateAttr(default=None)
    _rho_data: np.ndarray | None = PrivateAttr(default=None)

    # Different defaults than the WarpX solver
    method: Literal["FFT", "Multigrid"] | None = "Multigrid"
    required_precision: float | None = 1.0

    def solver_initialize_inputs(self):
        """Grab geometrical quantities from the grid."""
        self._right_voltage = self.grid.potential_xmax

        # set WarpX boundary potentials to None since we will handle it
        # ourselves in this solver
        self.grid.potential_xmin = None
        self.grid.potential_xmax = None
        self.grid.potential_ymin = None
        self.grid.potential_ymax = None
        self.grid.potential_zmin = None
        self.grid.potential_zmax = None

        super(PoissonSolverPseudo1D, self).solver_initialize_inputs()

        self._nx = self.grid.number_of_cells[0]
        self._nz = self.grid.number_of_cells[1]
        self._dx = (self.grid.upper_bound[0] - self.grid.lower_bound[0]) / self._nx
        self._dz = (self.grid.upper_bound[1] - self.grid.lower_bound[1]) / self._nz

        if not np.isclose(self._dx, self._dz):
            raise RuntimeError("Direct solver requires dx = dz.")

        self._nxguardphi = 1
        self._nzguardphi = 1

        self._phi = np.zeros(
            (self._nx + 1 + 2 * self._nxguardphi, self._nz + 1 + 2 * self._nzguardphi)
        )

        self.decompose_matrix()

        callbacks.installpoissonsolver(self._run_solve)

    def decompose_matrix(self):
        """Function to build the superLU object used to solve the linear
        system."""
        self._nxsolve = self._nx + 1
        self._nzsolve = self._nz + 3

        # Set up the computation matrix in order to solve A*phi = rho
        A = np.zeros((self._nzsolve * self._nxsolve, self._nzsolve * self._nxsolve))
        kk = 0
        for ii in range(self._nxsolve):
            for jj in range(self._nzsolve):
                temp = np.zeros((self._nxsolve, self._nzsolve))

                if ii == 0 or ii == self._nxsolve - 1:
                    temp[ii, jj] = 1.0
                elif ii == 1:
                    temp[ii, jj] = -2.0
                    temp[ii - 1, jj] = 1.0
                    temp[ii + 1, jj] = 1.0
                elif ii == self._nxsolve - 2:
                    temp[ii, jj] = -2.0
                    temp[ii + 1, jj] = 1.0
                    temp[ii - 1, jj] = 1.0
                elif jj == 0:
                    temp[ii, jj] = 1.0
                    temp[ii, -3] = -1.0
                elif jj == self._nzsolve - 1:
                    temp[ii, jj] = 1.0
                    temp[ii, 2] = -1.0
                else:
                    temp[ii, jj] = -4.0
                    temp[ii, jj + 1] = 1.0
                    temp[ii, jj - 1] = 1.0
                    temp[ii - 1, jj] = 1.0
                    temp[ii + 1, jj] = 1.0

                A[kk] = temp.flatten()
                kk += 1

        A = csc_matrix(A, dtype=np.float32)
        self._lu = sla.splu(A)

    def _run_solve(self):
        """Function run on every step to perform the required steps to solve
        Poisson's equation."""

        # get rho from WarpX
        self._rho_data = sim.fields.get("rho_fp", level=0)[...]

        self.solve()

        phi_wrapper = sim.fields.get("phi_fp", level=0)
        phi_wrapper[(), ()] = self._phi[...]

    def solve(self):
        """The solution step. Includes getting the boundary potentials and
        calculating phi from rho."""
        right_voltage = eval(
            self._right_voltage,
            {"t": sim.extension.warpx.gett_new(0), "sin": np.sin, "pi": np.pi},
        )
        left_voltage = 0.0

        rho = -self._rho_data / constants.ep0

        # Construct b vector
        nx, nz = np.shape(rho)
        source = np.zeros((nx, nz + 2), dtype=np.float32)
        source[:, 1:-1] = rho * self._dx**2

        source[0] = left_voltage
        source[-1] = right_voltage

        # Construct b vector
        b = source.flatten()

        flat_phi = self._lu.solve(b)
        self._phi[self._nxguardphi : -self._nxguardphi] = flat_phi.reshape(
            np.shape(source)
        )

        self._phi[: self._nxguardphi] = left_voltage
        self._phi[-self._nxguardphi :] = right_voltage

        # the electrostatic solver in WarpX keeps the ghost cell values as 0
        self._phi[:, : self._nzguardphi] = 0
        self._phi[:, -self._nzguardphi :] = 0


##########################
# physics components
##########################

v_rms_elec = np.sqrt(constants.kb * T_ELEC / constants.m_e)
v_rms_ion = np.sqrt(constants.kb * T_INERT / M_ION)

uniform_plasma_elec = picmi.UniformDistribution(
    density=PLASMA_DENSITY,
    upper_bound=[None] * 3,
    rms_velocity=[v_rms_elec] * 3,
    directed_velocity=[0.0] * 3,
)

uniform_plasma_ion = picmi.UniformDistribution(
    density=PLASMA_DENSITY,
    upper_bound=[None] * 3,
    rms_velocity=[v_rms_ion] * 3,
    directed_velocity=[0.0] * 3,
)

electrons = picmi.Species(
    particle_type="electron", name="electrons", initial_distribution=uniform_plasma_elec
)
ions = picmi.Species(
    particle_type="He",
    name="he_ions",
    charge="q_e",
    initial_distribution=uniform_plasma_ion,
)

# MCC collisions
cross_sec_direc = "../../../../warpx-data/MCC_cross_sections/He/"
mcc_electrons = picmi.MCCCollisions(
    name="coll_elec",
    species=electrons,
    background_density=N_INERT,
    background_temperature=T_INERT,
    background_mass=ions.mass,
    scattering_processes={
        "elastic": {"cross_section": cross_sec_direc + "electron_scattering.dat"},
        "excitation1": {
            "cross_section": cross_sec_direc + "excitation_1.dat",
            "energy": 19.82,
        },
        "excitation2": {
            "cross_section": cross_sec_direc + "excitation_2.dat",
            "energy": 20.61,
        },
        "ionization": {
            "cross_section": cross_sec_direc + "ionization.dat",
            "energy": 24.55,
            "species": ions,
        },
    },
)

mcc_ions = picmi.MCCCollisions(
    name="coll_ion",
    species=ions,
    background_density=N_INERT,
    background_temperature=T_INERT,
    scattering_processes={
        "elastic": {"cross_section": cross_sec_direc + "ion_scattering.dat"},
        "elastic_back": {
            "cross_section": cross_sec_direc + "ion_back_scatter.dat",
            "scattering_angle_model": "backward",
        },
        # 'charge_exchange' : {
        #    'cross_section' : cross_sec_direc+'charge_exchange.dat'
        # }
    },
)

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    warpx_max_grid_size=128,
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    bc_xmin="dirichlet",
    bc_xmax="dirichlet",
    bc_ymin="periodic",
    bc_ymax="periodic",
    warpx_potential_hi_x="%.1f*sin(2*pi*%.5e*t)" % (VOLTAGE, FREQ),
    lower_boundary_conditions_particles=["absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
)

# solver = picmi.ElectrostaticSolver(
#    grid=grid, method='Multigrid', required_precision=1e-6
# )
solver = PoissonSolverPseudo1D(grid=grid)

##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["rho_electrons", "rho_he_ions"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=max_steps,
    warpx_collisions=[mcc_electrons, mcc_ions],
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################

sim.step(max_steps)

# confirm that the external solver was run
assert hasattr(solver, "phi")
