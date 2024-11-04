#!/usr/bin/env python3
#
# --- Input file for MCC testing. There is already a test of the MCC
# --- functionality. This tests the PICMI interface for the MCC and
# --- provides an example of how an external Poisson solver can be
# --- used for the field solve step.

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as sla

from pywarpx import callbacks, fields, picmi

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
    def __init__(self, grid, **kwargs):
        """Direct solver for the Poisson equation using superLU. This solver is
        useful for pseudo 1D cases i.e. diode simulations with small x extent.

        Arguments:
            grid (picmi.Cartesian2DGrid): Instance of the grid on which the
            solver will be installed.
        """
        super(PoissonSolverPseudo1D, self).__init__(
            grid=grid,
            method=kwargs.pop("method", "Multigrid"),
            required_precision=1,
            **kwargs,
        )
        self.rho_wrapper = None
        self.phi_wrapper = None
        self.time_sum = 0.0

    def solver_initialize_inputs(self):
        """Grab geometrical quantities from the grid."""
        self.right_voltage = self.grid.potential_xmax

        # set WarpX boundary potentials to None since we will handle it
        # ourselves in this solver
        self.grid.potential_xmin = None
        self.grid.potential_xmax = None
        self.grid.potential_ymin = None
        self.grid.potential_ymax = None
        self.grid.potential_zmin = None
        self.grid.potential_zmax = None

        super(PoissonSolverPseudo1D, self).solver_initialize_inputs()

        self.nx = self.grid.number_of_cells[0]
        self.nz = self.grid.number_of_cells[1]
        self.dx = (self.grid.upper_bound[0] - self.grid.lower_bound[0]) / self.nx
        self.dz = (self.grid.upper_bound[1] - self.grid.lower_bound[1]) / self.nz

        if not np.isclose(self.dx, self.dz):
            raise RuntimeError("Direct solver requires dx = dz.")

        self.nxguardrho = 2
        self.nzguardrho = 2
        self.nxguardphi = 1
        self.nzguardphi = 1

        self.phi = np.zeros(
            (self.nx + 1 + 2 * self.nxguardphi, self.nz + 1 + 2 * self.nzguardphi)
        )

        self.decompose_matrix()

        callbacks.installpoissonsolver(self._run_solve)

    def decompose_matrix(self):
        """Function to build the superLU object used to solve the linear
        system."""
        self.nxsolve = self.nx + 1
        self.nzsolve = self.nz + 3

        # Set up the computation matrix in order to solve A*phi = rho
        A = np.zeros((self.nzsolve * self.nxsolve, self.nzsolve * self.nxsolve))
        kk = 0
        for ii in range(self.nxsolve):
            for jj in range(self.nzsolve):
                temp = np.zeros((self.nxsolve, self.nzsolve))

                if ii == 0 or ii == self.nxsolve - 1:
                    temp[ii, jj] = 1.0
                elif ii == 1:
                    temp[ii, jj] = -2.0
                    temp[ii - 1, jj] = 1.0
                    temp[ii + 1, jj] = 1.0
                elif ii == self.nxsolve - 2:
                    temp[ii, jj] = -2.0
                    temp[ii + 1, jj] = 1.0
                    temp[ii - 1, jj] = 1.0
                elif jj == 0:
                    temp[ii, jj] = 1.0
                    temp[ii, -3] = -1.0
                elif jj == self.nzsolve - 1:
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
        self.lu = sla.splu(A)

    def _run_solve(self):
        """Function run on every step to perform the required steps to solve
        Poisson's equation."""

        # get rho from WarpX
        if self.rho_wrapper is None:
            self.rho_wrapper = fields.RhoFPWrapper(0)
        self.rho_data = self.rho_wrapper[(), ()]

        self.solve()

        if self.phi_wrapper is None:
            self.phi_wrapper = fields.PhiFPWrapper(0)
        self.phi_wrapper[(), ()] = self.phi

    def solve(self):
        """The solution step. Includes getting the boundary potentials and
        calculating phi from rho."""
        right_voltage = eval(
            self.right_voltage,
            {"t": sim.extension.warpx.gett_new(0), "sin": np.sin, "pi": np.pi},
        )
        left_voltage = 0.0

        rho = (
            -self.rho_data[
                self.nxguardrho : -self.nxguardrho, self.nzguardrho : -self.nzguardrho
            ]
            / constants.ep0
        )

        # Construct b vector
        nx, nz = np.shape(rho)
        source = np.zeros((nx, nz + 2), dtype=np.float32)
        source[:, 1:-1] = rho * self.dx**2

        source[0] = left_voltage
        source[-1] = right_voltage

        # Construct b vector
        b = source.flatten()

        flat_phi = self.lu.solve(b)
        self.phi[self.nxguardphi : -self.nxguardphi] = flat_phi.reshape(
            np.shape(source)
        )

        self.phi[: self.nxguardphi] = left_voltage
        self.phi[-self.nxguardphi :] = right_voltage

        # the electrostatic solver in WarpX keeps the ghost cell values as 0
        self.phi[:, : self.nzguardphi] = 0
        self.phi[:, -self.nzguardphi :] = 0


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
        "back": {"cross_section": cross_sec_direc + "ion_back_scatter.dat"},
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
