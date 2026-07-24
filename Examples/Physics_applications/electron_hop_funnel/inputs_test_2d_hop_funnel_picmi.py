# -*- coding: utf-8 -*-

import numpy as np

from pywarpx import libwarpx, picmi
from pywarpx.LoadThirdParty import load_cupy

constants = picmi.constants


def energy_to_velocity(energy_ev, m0=9.1093837139e-31):
    """
    Converts beam kinetic energy to relativistic velocity.

    Parameters:
    energy_ev (float): Kinetic energy of the beam in electronvolts (eV).

    Returns:
    float: Velocity of the electron in meters per second (m/s).
    """

    E_rest_joules = m0 * (constants.c**2)
    E_k_joules = energy_ev * constants.q_e
    gamma_inv = E_rest_joules / (E_k_joules + E_rest_joules)
    velocity = constants.c * np.sqrt(1.0 - (gamma_inv**2))
    return velocity


##########################
# Funnel Parameters
##########################
blocking_factor = 16


class Geometry:
    def __init__(self, NDX=2.0, xmin=0.0, ymin=0.0):

        self.yFunnel = 1e-3
        self.yHopElectrode = 0.05e-3
        self.yHopAnodeGap = 0.5e-3
        self.yCathodeFunnelGap = 0.1e-3

        self.thetaFunnelDeg = 45.0
        self.xFunnelTop = 0.3e-3

        # dependant variables
        self.thetaFunnelRad = np.deg2rad(self.thetaFunnelDeg)
        self.xFunnelBuffer = self.yFunnel * 2
        self.xFunnelBottom = (
            self.yFunnel * np.tan(self.thetaFunnelRad) * 2 + self.xFunnelTop
        )

        # grid
        self.NDX = NDX
        self.xmin = xmin
        self.ymin = ymin
        self.dx, self.dy = self.calcdxy(NDX=NDX)
        self.xmax, self.ymax = self.calcDomain(xmin=xmin, ymin=ymin)
        self.nx, self.ny = self.calcCells(NDX=NDX)
        self.alignBlockingFactor()

    def calcdxy(self, NDX=2):
        dxy = self.yHopElectrode / NDX
        return dxy, dxy

    def calcDomain(self, xmin=0.0, ymin=0.0):
        ymax = (
            self.yFunnel
            + self.yHopElectrode
            + self.yHopAnodeGap
            + self.yCathodeFunnelGap
            + ymin
        )
        xmax = self.xFunnelBottom + self.xFunnelBuffer + xmin
        return xmax, ymax

    def calcCells(self, NDX=2):
        xmax, ymax = self.calcDomain()
        dx, dy = self.calcdxy(NDX=NDX)
        return xmax / dx, ymax / dy

    def alignBlockingFactor(
        self, xFudgeVariable="xFunnelBuffer", yFudgeVariable="yHopAnodeGap"
    ):
        nyAdd = blocking_factor - self.ny % blocking_factor
        nxAdd = blocking_factor - self.nx % blocking_factor
        xFudge = getattr(self, xFudgeVariable)
        yFudge = getattr(self, yFudgeVariable)
        setattr(self, xFudgeVariable, xFudge + nxAdd * self.dx)
        setattr(self, yFudgeVariable, yFudge + nyAdd * self.dy)
        self.xmax, self.ymax = self.calcDomain(xmin=self.xmin, ymin=self.ymin)
        self.nx, self.ny = self.calcCells(NDX=self.NDX)

    def redistributeCells(
        self,
    ):
        # this will ensure dx,nx, and xmax all are consistant
        # need way to align geometry to grid
        pass


gap = 0.1  # m

hopVoltage = 1000
anodeVoltage = hopVoltage + 200.0
cathodeVoltage = 0.0

plasma_density = 2.56e9
elec_temp = 3000.0  # K

beam_energy = 50.0  # [eV]
max_beam_energy = anodeVoltage + beam_energy  # [eV]

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 100
nDumps = 2
diagnostic_intervals = "::%i" % (max_steps / nDumps)

# --- Grid
xmin = 0.0
ymin = 0.0
geo = Geometry(xmin=xmin, ymin=ymin)
nx, ny = geo.calcCells()
xmax, ymax = geo.calcDomain(xmin=xmin, ymin=ymin)
dx, dy = geo.calcdxy()


number_per_cell = 5
number_per_cell_each_dim = [number_per_cell, number_per_cell]

dt_frac = 0.95
dt = dt_frac * min(dx, dy) / energy_to_velocity(max(max_beam_energy, 1.0))
total_time = max_steps * dt

##########################
# particle definitions
##########################
v_beam = energy_to_velocity(beam_energy)
v_rms_elec = np.sqrt(constants.kb * elec_temp / constants.m_e)
plasma_flux = plasma_density * v_beam

particle_flux = picmi.UniformFluxDistribution(
    flux=plasma_flux,
    flux_normal_axis="z",
    surface_flux_position=0.0,
    flux_direction=1,
    lower_bound=[xmin, ymin, None],
    upper_bound=[xmax, ymin + dy, None],
    rms_velocity=[v_rms_elec] * 3,
    directed_velocity=[0.0, v_beam, 0.0],
)

electrons = picmi.Species(
    particle_type="electron", name="electrons", initial_distribution=None
)

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    warpx_max_grid_size=64,
    warpx_blocking_factor=blocking_factor,
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    bc_xmin="neumann",
    bc_xmax="neumann",
    bc_ymin="dirichlet",
    bc_ymax="dirichlet",
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_potential_lo_z=cathodeVoltage,
    warpx_potential_hi_z=anodeVoltage,
)

solverType = "ES"
if solverType == "EM":
    solver = picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",
        cfl=dt_frac,
    )
    dt = None
else:
    solver = picmi.ElectrostaticSolver(
        grid=grid,
        method="Multigrid",
        required_precision=1e-5,
        warpx_absolute_tolerance=1e2,
        warpx_self_fields_verbosity=0,
    )
##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
    data_list=["position", "momentum", "weighting"],
    warpx_format="openpmd",
    # warpx_openpmd_backend="h5"
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["rho_electrons", "E", "phi"],
    warpx_format="openpmd",
    # warpx_openpmd_backend="h5"
)

##########################
# simulation setup
##########################
layout_elec = picmi.PseudoRandomLayout(
    n_macroparticles_per_cell=number_per_cell, grid=grid
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=False,
    time_step_size=dt,
)

sim.add_species(electrons, layout=layout_elec)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################
sim.initialize_inputs()
sim.initialize_warpx()

## add dielectric
xp, _ = load_cupy()
Ex = sim.fields.get("Efield_fp", dir="x", level=0)
X, Z = np.meshgrid(Ex.mesh("x"), Ex.mesh("z"))
epsilon_r = sim.fields.alloc_init(
    name="epsilon_r",
    level=0,
    ba=Ex.box_array(),
    dm=Ex.dm(),
    ncomp=1,
    ngrow=Ex.n_grow_vect,
    initial_value=1.0,  # Default to vacuum permittivity
    redistribute=True,
    redistribute_on_remake=True,
)
epsilon_r.level = 0
epsilon_r[..., 0] = xp.where((Z.T + X.T) > 0.001, 7.0, 1.0)
epsilon_r.finalize()
libwarpx.warpx.set_epsilon_r([epsilon_r])

sim.step(max_steps)
