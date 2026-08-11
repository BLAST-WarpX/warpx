#!/usr/bin/env python3
#
# --- Regression test for the hybrid-PIC (Ohm's law) solver with an embedded
# --- boundary that does NOT cover the (non-periodic) domain boundaries: a
# --- uniform, magnetized proton plasma surrounding a conducting cylinder at
# --- the center of the domain, with dirichlet field boundary conditions in x
# --- and y.
# ---
# --- In this configuration the `eb_update_E`/`eb_update_B` flags are read in
# --- the guard cells beyond the non-periodic domain faces: the plasma-current
# --- calculation (curl(B)/mu0) runs on a tilebox grown by one ghost cell, and
# --- the values it computes there feed the Ohm's-law E-field solve on the
# --- outermost valid cells (through the Hall-term interpolation and the
# --- hyper-resistivity Laplacian). The checksums of this test therefore guard
# --- the initialization of those flags: they fail if the flag guard cells
# --- beyond the domain boundary hold values that alter the update (see the
# --- pre-fill in MarkUpdateCellsStairCase).

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# Plasma parameters
n0 = 1e20  # plasma density (m^-3)
T_i = 10.0  # ion temperature (eV)
T_e = 10.0  # electron temperature (eV)
B0 = 0.1  # initial magnetic field strength (T)

# Geometry: conducting cylinder of radius R_eb at the center of the domain
R_eb = 0.2  # m
L = 1.0  # domain edge length (m)
N = 32  # cells per dimension

m_p = 1.67262192369e-27  # proton mass (kg)

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=False)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[N, N, N],
    lower_bound=[-0.5 * L, -0.5 * L, -0.5 * L],
    upper_bound=[0.5 * L, 0.5 * L, 0.5 * L],
    lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    warpx_max_grid_size=16,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=T_e,
    n0=n0,
    n_floor=0.05 * n0,
    plasma_resistivity=1e-6,
    plasma_hyper_resistivity=1e-9,
    substeps=10,
)
simulation.solver = solver

# Conducting cylinder: implicit function positive INSIDE r < R_eb, so the
# domain boundaries remain outside of the embedded boundary
simulation.embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="-(x**2+y**2-R_eb**2)", R_eb=R_eb
)

# Uniform Bz applied through the parsed initial-field path (this exercises
# the eb_update_B-gated external-field fill as well)
B_init = picmi.AnalyticInitialField(
    Bx_expression="0.0",
    By_expression="0.0",
    Bz_expression=f"{B0}",
    warpx_do_initial_div_cleaning=False,
)
simulation.add_applied_field(B_init)

ions = picmi.Species(
    particle_type="proton",
    name="ions",
    initial_distribution=picmi.UniformDistribution(
        density=n0,
        rms_velocity=[np.sqrt(constants.q_e * T_i / m_p)] * 3,
    ),
)
simulation.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=4, grid=grid),
)

# Time step: a small fraction of the ion cyclotron period
w_ci = constants.q_e * B0 / m_p
dt = 0.02 * 2 * np.pi / w_ci

simulation.time_step_size = dt
simulation.max_steps = 10
simulation.current_deposition_algo = "direct"
simulation.particle_shape = 1

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=["B", "E", "J"],
    warpx_format="plotfile",
)
simulation.add_diagnostic(field_diag)

simulation.step()
