#!/usr/bin/env python3
#
# 1D regression test for the macroscopic (dielectric) electromagnetic solver.
#
# A standing electromagnetic cavity mode is set up between two PEC walls along z.
# The mode is initialized through the external magnetic field By(z) = cos(k z),
# with E = 0 at t = 0, where k = p*pi/L is the cavity wavenumber. In a
# non-dispersive dielectric (epsilon_r = 1.5, mu = mu0, sigma = 0) the mode
# oscillates as By(z, t) = cos(k z) cos(omega t) with the reduced frequency
# omega = c * k / sqrt(epsilon_r).
#
# This exercises the 1D macroscopic E-update.

import numpy as np

from pywarpx import picmi

# Simulation parameters
max_steps = 200
nz = 128
zmin = 0.0
zmax = 1.0
max_grid_size = 128

# Problem specific parameters
epsilon_r = 1.5
L = zmax - zmin
kz = 1.0 * np.pi / L
By_expression = "cos(kz*z)"

# Define grid
grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["dirichlet"],
    upper_boundary_conditions=["dirichlet"],
    lower_boundary_conditions_particles=["absorbing"],
    upper_boundary_conditions_particles=["absorbing"],
)

# Define solver
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=1.0, divE_cleaning=0)

# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
)

# Define epsilon
epsilon = picmi.MacroscopicProperty(
    name="epsilon", implicit_function="epsilon_r*epsilon0", epsilon_r=epsilon_r
)
sigma = picmi.MacroscopicProperty(name="sigma", value=0.0, method="backwardeuler")
mu = picmi.MacroscopicProperty(name="mu", value=picmi.constants.mu0)

# Define diagnostics
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=[
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
    ],
)

# Define inital magnetic field
B_ext = picmi.AnalyticInitialField(
    Bx_expression="0.0", By_expression=By_expression, Bz_expression="0.0", kz=kz
)

# Add material properties, diagnostics, and inital magnetic field to the simulation
sim.add_macroscopic_property(epsilon)
sim.add_macroscopic_property(sigma)
sim.add_macroscopic_property(mu)

sim.add_diagnostic(field_diag)
sim.add_applied_field(B_ext)

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Advance simulation until last time step
sim.step(max_steps)
