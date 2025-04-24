#!/usr/bin/env python3

import numpy as np

from pywarpx import picmi

constants = picmi.constants
c = constants.c
q_e = constants.q_e
m_e = constants.m_e
ep0 = constants.ep0

max_step = 40
lx = 40.0e-6  # length of sides
dx = 6.25e-07  # grid cell size
nx = int(lx / dx)  # number of cells in each dimension
epsilon = 0.01
n0 = 2.0e24  # electron and positron densities, #/m^3
wp = np.sqrt(2.0 * n0 * q_e**2 / (ep0 * m_e))  # plasma frequency
kp = wp / c  # plasma wavenumber
k = 2.0 * 2.0 * np.pi / lx  # perturbation wavenumber

momentum_expressions_electrons = [
    """c * epsilon * k/kp * sin(k*x) * cos(k*y) * cos(k*z)""",
    """c * epsilon * k/kp * cos(k*x) * sin(k*y) * cos(k*z)""",
    """c * epsilon * k/kp * cos(k*x) * cos(k*y) * sin(k*z)""",
]
initial_distribution_electrons = picmi.AnalyticDistribution(
    density_expression=n0,
    lower_bound=[-20e-6, -20e-6, -20e-6],
    upper_bound=[20e-6, 20e-6, 20e-6],
    c=c,
    epsilon=epsilon,
    k=k,
    kp=kp,
    momentum_expressions=momentum_expressions_electrons,
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=initial_distribution_electrons,
)

momentum_expressions_positrons = [
    """(-1.) * c * epsilon * k/kp * sin(k*x) * cos(k*y) * cos(k*z)""",
    """(-1.) * c * epsilon * k/kp * cos(k*x) * sin(k*y) * cos(k*z)""",
    """(-1.) * c * epsilon * k/kp * cos(k*x) * cos(k*y) * sin(k*z)""",
]
initial_distribution_positrons = picmi.AnalyticDistribution(
    density_expression=n0,
    lower_bound=[-20e-6, -20e-6, -20e-6],
    upper_bound=[20e-6, 20e-6, 20e-6],
    c=c,
    epsilon=epsilon,
    k=k,
    kp=kp,
    momentum_expressions=momentum_expressions_positrons,
)
positrons = picmi.Species(
    particle_type="positron",
    name="positrons",
    initial_distribution=initial_distribution_positrons,
)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nx],
    lower_bound=[-lx / 2.0, -lx / 2.0, -lx / 2.0],
    upper_bound=[lx / 2.0, lx / 2.0, lx / 2.0],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    warpx_max_grid_size=[nx, nx, nx],
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="PSATD",
    cfl=0.5773502691896258,
    warpx_psatd_update_with_rho=1,
    warpx_psatd_JRhom="LL2",
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_step,
    verbose=1,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=0,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=[1, 1, 1],
        grid=grid,
    ),
)

sim.add_species(
    positrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=[1, 1, 1],
        grid=grid,
    ),
)

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_step,
    data_list=[
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
        "Jx",
        "Jy",
        "Jz",
        "part_per_cell",
        "rho",
    ],
)
sim.add_diagnostic(field_diag1)

particle_diag1_electrons = picmi.ParticleDiagnostic(
    name="diag1",
    period=max_step,
    species=[electrons],
    data_list=["x", "y", "z", "weighting", "ux"],
)
particle_diag1_positrons = picmi.ParticleDiagnostic(
    name="diag1",
    period=max_step,
    species=[positrons],
    data_list=["x", "y", "z", "uz"],
)
sim.add_diagnostic(particle_diag1_electrons)
sim.add_diagnostic(particle_diag1_positrons)

sim.write_input_file(file_name="inputs_test_3d_langmuir_multi_psatd_JRhom_LL2_picmi")

sim.step()
