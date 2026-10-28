#!/usr/bin/env python3
"""Constant-eta magnetic-diffusion analytic regression through PICMI."""

from pywarpx import picmi

grid = picmi.Cartesian2DGrid(
    number_of_cells=[64, 64],
    lower_bound=[0.0, 0.0],
    upper_bound=[1.0, 1.0],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=64,
)

# Dense, cold background keeps Hall and pressure effects negligible compared
# with diffusion of the divergence-free By Fourier mode initialized below.
solver = picmi.HybridPICSolver(
    grid=grid,
    Te=0.1,
    n0=1.0e26,
    gamma=1.0,
    n_floor=1.0e20,
    plasma_resistivity=0.0,
    plasma_hyper_resistivity=0.0,
    substeps=4,
    implicit_mag_diffusion=True,
    mag_diff_theta=1.0,
    mag_diff_eta_explicit_max=0.0,
    mag_diff_use_variable_eta=False,
    mag_diff_constant_eta=1.0e-2,
    mag_diff_linear_solver="amrex_gmres",
    mag_diff_rtol=1.0e-8,
    mag_diff_atol=0.0,
    mag_diff_max_iter=200,
    mag_diff_verbose=0,
)

protons = picmi.Species(
    particle_type="proton",
    name="protons",
    initial_distribution=picmi.UniformDistribution(
        density=1.0e26,
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)

initial_field = picmi.AnalyticInitialField(
    Bx_expression="0.0",
    By_expression="1.0e-4*cos(2.0*pi*x)",
    Bz_expression="0.0",
    warpx_do_initial_div_cleaning=False,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-9,
    max_steps=20,
    verbose=1,
    particle_shape=1,
    warpx_current_deposition_algo="direct",
)
simulation.add_species(
    protons,
    layout=picmi.GriddedLayout(
        grid=grid,
        n_macroparticle_per_cell=[2, 2],
    ),
)
simulation.add_applied_field(initial_field)
simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=20,
        data_list=["Bx", "By", "Bz"],
    )
)

simulation.step()
