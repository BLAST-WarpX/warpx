#!/usr/bin/env python3
"""Compact PICMI regression for prescribed current injection."""

from pathlib import Path

from pywarpx import picmi

test_dir = Path(__file__).resolve().parent

grid = picmi.Cartesian3DGrid(
    number_of_cells=[8, 8, 8],
    lower_bound=[-0.5, -0.5, -0.5],
    upper_bound=[0.5, 0.5, 0.5],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    lower_boundary_conditions_particles=["periodic", "periodic", "periodic"],
    upper_boundary_conditions_particles=["periodic", "periodic", "periodic"],
    warpx_max_grid_size=8,
    warpx_blocking_factor=8,
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)

drive = picmi.PrescribedCurrentDrive(
    lower_bound=[-0.25, -0.25, -0.25],
    upper_bound=[0.25, 0.25, 0.25],
    area=0.25,
    direction=0,
    sign=1,
)
current = picmi.PrescribedCurrentInjection(
    drives=[drive],
    file=str(test_dir / "current_profile.txt"),
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-12,
    max_steps=1,
    verbose=1,
    particle_shape="linear",
    warpx_current_deposition_algo="direct",
)
simulation.add_prescribed_current_injection(current)
try:
    simulation.add_prescribed_current_injection(current)
except ValueError:
    pass
else:
    raise AssertionError("A second PrescribedCurrentInjection must be rejected")

simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=-1,
        data_list=["Jx", "Jy", "Jz", "rho"],
        write_dir="diags",
        warpx_file_prefix="diag1",
    )
)
simulation.step()
