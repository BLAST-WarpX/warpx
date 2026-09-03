#!/usr/bin/env python3
"""PICMI regressions for the global paired current-controlled port."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pywarpx import libwarpx, picmi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=("2d", "3d", "rz-hybrid", "rz-implicit"), required=True
)
parser.add_argument("--with-eb", action="store_true")
args, remaining = parser.parse_known_args()
sys.argv = sys.argv[:1] + remaining

test_dir = Path(__file__).resolve().parent
waveform = str(test_dir / "current_profile.txt")
dt = 1.0e-12


def add_neutral_kinetic_plasma(
    simulation, grid, density=1.0e8, particles_per_cell=(1, 1, 1)
):
    distribution = picmi.UniformDistribution(
        density=density, directed_velocity=[0.0, 0.0, 0.0]
    )
    electrons = picmi.Species(
        particle_type="electron",
        name="electrons",
        initial_distribution=distribution,
    )
    protons = picmi.Species(
        particle_type="proton",
        name="protons",
        initial_distribution=distribution,
    )
    layout = picmi.GriddedLayout(
        grid=grid, n_macroparticle_per_cell=list(particles_per_cell)
    )
    simulation.add_species(electrons, layout=layout)
    simulation.add_species(protons, layout=layout)


if args.mode == "3d":
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[8, 8, 8],
        lower_bound=[-0.5, -0.5, -0.5],
        upper_bound=[0.5, 0.5, 0.5],
        lower_boundary_conditions=["periodic"] * 3,
        upper_boundary_conditions=["periodic"] * 3,
        lower_boundary_conditions_particles=["periodic"] * 3,
        upper_boundary_conditions_particles=["periodic"] * 3,
        warpx_max_grid_size=8,
        warpx_blocking_factor=8,
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
    simulation = picmi.Simulation(
        solver=solver,
        time_step_size=dt,
        max_steps=2,
        particle_shape="linear",
        warpx_current_deposition_algo="esirkepov",
        warpx_serialize_initial_conditions=1,
        warpx_use_filter=0,
    )
    add_neutral_kinetic_plasma(simulation, grid)

    # Exercise the material-current part of the fully kinetic total-current
    # constraint in addition to particle and displacement current.
    simulation.add_macroscopic_property(
        picmi.MacroscopicProperty(name="epsilon", value=picmi.constants.ep0)
    )
    simulation.add_macroscopic_property(
        picmi.MacroscopicProperty(name="mu", value=picmi.constants.mu0)
    )
    simulation.add_macroscopic_property(
        picmi.MacroscopicProperty(name="sigma", value=1.0, method="backwardeuler")
    )
    port = picmi.CurrentControlledPort(
        direction=0,
        terminal_0_lower_bound=[-0.1875, -0.25, -0.25],
        terminal_0_upper_bound=[-0.1875, 0.25, 0.25],
        terminal_1_lower_bound=[0.1875, -0.25, -0.25],
        terminal_1_upper_bound=[0.1875, 0.25, 0.25],
        file=waveform,
    )
    antenna = picmi.PrescribedCurrentInjection(
        drives=[
            picmi.PrescribedCurrentDrive(
                lower_bound=[-0.25, -0.25, -0.25],
                upper_bound=[0.25, 0.25, 0.25],
                area=0.25,
                direction=2,
            )
        ],
        file=waveform,
    )
    simulation.add_prescribed_current_injection(antenna)
    simulation.add_diagnostic(
        picmi.FieldDiagnostic(
            name="diag1",
            grid=grid,
            period=2,
            data_list=["Bx", "By", "Bz", "divB"],
            write_dir="diags",
            warpx_file_prefix="diag1",
        )
    )
elif args.mode == "2d":
    grid = picmi.Cartesian2DGrid(
        number_of_cells=[8, 8],
        lower_bound=[-0.5, -0.5],
        upper_bound=[0.5, 0.5],
        lower_boundary_conditions=["periodic", "periodic"],
        upper_boundary_conditions=["periodic", "periodic"],
        lower_boundary_conditions_particles=["periodic", "periodic"],
        upper_boundary_conditions_particles=["periodic", "periodic"],
        warpx_max_grid_size=8,
        warpx_blocking_factor=8,
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
    simulation = picmi.Simulation(
        solver=solver,
        time_step_size=dt,
        max_steps=2,
        particle_shape="linear",
        warpx_current_deposition_algo="esirkepov",
        warpx_serialize_initial_conditions=1,
        warpx_use_filter=0,
    )
    add_neutral_kinetic_plasma(simulation, grid, particles_per_cell=(1, 1))
    port = picmi.CurrentControlledPort(
        direction=0,
        terminal_0_lower_bound=[-0.1875, 0.0, -0.25],
        terminal_0_upper_bound=[-0.1875, 0.0, 0.25],
        terminal_1_lower_bound=[0.1875, 0.0, -0.25],
        terminal_1_upper_bound=[0.1875, 0.0, 0.25],
        file=waveform,
    )
elif args.mode == "rz-hybrid":
    grid = picmi.CylindricalGrid(
        number_of_cells=[16, 16],
        lower_bound=[0.0, 0.0],
        upper_bound=[0.5, 1.0],
        lower_boundary_conditions=["none", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["none", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic"],
        warpx_max_grid_size=16,
        warpx_blocking_factor=8,
    )
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=0.1,
        n0=1.0e12,
        gamma=1.0,
        n_floor=1.0e6,
        plasma_resistivity=0.0,
        plasma_hyper_resistivity=0.0,
        substeps=2,
    )
    embedded_boundary = None
    if args.with_eb:
        # The terminal surface cuts through this solid rod; its Ampere contour
        # remains at r=0.375 m in solver-active cells.
        embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="rod_radius-x", rod_radius=0.125
        )
    simulation = picmi.Simulation(
        solver=solver,
        time_step_size=dt,
        max_steps=2,
        particle_shape="linear",
        warpx_current_deposition_algo="direct",
        warpx_grid_type="collocated",
        warpx_embedded_boundary=embedded_boundary,
        warpx_serialize_initial_conditions=1,
        warpx_use_filter=0,
    )
    ions = picmi.Species(
        particle_type="proton",
        name="ions",
        initial_distribution=picmi.UniformDistribution(
            density=1.0e12, directed_velocity=[0.0, 0.0, 0.0]
        ),
    )
    simulation.add_species(
        ions,
        layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1, 1, 1]),
    )
    port = picmi.CurrentControlledPort(
        direction=2,
        terminal_0_lower_bound=[0.0, 0.0, 0.25],
        terminal_0_upper_bound=[0.375, 0.0, 0.25],
        terminal_1_lower_bound=[0.0, 0.0, 0.75],
        terminal_1_upper_bound=[0.375, 0.0, 0.75],
        file=waveform,
    )
else:
    assert args.mode == "rz-implicit"
    assert not args.with_eb
    grid = picmi.CylindricalGrid(
        number_of_cells=[16, 16],
        lower_bound=[0.0, 0.0],
        upper_bound=[0.5, 1.0],
        lower_boundary_conditions=["none", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["none", "periodic"],
        upper_boundary_conditions_particles=["reflecting", "periodic"],
        warpx_max_grid_size=16,
        warpx_blocking_factor=8,
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
    nonlinear_solver = picmi.PicardNonlinearSolver(
        verbose=False,
        require_convergence=False,
        max_iterations=2,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
        nonlinear_solver=nonlinear_solver, theta=0.5
    )
    simulation = picmi.Simulation(
        solver=solver,
        time_step_size=dt,
        max_steps=1,
        particle_shape="linear",
        warpx_evolve_scheme=evolve_scheme,
        warpx_current_deposition_algo="villasenor",
        warpx_field_gathering_algo="energy-conserving",
        warpx_serialize_initial_conditions=1,
        warpx_use_filter=0,
    )
    add_neutral_kinetic_plasma(simulation, grid)
    port = picmi.CurrentControlledPort(
        direction=2,
        terminal_0_lower_bound=[0.0, 0.0, 0.25],
        terminal_0_upper_bound=[0.375, 0.0, 0.25],
        terminal_1_lower_bound=[0.0, 0.0, 0.75],
        terminal_1_upper_bound=[0.375, 0.0, 0.75],
        file=waveform,
    )

simulation.add_current_controlled_port(port)
try:
    simulation.add_current_controlled_port(port)
except ValueError:
    pass
else:
    raise AssertionError("A second global CurrentControlledPort must be rejected")

simulation.step()
status = np.asarray(libwarpx.warpx.get_current_controlled_port_status())
expected = np.full(3, 1000.0)
relative_error = np.max(np.abs(status - expected)) / expected[0]
print(f"current-controlled port status [A or A/m]: {status}")
print(f"maximum relative port-current error: {relative_error:.6e}")
assert relative_error < 2.0e-12
