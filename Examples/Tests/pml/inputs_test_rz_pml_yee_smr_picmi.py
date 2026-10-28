#!/usr/bin/env python3
"""Cylindrical pulses cross radial, axial, and refined QCPML layers."""

import argparse
import sys

import numpy as np

from pywarpx import amr, picmi

parser = argparse.ArgumentParser()
parser.add_argument("--restart")
parser.add_argument("--annular", action="store_true")
parser.add_argument("--material", action="store_true")
parser.add_argument("--mode", type=int, choices=(0, 1), default=0)
args, remaining = parser.parse_known_args()
sys.argv = sys.argv[:1] + remaining
amr.n_error_buf = 0
cells = np.array([32, 64])
lower = np.array([0.0, -24.0e-6])
upper = np.array([24.0e-6, 24.0e-6])
grid = picmi.CylindricalGrid(
    number_of_cells=cells.tolist(),
    lower_bound=lower.tolist(),
    upper_bound=upper.tolist(),
    lower_boundary_conditions=["none", "open"],
    upper_boundary_conditions=["open", "open"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    n_azimuthal_modes=args.mode + 1,
    refined_regions=[[1, [6.0e-6, -12.0e-6], [18.0e-6, 12.0e-6]]]
    if args.annular
    else [[1, [0.0, -12.0e-6], [12.0e-6, 12.0e-6]]],
    warpx_max_grid_size=32,
    warpx_blocking_factor=8,
)
steps = 350
simulation = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.8),
    max_steps=steps,
    warpx_amr_restart=args.restart,
    warpx_use_filter=0,
    verbose=0,
)
if args.material:
    for name, value in (
        ("epsilon", picmi.constants.ep0),
        ("mu", picmi.constants.mu0),
        ("sigma", 0),
    ):
        simulation.add_macroscopic_property(
            picmi.MacroscopicProperty(name=name, value=value)
        )
simulation.add_diagnostic(
    picmi.ReducedDiagnostic(name="field_energy", diag_type="FieldEnergy", period=1)
)
simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=steps,
        data_list=["E", "B"],
        write_dir="diags",
        warpx_file_prefix="diag1",
    )
)
simulation.add_diagnostic(
    picmi.Checkpoint(
        name="chk", period=steps // 2, write_dir="diags", warpx_file_prefix="chk"
    )
)
simulation.initialize_inputs()
simulation.initialize_warpx()
if not args.restart:
    width = 2.0e-6
    for level, patch in ((0, "fp"), (1, "fp"), (1, "cp")):
        spacing = (upper - lower) / cells / (2 if level == 1 and patch == "fp" else 1)
        for kind in ("E", "B"):
            for direction in ("r", "theta", "z"):
                field = simulation.fields.get(
                    f"{kind}field_{patch}", dir=direction, level=level
                )
                values = np.asarray(field[...])
                if values.ndim == 2:
                    values = values[..., None]
                origin = np.array(field.box_array().minimal_box().small_end)
                staggering = np.array(
                    [field.box_array().ix_type().node_centered(i) for i in (0, 1)]
                )
                r = (
                    lower[0]
                    + (
                        np.arange(values.shape[0])
                        + origin[0]
                        + 0.5 * (1 - staggering[0])
                    )
                    * spacing[0]
                )
                z = (
                    lower[1]
                    + (
                        np.arange(values.shape[1])
                        + origin[1]
                        + 0.5 * (1 - staggering[1])
                    )
                    * spacing[1]
                )
                gaussian = np.exp(-(r[:, None] ** 2 + z[None, :] ** 2) / (2 * width**2))
                values[...] = 0
                if args.mode == 0 and direction == "theta":
                    values[..., 0] = (
                        r[:, None]
                        / width
                        * gaussian
                        / (picmi.constants.c if kind == "B" else 1)
                    )
                elif args.mode == 1 and kind == "E":
                    if direction == "r":
                        values[..., 1] = gaussian
                    elif direction == "theta":
                        rp, rm = r + spacing[0] / 2, r - spacing[0] / 2
                        derivative = (
                            rp * np.exp(-(rp**2) / (2 * width**2))
                            - rm * np.exp(-(rm**2) / (2 * width**2))
                        ) / spacing[0]
                        values[..., 2] = -derivative[:, None] * np.exp(
                            -(z[None, :] ** 2) / (2 * width**2)
                        )
                        if origin[0] == 0:
                            values[0, :, 2] = -np.exp(
                                -((spacing[0] / 2) ** 2 + z**2) / (2 * width**2)
                            )
                field[...] = values[..., 0] if args.mode == 0 else values
simulation.step(steps // 2 if args.restart else steps)
