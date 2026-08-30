#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "Tools" / "PostProcessing"))
from read_raw_data import _read_buffer  # noqa: E402

parser = argparse.ArgumentParser(
    description="Plot the final four-group RZ hohlraum demonstration."
)
parser.add_argument("plotfile", type=Path)
parser.add_argument("--output", type=Path, default=Path("hohlraum_multigroup.png"))
args = parser.parse_args()

with open(args.plotfile / "Header") as header:
    header.readline()
    n_fields = int(header.readline())
    field_names = [header.readline().strip() for _ in range(n_fields)]
fields = _read_buffer(
    str(args.plotfile), str(args.plotfile / "Level_0" / "Cell_H"), field_names
)

group_names = [f"radiation_diffusion_energy_g{group}" for group in range(4)]
missing_fields = [name for name in ["rho", "Te", *group_names] if name not in fields]
if missing_fields:
    raise RuntimeError(
        "The plotfile is not a four-group hybrid hohlraum result; missing "
        + ", ".join(missing_fields)
    )

# Geometry matches inputs_base_rz. Convert cell-integrated group energies to
# physical RZ energy densities before mirroring the axisymmetric half-plane.
r_max = 1.0e-3
z_min = -1.5e-3
z_max = 1.5e-3
n_r, n_z = fields["Te"].shape
dr = r_max / n_r
dz = (z_max - z_min) / n_z
r_lo = np.arange(n_r)[:, None] * dr
cell_volume = np.pi * ((r_lo + dr) ** 2 - r_lo**2) * dz
group_energy_density = [fields[name] / cell_volume for name in group_names]


def mirror_rz(values):
    """Mirror an RZ field around the symmetry axis for visualization."""
    return np.concatenate((np.flip(values, axis=0), values), axis=0).T


temperature_ev = fields["Te"] * 8.617333262145e-5
total_energy_density = np.sum(group_energy_density, axis=0)
panels = [temperature_ev, total_energy_density, *group_energy_density]
titles = [
    "Electron temperature [eV]",
    "Total radiation energy density [J m$^{-3}$]",
    "Radiation: 0–100 eV [J m$^{-3}$]",
    "Radiation: 100–300 eV [J m$^{-3}$]",
    "Radiation: 300–1000 eV [J m$^{-3}$]",
    "Radiation: >1000 eV [J m$^{-3}$]",
]
extent_mm = (-1.0, 1.0, -1.5, 1.5)
wall_density = mirror_rz(fields["rho"])

plt.switch_backend("Agg")
figure, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
for axis, values, title in zip(axes.flat, panels, titles):
    mirrored = mirror_rz(values)
    positive = mirrored[mirrored > 0.0]
    value_max = np.max(positive)
    value_min = max(np.min(positive), value_max * 1.0e-7)
    image = axis.imshow(
        mirrored,
        origin="lower",
        extent=extent_mm,
        aspect="equal",
        cmap="inferno",
        norm=LogNorm(vmin=value_min, vmax=value_max),
    )
    axis.contour(
        wall_density,
        levels=[0.5 * np.max(wall_density)],
        colors="cyan",
        linewidths=0.7,
        origin="lower",
        extent=extent_mm,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("r [mm]")
    axis.set_ylabel("z [mm]")
    figure.colorbar(image, ax=axis, shrink=0.78)

figure.suptitle(
    "WarpX hybrid multigroup hohlraum demonstration\n"
    "20 steps, cell-local implicit LTE exchange",
    fontsize=14,
)
figure.savefig(args.output, dpi=180)
print(f"wrote {args.output}")
