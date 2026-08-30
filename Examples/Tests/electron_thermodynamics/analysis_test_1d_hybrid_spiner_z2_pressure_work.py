#!/usr/bin/env python3
"""Validate fixed-Z=2 SP5 conservative pressure work."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yt

ELEMENTARY_CHARGE = 1.602176634e-19
FORMULA_MASS = 2.1618558138269478e-26
CV0_CGS = 2.0e8
CV_SLOPE_CGS = 1.0e5


def field_key(dataset, name: str):
    matches = [field for field in dataset.field_list if field[1] == name]
    if len(matches) != 1:
        raise KeyError(f"expected one field named {name!r}, got {matches}")
    return matches[0]


def load_cell_field(dataset, name: str) -> np.ndarray:
    grid = dataset.covering_grid(
        level=0,
        left_edge=dataset.domain_left_edge,
        dims=dataset.domain_dimensions,
    )
    return np.asarray(grid[field_key(dataset, name)], dtype=np.float64).squeeze()


def load_particle_energy(path: Path) -> dict[int, float]:
    lines = path.read_text().splitlines()
    header = next((line for line in lines if line.startswith("#")), None)
    if header is None:
        raise ValueError(f"missing ParticleEnergy header in {path}")
    labels = re.findall(r"\[\d+\]([^\s]+)", header)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if len(labels) != data.shape[1]:
        raise ValueError("ParticleEnergy header/data width mismatch")
    columns = {label: index for index, label in enumerate(labels)}
    if "step()" not in columns or "ions(J)" not in columns:
        raise ValueError(f"ParticleEnergy labels omit step()/ions(J): {labels}")
    return {
        int(round(row[columns["step()"]])): float(row[columns["ions(J)"]])
        for row in data
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_plot", type=Path)
    parser.add_argument("final_plot", type=Path)
    parser.add_argument("particle_energy", type=Path)
    args = parser.parse_args()

    yt.funcs.mylog.setLevel("ERROR")
    initial = yt.load(args.initial_plot)
    final = yt.load(args.final_plot)
    rho_charge = load_cell_field(initial, "rho")
    initial_temperature = load_cell_field(initial, "Te")
    final_rho_charge = load_cell_field(final, "rho")
    final_temperature = load_cell_field(final, "Te")
    final_pressure = load_cell_field(final, "Pe")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            rho_charge,
            initial_temperature,
            final_rho_charge,
            final_temperature,
            final_pressure,
        )
    ):
        raise AssertionError("pressure-work diagnostics contain non-finite values")
    if not np.all(final_temperature > 0.0) or not np.all(final_pressure > 0.0):
        raise AssertionError("pressure-work endpoint must have positive Te and Pe")

    material_density = rho_charge * FORMULA_MASS / (2.0 * ELEMENTARY_CHARGE)
    minimum_density = float(np.min(material_density))
    maximum_density = float(np.max(material_density))
    if not minimum_density > 5000.0:
        raise AssertionError(
            f"material-density minimum is too small: {minimum_density}"
        )
    if not maximum_density < 9000.0:
        raise AssertionError(
            f"material-density maximum is too large: {maximum_density}"
        )
    if not 2.0 * maximum_density > 15500.0:
        raise AssertionError(
            "density sentinel no longer excludes the historical Z scaling"
        )

    domain_length = float(
        (initial.domain_right_edge[0] - initial.domain_left_edge[0]).to_value("m")
    )
    cell_size = domain_length / int(initial.domain_dimensions[0])
    # The step-zero plot precedes nonlinear-U initialization, so it cannot be
    # one endpoint of an energy ledger.  Use the independently manufactured
    # fixture EOS only to scale the required nonzero particle-work signal.
    initial_specific_energy_cgs = (
        CV0_CGS * initial_temperature + 0.5 * CV_SLOPE_CGS * initial_temperature**2
    )
    initial_energy_density = material_density * 1.0e-4 * initial_specific_energy_cgs
    initial_total_energy = float(np.sum(initial_energy_density) * cell_size)
    particle_energy = load_particle_energy(args.particle_energy)
    if 0 not in particle_energy or 1 not in particle_energy:
        raise AssertionError("ParticleEnergy output omits step 0 or step 1")
    delta_ion_energy = particle_energy[1] - particle_energy[0]
    if not initial_total_energy > 0.0:
        raise AssertionError("initial electron energy must be positive")
    if not delta_ion_energy > 0.0:
        raise AssertionError("pressure work did not increase ion kinetic energy")

    signal = delta_ion_energy / initial_total_energy
    if not signal >= 1.0e-9:
        raise AssertionError(f"pressure-work signal is too small: {signal}")

    print(
        json.dumps(
            {
                "material_density_min_kg_m3": minimum_density,
                "material_density_max_kg_m3": maximum_density,
                "initial_electron_energy_scale_J_m2": initial_total_energy,
                "delta_ion_energy_J_m2": delta_ion_energy,
                "particle_work_relative_to_initial_electron_energy": signal,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
