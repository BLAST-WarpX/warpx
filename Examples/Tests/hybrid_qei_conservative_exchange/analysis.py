#!/usr/bin/env python3
"""Check exact two-temperature exchange and conservative ion realization."""

import argparse

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)

ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN = 1.380649e-23
KELVIN_PER_EV = ELEMENTARY_CHARGE / BOLTZMANN
PROTON_MASS = 1.67262192369e-27
GAMMA_E = 5.0 / 3.0
NU_EI = 1.0e6
DT = 1.0e-8


def load_fields(step):
    dataset = yt.load(f"diags/plt{step:06d}")
    grid = dataset.covering_grid(
        level=0,
        left_edge=dataset.domain_left_edge,
        dims=dataset.domain_dimensions,
    )
    names = (
        "rho",
        "ni_charge_fp_ions",
        "Te",
        "hybrid_qei_ion_temperature_fp_ions",
        "hybrid_qei_electron_energy_fp",
        "hybrid_qei_electron_energy_fp_ions",
        "hybrid_qei_ion_energy_cc_ions",
        "hybrid_qei_electron_energy_cumulative_fp",
    )
    fields = {
        name: np.asarray(grid["boxlib", name].v, dtype=np.float64).squeeze()
        for name in names
    }
    return dataset, fields


def particle_moments(dataset):
    particles = dataset.all_data()
    weights = np.asarray(particles["ions", "particle_weight"].v, dtype=np.float64)
    momenta = np.stack(
        [
            np.asarray(
                particles["ions", f"particle_momentum_{axis}"].to_value("kg*m/s"),
                dtype=np.float64,
            )
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    kinetic_energy = float(
        np.sum(weights * np.sum(momenta * momenta, axis=1) / (2.0 * PROTON_MASS))
    )
    total_momentum = np.sum(weights[:, None] * momenta, axis=0)
    momentum_scale = float(np.sum(weights * np.linalg.norm(momenta, axis=1)))
    return kinetic_energy, total_momentum, momentum_scale


def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=("SINGLE", "DOUBLE"), default="DOUBLE")
parser.add_argument(
    "--particle-precision", choices=("SINGLE", "DOUBLE"), default="DOUBLE"
)
args = parser.parse_args()

initial_dataset, initial = load_fields(0)
final_dataset, final = load_fields(1)
ion_energy_initial, momentum_initial, momentum_scale_initial = particle_moments(
    initial_dataset
)
ion_energy_final, momentum_final, momentum_scale_final = particle_moments(final_dataset)

electron_density = initial["rho"] / ELEMENTARY_CHARGE
ion_density = final["ni_charge_fp_ions"] / ELEMENTARY_CHARGE
electron_temperature_initial = weighted_mean(
    initial["Te"] / KELVIN_PER_EV, electron_density
)
ion_temperature_initial = weighted_mean(
    final["hybrid_qei_ion_temperature_fp_ions"], ion_density
)
electron_temperature_final = weighted_mean(
    final["Te"] / KELVIN_PER_EV, electron_density
)

electron_capacity = 1.0 / (GAMMA_E - 1.0)
ion_capacity = 1.5
equilibrium_temperature = (
    electron_capacity * electron_temperature_initial
    + ion_capacity * ion_temperature_initial
) / (electron_capacity + ion_capacity)
decay_rate = 3.0 * NU_EI * (1.0 / electron_capacity + 1.0 / ion_capacity)
analytic_electron_temperature = equilibrium_temperature + (
    ion_capacity / (electron_capacity + ion_capacity)
) * (electron_temperature_initial - ion_temperature_initial) * np.exp(-decay_rate * DT)
temperature_error = abs(
    electron_temperature_final - analytic_electron_temperature
) / abs(analytic_electron_temperature)

cell_volume = 1.0e-6
electron_energy_change = float(
    np.sum(
        electron_density
        * BOLTZMANN
        / (GAMMA_E - 1.0)
        * (final["Te"] - initial["Te"])
        * cell_volume
    )
)
electron_ledger = float(np.sum(final["hybrid_qei_electron_energy_fp"] * cell_volume))
species_electron_ledger = float(
    np.sum(final["hybrid_qei_electron_energy_fp_ions"] * cell_volume)
)
cumulative_electron_ledger = float(
    np.sum(final["hybrid_qei_electron_energy_cumulative_fp"] * cell_volume)
)
mapped_ion_request = float(np.sum(final["hybrid_qei_ion_energy_cc_ions"] * cell_volume))
ion_energy_change = ion_energy_final - ion_energy_initial

energy_scale = max(
    abs(electron_energy_change), abs(ion_energy_change), np.finfo(float).tiny
)
ledger_error = (
    max(
        abs(electron_ledger - electron_energy_change),
        abs(species_electron_ledger - electron_ledger),
        abs(cumulative_electron_ledger - electron_ledger),
    )
    / energy_scale
)
mapping_error = abs(mapped_ion_request + species_electron_ledger) / energy_scale
ion_realization_error = abs(ion_energy_change - mapped_ion_request) / energy_scale
pair_energy_error = abs(ion_energy_change + electron_energy_change) / energy_scale

momentum_change = momentum_final - momentum_initial
momentum_error = float(
    np.linalg.norm(momentum_change)
    / max(momentum_scale_initial, momentum_scale_final, np.finfo(float).tiny)
)

if args.precision == "SINGLE" or args.particle_precision == "SINGLE":
    temperature_tolerance = 5.0e-5
    energy_tolerance = 8.0e-5
    momentum_tolerance = 2.0e-5
else:
    temperature_tolerance = 5.0e-12
    energy_tolerance = 5.0e-8
    momentum_tolerance = 5.0e-12

print(f"analytic final electron temperature [eV]: {analytic_electron_temperature:.16e}")
print(f"WarpX final electron temperature [eV]:   {electron_temperature_final:.16e}")
print(f"electron temperature relative error:     {temperature_error:.16e}")
print(f"electron EOS energy change [J/m]:        {electron_energy_change:.16e}")
print(f"electron aggregate ledger [J/m]:         {electron_ledger:.16e}")
print(f"electron species ledger [J/m]:           {species_electron_ledger:.16e}")
print(f"mapped ion request [J/m]:                 {mapped_ion_request:.16e}")
print(f"realized ion energy change [J/m]:         {ion_energy_change:.16e}")
print(f"ledger relative error:                    {ledger_error:.16e}")
print(f"mapping relative error:                   {mapping_error:.16e}")
print(f"ion realization relative error:           {ion_realization_error:.16e}")
print(f"electron-ion pair relative error:         {pair_energy_error:.16e}")
print(f"ion momentum relative change:             {momentum_error:.16e}")

assert temperature_error <= temperature_tolerance
assert ledger_error <= energy_tolerance
assert mapping_error <= energy_tolerance
assert ion_realization_error <= energy_tolerance
assert pair_energy_error <= energy_tolerance
assert momentum_error <= momentum_tolerance
