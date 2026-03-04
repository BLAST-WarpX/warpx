#!/usr/bin/env python3

"""
This script analyzes the results of a 2D electrostatic simulation with collisions
between electrons and protons placed in the middle of the position push. It computes the
total energy (field + particle) variation over time and verifies that it is conserved
within a given tolerance, as well as that the final value of the field energy variation
is close to a reference (so-called equipartition) value. If collisions are placed before
the position push (standard algorithm), the tolerance on the total energy conservation
needs to be relaxed to one order of magnitude higher.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

sys.path.append("../../../../warpx/Tools/Parser/")
from input_file_parser import parse_input_file


def analyze(args: argparse.Namespace) -> None:
    # compute energies from the reduced diagnostics
    field_energy_data = np.loadtxt(fname=f"{args.path}/field_energy.txt", skiprows=1)
    particle_energy_data = np.loadtxt(
        fname=f"{args.path}/particle_energy.txt", skiprows=1
    )
    field_energy = field_energy_data[:, 2]
    particle_energy = particle_energy_data[:, 2]

    # compute the total energy variation
    total_energy = field_energy + particle_energy
    total_energy_error = (total_energy - total_energy[0]) / total_energy[0]

    # compute the reference equipartition value
    input_dict = parse_input_file("warpx_used_inputs")
    num_particles_per_cell_list = input_dict[
        "electrons.num_particles_per_cell_each_dim"
    ]
    num_particles_per_cell_array = np.array(
        [int(nppc) for nppc in num_particles_per_cell_list]
    )
    num_particles_per_cell = np.prod(num_particles_per_cell_array)
    equipartition_value = 1 / (6 * num_particles_per_cell + 1)

    # normalize the field energy variation
    electron_temperature = float(input_dict["my_constants.Te"][0])
    plasma_density = float(input_dict["my_constants.n0"][0])
    plasma_frequency = np.sqrt(
        plasma_density * constants.e**2 / (constants.m_e * constants.epsilon_0)
    )
    normalization_factor = (
        3
        * plasma_density
        * (electron_temperature * constants.eV)
        * (10 * constants.c / plasma_frequency) ** 2
    )
    field_energy_error = (field_energy - field_energy[0]) / normalization_factor

    # plot the results before raising exceptions
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    ax[0].plot(
        field_energy_data[:, 1] * plasma_frequency,
        total_energy_error,
        label="Total energy error",
    )
    ax[0].set_xlabel(r"$\omega_p t$")
    ax[0].set_ylabel(r"$\Delta U_{tot} / U_{tot}(t=0)$")
    ax[0].set_yscale("symlog", linthresh=1e-6)
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(
        field_energy_data[:, 1] * plasma_frequency,
        field_energy_error,
        label="Field energy error",
    )
    ax[1].hlines(
        y=equipartition_value,
        xmin=field_energy_data[0, 1] * plasma_frequency,
        xmax=field_energy_data[-1, 1] * plasma_frequency,
        colors="C1",
        linestyles="dashed",
        label="Equipartition value",
    )
    ax[1].set_xlabel(r"$\omega_p t$")
    ax[1].set_ylabel(r"$\Delta U_{field} / (3 n T_e (10 c / \omega_p)^2)$")
    ax[1].set_yscale("symlog", linthresh=1e-6)
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("energy_conservation.png", dpi=300)
    plt.close()

    print(f"Total energy error: \n{total_energy_error}")
    print(f"Field energy error: \n{field_energy_error}")
    print(f"Equipartition value: {equipartition_value}")

    # verify the total energy conservation
    total_energy_error_norm = np.max(np.abs(total_energy_error))
    relative_tolerance = 1e-5
    if total_energy_error_norm >= relative_tolerance:
        raise ValueError(
            f"Total energy conservation failed with a maximum relative error of {total_energy_error_norm}"
        )

    # compare the final value of the field energy variation to the reference equipartition value
    relative_tolerance = 1e-1
    # average over the last 100 time steps to reduce noise
    field_energy_error_avg = np.mean(field_energy_error[-100:])
    if not np.isclose(
        field_energy_error_avg, equipartition_value, rtol=relative_tolerance
    ):
        raise ValueError(
            f"Averaged field energy error {field_energy_error_avg} is not close to the reference equipartition value {equipartition_value}"
        )


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()

    # add arguments: output path
    parser.add_argument(
        "--path",
        help="path to output file(s)",
        required=True,
        type=str,
    )

    # parse arguments
    args = parser.parse_args()

    # run analysis
    analyze(args)
