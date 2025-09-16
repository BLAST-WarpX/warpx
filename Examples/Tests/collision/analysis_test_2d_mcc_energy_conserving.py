#!/usr/bin/env python3

import argparse
import sys

import numpy as np
from scipy import constants

sys.path.append("../../../../warpx/Tools/Parser/")
from input_file_parser import parse_input_file


def analyze(args: argparse.Namespace) -> None:
    # compute energies from the reduced diagnostics
    data_fields = np.loadtxt(fname=f"{args.path}/energy_fields.txt", skiprows=1)
    energy_fields_total = data_fields[:, 2]
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
    # normalize the total field energy variation
    temperature_electrons = float(input_dict["my_constants.Te"][0])
    plasma_density = float(input_dict["my_constants.n0"][0])
    plasma_frequency = np.sqrt(
        plasma_density * constants.e**2 / (constants.m_e * constants.epsilon_0)
    )
    normalization_factor = (
        3
        * plasma_density
        * (temperature_electrons * constants.eV)
        * (10 * constants.c / plasma_frequency) ** 2
    )
    delta_energy_fields_total_normalized = (
        energy_fields_total - energy_fields_total[0]
    ) / normalization_factor
    print(f"Equipartition value: {equipartition_value}")
    print(f"Normalized energy variation: {delta_energy_fields_total_normalized}")
    assert np.isclose(
        delta_energy_fields_total_normalized[-1],
        equipartition_value,
        rtol=1e-1,
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
