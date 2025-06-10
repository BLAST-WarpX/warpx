#!/usr/bin/env python3

# This test checks that the linear Breit-Wheeler infrastructure is working ok.
# It initializes two photons in the same cell with a certain energy (above threshold) and weight.
# The two photons are not pushed, so that the keep colliding until they disappear
# upon having been completely converted into electron-positron pairs.
# It also checks charge, energy, and momentum conservation.

import numpy as np
from analysis_base import (
    check_charge_conservation,
    check_energy_conservation,
    check_momentum_conservation,
    find_num_in_line,
)


# get input parameters from warpx_used_inputs
def get_input_parameters():
    with open("./warpx_used_inputs", "rt") as f:
        lines = f.readlines()
        for line in lines:
            if "photonA.single_particle_weight" in line:
                w1 = find_num_in_line(line)
            if "photonB.single_particle_weight" in line:
                w2 = find_num_in_line(line)
    return (w1, w2)


# check that the photons have been completely transformed into pairs:
# because the event multiplier is 1, as soon as a linear Breit-Wheeler event occurs,
# the two photons must disappear and 2 electron-positron pairs must be generated
def check_final_macroparticles():
    (w1, w2) = get_input_parameters()
    macro_photonA_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 3]
    macro_photonA_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 8]
    macro_photonB_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 4]
    macro_photonB_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 9]
    assert macro_photonA_number == macro_photonB_number == 0.0
    assert macro_photonA_weight == macro_photonB_weight == 0.0
    macro_positron_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 6]
    macro_positron_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 11]
    macro_electron_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 5]
    macro_electron_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 10]
    assert macro_electron_number == macro_positron_number == 2.0
    assert macro_electron_weight == macro_positron_weight == w1 == w2


def main():
    check_final_macroparticles()
    check_energy_conservation()
    check_momentum_conservation()
    check_charge_conservation()


if __name__ == "__main__":
    main()
