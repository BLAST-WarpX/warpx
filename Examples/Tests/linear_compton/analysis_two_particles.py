#!/usr/bin/env python3

# This test checks that the linear Compton infrastructure is working ok.
# It initializes one electron and one photon in the same cell with a certain energy and weight.
# The two particles are not pushed, so that the keep colliding.
# It checks charge, energy, and momentum conservation.

import numpy as np
from analysis_base import (
    check_charge_conservation,
    check_energy_conservation,
    check_momentum_conservation,
)


# check that all the original electrons and photons have been completely scattered,
# i.e. all the electrons and photons are now in the species labelled "electron2" and "photon2"
def check_final_macroparticles():
    macro_photon1_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 3]
    macro_photon1_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 8]
    macro_electron1_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 4]
    macro_electron1_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 9]
    assert macro_photon1_number == macro_electron1_number == 0.0
    assert macro_photon1_weight == macro_electron1_weight == 0.0
    macro_electron2_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 6]
    macro_electron2_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 11]
    macro_photon2_number = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 5]
    macro_photon2_weight = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")[-1, 10]
    assert macro_electron2_number == macro_photon2_number == 2.0
    assert macro_electron2_weight == macro_photon2_weight == 1e14


def main():
    check_final_macroparticles()
    check_energy_conservation()
    check_momentum_conservation()
    check_charge_conservation()


if __name__ == "__main__":
    main()
