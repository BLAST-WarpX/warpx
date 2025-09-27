#!/usr/bin/env python3

import re

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, m_e

# default relative tolerance
rtol = 5e-10


# extract numbers from a string
def find_num_in_line(line):
    items = re.findall("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?", line)
    fitems = [float(it) for it in items]
    if len(fitems) == 1:
        return fitems[0]
    else:
        return fitems


def check_energy_conservation():
    ekin_data = np.loadtxt("diags/reducedfiles/ParticleEnergy.txt")
    ekin_photon1 = ekin_data[:, 3]
    ekin_photon2 = ekin_data[:, 5]
    ekin_electron1 = ekin_data[:, 4]
    ekin_electron2 = ekin_data[:, 6]
    num_data = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")
    num_phys_electron = num_data[:, 9] + num_data[:, 11]
    total_energy = (
        ekin_photon1
        + ekin_photon2
        + ekin_electron1
        + ekin_electron2
        + m_e * c**2 * num_phys_electron
    )
    assert np.all(np.isclose(total_energy, total_energy[0], rtol=rtol, atol=0.0))


def check_momentum_conservation():
    total_momentum_x = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 2]
    total_momentum_y = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 3]
    total_momentum_z = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 4]
    assert np.allclose(total_momentum_x, total_momentum_x[0], rtol=rtol, atol=1e-12)
    assert np.allclose(total_momentum_y, total_momentum_y[0], rtol=rtol, atol=1e-12)
    assert np.allclose(total_momentum_z, total_momentum_z[0], rtol=rtol, atol=1e-12)


def check_charge_conservation():
    rho_max = np.loadtxt("diags/reducedfiles/RhoMaximum.txt")[:, 2]
    assert np.allclose(rho_max, rho_max[0], rtol=rtol, atol=0.0)

    series = OpenPMDTimeSeries("diags/diag1")

    rho_end, info = series.get_field("rho", iteration=series.iterations[-1])

    rho_start, info = series.get_field("rho", iteration=series.iterations[0])

    assert np.allclose(rho_start, rho_end, rtol=rtol, atol=0.0)
