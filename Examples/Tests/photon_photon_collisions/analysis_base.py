#!/usr/bin/env python3

import re
import sys

import numpy as np
import yt
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
    ekin_photonA = ekin_data[:, 3]
    ekin_photonB = ekin_data[:, 4]
    ekin_electron = ekin_data[:, 5]
    ekin_positron = ekin_data[:, 6]
    num_data = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")
    num_phys_electron = num_data[:, 10]
    num_phys_positron = num_data[:, 11]
    field_energy = np.loadtxt("diags/reducedfiles/FieldEnergy.txt")[:, 2]
    total_energy = (
        ekin_photonA
        + ekin_photonB
        + ekin_electron
        + ekin_positron
        + m_e * c**2 * (num_phys_electron + num_phys_positron)
        + field_energy
    )
    assert np.all(np.isclose(total_energy, total_energy[0], rtol=rtol, atol=0.0))


def check_momentum_conservation():
    total_momentum_x = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 2]
    total_momentum_y = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 3]
    total_momentum_z = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")[:, 4]
    assert np.all(
        np.isclose(total_momentum_x, total_momentum_x[0], rtol=rtol, atol=0.0)
    )
    assert np.all(
        np.isclose(total_momentum_y, total_momentum_y[0], rtol=rtol, atol=0.0)
    )
    assert np.all(
        np.isclose(total_momentum_z, total_momentum_z[0], rtol=rtol, atol=0.0)
    )


def check_charge_conservation():
    rho_max = np.loadtxt("diags/reducedfiles/RhoMaximum.txt")[:, 2]
    assert np.all(np.isclose(rho_max, rho_max[0], rtol=rtol, atol=0.0))

    filename_end = sys.argv[1]
    filename_start = filename_end[:-4] + "0000"
    ds_end = yt.load(filename_end)
    ds_start = yt.load(filename_start)
    field_data_end = ds_end.covering_grid(
        level=0, left_edge=ds_end.domain_left_edge, dims=ds_end.domain_dimensions
    )
    field_data_start = ds_start.covering_grid(
        level=0, left_edge=ds_start.domain_left_edge, dims=ds_start.domain_dimensions
    )
    rho_start = field_data_start["rho"].to_ndarray()
    rho_end = field_data_end["rho"].to_ndarray()
    assert np.all(np.isclose(rho_start, rho_end, rtol=rtol, atol=0.0))
