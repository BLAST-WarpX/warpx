#!/usr/bin/env python3

import numpy as np
from scipy.constants import elementary_charge

Q_eV = -1.66  # in [eV]
Q_J = Q_eV * elementary_charge  # in [J]

ekin_data = np.loadtxt("./diags/reducedfiles/EP.txt")
ekin_total = ekin_data[:, 2]
H_count = np.loadtxt("./diags/reducedfiles/PN.txt")[:, 3]
total_energy = ekin_total + Q_J * H_count

# Check energy and momentum conservation with for an absoluete tolerance (atol)
assert np.all(np.isclose(total_energy, total_energy[1], rtol=0.0, atol=1e-3))


def check_momentum_conservation():
    total_momentum_x = np.loadtxt("./diags/reducedfiles/PP.txt")[:, 2]
    total_momentum_y = np.loadtxt("./diags/reducedfiles/PP.txt")[:, 3]
    total_momentum_z = np.loadtxt("./diags/reducedfiles/PP.txt")[:, 4]
    assert np.allclose(total_momentum_x, total_momentum_x[0], rtol=0.0, atol=1e-7)
    assert np.allclose(total_momentum_y, total_momentum_y[0], rtol=0.0, atol=1e-7)
    assert np.allclose(total_momentum_z, total_momentum_z[0], rtol=0.0, atol=1e-7)


check_momentum_conservation()
