"""
This script prepares the inputs for the test_3d_focusing_gaussian_beam_from_openpmd script.

It generates a set of particles and saves in an openPMD file, to be read as input by the WarpX simulation.
"""

import numpy as np
from scipy.constants import c, e, m_e

np.random.seed(0)

# Beam properties

mc2 = m_e * c * c
nano = 1.0e-9
micro = 1.0e-6
GeV = e * 1.0e9
energy = 125.0 * GeV
gamma = energy / mc2
npart = 2.0e10
nmacropart = 2000000
charge = e * npart
sigmax = 516.0 * nano
sigmay = 7.7 * nano
sigmaz = 300.0 * micro
ux = 0.0
uy = 0.0
uz = gamma
emitx = 50 * micro
emity = 20 * nano
emitz = 0.0
dux = emitx / sigmax
duy = emity / sigmay
duz = emitz / sigmaz
focal_distance = 4 * sigmaz

# Generate arrays of particle quantities using numpy

# Generate Gaussian distribution (corresponds to the beam at focus)
x = np.random.normal(0, sigmax, nmacropart)
y = np.random.normal(0, sigmay, nmacropart)
z = np.random.normal(0, sigmaz, nmacropart)
ux = np.random.normal(ux, dux, nmacropart)
uy = np.random.normal(uy, duy, nmacropart)
uz = np.random.normal(uz, duz, nmacropart)

# Change transverse beam positions, in way consistent with
# the beam being focused to focal_distance downstream
x = x - (focal_distance - z) * ux / uz
y = y - (focal_distance - z) * uy / uz

# Write particles to file using openPMD API

from openpmd_api import (
    Access_Type,
    Dataset,
    Mesh_Record_Component,
    Series,
)

SCALAR = Mesh_Record_Component.SCALAR

# open file for writing
f = Series("openpmd_generated_particles.h5", Access_Type.create)
f.set_meshes_path("fields")
f.set_particles_path("particles")
cur_it = f.iterations[0]
electrons = cur_it.particles["electrons"]

# All particle quantities will be written as float
d = Dataset(np.dtype("float64"), extent=(nmacropart,))

# The weight is the number of physical particles per macro particle.
# In this particular case, it is the same for all macro particles.
electrons["weighting"].reset_dataset(d)
electrons["weighting"][SCALAR].make_constant(npart / nmacropart)

# Write mass and charge
electrons["mass"].reset_dataset(d)
electrons["mass"][SCALAR].make_constant(m_e)
electrons["charge"].reset_dataset(d)
electrons["charge"][SCALAR].make_constant(-e)

# Write particle positions
electrons["position"]["x"].reset_dataset(d)
electrons["position"]["x"].store_chunk(x)
electrons["position"]["y"].reset_dataset(d)
electrons["position"]["y"].store_chunk(y)
electrons["position"]["z"].reset_dataset(d)
electrons["position"]["z"].store_chunk(z)

# Write particle momenta
electrons["momentum"]["x"].reset_dataset(d)
electrons["momentum"]["x"].store_chunk(ux)
electrons["momentum"]["x"].unit_SI = m_e * c  # Convert from unitless (gamma*beta) to SI
electrons["momentum"]["y"].reset_dataset(d)
electrons["momentum"]["y"].store_chunk(uy)
electrons["momentum"]["y"].unit_SI = m_e * c  # Convert from unitless (gamma*beta) to SI
electrons["momentum"]["z"].reset_dataset(d)
electrons["momentum"]["z"].store_chunk(uz)
electrons["momentum"]["z"].unit_SI = m_e * c  # Convert from unitless (gamma*beta) to SI

# Write the particle offset
# (required by the openPMD standard but set to 0 here)
electrons["positionOffset"]["x"].reset_dataset(d)
electrons["positionOffset"]["x"].make_constant(0.0)
electrons["positionOffset"]["y"].reset_dataset(d)
electrons["positionOffset"]["y"].make_constant(0.0)
electrons["positionOffset"]["z"].reset_dataset(d)
electrons["positionOffset"]["z"].make_constant(0.0)

# at any point in time you may decide to dump already created output to
# disk note that this will make some operations impossible (e.g. renaming
# files)
f.flush()

# now the file is closed
del f
