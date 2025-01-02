#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV

mpl.use("Agg")
mpl.rcParams.update({"font.size": 18})

MeV = 1e6 * eV

# open the BoundaryScrapingDiagnostic that represents the detector
series = OpenPMDTimeSeries("./diags/screen/particles_at_zhi/")
# open the Full diagnostic at time zero
series0 = OpenPMDTimeSeries("./diags/diag0/")
# we use the data at time 0 to retrieve the initial energy
# of all the particles the boundary

# timesteps and real times
it = series.iterations
time = series.t  # s
N_iterations = len(it)

# list of species names
species = series.avail_species
N_species = len(species)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), dpi=300)

# some stuff for plotting
vmin = 0
vmax = 50
cmap = ["Reds", "Greens", "Blues"]

# loop through the species
for s in range(N_species):
    print(species[s])

    # arrays of positions and energies
    X, Y, E = [], [], []
    for i in range(N_iterations):
        # get particles at detector location
        x, y, z, ids = series.get_particle(
            ["x", "y", "z", "id"], iteration=it[i], species=species[s], plot=False
        )
        # get particles at initialization
        uz0, ids0, m = series0.get_particle(
            ["uz", "id", "mass"],
            iteration=series0.iterations[0],
            species=species[s],
            plot=False,
        )

        indeces = np.where(np.in1d(ids0, ids))[0]

        E = np.append(E, 0.5 * m[indeces] * (uz0[indeces] * c) ** 2 / MeV)
        X = np.append(X, x)
        Y = np.append(Y, y)
    print(np.min(E), np.max(E))

    # sort particles according to energy for nicer plot
    sorted_indeces = np.argsort(E)
    ax.scatter(
        X[sorted_indeces],
        Y[sorted_indeces],
        c=E[sorted_indeces],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap[s],
    )
    sorted_indeces = np.argsort(E)
    ax.scatter(
        X[sorted_indeces],
        Y[sorted_indeces],
        c=E[sorted_indeces],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap[s],
    )

# dummy plot just to have a neutral colorbar
im = ax.scatter(np.nan, np.nan, c=np.nan, cmap="Greys_r", vmin=vmin, vmax=vmax)
plt.colorbar(im, label="E [MeV]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

plt.tight_layout()
fig.savefig("detect.png", dpi=300)
plt.close()
