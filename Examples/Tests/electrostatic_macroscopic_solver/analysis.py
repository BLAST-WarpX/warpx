# -*- coding: utf-8 -*-

import matplotlib
import numpy as np

from pywarpx import picmi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

filename = "./diags/diag1/"

ts = OpenPMDTimeSeries(filename)
iterations = ts.iterations
phi, info = ts.get_field("phi", iteration=iterations[-1], plot=False)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(phi, origin="lower", aspect="auto", cmap="hsv")
fig.colorbar(im, ax=ax)
fig.savefig("./phi.png", dpi=300, bbox_inches="tight")

Esim, info = ts.get_field("E", coord="z", iteration=iterations[-1], plot=False)
Esim = Esim[:, 4]

constants = picmi.constants
nz = 128
voltage = -1000
gap = 0.1
tVac = gap / 2
epsilon_r = 10

Ediel = voltage / (epsilon_r * (gap - tVac) + tVac)
Evac = voltage / ((gap - tVac) + tVac / epsilon_r)

z = np.linspace(0.0, gap, nz)
Eth = (z > tVac) * Ediel + (z <= tVac) * Evac

plt.figure()
plt.plot(z, Eth, "k-", label="Theory")
plt.plot(z, Esim, "r--", label="WarpX")
plt.xlabel("z (m)")
plt.ylabel(r"Electric Field (V/m)")
plt.title("Partially Filled Capacitor: Electric Field")
plt.legend()
plt.tight_layout()
plt.savefig("electricField.png")


rel_err = np.abs(Esim - Eth) / np.abs(Eth)
rms_rel_err = np.sqrt(np.mean(rel_err**2))
print(f"Max relative error: {rel_err.max():0.2e}")
print(f"RMS relative error: {rms_rel_err:0.2e}")
tolerance = 1e-6
assert rms_rel_err < tolerance, (
    f"RMS relative error {rms_rel_err} % exceeds tolerance {tolerance}"
)
