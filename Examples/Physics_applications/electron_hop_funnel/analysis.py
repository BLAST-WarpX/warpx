# -*- coding: utf-8 -*-

import matplotlib

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
