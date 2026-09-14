#!/usr/bin/env python3

# Copyright 2022 Modern Electron, David Grote

# This script checks the time-averaged ion density profile of the 1D
# capacitive discharge (case 1) against the benchmark data published in
# Turner et al. (2013) - https://doi.org/10.1063/1.4775084

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Turner et al. (2013) benchmark, case 1: ion density (m^-3), averaged over
# time, on the 129-point grid spanning the 6.7 cm gap
# fmt: off
ref_density = np.array([
    3.615140e+13,  3.649310e+13,  3.700160e+13,  3.750370e+13,
    3.803490e+13,  3.858030e+13,  3.914140e+13,  3.971730e+13,
    4.033450e+13,  4.094000e+13,  4.156350e+13,  4.220910e+13,
    4.287840e+13,  4.357200e+13,  4.429100e+13,  4.503520e+13,
    4.578170e+13,  4.656380e+13,  4.737210e+13,  4.819220e+13,
    4.905750e+13,  4.993360e+13,  5.085470e+13,  5.179680e+13,
    5.277570e+13,  5.379100e+13,  5.486320e+13,  5.595360e+13,
    5.710280e+13,  5.829460e+13,  5.952200e+13,  6.082260e+13,
    6.220160e+13,  6.368110e+13,  6.520280e+13,  6.680680e+13,
    6.853560e+13,  7.039070e+13,  7.234350e+13,  7.443040e+13,
    7.664670e+13,  7.901410e+13,  8.159660e+13,  8.436080e+13,
    8.731620e+13,  9.051020e+13,  9.386320e+13,  9.742330e+13,
    1.010890e+14,  1.048750e+14,  1.087980e+14,  1.127630e+14,
    1.165890e+14,  1.202870e+14,  1.238340e+14,  1.271580e+14,
    1.300660e+14,  1.326790e+14,  1.348810e+14,  1.366620e+14,
    1.382180e+14,  1.393130e+14,  1.399380e+14,  1.404460e+14,
    1.404550e+14,  1.404750e+14,  1.400490e+14,  1.392120e+14,
    1.381080e+14,  1.367480e+14,  1.349150e+14,  1.326120e+14,
    1.300270e+14,  1.271080e+14,  1.238330e+14,  1.202510e+14,
    1.165870e+14,  1.127130e+14,  1.087740e+14,  1.049080e+14,
    1.010750e+14,  9.741190e+13,  9.389870e+13,  9.056250e+13,
    8.741220e+13,  8.444080e+13,  8.169930e+13,  7.911350e+13,
    7.669090e+13,  7.445670e+13,  7.240210e+13,  7.042020e+13,
    6.857700e+13,  6.684000e+13,  6.523480e+13,  6.370320e+13,
    6.226310e+13,  6.089210e+13,  5.959620e+13,  5.833490e+13,
    5.713550e+13,  5.599560e+13,  5.488050e+13,  5.383350e+13,
    5.281440e+13,  5.182910e+13,  5.087840e+13,  4.996970e+13,
    4.907640e+13,  4.821260e+13,  4.738490e+13,  4.659190e+13,
    4.581770e+13,  4.505920e+13,  4.431460e+13,  4.359860e+13,
    4.290670e+13,  4.222390e+13,  4.157240e+13,  4.093110e+13,
    4.031290e+13,  3.973250e+13,  3.915630e+13,  3.860640e+13,
    3.805990e+13,  3.752300e+13,  3.701210e+13,  3.651580e+13,
    3.618610e+13])
# fmt: on

density_data = np.load("ion_density_case_1.npy")
print(repr(density_data))

# The Turner benchmark above is tabulated on a 129-node grid (128 cells over
# the 6.7 cm gap). The test may run on a coarser grid, so interpolate the
# benchmark onto the simulation nodes before comparing.
gap = 0.067
z = np.linspace(0.0, gap, density_data.size)
z_ref = np.linspace(0.0, gap, ref_density.size)
ref_on_grid = np.interp(z, z_ref, ref_density)

# Plot the simulated ion density profile against the Turner benchmark.
plt.figure()
plt.plot(z_ref, ref_density, "k-", label="Turner et al. (2013)")
plt.plot(z, density_data, "r--o", markersize=3, label="WarpX")
plt.xlabel("z (m)")
plt.ylabel(r"Ion density (m$^{-3}$)")
plt.title("Capacitive discharge (case 1): time-averaged ion density")
plt.legend()
plt.tight_layout()
plt.savefig("ion_density_case_1.png")

# Compare with the benchmark. The two boundary nodes are excluded: at the
# absorbing walls the nodal charge deposition only sees half a cell, so the
# density there is a grid artifact (~half the physical value) rather than a
# physical disagreement with the (cell-averaged) benchmark.
rel_err = np.abs(density_data[1:-1] - ref_on_grid[1:-1]) / ref_on_grid[1:-1]
rms_rel_err = np.sqrt(np.mean(rel_err**2))
print(f"Max relative error (interior): {rel_err.max() * 100:.2f} %")
print(f"RMS relative error (interior): {rms_rel_err * 100:.2f} %")
tolerance = 0.06
assert rms_rel_err < tolerance, (
    f"RMS relative error {rms_rel_err * 100:.2f} % exceeds tolerance "
    f"{tolerance * 100:.2f} %"
)
