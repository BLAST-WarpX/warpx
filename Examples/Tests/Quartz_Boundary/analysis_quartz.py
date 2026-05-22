#!/usr/bin/env python3

# Analysis script for 3D Quartz Boundary Condition Test
# This script loads the simulation output, extracts Ey and Bx along the z direction,
# and visualizes the field distribution near the quartz boundary.

import sys

import matplotlib
import numpy as np
import yt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

yt.funcs.mylog.setLevel(50)

# Usage: python analysis_quartz.py <output_plotfile>
if len(sys.argv) < 2:
    print("Usage: python analysis_quartz.py <output_plotfile>")
    sys.exit(1)

plotfile = sys.argv[1]

ds = yt.load(plotfile)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

# Extract field data
Ey = data[("mesh", "Ey")].to_ndarray()
Bx = data[("mesh", "Bx")].to_ndarray()
z = data["z"].to_ndarray()

# Take a slice at the center of x and y
dim_x, dim_y, dim_z = Ey.shape
mid_x = dim_x // 2
mid_y = dim_y // 2
Ey_z = Ey[mid_x, mid_y, :]
Bx_z = Bx[mid_x, mid_y, :]
z_line = z[mid_x, mid_y, :]

# Plot Ey and Bx along z
title = "Quartz Boundary Test: Ey and Bx along z (center x, y)"
plt.figure(figsize=(10, 6))
plt.plot(z_line, Ey_z, label="Ey (center)", color="b")
plt.plot(z_line, Bx_z, label="Bx (center)", color="g")
plt.xlabel("z (m)")
plt.ylabel("Field Value")
plt.title(title)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("quartz_boundary_fields_z.png", dpi=150)

# Simple numerical check: field amplitude and variation
max_Ey = np.max(np.abs(Ey_z))
min_Ey = np.min(np.abs(Ey_z))
std_Ey = np.std(Ey_z)
print(
    f"Max |Ey|: {max_Ey:.2e} V/m, Min |Ey|: {min_Ey:.2e} V/m, Std |Ey|: {std_Ey:.2e} V/m"
)

# Assert field is not zero everywhere (sanity check)
assert max_Ey > 1e4, "Ey field amplitude too small, check simulation setup."
assert std_Ey > 1e3, "Ey field variation too small, check simulation setup."

print("Analysis complete. Field plot saved as quartz_boundary_fields_z.png.")
