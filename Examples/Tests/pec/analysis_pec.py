#!/usr/bin/env python3

#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# This is a script that analyses the simulation results from
# the script `inputs_field_PEC_3d`. This simulates a sinusoidal wave.
# The electric field (Ey) is a standing wave due to the PEC boundary condition,
# and as a result, the minimum and maximum value after reflection would be two times the value at initialization due to constructive interference.
# Additionally, the value of Ey at the boundary must be equal to zero.
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(50)

import numpy as np

# this will be the name of the plot file
fn = sys.argv[1]

# Parameters (these parameters must match the parameters in `inputs.multi.rt`)
Ey_in = 1.0e5
E_th = 2.0 * Ey_in

# Read the file
ds = yt.load(fn)

t0 = ds.current_time.to_value()
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
Ey_array = data[("mesh", "Ey")].to_ndarray()
max_Ey_sim = Ey_array.max()
min_Ey_sim = Ey_array.min()
max_Ey_error = abs(max_Ey_sim - E_th) / abs(E_th)
min_Ey_error = abs(min_Ey_sim - (-E_th)) / abs(E_th)
print(
    "Ey_max is %s: Max Eth is %s : Max error for Ey_max: %.2e"
    % (max_Ey_sim, E_th, max_Ey_error)
)
print(
    "Ey_min is %s: Min Eth is %s : Max error for Ey_min: %.2e"
    % (min_Ey_sim, (-E_th), min_Ey_error)
)
error_rel = 0.0
max_Ey_error_rel = max(error_rel, max_Ey_error)
min_Ey_error_rel = max(error_rel, min_Ey_error)

# Plot the last field from the loop (Ez at iteration 40)
field = "Ey"
xCell = ds.domain_dimensions[0]
yCell = ds.domain_dimensions[1]
zCell = ds.domain_dimensions[2]
z_array = data["z"].to_ndarray()

plt.figure(figsize=(8, 4))
plt.plot(
    z_array[int(xCell / 2), int(yCell / 2), :]
    - z_array[int(xCell / 2), int(yCell / 2), int(zCell / 2)],
    Ey_array[int(xCell / 2), int(yCell / 2), :],
)
plt.ylim(-1.2e-3, 1.2e-3)
plt.ylim(-2.2e5, 2.2e5)
plt.xlim(-4.0e-6, 4.000001e-6)
plt.xticks(np.arange(-4.0e-6, 4.000001e-6, step=1e-6))
plt.xlabel("z (m)")
plt.ylabel(field + " (V/m)")
plt.tight_layout()
plt.savefig("Ey_pec_analysis.png")


tolerance_rel = 0.01

print("min_Ey_error_rel    : " + str(min_Ey_error_rel))
print("max_Ey_error_rel    : " + str(max_Ey_error_rel))
print("tolerance_rel: " + str(tolerance_rel))

assert max_Ey_error_rel < tolerance_rel
assert min_Ey_error_rel < tolerance_rel
