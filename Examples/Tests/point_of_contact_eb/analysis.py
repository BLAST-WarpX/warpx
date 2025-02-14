#!/usr/bin/env python

"""
This script tests the coordinates of the point of contact of an electron hitting a sphere in 3D.
It compares the numerical results with the analytical solutions.
The sphere is centered on O and has a radius of 0.2 (EB)
The electron is initially at: (-0.25,0,0) and moves with a normalized momentum: (1,0.5,0)
An input file inputs_test_3d_point_of_contact_eb is used.
"""

import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries

yt.funcs.mylog.setLevel(0)

ts_scraping = OpenPMDTimeSeries("./diags/diag2/particles_at_eb/")

it = ts_scraping.iterations
step_scraped, delta, x, y, z, nx, ny, nz = ts_scraping.get_particle(
    ["stepScraped", "deltaTimeScraped", "x", "y", "z", "nx", "ny", "nz"],
    species="electron",
    iteration=it,
)
delta_reduced = delta[0] * 1e10

# Analytical results calculated
x_analytic = -0.1983
y_analytic = 0.02584
z_analytic = 0.0000
nx_analytic = -0.99
ny_analytic = 0.13
nz_analytic = 0.0

# result obtained by analysis of simulations
step_ref = 3
delta_reduced_ref = 0.59

print("NUMERICAL coordinates of the point of contact:")
print(
    "step_scraped=%d, time_stamp=%5.4f e-10, x=%5.4f, y=%5.4f, z=%5.4f, nx=%5.4f, ny=%5.4f, nz=%5.4f"
    % (step_scraped[0], delta_reduced, x[0], y[0], z[0], nx[0], ny[0], nz[0])
)
print("\n")
print("ANALYTICAL coordinates of the point of contact:")
print(
    "step_scraped=%d, time_stamp=%5.4f e-10, x=%5.4f, y=%5.4f, z=%5.4f, nx=%5.4f, ny=%5.4f, nz=%5.4f"
    % (
        step_ref,
        delta_reduced_ref,
        x_analytic,
        y_analytic,
        z_analytic,
        nx_analytic,
        ny_analytic,
        nz_analytic,
    )
)

tolerance = 0.001
tolerance_t = 0.01
tolerance_n = 0.01
print("tolerance = " + str(tolerance * 100) + "%")
print("tolerance for the time = " + str(tolerance_t * 100) + "%")
print("tolerance for the normal components = " + str(tolerance_n * 100) + "%")

diff_step = np.abs((step_scraped[0] - step_ref) / step_ref)
diff_delta = np.abs((delta_reduced - delta_reduced_ref) / delta_reduced_ref)
diff_x = np.abs((x[0] - x_analytic) / x_analytic)
diff_y = np.abs((y[0] - y_analytic) / y_analytic)
diff_nx = np.abs((nx[0] - nx_analytic) / nx_analytic)
diff_ny = np.abs((ny[0] - ny_analytic) / ny_analytic)

print("percentage error for x = %5.4f %%" % (diff_x * 100))
print("percentage error for y = %5.4f %%" % (diff_y * 100))
print("percentage error for nx = %5.2f %%" % (diff_nx * 100))
print("percentage error for ny = %5.2f %%" % (diff_ny * 100))
print("nz = %5.2f " % (nz[0]))

assert (
    (diff_x < tolerance)
    and (diff_y < tolerance)
    and (np.abs(z[0]) < 1e-8)
    and (diff_step < 1e-8)
    and (diff_delta < tolerance_t)
    and (diff_nx < tolerance_n)
    and (diff_ny < tolerance_n)
    and (np.abs(nz) < 1e-8)
), "Test point_of_contact did not pass"
