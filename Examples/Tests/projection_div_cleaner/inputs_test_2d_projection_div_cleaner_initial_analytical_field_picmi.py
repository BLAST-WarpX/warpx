#!/usr/bin/env python3
#
# --- Input file for loading initial field from openPMD file.

from pywarpx import picmi

constants = picmi.constants

import numpy as np
import yt

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 300

max_grid_size = 128
nx = max_grid_size
nz = max_grid_size

xmin = 0
xmax = 10
zmin = 0
zmax = 10

#################################
############ NUMERICS ###########
#################################

verbose = 1
dt = 1e-12
use_filter = 0

#################################
######## INITIAL FIELD ##########
#################################

B0 = 1
dB = 0.1
k1 = 1
k2 = 1
phi = np.zeros(shape=(8,))

# Analytical field expressions
Bx = f"""
    {dB} * {B0} * (
        1/sqrt(2)*cos({k1}*x + {k1}*z + {phi[4]})
        +1/sqrt(2)*cos(-{k1}*x - {k1}*z + {phi[5]})
        +1/sqrt(2)*cos({k1}*x - {k1}*z + {phi[6]})
        +1/sqrt(2)*cos(-{k1}*x + {k1}*z + {phi[7]})
    )
"""
By = f"{B0}"
Bz = f"""
    {dB} * {B0} * (
        cos({k1}*x + {phi[0]})
        +cos(-{k1}*x + {phi[1]})
        +cos({k2}*x + {phi[2]})
        +cos(-{k2}*x + {phi[3]})
        +1/sqrt(2)*cos({k1}*x + {k1}*z + {phi[4]})
        +1/sqrt(2)*cos(-{k1}*x - {k1}*z + {phi[5]})
        +1/sqrt(2)*cos({k1}*x - {k1}*z + {phi[6]})
        +1/sqrt(2)*cos(-{k1}*x + {k1}*z + {phi[7]})
    )
"""

# Field initialization
B_ext = picmi.AnalyticInitialField(
    Bx_expression=Bx,
    By_expression=By,
    Bz_expression=Bz,
    warpx_do_initial_div_cleaning=True,
    warpx_projection_div_cleaner_rtol=5e-12,
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)
solver = picmi.HybridPICSolver(grid=grid, n0=1e6, Te=0.0)

#################################
######### DIAGNOSTICS ###########
#################################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["Bx", "By", "Bz", "divB"],
    warpx_plot_raw_fields=True,
    warpx_plot_raw_fields_guards=True,
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=verbose,
    warpx_serialize_initial_conditions=True,
    warpx_do_dynamic_scheduling=False,
    warpx_use_filter=use_filter,
    time_step_size=dt,
)

sim.add_applied_field(B_ext)

sim.add_diagnostic(field_diag)

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

#################################
##### SIMULATION EXECUTION ######
#################################

# Simulation is not stepped since the field load is not self-consistent with
# the currents. This is just testing the div cleaner for loading an analytical
# initial field with the Hybrid solver.

#################################
##### SIMULATION ANALYSIS ######
#################################

filename = "diags/diag1000000"

ds = yt.load(filename)
grid0 = ds.index.grids[0]

dBxdx = (
    grid0["raw", "Bx_aux"].v[:, :, 0, 1] - grid0["raw", "Bx_aux"].v[:, :, 0, 0]
) / grid0.dds.v[0]
dBzdz = (
    grid0["raw", "Bz_aux"].v[:, :, 0, 1] - grid0["raw", "Bz_aux"].v[:, :, 0, 0]
) / grid0.dds.v[1]

divB = dBxdx + dBzdz

import matplotlib.pyplot as plt

plt.imshow(np.log10(np.abs(divB[:, :])))
plt.title("log10(|div(B)|)")
plt.colorbar()
plt.savefig("divb.png")

error = np.sqrt((divB[2:-2, 2:-2] ** 2).sum())
tolerance = 5e-12

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
