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

max_grid_size = 48
nx = max_grid_size
ny = max_grid_size
nz = max_grid_size

xmin = -1
xmax = 1
ymin = xmin
ymax = xmax
zmin = 0
zmax = 5

#################################
############ NUMERICS ###########
#################################

verbose = 1
dt = 4.4e-7
use_filter = 0

#################################
######## INITIAL FIELD ##########
#################################

initial_field = picmi.LoadInitialField(
    read_fields_from_path="../../../../openPMD-example-datasets/example-femm-3d.h5",
    load_E=False,
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)
solver = picmi.ElectrostaticSolver(grid=grid)

#################################
######### DIAGNOSTICS ###########
#################################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["Bx", "By", "Bz"],
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
    warpx_serialize_initial_conditions=False,
    warpx_do_dynamic_scheduling=False,
    warpx_use_filter=use_filter,
    time_step_size=dt,
)

sim.add_applied_field(initial_field)

sim.add_diagnostic(field_diag)

# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

#################################
##### SIMULATION EXECUTION ######
#################################

sim.step(1)

#################################
##### SIMULATION ANALYSIS ######
#################################

filename = "diags/diag1000001"

ds = yt.load(filename)
grid0 = ds.index.grids[0]

dBxdx = (
    grid0["raw", "Bx_aux"].v[:, :, :, 1] - grid0["raw", "Bx_aux"].v[:, :, :, 0]
) / grid0.dds.v[0]
dBydy = (
    grid0["raw", "By_aux"].v[:, :, :, 1] - grid0["raw", "By_aux"].v[:, :, :, 0]
) / grid0.dds.v[1]
dBzdz = (
    grid0["raw", "Bz_aux"].v[:, :, :, 1] - grid0["raw", "Bz_aux"].v[:, :, :, 0]
) / grid0.dds.v[2]

divB = dBxdx + dBydy + dBzdz

import matplotlib.pyplot as plt

plt.imshow(np.log10(np.abs(divB[:, 24, :])))
plt.title("log10(|div(B)|)")
plt.colorbar()
plt.savefig("divb.png")

error = np.sqrt((divB[2:-2, 2:-2, 2:-2] ** 2).sum())
tolerance = 1e-12

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
