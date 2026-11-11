#!/usr/bin/env python3
#
# --- Efficacy test for the projection-based B-field divergence cleaner acting on an
# --- externally *applied* particle field (picmi.LoadAppliedField, i.e.
# --- particles.B_ext_particle_init_style = read_from_file) in electrostatic mode.
# ---
# --- This is the scenario from https://github.com/BLAST-WarpX/warpx/issues/6967, but with a
# --- deliberately divergent B field written to an openPMD file. We load it, enable the
# --- divergence cleaner, and check that the cleaner drives the discrete divergence of the
# --- loaded field to ~0 (efficacy), whereas the loaded field is strongly divergent to begin
# --- with.

import numpy as np
import openpmd_api as io
from mpi4py import MPI as mpi

from pywarpx import picmi

comm = mpi.COMM_WORLD

#################################
####### GENERAL PARAMETERS ######
#################################

# Cubic domain centered on the origin
Lx = Ly = Lz = 1.0
Nx = Ny = Nz = 32

xmin, xmax = -Lx / 2.0, Lx / 2.0
ymin, ymax = -Ly / 2.0, Ly / 2.0
zmin, zmax = -Lz / 2.0, Lz / 2.0

DX, DY, DZ = Lx / Nx, Ly / Ny, Lz / Nz

# Divergent applied field: Bx = B0 exp(-r^2 / (2 sigma^2)), By = Bz = 0
# -> div(B) = dBx/dx = -B0 (x / sigma^2) exp(-r^2 / (2 sigma^2)) != 0
B0 = 1.0
sigma = 0.2
field_file = "Bfield_divergent_applied.h5"

#################################
##### WRITE THE INPUT FIELD #####
#################################


def write_divergent_b_field(filename):
    """Write a divergent B field to an openPMD file readable by WarpX's external-field reader.

    The mesh "B" has cartesian geometry with components "x", "y", "z" and the data is laid out
    in (x, y, z) order, matching what Source/Initialization/ExternalField.cpp expects.
    """
    # File grid is a little larger than the simulation domain so that interpolation onto the
    # (staggered) WarpX grid, including ghost cells, is well defined everywhere.
    pad = 0.1
    nfx = nfy = nfz = 64
    xf = np.linspace(xmin - pad, xmax + pad, nfx)
    yf = np.linspace(ymin - pad, ymax + pad, nfy)
    zf = np.linspace(zmin - pad, zmax + pad, nfz)
    XF, YF, ZF = np.meshgrid(xf, yf, zf, indexing="ij")
    R2 = XF**2 + YF**2 + ZF**2
    Bx = B0 * np.exp(-R2 / (2.0 * sigma**2))
    By = np.zeros_like(Bx)
    Bz = np.zeros_like(Bx)

    series = io.Series(filename, io.Access.create)
    it = series.iterations[1]
    mesh = it.meshes["B"]
    mesh.geometry = io.Geometry.cartesian
    mesh.grid_spacing = [xf[1] - xf[0], yf[1] - yf[0], zf[1] - zf[0]]
    mesh.grid_global_offset = [xf[0], yf[0], zf[0]]
    mesh.axis_labels = ["x", "y", "z"]
    for comp, data in zip(["x", "y", "z"], [Bx, By, Bz]):
        data = np.ascontiguousarray(data.astype(np.float64))
        rc = mesh[comp]
        rc.position = [0.0, 0.0, 0.0]
        rc.reset_dataset(io.Dataset(data.dtype, data.shape))
        rc.store_chunk(data)
    series.flush()


# Single rank writes the field map, then everyone waits for it to be on disk.
if comm.rank == 0:
    write_divergent_b_field(field_file)
comm.Barrier()

#################################
######## APPLIED FIELD ##########
#################################

applied_field = picmi.LoadAppliedField(
    read_fields_from_path=field_file,
    load_E=False,
    load_B=True,
    warpx_do_initial_div_cleaning=True,
    warpx_projection_div_cleaner_rtol=1e-11,
)

#################################
###### GRID AND SOLVER ##########
#################################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[Nx, Ny, Nz],
    warpx_max_grid_size=32,
    warpx_blocking_factor=8,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
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
    data_list=["B"],
    warpx_format="plotfile",
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=1,
    verbose=1,
    time_step_size=1e-9,
    particle_shape=1,
)
sim.add_applied_field(applied_field)
sim.add_diagnostic(field_diag)

sim.initialize_inputs()
sim.initialize_warpx()
sim.step(1)

#################################
######### EFFICACY CHECK ########
#################################

# Read back the cleaned applied B field (the field the cleaner operated on).
Bxg = sim.fields.get("B_external_particle_field", dir="x", level=0)
Byg = sim.fields.get("B_external_particle_field", dir="y", level=0)
Bzg = sim.fields.get("B_external_particle_field", dir="z", level=0)

Bx_local = Bxg[(), (), ()]
By_local = Byg[(), (), ()]
Bz_local = Bzg[(), (), ()]

# Discrete (Yee) divergence at cell centers; the staggered components collapse to a common shape.
dBxdx = (Bx_local[1:, :, :] - Bx_local[:-1, :, :]) / DX
dBydy = (By_local[:, 1:, :] - By_local[:, :-1, :]) / DY
dBzdz = (Bz_local[:, :, 1:] - Bz_local[:, :, :-1]) / DZ
divB = dBxdx + dBydy + dBzdz

# Trim ghost cells / boundary rows before measuring.
div_after = np.sqrt((divB[2:-2, 2:-2, 2:-2] ** 2).sum())

# Analytic "before" estimate: divergence of the loaded Gaussian, sampled at cell centers,
# to confirm the input field really is strongly divergent (so the test is meaningful).
xc = np.linspace(xmin + DX / 2, xmax - DX / 2, Nx)
yc = np.linspace(ymin + DY / 2, ymax - DY / 2, Ny)
zc = np.linspace(zmin + DZ / 2, zmax - DZ / 2, Nz)
XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing="ij")
RC2 = XC**2 + YC**2 + ZC**2
divB_analytic = -B0 * (XC / sigma**2) * np.exp(-RC2 / (2.0 * sigma**2))
div_before = np.sqrt((divB_analytic[2:-2, 2:-2, 2:-2] ** 2).sum())

# Sanity: the field actually loaded and is non-trivial.
b_max = np.abs(Bx_local).max()

if comm.rank == 0:
    print(f"max|Bx| (loaded)          = {b_max:.6e}")
    print(f"L2 div(B) before (analytic) = {div_before:.6e}")
    print(f"L2 div(B) after  (cleaned)  = {div_after:.6e}")

# The applied field loaded and is non-trivial.
assert b_max > 0.1
# The input field really is strongly divergent ...
assert div_before > 1.0
# ... and the divergence cleaner scrubbed it down to ~0.
assert div_after < 1e-6
