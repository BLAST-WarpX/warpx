#!/usr/bin/env python3
#
# --- Regression (smoke) test for https://github.com/BLAST-WarpX/warpx/issues/6967.
# ---
# --- A zero B field is written to an openPMD file and loaded via picmi.LoadAppliedField with an
# --- ElectrostaticSolver and warpx_do_initial_div_cleaning=True. Before the fix this aborted with
# ---   "MultiFabRegister::get name does not exist in register: Bfield_fp_external[dir=x][level=0]"
# --- because the cleaner targeted an (unallocated) grid field rather than the applied particle
# --- field. This test simply checks that initialization completes and the (trivially zero)
# --- divergence is ~0.

import numpy as np
import openpmd_api as io
from mpi4py import MPI as mpi

from pywarpx import picmi

comm = mpi.COMM_WORLD

Lx = Ly = Lz = 1.0
Nx = Ny = Nz = 16

xmin, xmax = -Lx / 2.0, Lx / 2.0
ymin, ymax = -Ly / 2.0, Ly / 2.0
zmin, zmax = -Lz / 2.0, Lz / 2.0

DX, DY, DZ = Lx / Nx, Ly / Ny, Lz / Nz

field_file = "Bfield_zero_applied.h5"


def write_zero_b_field(filename):
    """Write a zero B field to an openPMD file (mirrors the issue reproducer)."""
    pad = 0.1
    nf = 24
    xf = np.linspace(xmin - pad, xmax + pad, nf)
    yf = np.linspace(ymin - pad, ymax + pad, nf)
    zf = np.linspace(zmin - pad, zmax + pad, nf)
    data = np.zeros((nf, nf, nf), dtype=np.float64)

    series = io.Series(filename, io.Access.create)
    mesh = series.iterations[1].meshes["B"]
    mesh.geometry = io.Geometry.cartesian
    mesh.grid_spacing = [xf[1] - xf[0], yf[1] - yf[0], zf[1] - zf[0]]
    mesh.grid_global_offset = [xf[0], yf[0], zf[0]]
    mesh.axis_labels = ["x", "y", "z"]
    for comp in ["x", "y", "z"]:
        rc = mesh[comp]
        rc.position = [0.0, 0.0, 0.0]
        rc.reset_dataset(io.Dataset(data.dtype, data.shape))
        rc.store_chunk(np.ascontiguousarray(data))
    series.flush()
    del series


if comm.rank == 0:
    write_zero_b_field(field_file)
comm.Barrier()

applied_field = picmi.LoadAppliedField(
    read_fields_from_path=field_file,
    load_E=False,
    load_B=True,
    warpx_do_initial_div_cleaning=True,
)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[Nx, Ny, Nz],
    warpx_max_grid_size=16,
    warpx_blocking_factor=8,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "neumann"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)
solver = picmi.ElectrostaticSolver(grid=grid)

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["B"],
    warpx_format="plotfile",
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=1,
    verbose=1,
    time_step_size=1e-9,
    particle_shape=1,
)
sim.add_applied_field(applied_field)
sim.add_diagnostic(field_diag)

# The key assertion of this regression test is simply that initialization completes
# without aborting in the divergence cleaner.
sim.initialize_inputs()
sim.initialize_warpx()
sim.step(1)

Bxg = sim.fields.get("B_external_particle_field", dir="x", level=0)
Byg = sim.fields.get("B_external_particle_field", dir="y", level=0)
Bzg = sim.fields.get("B_external_particle_field", dir="z", level=0)

Bx_local = Bxg[(), (), ()]
By_local = Byg[(), (), ()]
Bz_local = Bzg[(), (), ()]

dBxdx = (Bx_local[1:, :, :] - Bx_local[:-1, :, :]) / DX
dBydy = (By_local[:, 1:, :] - By_local[:, :-1, :]) / DY
dBzdz = (Bz_local[:, :, 1:] - Bz_local[:, :, :-1]) / DZ
divB = dBxdx + dBydy + dBzdz

div_norm = np.sqrt((divB[2:-2, 2:-2, 2:-2] ** 2).sum())
if comm.rank == 0:
    print(f"L2 div(B) = {div_norm:.6e}")

# A zero input field stays divergence free; the point is that we got here without aborting.
assert div_norm < 1e-12
