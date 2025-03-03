#!/usr/bin/env python3
import numpy as np

from pywarpx import fields, picmi

max_steps = 1
unit = 1e-3

##################################
# Define the mesh
##################################
# mesh cells per direction
nx = 64
ny = 64
nz = 64

# mesh bounds for domain
xmin = -32 * unit
xmax = 32 * unit
ymin = -32 * unit
ymax = 32 * unit
zmin = -32 * unit
zmax = 32 * unit

##########################
# numerics components
##########################
lower_boundary_conditions = ["open", "dirichlet", "periodic"]
upper_boundary_conditions = ["open", "dirichlet", "periodic"]

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=lower_boundary_conditions,
    upper_boundary_conditions=upper_boundary_conditions,
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    moving_window_velocity=None,
    warpx_max_grid_size=64,
)


flag_correct_div = False

solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=1.0)

n_cavity = 30
L_cavity = n_cavity * unit

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="max(max(max(x-L_cavity/2,-L_cavity/2-x),max(y-L_cavity/2,-L_cavity/2-y)),max(z-L_cavity/2,-L_cavity/2-z))",
    L_cavity=L_cavity,
)


##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=1,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["Ex"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    warpx_embedded_boundary=embedded_boundary,
    verbose=1,
)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

sim.initialize_inputs()

sim.step(1)

edge_lengths_x = fields.EdgeLengthsxWrapper()
edge_lengths_y = fields.EdgeLengthsyWrapper()
edge_lengths_z = fields.EdgeLengthszWrapper()
face_areas_x = fields.FaceAreasxWrapper()
face_areas_y = fields.FaceAreasyWrapper()
face_areas_z = fields.FaceAreaszWrapper()

print("======== Testing the wrappers of edge_lengths =========")

ly_slice_x = edge_lengths_y[nx // 2, :, :]
lz_slice_x = edge_lengths_z[nx // 2, :, :]

n_edge_y_lo = (ny - 30) // 2
n_edge_y_hi = ny - (ny - 30) // 2
n_edge_z_lo = (nz - 30) // 2
n_edge_z_hi = nz - (nz - 30) // 2

perimeter_slice_x = (
    np.sum(ly_slice_x[n_edge_y_lo:n_edge_y_hi, n_edge_z_lo + 1])
    + np.sum(ly_slice_x[n_edge_y_lo:n_edge_y_hi, n_edge_z_hi - 1])
    + np.sum(lz_slice_x[n_edge_y_lo + 1, n_edge_z_lo:n_edge_z_hi])
    + np.sum(lz_slice_x[n_edge_y_hi - 1, n_edge_z_lo:n_edge_z_hi])
)

perimeter_slice_x_true = L_cavity * 4

print("Perimeter of the middle x-slice:", perimeter_slice_x)
assert np.isclose(perimeter_slice_x, perimeter_slice_x_true, rtol=1e-05, atol=1e-08)


lx_slice_y = edge_lengths_x[:, ny // 2, :]
lz_slice_y = edge_lengths_z[:, ny // 2, :]

n_edge_x_lo = (nx - 30) // 2
n_edge_x_hi = nx - (nx - 30) // 2
n_edge_z_lo = (nz - 30) // 2
n_edge_z_hi = nz - (nz - 30) // 2

perimeter_slice_y = (
    np.sum(lx_slice_y[n_edge_x_lo:n_edge_x_hi, n_edge_z_lo + 1])
    + np.sum(lx_slice_y[n_edge_x_lo:n_edge_x_hi, n_edge_z_hi - 1])
    + np.sum(lz_slice_y[n_edge_x_lo + 1, n_edge_z_lo:n_edge_z_hi])
    + np.sum(lz_slice_y[n_edge_x_hi - 1, n_edge_z_lo:n_edge_z_hi])
)

perimeter_slice_y_true = L_cavity * 4


print("Perimeter of the middle y-slice:", perimeter_slice_y)
assert np.isclose(perimeter_slice_y, perimeter_slice_y_true, rtol=1e-05, atol=1e-08)


lx_slice_z = edge_lengths_x[:, :, nz // 2]
ly_slice_z = edge_lengths_y[:, :, nz // 2]

n_edge_x_lo = (nx - 30) // 2
n_edge_x_hi = nx - (nx - 30) // 2
n_edge_y_lo = (ny - 30) // 2
n_edge_y_hi = ny - (ny - 30) // 2

perimeter_slice_z = (
    np.sum(lx_slice_z[n_edge_x_lo:n_edge_x_hi, n_edge_y_lo + 1])
    + np.sum(lx_slice_z[n_edge_x_lo:n_edge_x_hi, n_edge_y_hi - 1])
    + np.sum(ly_slice_z[n_edge_x_lo + 1, n_edge_y_lo:n_edge_y_hi])
    + np.sum(ly_slice_z[n_edge_x_hi - 1, n_edge_y_lo:n_edge_y_hi])
)

perimeter_slice_z_true = L_cavity * 4

print("Perimeter of the middle z-slice:", perimeter_slice_z)
assert np.isclose(perimeter_slice_z, perimeter_slice_z_true, rtol=1e-05, atol=1e-08)

print("======== Testing the wrappers of face_areas =========")

Sx_slice = np.sum(face_areas_x[nx // 2, :, :])
dx = (xmax - xmin) / nx
Ax = dx * dx
Sx_slice_true = L_cavity * L_cavity - 2 * Ax
print("Area of the middle x-slice:", Sx_slice)
assert np.isclose(Sx_slice, Sx_slice_true, rtol=1e-05, atol=1e-08)

Sy_slice = np.sum(face_areas_y[:, ny // 2, :])
dy = (ymax - ymin) / ny
Ay = dy * dy
Sy_slice_true = L_cavity * L_cavity - 2 * Ay
print("Area of the middle y-slice:", Sx_slice)
assert np.isclose(Sy_slice, Sy_slice_true, rtol=1e-05, atol=1e-08)

Sz_slice = np.sum(face_areas_z[:, :, nz // 2])
dz = (zmax - zmin) / nz
Az = dz * dz
Sz_slice_true = L_cavity * L_cavity - 2 * Az
print("Area of the middle z-slice:", Sz_slice)
assert np.isclose(Sz_slice, Sz_slice_true, rtol=1e-05, atol=1e-08)

sim.step(1)
