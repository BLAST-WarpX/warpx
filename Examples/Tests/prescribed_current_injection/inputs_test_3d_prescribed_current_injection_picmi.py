#!/usr/bin/env python3
#
# PICMI interface for WarpX prescribed current injection.
#
# Demonstrates the WarpX-specific PICMI classes:
#   - picmi.PrescribedCurrentDrive
#   - picmi.PrescribedCurrentInjection
#   - Simulation.add_prescribed_current_injection
#
# Geometry: short copper-free vacuum box with a single drive face (Jx).
# The classic macroscopic-coil regression remains in
# inputs_test_3d_prescribed_current_injection (non-PICMI).
#
# Usage (pywarpx 3D build installed):
#   python inputs_test_3d_prescribed_current_injection_picmi.py
#
# Write classic inputs only (no C++ library load beyond geometry setup):
#   WARPX_PICMI_WRITE_ONLY=1 python inputs_test_3d_prescribed_current_injection_picmi.py

import os

from pywarpx import picmi

this_dir = os.path.dirname(os.path.abspath(__file__))
waveform = os.path.join(this_dir, "current_profile.txt")

##########################
# numerics
##########################

max_steps = 50

nx = ny = 40
nz = 24
xmin = ymin = -0.10
xmax = ymax = 0.10
zmin = 0.00
zmax = 0.12

# 1-cell-thick drive face (x) near the origin
xlo, xhi = -0.005, 0.000
ylo, yhi = -0.010, 0.010
zlo, zhi = 0.050, 0.070
# Face area A = dy * dz
A_drive = (yhi - ylo) * (zhi - zlo)

##########################
# grid / solver
##########################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_max_grid_size=32,
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
    cfl=0.9,
    warpx_pml_ncell=4,
)

##########################
# prescribed current (WarpX PICMI extension)
##########################

drive = picmi.PrescribedCurrentDrive(
    lower_bound=[xlo, ylo, zlo],
    upper_bound=[xhi, yhi, zhi],
    area=A_drive,
    direction=0,  # Jx
    sign=1,
)

current_injection = picmi.PrescribedCurrentInjection(
    drives=[drive],
    file=waveform,
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=-1,
    data_list=["Bz", "jx"],
    write_dir="diags",
    warpx_file_prefix="diag1",
)

##########################
# simulation
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    # Required: injection deposits through the particle current path.
    particle_shape="linear",
    warpx_current_deposition_algo="direct",
)

sim.add_prescribed_current_injection(current_injection)
sim.add_diagnostic(field_diag)

##########################
# run
##########################

if os.environ.get("WARPX_PICMI_WRITE_ONLY", "0") == "1":
    out = os.path.join(this_dir, "inputs_from_picmi")
    sim.write_input_file(file_name=out)
    print(f"[PICMI] wrote {out}")
else:
    sim.step(max_steps)
