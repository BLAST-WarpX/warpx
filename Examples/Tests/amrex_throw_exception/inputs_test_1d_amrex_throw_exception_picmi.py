#!/usr/bin/env python3
#
# --- Regression test for https://github.com/BLAST-WarpX/warpx/issues/6741
# --- Verify that amrex.throw_exception defaults to true, so that AMReX
# --- assertions raise a Python RuntimeError instead of calling amrex::Abort().

from pywarpx import picmi, warpx

# Minimal 1D simulation
grid = picmi.Cartesian1DGrid(
    number_of_cells=[8],
    lower_bound=[0.0],
    upper_bound=[1.0],
    lower_boundary_conditions=["periodic"],
    upper_boundary_conditions=["periodic"],
)
solver = picmi.ElectromagneticSolver(grid=grid)
sim = picmi.Simulation(
    solver=solver,
    max_steps=1,
    time_step_size=1e-12,
    particle_shape=1,
)

# Set an invalid parameter to trigger an AMReX assertion.
# v_particle_pml must be in (0, 1]; 2.0 is out of range.
# This assertion is in the WarpX constructor (Source/WarpX.cpp:1004).
warpx.v_particle_pml = 2.0

# With amrex.throw_exception=true (the new default per #6741),
# this assertion raises a Python RuntimeError.
# With amrex.throw_exception=false (old default), it would call
# amrex::Abort() and kill the process.
try:
    sim.initialize_inputs()
    sim.initialize_warpx()
    raise AssertionError("Expected RuntimeError was not raised")
except RuntimeError as e:
    assert "v_particle_pml" in str(e), (
        f"Error message should mention 'v_particle_pml', got: {e}"
    )
    print(f"PASS: RuntimeError raised as expected: {e}")
