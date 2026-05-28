#!/usr/bin/env python3
"""
Test for add_field_to_diagnostic: verify that custom fields added via
add_field_to_diagnostic() appear in openPMD output with correct values.

This test verifies:
1. A custom scalar MultiFab added to diagnostic output appears in the openPMD file
2. A custom vector MultiFab (x/y/z components) appears as separate scalar meshes
3. Field values in the file match what was set before the step
4. sim.fields.has() and sim.fields.has_vector() return correct results
"""

import numpy as np
import openpmd_api as io

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

# Minimal simulation setup
nx, ny, nz = 8, 8, 8
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1.0, 1.0, 1.0],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=0.99)

sim = picmi.Simulation(solver=solver, max_steps=1, verbose=0)

# FieldDiagnostic in openPMD/HDF5 format, writing at every step
diag = picmi.FieldDiagnostic(
    name="field_output_test",
    grid=grid,
    period=1,
    data_list=["Ex"],  # at least one standard field
    write_dir="diags",
    warpx_format="openpmd",
    warpx_file_prefix="field_output_test",
    warpx_openpmd_backend="h5",
)
sim.add_diagnostic(diag)

# Known values used to initialise the custom fields
SCALAR_VALUE = 7.0
VECTOR_VALUES = {"x": 1.0, "y": 2.0, "z": 3.0}


def setup_custom_fields():
    """Allocate custom fields and register them for diagnostic output."""
    Ex = sim.fields.get("Efield_fp", dir="x", level=0)
    ba = Ex.box_array()
    dm = Ex.dm()
    ng = Ex.n_grow_vect

    # --- has() on non-existent field must return False before allocation ---
    assert not sim.fields.has("custom_scalar", level=0), (
        "custom_scalar should not exist yet"
    )
    assert not sim.fields.has_vector("custom_vector", level=0), (
        "custom_vector should not exist yet"
    )

    # --- Allocate scalar field ---
    sim.fields.alloc_init(
        name="custom_scalar",
        level=0,
        ba=ba,
        dm=dm,
        ncomp=1,
        ngrow=ng,
        initial_value=SCALAR_VALUE,
        redistribute=True,
        redistribute_on_remake=True,
    )
    assert sim.fields.has("custom_scalar", level=0), (
        "custom_scalar must exist after alloc_init"
    )
    print(f"  [OK] custom_scalar allocated with value {SCALAR_VALUE}")

    # --- Allocate vector field ---
    for dirstr, value in VECTOR_VALUES.items():
        sim.fields.alloc_init(
            name="custom_vector",
            dir=dirstr,
            level=0,
            ba=ba,
            dm=dm,
            ncomp=1,
            ngrow=ng,
            initial_value=value,
            redistribute=True,
            redistribute_on_remake=True,
        )
    assert sim.fields.has_vector("custom_vector", level=0), (
        "custom_vector must exist after allocating all three components"
    )
    print(f"  [OK] custom_vector allocated with values {VECTOR_VALUES}")

    # --- Register both fields for diagnostic output ---
    wx = sim.extension.warpx
    wx.add_field_to_diagnostic("field_output_test", "custom_scalar", 0)
    wx.add_field_to_diagnostic("field_output_test", "custom_vector", 0)
    print("  [OK] Fields registered with diagnostic 'field_output_test'")


installafterInitEsolve(setup_custom_fields)

# -------------------------------------------------------------------------
# Run 1 step — diagnostic writes at step 1
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Testing add_field_to_diagnostic with openPMD/HDF5 output")
print("=" * 60)

sim.step(1)

# -------------------------------------------------------------------------
# Read back the openPMD file and verify field names + values
# -------------------------------------------------------------------------
print("\nVerifying openPMD output ...")
series = io.Series("diags/field_output_test/openpmd_%06T.h5", io.Access.read_only)
it = series.iterations[1]

mesh_names = list(it.meshes)
print(f"  Meshes in output: {mesh_names}")

# --- Scalar field ---
assert "custom_scalar" in mesh_names, (
    f"custom_scalar not found in meshes. Available: {mesh_names}"
)

scalar_data = it.meshes["custom_scalar"][io.Mesh_Record_Component.SCALAR].load_chunk()
series.flush()
scalar_mean = float(np.mean(scalar_data))
assert abs(scalar_mean - SCALAR_VALUE) < 1e-10, (
    f"custom_scalar mean {scalar_mean} != expected {SCALAR_VALUE}"
)
print(f"  [OK] custom_scalar: mean = {scalar_mean} (expected {SCALAR_VALUE})")

# --- Vector field components (stored as flat scalar meshes) ---
for dirstr, expected in VECTOR_VALUES.items():
    comp_name = f"custom_vector_{dirstr}"
    assert comp_name in mesh_names, (
        f"{comp_name} not found in meshes. Available: {mesh_names}"
    )

    comp_data = it.meshes[comp_name][io.Mesh_Record_Component.SCALAR].load_chunk()
    series.flush()
    comp_mean = float(np.mean(comp_data))
    assert abs(comp_mean - expected) < 1e-10, (
        f"{comp_name} mean {comp_mean} != expected {expected}"
    )
    print(f"  [OK] {comp_name}: mean = {comp_mean} (expected {expected})")

series.close()

print("\n" + "=" * 60)
print("add_field_to_diagnostic test PASSED")
print("=" * 60 + "\n")
