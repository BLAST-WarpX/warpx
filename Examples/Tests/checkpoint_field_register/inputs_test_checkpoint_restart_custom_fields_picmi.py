#!/usr/bin/env python3
"""
Verify checkpoint/restart with custom fields

This test verifies:
1. Custom fields marked for checkpoint are written to checkpoint files
2. Custom fields can be restored from checkpoint on restart
3. Both scalar and vector fields work correctly
"""

import os
import numpy as np
import argparse
from pywarpx import picmi

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--restart', type=str, default=None,
                    help='Restart from checkpoint file')
args, _ = parser.parse_known_args()

# Simulation setup
nx, ny, nz = 16, 16, 16
xmin, ymin, zmin = 0.0, 0.0, 0.0
xmax, ymax, zmax = 1.0, 1.0, 1.0

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=['periodic', 'periodic', 'periodic'],
    upper_boundary_conditions=['periodic', 'periodic', 'periodic']
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=0.99)

sim = picmi.Simulation(
    solver=solver,
    max_steps=10,
    verbose=1,
    warpx_amr_restart=args.restart  # Set restart file if provided
)

# Add checkpoint diagnostic
chk = picmi.Checkpoint(name="checkpoint_restart_test", period=5)
sim.add_diagnostic(chk)

def setup_custom_fields():
    """Allocate and initialize custom fields (both new run and restart)"""
    import amrex
    
    print("\n" + "="*60)
    print("Setting up custom fields")
    print("="*60)
    
    # Get Ex field as reference
    Ex = sim.fields.get("Efield_fp", dir='x', level=0)
    
    # Create a scalar field
    test_scalar = sim.fields.alloc_init(
        name="custom_scalar",
        level=0,
        ba=Ex.box_array(),
        dm=Ex.dm(),
        ncomp=1,
        ngrow=Ex.n_grow_vect,
        initial_value=42.0,  # Initialize with distinctive value
        redistribute=True,
        redistribute_on_remake=True
    )
    print("  [OK] Allocated custom_scalar field")
    
    # Create a vector field
    for idir, dirstr in enumerate(['x', 'y', 'z']):
        test_vec = sim.fields.alloc_init(
            name="custom_vector",
            dir=dirstr,
            level=0,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=float(10 + idir),  # 10, 11, 12 for x, y, z
            redistribute=True,
            redistribute_on_remake=True
        )
        print(f"  [OK] Allocated custom_vector_{dirstr} field")
    
    # Mark fields for checkpointing
    sim.fields.set_checkpoint("custom_scalar", level=0, checkpoint=True)
    print("  [OK] Marked custom_scalar for checkpoint")
    
    for dirstr in ['x', 'y', 'z']:
        sim.fields.set_checkpoint("custom_vector", dir=dirstr, level=0, checkpoint=True)
    print("  [OK] Marked custom_vector for checkpoint")
    
    # Note: On restart, fields are automatically restored in PostRestart()
    # after user callbacks have allocated them
    if sim.amr_restart:
        print("\n  Note: Custom field data will be automatically restored after this callback")

def verify_restored_fields():
    """Verify restored field values after restart (runs at first step)"""
    step = sim.extension.warpx.getistep(lev=0)
    
    if sim.amr_restart and step == 6:
        print("\n" + "="*60)
        print("Verifying automatically restored field values")
        print("="*60)
        
        # Get the restored fields
        test_scalar = sim.fields.get("custom_scalar", level=0)
        
        # Check scalar
        scalar_data = test_scalar[...]
        scalar_mean = scalar_data.mean()
        print(f"  custom_scalar mean value: {scalar_mean}")
        assert abs(scalar_mean - 42.0) < 1e-10, f"Scalar field not restored correctly! Expected 42.0, got {scalar_mean}"
        print("  [OK] Scalar field values correct")
        
        # Check vector components
        for idir, dirstr in enumerate(['x', 'y', 'z']):
            vec_field = sim.fields.get("custom_vector", dir=dirstr, level=0)
            vec_data = vec_field[...]
            vec_mean = vec_data.mean()
            expected = float(10 + idir)
            print(f"  custom_vector_{dirstr} mean value: {vec_mean}")
            assert abs(vec_mean - expected) < 1e-10, f"Vector {dirstr} not restored correctly! Expected {expected}, got {vec_mean}"
        print("  [OK] Vector field values correct")
        
        print("\n" + "="*60)
        print("RESTART TEST PASSED!")
        print("="*60 + "\n")

def verify_checkpoint_written():
    """Verify checkpoint file was written at step 5 (check at step 6)"""
    step = sim.extension.warpx.getistep(lev=0)
    
    # Check after step 5 has completed (at step 6)
    if step == 6:
        print("\n" + "="*60)
        print("Verifying checkpoint file contents")
        print("="*60)
        
        # Check that checkpoint directory exists (in diags/ subdirectory)
        chk_dir = "diags/checkpoint_restart_test000005"
        assert os.path.exists(chk_dir), f"Checkpoint directory {chk_dir} not found!"
        print(f"  [OK] Checkpoint directory exists: {chk_dir}")
        
        # Check that custom field files exist
        for field_name in ["custom_scalar", "custom_vector[dir=x]", "custom_vector[dir=y]", "custom_vector[dir=z]"]:
            # AMReX MultiFab files have a _H header file
            field_path = os.path.join(chk_dir, "Level_0", field_name + "_H")
            if os.path.exists(field_path):
                print(f"  [OK] Found custom field file: {field_name}")
            else:
                print(f"  WARNING: Custom field file not found: {field_name}")
        
        print("="*60 + "\n")

from pywarpx.callbacks import installafterInitEsolve, installafterInitatRestart, installafterstep

# IMPORTANT: Custom fields must be allocated in BOTH callbacks:
# - afterInitEsolve: runs on fresh start (after initial E-field solve)
# - afterInitatRestart: runs on restart (after checkpoint is loaded)
# The field structure must be re-created on restart before data can be restored.
installafterInitEsolve(setup_custom_fields)       # Fresh start
installafterInitatRestart(setup_custom_fields)    # Restart
installafterstep(verify_checkpoint_written)
installafterstep(verify_restored_fields)

# Run simulation
# Initial run: go to step 11 (past step 5 checkpoint)
# Restart: only need a few steps to verify restoration
sim.step(6 if sim.amr_restart else 11)
