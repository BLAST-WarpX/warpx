#!/usr/bin/env python3
"""
Test for PR1: Verify MultiFabRegister checkpoint flag functionality

This test verifies:
1. Scalar fields can be marked for checkpointing
2. Vector field components can be marked for checkpointing
3. get_checkpoint_fields() returns correct list
4. Alias fields are excluded from checkpoint list
"""

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

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
    max_steps=1,
    verbose=1
)

chk = picmi.Checkpoint(name="checkpoint_flag_test", period=1)
sim.add_diagnostic(chk)

def test_checkpoint_flags():
    """Test checkpoint flag functionality"""
    print("\n" + "="*60)
    print("Testing PR1: Checkpoint Flag Functionality")
    print("="*60)

    # Test 0: has() / has_vector() on non-existent fields
    print("\nTest 0: has() / has_vector() on non-existent fields")
    assert not sim.fields.has("nonexistent_field", level=0), \
        "has() should return False for non-existent scalar field"
    assert not sim.fields.has_vector("nonexistent_vector", level=0), \
        "has_vector() should return False for non-existent vector field"
    print("  [OK] has() and has_vector() return False for non-existent fields")

    # Get Ex field as reference
    Ex = sim.fields.get("Efield_fp", dir='x', level=0)

    # Test 1: Allocate a scalar field and mark for checkpoint
    print("\nTest 1: Scalar field checkpoint flag")
    sim.fields.alloc_init(
        name="test_scalar",
        level=0,
        ba=Ex.box_array(),
        dm=Ex.dm(),
        ncomp=1,
        ngrow=Ex.n_grow_vect,
        initial_value=0.,
        redistribute=True,
        redistribute_on_remake=True
    )

    # Verify has() returns True now that the field is allocated
    assert sim.fields.has("test_scalar", level=0), \
        "has() should return True after alloc_init"
    print("  [OK] has('test_scalar') returns True")

    # Mark for checkpoint
    sim.fields.set_checkpoint("test_scalar", level=0, checkpoint=True)
    print("  [OK] Scalar field 'test_scalar' marked for checkpoint")

    # Test 2: Allocate vector field components and mark for checkpoint
    print("\nTest 2: Vector field checkpoint flags")
    for idir in range(3):
        dirstr = ['x', 'y', 'z'][idir]
        sim.fields.alloc_init(
            name="test_vector",
            dir=dirstr,
            level=0,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=0.,
            redistribute=True,
            redistribute_on_remake=True
        )
        sim.fields.set_checkpoint("test_vector", dir=dirstr, level=0, checkpoint=True)
        print(f"  [OK] Vector component 'test_vector_{dirstr}' marked for checkpoint")

    # Verify has() with direction and has_vector()
    assert sim.fields.has("test_vector", dir='x', level=0), \
        "has('test_vector', dir='x') should be True"
    assert sim.fields.has_vector("test_vector", level=0), \
        "has_vector('test_vector') should be True after all 3 components are allocated"
    print("  [OK] has() with direction and has_vector() return True")

    # Test 3: Query checkpoint fields
    print("\nTest 3: Query checkpoint fields")
    checkpoint_fields = sim.fields.get_checkpoint_fields(level=0)
    print(f"  Found {len(checkpoint_fields)} fields marked for checkpoint at level 0:")

    for field_name, dir_opt in checkpoint_fields:
        if dir_opt is None:
            print(f"    - {field_name} (scalar)")
        else:
            print(f"    - {field_name} (vector, dir={str(dir_opt)})")

    # Verify our test fields are in the list
    # Scalar field: name only
    field_names = [name for name, _ in checkpoint_fields]
    assert "test_scalar" in field_names, "test_scalar not in checkpoint list!"

    # Vector field: check base name with each direction
    vector_entries = [(name, dir_opt) for name, dir_opt in checkpoint_fields if name == "test_vector"]
    assert len(vector_entries) == 3, f"Expected 3 test_vector components, found {len(vector_entries)}"

    directions_found = [str(dir_opt) for _, dir_opt in vector_entries if dir_opt is not None]
    assert "x" in directions_found, "test_vector direction x not found!"
    assert "y" in directions_found, "test_vector direction y not found!"
    assert "z" in directions_found, "test_vector direction z not found!"

    print("\n  [OK] All test fields found in checkpoint list")

    # Test 4: Create an alias and verify it's not in checkpoint list
    print("\nTest 4: Verify aliases are excluded from checkpoints")
    sim.fields.alias_init(
        new_name="test_scalar_alias",
        alias_name="test_scalar",
        level=0,
        #initial_value=0.
    )

    # Get checkpoint fields again
    checkpoint_fields_after = sim.fields.get_checkpoint_fields(level=0)
    field_names_after = [name for name, _ in checkpoint_fields_after]

    # Alias should NOT be in the list (even though owner is)
    assert "test_scalar_alias" not in field_names_after, "Alias should not be in checkpoint list!"
    assert "test_scalar" in field_names_after, "Owner should still be in checkpoint list!"

    print("  [OK] Alias correctly excluded from checkpoint list")


def test_remove_field_from_checkpoint():
    """Remove fields from checkpoint after step 1"""

    step = sim.extension.warpx.getistep(lev=0)
    if step == 1:

        # Test 5: Unmark a field
        print("\nTest 5: Unmark field from checkpoint")
        sim.fields.set_checkpoint("test_scalar", level=0, checkpoint=False)
        checkpoint_fields_unmarked = sim.fields.get_checkpoint_fields(level=0)
        field_names_unmarked = [name for name, _ in checkpoint_fields_unmarked]
        assert "test_scalar" not in field_names_unmarked, "Unmarked field still in checkpoint list!"
        print("  [OK] Field successfully unmarked")

        print("\n" + "="*60)
        print("All PR1 tests PASSED!")
        print("="*60 + "\n")


installafterInitEsolve(test_checkpoint_flags)
installafterstep(test_remove_field_from_checkpoint)

# Run simulation
sim.step(2)
