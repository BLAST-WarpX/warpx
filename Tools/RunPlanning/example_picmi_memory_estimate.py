#!/usr/bin/env python3
# Copyright 2025 Marco Garten
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Example: Memory estimation from PICMI simulation object.

Demonstrates automatic memory estimation before simulation initialization,
allowing users to adjust parameters based on memory constraints.
"""

# Import memory estimator
from memory_calculator_picmi import estimate_from_picmi

from pywarpx import picmi

# ============================================================================
# Setup PICMI simulation
# ============================================================================

# Grid
nx, ny, nz = 256, 256, 256
xmin, ymin, zmin = -20e-6, -20e-6, -20e-6
xmax, ymax, zmax = 20e-6, 20e-6, 20e-6

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)

# Electromagnetic solver
solver = picmi.ElectromagneticSolver(
    grid=grid, method="Yee", cfl=1.0, divE_cleaning=1, warpx_pml_ncell=10
)

# Simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=1000,
    verbose=1,
    particle_shape="cubic",
    warpx_serialize_initial_conditions=1,
)

# Species
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=picmi.UniformDistribution(
        density=1e25,
        lower_bound=[xmin, ymin, zmin],
        upper_bound=[xmax, ymax, zmax],
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)

ions = picmi.Species(
    particle_type="H",
    name="H_ions",
    charge_state=0,
    initial_distribution=picmi.UniformDistribution(
        density=1e25,
        lower_bound=[xmin, ymin, zmin],
        upper_bound=[xmax, ymax, zmax],
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)

# Add species with layouts
sim.add_species(
    electrons, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2, 2])
)

sim.add_species(
    ions, layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[2, 2, 2])
)

# ============================================================================
# Estimate memory BEFORE initialization
# ============================================================================

print("\n" + "=" * 70)
print("Memory Estimation from PICMI Simulation")
print("IMPORTANT: Called BEFORE sim.initialize_inputs()")
print("=" * 70)

# Estimate memory
mem_info = estimate_from_picmi(sim, verbose=True)

# Check against target
target_memory_gb = 10.0  # Example: 10 GB per device
target_memory_bytes = target_memory_gb * 1e9

if mem_info["total"] > target_memory_bytes:
    print(f"\n⚠️  WARNING: Estimated memory ({mem_info['total'] / 1e9:.2f} GB) ")
    print(f"   exceeds target ({target_memory_gb:.2f} GB)")
    print("\n   Consider:")
    print("   - Reducing grid resolution")
    print("   - Decreasing particles per cell")
    print("   - Using fewer species")
    print("   - Increasing number of MPI ranks")
else:
    print(f"\n✓ Memory estimate ({mem_info['total'] / 1e9:.2f} GB) within target")
    print(f"  ({target_memory_gb:.2f} GB)")

# ============================================================================
# Access detailed information
# ============================================================================

print("\n" + "-" * 70)
print("Detailed Breakdown:")
print("-" * 70)

# Get field breakdown
mc = mem_info["calculator"]
field_breakdown = mc.get_field_breakdown()
print(f"\nField components: {field_breakdown}")
print(f"Total fields: {sum(field_breakdown.values())} MultiFabs")

# Species breakdown
print("\nSpecies breakdown:")
for species_name, mem in mem_info["species"].items():
    print(f"  {species_name:<20s}: {mem / 1e6:>10.2f} MB")

# ============================================================================
# Now initialize if memory is acceptable
# ============================================================================

if mem_info["total"] <= target_memory_bytes:
    print("\n" + "=" * 70)
    print("Proceeding with simulation initialization...")
    print("=" * 70)
    # sim.initialize_inputs()
    # sim.initialize_warpx()
    # sim.step(1000)
else:
    print("\nNot initializing due to memory constraints.")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
