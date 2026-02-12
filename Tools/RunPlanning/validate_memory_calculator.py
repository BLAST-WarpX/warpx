#!/usr/bin/env python3
# Copyright 2025 Marco Garten
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Validation script for WarpX memory calculator.

This script demonstrates how to use the MemoryCalculator to estimate
memory requirements for WarpX simulations. It uses the 2D Langmuir
test as an example.

Usage:
    python validate_memory_calculator.py
"""

import numpy as np

from memory_calculator import MemoryCalculator as MC


def validate_langmuir_2d():
    """
    Validate memory calculator against the 2D Langmuir test.

    This test has:
    - 128x128 cells
    - 2 species (electrons and positrons)
    - 2x2 = 4 particles per cell per species
    - Periodic boundaries (no PML)
    - 2D geometry
    """
    print("=" * 60)
    print("Memory Calculator Validation: 2D Langmuir Test")
    print("=" * 60)

    # Grid parameters from inputs_base_2d
    Nx, Ny, Nz = 128, 128, 1  # 2D simulation
    sim_dim = 2

    # Create memory calculator
    mc = MC(Nx, Ny, Nz, build_dim=sim_dim, particle_shape_order=1, precision="double")

    # Field memory (periodic boundaries, no PML)
    field_mem = mc.mem_req_by_fields(
        Nx,
        Ny,
        Nz,
        divb_cleaning=False,
        dive_cleaning=False,
        pml_ncell=0,  # periodic boundaries
        num_mr_levels=1,  # no mesh refinement
        use_psatd=False,
    )

    # Particle memory
    # 2x2 particles per cell per species
    ppc = 2 * 2
    species_e_mem = mc.mem_req_by_species(
        Nx,
        Ny,
        Nz,
        particles_per_cell=ppc,
        enable_qed=False,
        enable_ionization=False,
    )
    species_p_mem = mc.mem_req_by_species(
        Nx,
        Ny,
        Nz,
        particles_per_cell=ppc,
        enable_qed=False,
        enable_ionization=False,
    )

    # RNG memory (for CPU test with OMP)
    rng_mem = mc.mem_req_by_rng(warpx_compute="OMP", omp_num_threads=4)

    # Total memory
    total_mem = field_mem + species_e_mem + species_p_mem + rng_mem

    # Convert to MB and GB
    MB = 1e6
    GB = 1e9

    print("\nMemory Breakdown:")
    print("-" * 60)
    print(f"  Fields:              {field_mem / MB:>10.2f} MB")
    print(f"  Electrons:           {species_e_mem / MB:>10.2f} MB")
    print(f"  Positrons:           {species_p_mem / MB:>10.2f} MB")
    print(f"  RNG states:          {rng_mem / MB:>10.2f} MB")
    print("-" * 60)
    print(f"  Total (per box):     {total_mem / MB:>10.2f} MB")
    print(f"                       {total_mem / GB:>10.4f} GB")
    print("=" * 60)

    print("\nNotes:")
    print("  - This is a lower-bound estimate (typically 70-90% of actual usage)")
    print("  - Actual usage includes AMReX overhead, load balancing, etc.")
    print("  - For multi-box simulations, multiply by number of boxes")
    print("  - Use LoadBalanceCosts diagnostic for runtime measurements")

    return total_mem


def demonstrate_gpu_variants():
    """
    Demonstrate memory calculator with different GPU models.
    """
    print("\n" + "=" * 60)
    print("GPU Model Comparison")
    print("=" * 60)

    Nx, Ny, Nz = 512, 512, 512
    mc = MC(Nx, Ny, Nz, build_dim=3, precision="double")

    gpu_models = ["A100", "H100", "V100", "MI250X"]

    print("\nRNG Memory for different GPU models (3D, 512^3 cells):")
    print("-" * 60)

    for gpu_model in gpu_models:
        rng_mem = mc.mem_req_by_rng(warpx_compute="CUDA", gpu_model=gpu_model)
        print(f"  {gpu_model:<10s}: {rng_mem / 1e6:>8.2f} MB")

    # Custom GPU specification
    custom_rng_mem = mc.mem_req_by_rng(
        warpx_compute="CUDA",
        gpu_multiprocessors=100,
        gpu_threads_per_multiprocessor=2048,
    )
    print(f"  {'Custom':<10s}: {custom_rng_mem / 1e6:>8.2f} MB  (100 MPs, 2048 threads/MP)")

    print("=" * 60)


def demonstrate_advanced_features():
    """
    Demonstrate advanced features: MR, QED, ionization, PSATD.
    """
    print("\n" + "=" * 60)
    print("Advanced Features Demonstration")
    print("=" * 60)

    Nx, Ny, Nz = 256, 256, 256
    mc = MC(Nx, Ny, Nz, build_dim=3, precision="double")

    # Base case
    base_field = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=0)

    # With mesh refinement
    mr_field = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=0, num_mr_levels=2)

    # With PSATD
    psatd_field = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=0, use_psatd=True)

    # With MR + PSATD
    mr_psatd_field = mc.mem_req_by_fields(
        Nx, Ny, Nz, pml_ncell=0, num_mr_levels=2, use_psatd=True
    )

    print("\nField Memory Comparison (256^3 cells):")
    print("-" * 60)
    print(f"  Base (no MR, FDTD):  {base_field / 1e6:>8.2f} MB")
    print(f"  With 2-level MR:     {mr_field / 1e6:>8.2f} MB  (+{(mr_field/base_field-1)*100:.1f}%)")
    print(f"  With PSATD:          {psatd_field / 1e6:>8.2f} MB  (+{(psatd_field/base_field-1)*100:.1f}%)")
    print(
        f"  With MR + PSATD:     {mr_psatd_field / 1e6:>8.2f} MB  (+{(mr_psatd_field/base_field-1)*100:.1f}%)"
    )

    # Particle features
    ppc = 10
    base_species = mc.mem_req_by_species(Nx, Ny, Nz, particles_per_cell=ppc)
    qed_species = mc.mem_req_by_species(
        Nx, Ny, Nz, particles_per_cell=ppc, enable_qed=True
    )
    ion_species = mc.mem_req_by_species(
        Nx, Ny, Nz, particles_per_cell=ppc, enable_ionization=True
    )

    print("\nParticle Memory per Species (256^3 cells, 10 ppc):")
    print("-" * 60)
    print(f"  Base:                {base_species / 1e6:>8.2f} MB")
    print(
        f"  With QED:            {qed_species / 1e6:>8.2f} MB  (+{(qed_species/base_species-1)*100:.1f}%)"
    )
    print(
        f"  With ionization:     {ion_species / 1e6:>8.2f} MB  (+{(ion_species/base_species-1)*100:.1f}%)"
    )

    print("=" * 60)


def demonstrate_solver_types():
    """
    Demonstrate memory differences between solver types.
    """
    print("\n" + "=" * 60)
    print("Solver Type Comparison")
    print("=" * 60)

    Nx, Ny, Nz = 256, 256, 256

    solver_types = [
        ("electromagnetic", "Full EM solver (E, B, J)"),
        ("electrostatic", "Electrostatic (E, phi, rho only)"),
        ("magnetostatic", "Magnetostatic (E, B from J)"),
        ("hybrid", "Hybrid-PIC (kinetic ions + fluid e-)"),
    ]

    print("\nField Memory for different solvers (256^3 cells):")
    print("-" * 60)

    for solver_type, description in solver_types:
        mc = MC(Nx, Ny, Nz, build_dim=3, precision="double", solver_type=solver_type)

        # Calculate field memory
        field_mem = mc.mem_req_by_fields(
            Nx, Ny, Nz, pml_ncell=0, dive_cleaning=False, divb_cleaning=False
        )

        # Get detailed breakdown
        breakdown = mc.get_field_breakdown()
        total_components = sum(breakdown.values())

        print(f"\n  {solver_type.upper():<20s} {description}")
        print(f"    Memory: {field_mem / 1e6:>8.2f} MB")
        print(f"    Components: {total_components} fields")
        print(f"    Breakdown: {', '.join(f'{k}={v}' for k, v in breakdown.items())}")

    print("\n" + "=" * 60)

    # Detailed example for electromagnetic
    print("\nDetailed breakdown for ELECTROMAGNETIC solver:")
    print("-" * 60)
    mc = MC(Nx, Ny, Nz, build_dim=3, solver_type="electromagnetic")

    # Base case
    base = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=0)
    base_breakdown = mc.get_field_breakdown()

    # With dive cleaning (adds F field)
    dive = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=0, dive_cleaning=True)
    dive_breakdown = mc.get_field_breakdown()

    # With PML
    pml = mc.mem_req_by_fields(Nx, Ny, Nz, pml_ncell=10, dive_cleaning=True)

    print(f"  Base (E,B,J):        {base / 1e6:>8.2f} MB - {list(base_breakdown.keys())}")
    print(
        f"  + div(E) cleaning:   {dive / 1e6:>8.2f} MB - {list(dive_breakdown.keys())}"
    )
    print(f"  + PML (10 cells):    {pml / 1e6:>8.2f} MB")

    print("=" * 60)


if __name__ == "__main__":
    # Run validation
    validate_langmuir_2d()

    # Demonstrate solver types
    demonstrate_solver_types()

    # Demonstrate GPU variants
    demonstrate_gpu_variants()

    # Demonstrate advanced features
    demonstrate_advanced_features()

    print("\nValidation complete!")
