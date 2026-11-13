#!/usr/bin/env python3
# Copyright 2025 Marco Garten
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
PICMI integration for WarpX memory calculator.

Provides utilities to automatically extract simulation parameters from
PICMI Simulation objects and estimate memory requirements.
"""

import numpy as np
from memory_calculator import MemoryCalculator


def estimate_from_picmi(sim, verbose=True):
    """
    Estimate memory requirements from a PICMI Simulation object.

    This should be called BEFORE sim.initialize_inputs() to allow
    users to adjust simulation parameters based on memory estimates.

    Parameters
    ----------
    sim : picmi.Simulation
        PICMI Simulation object (before initialize_inputs)
    verbose : bool, optional
        Print detailed memory breakdown (default: True)

    Returns
    -------
    dict
        Dictionary with memory estimates:
        {
            'fields': field_memory,
            'species': {'species_name': memory, ...},
            'rng': rng_memory,
            'total': total_memory,
            'calculator': MemoryCalculator instance
        }

    Example
    -------
    >>> from pywarpx import picmi
    >>> # Set up simulation
    >>> sim = picmi.Simulation(solver=solver, ...)
    >>> sim.add_species(electrons, layout=layout)
    >>>
    >>> # Estimate memory BEFORE initialization
    >>> from Tools.RunPlanning.memory_calculator_picmi import estimate_from_picmi
    >>> mem_info = estimate_from_picmi(sim)
    >>>
    >>> # Check if memory is acceptable
    >>> if mem_info['total'] > target_memory:
    >>> # Adjust parameters
    >>>     ...
    >>>
    >>> # Now initialize
    >>> sim.initialize_inputs()
    >>> sim.initialize_warpx()
    """
    # Extract solver type
    solver_type = _extract_solver_type(sim)

    # Extract grid info
    grid_info = _extract_grid_info(sim)

    # Determine build dimension
    build_dim = _get_build_dimension(grid_info)

    # Determine precision
    precision = _extract_precision(sim)

    # Create memory calculator
    mc = MemoryCalculator(
        n_x=grid_info["nx"],
        n_y=grid_info["ny"],
        n_z=grid_info["nz"],
        build_dim=build_dim,
        particle_shape_order=_extract_particle_shape(sim),
        precision=precision,
        solver_type=solver_type,
    )

    # Calculate field memory
    field_params = _extract_field_params(sim)
    field_mem = mc.mem_req_by_fields(
        grid_info["nx"],
        grid_info["ny"],
        grid_info["nz"],
        dive_cleaning=field_params.get("dive_cleaning", False),
        divb_cleaning=field_params.get("divb_cleaning", False),
        pml_ncell=field_params.get("pml_ncell", 0),
        num_mr_levels=grid_info.get("num_levels", 1),
        use_psatd=field_params.get("use_psatd", False),
    )

    # Calculate particle memory for each species
    species_mems = {}
    if hasattr(sim, "species"):
        for species in sim.species:
            species_name = species.name if hasattr(species, "name") else "unnamed"
            species_info = _extract_species_info(species, sim, grid_info)

            species_mem = mc.mem_req_by_species(
                target_n_x=species_info["target_nx"],
                target_n_y=species_info["target_ny"],
                target_n_z=species_info["target_nz"],
                particles_per_cell=species_info["ppc"],
                enable_qed=species_info.get("enable_qed", False),
                enable_ionization=species_info.get("enable_ionization", False),
                num_additional_reals=species_info.get("num_additional_reals", 0),
                num_additional_ints=species_info.get("num_additional_ints", 0),
            )
            species_mems[species_name] = species_mem

    # Calculate RNG memory
    rng_params = _extract_rng_params(sim)
    rng_mem = mc.mem_req_by_rng(**rng_params)

    # Total
    total_mem = field_mem + sum(species_mems.values()) + rng_mem

    # Print summary if requested
    if verbose:
        print("\n" + "=" * 60)
        print("PICMI Memory Estimate")
        print(f"Solver: {solver_type}, Dimension: {build_dim}D, Precision: {precision}")
        print("=" * 60)

        mc.print_summary(
            field_mem=field_mem, species_mems=species_mems, rng_mem=rng_mem
        )

        print("\nNote: Estimate before initialization. Adjust parameters if needed.")
        print("=" * 60)

    return {
        "fields": field_mem,
        "species": species_mems,
        "rng": rng_mem,
        "total": total_mem,
        "calculator": mc,
    }


def _extract_solver_type(sim):
    """Extract solver type from PICMI simulation."""
    # Check if solver exists and what type it is
    if hasattr(sim, "solver"):
        solver = sim.solver
        solver_class_name = solver.__class__.__name__

        if "Electrostatic" in solver_class_name:
            # Check if magnetostatic
            if hasattr(solver, "warpx_magnetostatic") and solver.warpx_magnetostatic:
                return "magnetostatic"
            return "electrostatic"
        elif "HybridPIC" in solver_class_name:
            return "hybrid"
        elif "Electromagnetic" in solver_class_name:
            return "electromagnetic"

    # Default to electromagnetic
    return "electromagnetic"


def _extract_grid_info(sim):
    """Extract grid dimensions from PICMI simulation."""
    grid = sim.solver.grid if hasattr(sim.solver, "grid") else None

    if grid is None:
        raise ValueError(
            "Cannot extract grid information from simulation. "
            "Make sure solver has a grid attribute."
        )

    # Extract number of cells
    nx = grid.number_of_cells[0] if len(grid.number_of_cells) > 0 else 1
    ny = grid.number_of_cells[1] if len(grid.number_of_cells) > 1 else 1
    nz = grid.number_of_cells[2] if len(grid.number_of_cells) > 2 else 1

    # Get number of refinement levels if available
    num_levels = 1
    if hasattr(grid, "max_level"):
        num_levels = grid.max_level + 1

    return {"nx": nx, "ny": ny, "nz": nz, "num_levels": num_levels}


def _get_build_dimension(grid_info):
    """Determine build dimension from grid info."""
    # Check which dimensions are active (> 1 cell)
    dims = sum([grid_info["nx"] > 1, grid_info["ny"] > 1, grid_info["nz"] > 1])

    if dims == 3:
        return 3
    elif dims == 2:
        return 2
    elif dims == 1:
        return 1
    else:
        return 3  # Default


def _extract_particle_shape(sim):
    """Extract particle shape order from simulation."""
    if hasattr(sim, "particle_shape"):
        shape_map = {"linear": 1, "quadratic": 2, "cubic": 3}
        return shape_map.get(sim.particle_shape, 3)
    return 3  # Default cubic


def _extract_precision(sim):
    """Extract precision from simulation (default: double)."""
    # PICMI typically uses double precision
    # Could be extended to check for specific flags if available
    return "double"


def _extract_field_params(sim):
    """Extract field-related parameters."""
    params = {}

    # Check for divergence cleaning
    if hasattr(sim.solver, "divE_cleaning"):
        params["dive_cleaning"] = sim.solver.divE_cleaning
    if hasattr(sim.solver, "divB_cleaning"):
        params["divb_cleaning"] = sim.solver.divB_cleaning

    # Check for PML
    params["pml_ncell"] = 0
    if hasattr(sim.solver, "warpx_pml_ncell"):
        params["pml_ncell"] = sim.solver.warpx_pml_ncell

    # Check for PSATD
    params["use_psatd"] = False
    if hasattr(sim.solver, "method"):
        params["use_psatd"] = sim.solver.method == "PSATD"

    return params


def _extract_species_info(species, sim, grid_info):
    """Extract species information including particles per cell."""
    info = {
        "target_nx": grid_info["nx"],
        "target_ny": grid_info["ny"],
        "target_nz": grid_info["nz"],
        "ppc": 1,
    }

    # Extract particles per cell from layout
    if hasattr(species, "layout"):
        layouts = (
            species.layout if isinstance(species.layout, list) else [species.layout]
        )

        # Sum particles from all layouts
        total_ppc = 0
        for layout in layouts:
            if hasattr(layout, "n_macroparticle_per_cell"):
                ppc = layout.n_macroparticle_per_cell
                if isinstance(ppc, (list, tuple)):
                    total_ppc += np.prod(ppc)
                else:
                    total_ppc += ppc
            elif hasattr(layout, "n_macroparticles_per_cell"):
                total_ppc += layout.n_macroparticles_per_cell

        if total_ppc > 0:
            info["ppc"] = int(total_ppc)

    # Check for ionization
    if hasattr(species, "particle_type"):
        if species.particle_type not in ["electron", "positron"]:
            info["enable_ionization"] = True

    # Check for QED (if species has QED-related attributes)
    if hasattr(species, "do_qed") and species.do_qed:
        info["enable_qed"] = True

    return info


def _extract_rng_params(sim):
    """Extract RNG parameters."""
    params = {"warpx_compute": "OMP", "omp_num_threads": 1}

    # Try to determine compute backend
    # This would need to be extended based on actual PICMI capabilities

    return params
