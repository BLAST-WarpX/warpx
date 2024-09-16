#!/usr/bin/env python
# Copyright 2022 Marco Garten
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# requirements:
# - python, numpy

import numpy as np

from Tools.RunPlanning.memory_calculator import MemoryCalculator as MC

"""
@file

This file contains a usage example of the ``MemoryCalculator`` class for WarpX'
:ref:`laser-ion acceleration example <usage-examples>`.

It assume a single box within the whole simulation domain. Naturally, for a
production scale simulation, multiple MPI ranks would be assigned to subdomains
and the complex plasma dynamics make accurate estimations of later memory load
very difficult.
"""

# simulation dimension
sim_dim = 2
# density cut-off, where the target is centered around z = 0
L_cut = 2.0e-6
prob_lo = np.array([-7.5, -5.0]) * 1e-6  # simulation box: lower bounds
prob_hi = np.array([7.5, 25.0]) * 1e-6  # simulation box: upper bounds

Nx, Ny, Nz = 2688, 1, 3712  # box size in cells

cell_size_z = (prob_hi[1] - prob_lo[1]) / Nz

# extent of the target in cells
target_x = Nx
target_y = Ny
target_z = np.ceil(2 * L_cut / cell_size_z).astype(int)

mc = MC(Nx, Ny, Nz, build_dim=sim_dim)

# memory allocated for electromagnetic and helper fields
field_mem = mc.mem_req_by_fields(
    Nx,
    Ny,
    Nz,
    divb_cleaning=True,
    dive_cleaning=True,
    pml_ncell=10,  # PML boundary conditions are active in this example
)

# particles per cell
species_e_ppc = 2 * 2 * 4
species_H_ppc = species_e_ppc
# memory allocated per particle species
species_e_mem = mc.mem_req_by_species(
    target_x, target_y, target_z, particles_per_cell=species_e_ppc
)
species_H_mem = mc.mem_req_by_species(
    target_x,
    target_y,
    target_z,
    particles_per_cell=species_H_ppc,
    num_additional_ints=1,  # ionization state
)
# memory allocated mainly for the states of RNGs
rng_mem = mc.mem_req_by_rng(warpx_compute="CUDA", gpu_model="A100")


megabyte = 1e6
print("Memory requirement per box:")
print(r" + fields:")
print(r"   - total: {:.2f} MB".format(field_mem / megabyte))
print(" + species:")
print(r"   - e: {:.2f} MB".format(species_e_mem / megabyte))
print(r"   - H: {:.2f} MB".format(species_H_mem / megabyte))
print(r" + random number generation: states:")
print(r"   - states: {:.2f} MB".format(rng_mem / megabyte))
print(r"---------")
print(r"Total sum")
mem_sum = species_e_mem + species_H_mem + field_mem + rng_mem
print(r" {:.2f} MB".format(mem_sum / megabyte))
