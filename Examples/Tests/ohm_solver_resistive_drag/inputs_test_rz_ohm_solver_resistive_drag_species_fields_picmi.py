#!/usr/bin/env python3

import shutil
from pathlib import Path

from mpi4py import MPI

from pywarpx import picmi


class HybridResistiveDragCollision:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def collision_initialize_inputs(self):
        import pywarpx

        collision = pywarpx.Collisions.newcollision(self.name)
        collision.type = "hybrid_resistive_drag"
        collision.species = [species.name for species in self.species]


comm = MPI.COMM_WORLD
n0 = 2.0e20

grid = picmi.CylindricalGrid(
    number_of_cells=[16, 8],
    lower_bound=[0.0, -0.0025],
    upper_bound=[0.01, 0.0025],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["reflecting", "periodic"],
    warpx_max_grid_size=8,
)

# The per-species parser deliberately ignores rho_s and J_s, as a Te-only
# Spitzer parser does. Registering it and the drag activates the per-species
# deposition path whose SI units this test checks.
solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=100.0,
    n0=n0,
    n_floor=0.05 * n0,
    plasma_resistivity=0.0,
    plasma_resistivity_species={"ions": "1.0e-5 + 0.0*Te"},
    plasma_hyper_resistivity=0.0,
    substeps=2,
)

simulation = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-10,
    max_steps=1,
    particle_shape=1,
    warpx_current_deposition_algo="direct",
    warpx_serialize_initial_conditions=True,
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=picmi.constants.m_p,
    initial_distribution=picmi.UniformDistribution(
        density=n0,
        directed_velocity=[0.0, 0.0, 0.0],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.GriddedLayout(
        grid=grid,
        n_macroparticle_per_cell=[2, 4, 2],
    ),
)
simulation.collisions = [HybridResistiveDragCollision(name="ion_drag", species=[ions])]

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=1,
    data_list=["rho_ions"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

simulation.initialize_inputs()
field_diag.diagnostic.additional_fields_to_plot = ["hybrid_rho_species_sum_fp"]
simulation.initialize_warpx()
simulation.step()
