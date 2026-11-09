#!/usr/bin/env python3

import shutil
from pathlib import Path

from mpi4py import MPI

from pywarpx import picmi

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

# A uniform axial drift makes the deposited per-species current J_s finite,
# so its RZ inverse-volume scaling (including the on-axis cells) is covered
# alongside rho_s. The drift is non-relativistic and moves particles by
# v*dt = 1e-5 m << dz in the single step taken.
v_drift = 1.0e5  # m/s

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=picmi.constants.m_p,
    initial_distribution=picmi.UniformDistribution(
        density=n0,
        directed_velocity=[0.0, 0.0, v_drift],
    ),
)
simulation.add_species(
    ions,
    layout=picmi.GriddedLayout(
        grid=grid,
        n_macroparticle_per_cell=[2, 4, 2],
    ),
)
simulation.collisions = [
    picmi.HybridResistiveDragCollisions(name="ion_drag", species=ions)
]

if comm.rank == 0 and Path("diags").exists():
    shutil.rmtree("diags")
comm.Barrier()

field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=1,
    data_list=["rho_ions", "J"],
    write_dir="diags",
    warpx_file_prefix="field_diags",
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
simulation.add_diagnostic(field_diag)

simulation.initialize_inputs()
# The per-species current is registered as a vector field, which the
# diagnostics resolve per component through the register-mangled name
# (only the axial component is needed: the drift is purely axial). The
# name is quoted so the embedded '=' survives the AMReX inputs parser.
field_diag.diagnostic.additional_fields_to_plot = [
    "hybrid_rho_species_sum_fp",
    '"current_fp_ions[dir=z]"',
]
simulation.initialize_warpx()
simulation.step()
