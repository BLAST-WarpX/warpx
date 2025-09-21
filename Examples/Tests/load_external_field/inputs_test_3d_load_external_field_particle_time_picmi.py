#!/usr/bin/env python3
from math import pi

from pywarpx import picmi

# -------------------
# General parameters
# -------------------
max_steps = 300
max_grid_size = 40
nx = ny = nz = max_grid_size

xmin, xmax = -1.0, 1.0
ymin, ymax = xmin, xmax
zmin, zmax = 0.0, 5.0

number_per_cell = 200
verbose = 1
dt = 4.4e-7  # seconds

# Choose omega so that cos(omega * (max_steps*dt)) = 0.5
t_end = max_steps * dt
omega = (pi / 3.0) / t_end  # rad/s
phase = 0.0

# -------------------
# Species
# -------------------
constants = picmi.constants
ion_dist = picmi.ParticleListDistribution(
    x=0.0,
    y=0.2,
    z=2.5,
    ux=9.5e-05 * constants.c,
    uy=0.0 * constants.c,
    uz=1.34e-4 * constants.c,
    weight=1.0,
)
ions = picmi.Species(
    particle_type="H",
    name="proton",
    charge="q_e",
    mass="m_p",
    warpx_do_not_deposit=1,
    initial_distribution=ion_dist,
)

# -------------------
# Applied field (file)
# -------------------
applied_field = picmi.LoadAppliedField(
    read_fields_from_path="../../../../openPMD-example-datasets/example-femm-3d.h5",
    load_E=False,
    load_B=True,
    warpx_B_time_function=f"cos({omega}*t + {phase})",
)

print(
    f"[PICMI] B time function: warpx_B_time_function = cos({omega}*t + {phase}) ; t_end={t_end}"
)

# -------------------
# Grid & solver
# -------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)
solver = picmi.ElectrostaticSolver(grid=grid)

# -------------------
# Diagnostics
# -------------------
field_diag = picmi.FieldDiagnostic(
    name="diag1", grid=grid, period=max_steps, data_list=["Bx", "By", "Bz"]
)
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=max_steps,
    species=[ions],
    data_list=["ux", "uy", "uz", "x", "y", "z", "weighting"],
)

# -------------------
# Simulation
# -------------------
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=verbose,
    warpx_serialize_initial_conditions=False,
    warpx_grid_type="collocated",
    warpx_do_dynamic_scheduling=False,
    warpx_use_filter=0,
    time_step_size=dt,
    particle_shape=1,
)
sim.add_applied_field(applied_field)
sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=number_per_cell, grid=grid
    ),
)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_diag)

# Run
sim.step(max_steps)
