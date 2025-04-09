#!/usr/bin/env python3

from pywarpx import picmi

# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

# Number of time steps
max_steps = 84

# Number of cells
nx = 16
ny = 16
nz = 16

# Physical domain
xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0
zmin = 0.0
zmax = 2.0

# Create grid
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
)

# Particles
vel_z = 0.5 * c
offset_x_particle = picmi.ParticleListDistribution(
    x=[0.05], y=[0.0], z=[0.05], ux=[0.0], uy=[0.0], uz=[vel_z], weight=[1.0]
)

offset_y_particle = picmi.ParticleListDistribution(
    x=[0.0], y=[0.04], z=[0.05], ux=[0.0], uy=[0.0], uz=[vel_z], weight=[1.0]
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=[offset_x_particle, offset_y_particle],
)

# Plasma lenses
plasma_lenses = picmi.PlasmaLens(
    period=0.5,
    starts=[0.1, 0.11, 0.12, 0.13],
    lengths=[0.1, 0.11, 0.12, 0.13],
    strengths_E=[600000.0, 800000.0, 600000.0, 200000.0],
    strengths_B=[0.0, 0.0, 0.0, 0.0],
)

# Electromagnetic solver
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.7)

# Diagnostics
part_diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    period=max_steps,
    species=[electrons],
    data_list=["ux", "uy", "uz", "x", "y", "z"],
)

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
)
# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape="linear",
    warpx_serialize_initial_conditions=1,
    warpx_do_dynamic_scheduling=0,
)

# Add plasma electrons
sim.add_species(electrons, layout=None)

# Add the plasma lenses
sim.add_applied_field(plasma_lenses)

# Add diagnostics
sim.add_diagnostic(part_diag1)
sim.add_diagnostic(field_diag1)

# Write input file that can be used to run with the compiled version
# sim.write_input_file(file_name = 'inputs_3d_picmi')

# Advance simulation until last time step
sim.step(max_steps)
