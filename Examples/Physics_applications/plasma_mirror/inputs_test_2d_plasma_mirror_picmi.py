from pywarpx import picmi

# -----------------------------
# Simulation Domain Setup
# -----------------------------

# 2D grid parameters (x, z)
nx = 256            # number of cells in x
nz = 512            # number of cells in z

# Domain extents (in meters)
xmin = -30e-6
xmax =  30e-6
zmin = -56e-6
zmax =  12e-6

# Boundary conditions for fields:
# Here we use periodic in x and Dirichlet in z (uncertain – adjust if needed)
lower_field_bc = ['periodic', 'dirichlet']
upper_field_bc = ['periodic', 'dirichlet']

# Boundary conditions for particles:
# Particles are set to be periodic in x and absorbing in z
lower_particle_bc = ['periodic', 'absorbing']
upper_particle_bc = ['periodic', 'absorbing']

# Domain decomposition parameters (tunable for performance)
max_grid_size = 64       # maximum grid block size
blocking_factor = 32     # blocking factor (may be tuned)

# Create the 2D Cartesian grid for simulation
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=lower_field_bc,
    upper_boundary_conditions=upper_field_bc,
    lower_boundary_conditions_particles=lower_particle_bc,
    upper_boundary_conditions_particles=upper_particle_bc,
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor,
    # No moving window is used in this plasma mirror setup
)

