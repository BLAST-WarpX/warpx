# -*- coding: utf-8 -*-


from pywarpx import picmi

constants = picmi.constants

gap = 0.1  # m
tVacuum = gap / 2
voltage = 1000

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 1

# --- Grid
nx = 8
nz = 128

xmin = 0.0
zmin = 0.0
xmax = gap * nx / nz
zmax = gap


##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    warpx_max_grid_size=64,
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    bc_xmin="neumann",
    bc_xmax="neumann",
    bc_ymin="dirichlet",
    bc_ymax="dirichlet",
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_potential_lo_z=0.0,
    warpx_potential_hi_z=voltage,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-5,
    warpx_absolute_tolerance=1e2,
    warpx_self_fields_verbosity=0,
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["E", "phi"],
    warpx_format="openpmd",
    # warpx_openpmd_backend="h5"
)

epsilon0 = constants.ep0
epsilon1 = constants.ep0 * 10.0
expression = "if(z>(tVacuum),10.0,1.0)"
epsilon = picmi.MacroscopicProperty(
    name="epsilon", implicit_function=expression, tVacuum=tVacuum
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=False,
    time_step_size=1e-9,
)

sim.add_diagnostic(field_diag)
sim.add_macroscopic_property(epsilon)
##########################
# simulation run
##########################
sim.initialize_inputs()
sim.initialize_warpx()

sim.step(max_steps)
