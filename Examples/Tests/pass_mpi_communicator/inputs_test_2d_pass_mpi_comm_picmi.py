#!/usr/bin/env python3
#
# --- Test if MPI_communicator is correctly passed from Python to CPP
# --- This test should be run with MPI enabled
# --- Inputs taken from langmuir_2d. Runs 10 steps, and will check
# --- if the correct amount of processors are initialized in AMReX.

from mpi4py import MPI

from pywarpx import picmi

constants = picmi.constants

##########################
# MPI communicator setup
##########################

# split processor 0 into separate communicator from others
comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()
if rank == 0:
    color = 0
else:
    color = 1
new_comm = comm_world.Split(color)

##########################
# physics parameters
##########################

# different communicators will be passed different plasma density
plasma_density = [1.0e18, 1.0e19]
plasma_xmin = 0.0
plasma_x_velocity = 0.1 * constants.c

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 10
diagnostic_intervals = "::10"

# --- Grid
nx = 64
ny = 64

xmin = -20.0e-6
ymin = -20.0e-6
xmax = +20.0e-6
ymax = +20.0e-6

number_per_cell_each_dim = [2, 2]

##########################
# physics components
##########################

uniform_plasma = picmi.UniformDistribution(
    density=plasma_density[color],
    upper_bound=[0.0, None, None],
    directed_velocity=[0.1 * constants.c, 0.0, 0.0],
)

electrons = picmi.Species(
    particle_type="electron", name="electrons", initial_distribution=uniform_plasma
)

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    moving_window_velocity=[0.0, 0.0, 0.0],
    warpx_max_grid_size=32,
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=1.0)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name=f"diag{color + 1}",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["Ex", "Jx"],
    warpx_file_prefix=f"Python_pass_mpi_comm_plt{color + 1}_",
)

part_diag = picmi.ParticleDiagnostic(
    name=f"diag{color + 1}",
    period=diagnostic_intervals,
    species=[electrons],
    data_list=["weighting", "ux"],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    warpx_current_deposition_algo="direct",
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    ),
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)

##########################
# simulation run
##########################

# TODO: Enable in pyAMReX, then enable lines in inputs_test_2d_pass_mpi_comm_picmi.py again
# https://github.com/AMReX-Codes/pyamrex/issues/163

# sim.step(max_steps, mpi_comm=new_comm)

##########################
# test
##########################

# NOTE: Some tests are done in this input PICMI file. These tests
# check that amrex initialized the correct amount of procs and
# that the procs ranks in amrex are correct.
# If any of these tests fail, the terminal will print that the
# program crashed.

# TODO: Enable in pyAMReX, then enable lines in inputs_test_2d_pass_mpi_comm_picmi.py again
# https://github.com/AMReX-Codes/pyamrex/issues/163
# comm_world_size = comm_world.size
# new_comm_size = new_comm.size

# if color == 0:
#    # verify that communicator contains correct number of procs (1)
#    assert sim.extension.getNProcs() == comm_world_size - 1
#    assert sim.extension.getNProcs() == new_comm_size

# else:
#    # verify that amrex initialized with 1 fewer proc than comm world
#    assert sim.extension.getNProcs() == comm_world_size - 1
#    assert sim.extension.getNProcs() == new_comm_size

# verify that amrex proc ranks are offset by -1 from
# world comm proc ranks
#    assert libwarpx.amr.ParallelDescriptor.MyProc() == rank - 1
