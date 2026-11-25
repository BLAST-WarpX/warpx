#!/usr/bin/env python3
#
# A single proton gyrates in a uniform, linearly ramped B_z applied to the
# particles, with the theta-implicit scheme. By default the field is read from
# a file written by this script (with a time-dependent scaling); with
# --parsed it is given as an analytic expression. Both must give the same
# trajectory, with the field evaluated at the implicit stage time.

import argparse

import numpy as np
import openpmd_api as io
from mpi4py import MPI as mpi

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--parsed",
    action="store_true",
    help="Apply B_z as a parsed expression instead of a file map",
)
args = parser.parse_args()

# -------------------
# General parameters
# -------------------
max_steps = 200
dt = 1.0e-8  # seconds
nz = 8
zmin, zmax = 0.0, 1.0

B0 = 0.01  # tesla
t_ramp = max_steps * dt  # B_z doubles over the run
v0 = 1.0e4  # m/s

constants = picmi.constants

# -------------------
# Applied field
# -------------------
if args.parsed:
    applied_field = picmi.AnalyticAppliedField(
        Bx_expression=0.0,
        By_expression=0.0,
        Bz_expression=f"{B0}*(1 + t/{t_ramp})",
    )
else:
    field_file = "uniform_Bz_1d.h5"
    if mpi.COMM_WORLD.rank == 0:
        series = io.Series(field_file, io.Access.create)
        B = series.iterations[0].meshes["B"]
        B.geometry = io.Geometry.cartesian
        B.axis_labels = ["z"]
        B.grid_spacing = [(zmax - zmin) / nz]
        B.grid_global_offset = [zmin - 2.0 * (zmax - zmin) / nz]
        B.data_order = "C"
        B.unit_dimension = {
            io.Unit_Dimension.M: 1.0,
            io.Unit_Dimension.I: -1.0,
            io.Unit_Dimension.T: -2.0,
        }
        for comp, value in (("x", 0.0), ("y", 0.0), ("z", B0)):
            data = np.full(nz + 5, value)
            B[comp].reset_dataset(io.Dataset(data.dtype, data.shape))
            B[comp].position = [0.0]
            B[comp].unit_SI = 1.0
            B[comp].store_chunk(data)
        series.close()
    mpi.COMM_WORLD.Barrier()

    applied_field = picmi.LoadAppliedField(
        read_fields_from_path=field_file,
        load_E=False,
        load_B=True,
        warpx_B_time_function=f"1 + t/{t_ramp}",
    )

# -------------------
# Species
# -------------------
proton_dist = picmi.ParticleListDistribution(
    x=0.0, y=0.0, z=0.5, ux=v0, uy=0.0, uz=0.0, weight=1.0
)
protons = picmi.Species(
    particle_type="H",
    name="proton",
    charge="q_e",
    mass="m_p",
    warpx_do_not_deposit=1,
    initial_distribution=proton_dist,
)

# -------------------
# Grid & solver
# -------------------
grid = picmi.Cartesian1DGrid(
    number_of_cells=[nz],
    warpx_max_grid_size=nz,
    lower_bound=[zmin],
    upper_bound=[zmax],
    lower_boundary_conditions=["periodic"],
    upper_boundary_conditions=["periodic"],
    lower_boundary_conditions_particles=["periodic"],
    upper_boundary_conditions_particles=["periodic"],
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")

picard_solver = picmi.PicardNonlinearSolver(
    max_iterations=5,
    relative_tolerance=1.0e-12,
    absolute_tolerance=0.0,
    require_convergence=False,
)
evolve_scheme = picmi.ThetaImplicitEMEvolveScheme(
    theta=0.5, nonlinear_solver=picard_solver
)

# -------------------
# Diagnostics
# -------------------
# No output before the last step: a diagnostic flush refreshes the aux fields
# and would mask a missing aux update in the implicit solver.
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=f"{max_steps}:{max_steps}",
    species=[protons],
    data_list=["ux", "uy", "uz", "z", "weighting"],
)

# -------------------
# Simulation
# -------------------
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    time_step_size=dt,
    particle_shape=1,
    warpx_evolve_scheme=evolve_scheme,
    warpx_current_deposition_algo="direct",
    warpx_serialize_initial_conditions=1,
    warpx_use_filter=0,
)
sim.add_applied_field(applied_field)
sim.add_species(protons, layout=None)
sim.add_diagnostic(particle_diag)

sim.step(max_steps)
